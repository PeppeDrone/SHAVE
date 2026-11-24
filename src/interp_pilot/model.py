"""Model loading, hooked forward passes, weight folds and LoRA training helpers.

The classifier reads the last prompt token: margin = logit(" 1") - logit(" 0").
Sites: "embed" (input-embedding output), int L (output of decoder layer L), "readout" (lm_head input,
last position). Positions: "last", "all", a field name (all token positions of the field), or
"<field>@k" (k-th token of the field span, clipped to the span).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from . import civil as C


# ---------------------------------------------------------------- discovery helpers

def single_token_id(tokenizer, candidates):
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], cand
    raise RuntimeError(f"No single-token candidate in {candidates}")


def discover_layers(model):
    """(name, ModuleList, n) of the language-model decoder layers (vision towers skipped)."""
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = []
    for name, module in root.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4 and name.endswith("layers"):
            if any(part in name.lower() for part in ("vision", "visual", "vit", "patch")):
                continue
            candidates.append((name, module, len(module)))
    if not candidates:
        raise RuntimeError("Could not find a language layer ModuleList")
    candidates.sort(key=lambda item: (-item[2], len(item[0])))
    return candidates[0]


def _named_proj(module: nn.Module, suffixes: Sequence[str]):
    """Find a projection by name suffix (PEFT wraps nn.Linear, so isinstance is not required)."""
    matches = []
    for name, child in module.named_modules():
        if name.endswith("base_layer"):
            continue
        if any(name == s or name.endswith("." + s) for s in suffixes):
            matches.append((name, child))
    if not matches:
        return None
    matches.sort(key=lambda item: len(item[0]))
    return matches[0][1]


def unwrap(output):
    return output[0] if isinstance(output, tuple) else output


def rewrap(output, hidden):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


# ---------------------------------------------------------------- loading

def load_tokenizer(base: str):
    """Tokenizer of the base model, pad = eos if unset, right padding."""
    from transformers import AutoTokenizer
    if (Path(base) / "TINY_TEST_MODEL").exists():  # tests only: transformers 5 would rebuild a Qwen2 BPE tokenizer from model_type
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(base) / "tokenizer.json"), unk_token="[UNK]", pad_token="[PAD]",
                                      eos_token="[EOS]", bos_token="[EOS]", model_max_length=128)
    else:
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(base: str, dtype=None, device: str = "cuda"):
    """Load tokenizer + causal LM. Qwen3.5 checkpoints are vision-language; the causal-LM loader is
    tried first, then AutoModelForImageTextToText."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText
    except Exception:  # noqa: BLE001
        AutoModelForImageTextToText = None
    if dtype is None:
        dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
    tok = load_tokenizer(base)
    kwargs = dict(trust_remote_code=False, dtype=dtype, low_cpu_mem_usage=True)
    if device.startswith("cuda") and torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    errors = []
    for loader in [AutoModelForCausalLM] + ([AutoModelForImageTextToText] if AutoModelForImageTextToText else []):
        try:
            return tok, loader.from_pretrained(base, **kwargs), loader.__name__
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not load base model {base}: {errors}")


def load_classifier(base: str, adapter: str | None, device: str = "cuda", max_length: int = 128, fp32: bool = True):
    """Base + trained LoRA adapter as a HookedClassifier, eval mode, gradients off.
    fp32=True casts the model to float32 (all editing and scoring runs in fp32)."""
    tok, model, loader = load_base_model(base, device=device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    if fp32:
        model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hc = HookedClassifier(model, tok, max_length=max_length)
    hc.loader = loader
    return hc


# ---------------------------------------------------------------- linear algebra

def orthonormal_rows(u) -> torch.Tensor:
    """(d,) or (k,d) -> (k',d) float32 orthonormal rows spanning the same space (near-zero directions dropped)."""
    t = torch.as_tensor(np.asarray(u, dtype=np.float32)) if not torch.is_tensor(u) else u.detach().float().cpu()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    q, r = torch.linalg.qr(t.T)
    keep = torch.abs(torch.diagonal(r)) > 1e-6 * max(float(torch.abs(torch.diagonal(r)).max()), 1e-12)
    return q[:, keep].T.contiguous()


def _proj(h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if u.dim() == 1:
        vec = u.to(device=h.device, dtype=h.dtype)
        vec = vec / torch.clamp(vec.norm(), min=1e-12)
        return h - (h @ vec).unsqueeze(-1) * vec
    U = u.to(device=h.device, dtype=h.dtype)
    return h - (h @ U.T) @ U


def projector(u, device, dtype=torch.float32) -> torch.Tensor:
    """I - U^T U for the orthonormalised rows U of u."""
    U = orthonormal_rows(u)
    P = torch.eye(U.size(1)) - U.T @ U
    return P.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------- hooked classifier

class HookedClassifier:
    def __init__(self, model, tokenizer, max_length: int = 128):
        self.model = model
        self.tok = tokenizer
        self.device = next(model.parameters()).device
        self.id0, _ = single_token_id(tokenizer, [" 0", "0"])
        self.id1, _ = single_token_id(tokenizer, [" 1", "1"])
        self.layer_name, self.layers, self.n_layers = discover_layers(model)
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        self.hidden = int(getattr(cfg, "hidden_size", None) or getattr(text_cfg, "hidden_size", None) or 1024)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.max_length = int(max_length)
        self.embed = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()

    # -- module lookup
    def attn_out_proj(self, i: int):
        return _named_proj(self.layers[i], ("o_proj", "out_proj"))

    def mlp_out_proj(self, i: int):
        return _named_proj(self.layers[i], ("down_proj",))

    def module_for(self, site):
        if site == "embed":
            return self.embed
        if site == "readout":
            return self.lm_head
        return self.layers[int(site)]

    # -- tokenisation
    def encode(self, texts: Sequence[str]):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
                       add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        return {k: v.to(self.device) for k, v in enc.items()}, offsets

    def span_positions(self, texts, offsets, field: str) -> list[list[int]]:
        """Token positions of a field: every occurrence of a word field; the whole comment for cc_comment."""
        out = []
        for text, offs in zip(texts, offsets):
            if C.is_word_field(field):
                spans = C.all_word_char_spans(text, field)
            else:
                m = C.field_match(text, field)
                spans = [] if not m else [(m.start(), m.end())]
            idx = set()
            for s, e in spans:
                idx.update(i for i, (a, b) in enumerate(offs) if b > a and b > s and a < e)
            out.append(sorted(idx))
        return out

    # -- forward with interventions / captures
    @torch.no_grad()
    def run(self, texts: Sequence[str], interventions: Sequence[dict] = (), capture: Sequence[tuple] = (), batch_size: int = 16) -> dict:
        interventions, capture, texts = list(interventions or []), list(capture or []), list(texts)
        margins, logits2 = [], []
        captured = {key: [] for key in capture}
        for start in range(0, len(texts), int(batch_size)):
            m, lg, cap = self._run_batch(texts[start:start + int(batch_size)], interventions, capture)
            margins.extend(m); logits2.append(lg)
            for key, val in cap.items():
                captured[key].append(val)
        return {"margins": np.array(margins, dtype=np.float64),
                "logits2": np.concatenate(logits2, 0) if logits2 else np.zeros((0, 2)),
                "captured": {k: (np.concatenate(v, 0) if v else np.zeros((0, self.hidden))) for k, v in captured.items()}}

    def _run_batch(self, texts, interventions, capture):
        enc, offsets = self.encode(texts)
        last = enc["attention_mask"].long().sum(dim=1) - 1
        bidx = torch.arange(len(texts), device=self.device)
        pos_cache: dict[str, list[list[int]]] = {}

        def positions(pos):
            if pos not in pos_cache:
                pos_cache[pos] = self.span_positions(texts, offsets, pos)
            return pos_cache[pos]

        cap_out: dict[tuple, np.ndarray] = {}

        def apply_interventions(site, h):
            for iv in interventions:
                if iv["site"] != site:
                    continue
                if iv["type"] == "proj":
                    pos = iv.get("pos", "all")
                    if pos == "all":
                        h = _proj(h, iv["unit"])
                    elif pos == "last":
                        h = h.clone(); h[bidx, last] = _proj(h[bidx, last], iv["unit"])
                    else:
                        h = h.clone()
                        for b, idxs in enumerate(positions(pos)):
                            if idxs:
                                h[b, idxs] = _proj(h[b, idxs], iv["unit"])
                elif iv["type"] == "affine":  # h <- h - (h - mu) A^T (LEACE form)
                    A = iv["A"].to(device=h.device, dtype=h.dtype); mu = iv["mu"].to(device=h.device, dtype=h.dtype)
                    pos = iv.get("pos", "all")
                    if pos == "all":
                        h = h - (h - mu) @ A.T
                    else:
                        h = h.clone()
                        idx = [[int(last[b])] for b in range(h.size(0))] if pos == "last" else positions(pos)
                        for b, idxs in enumerate(idx):
                            if idxs:
                                h[b, idxs] = h[b, idxs] - (h[b, idxs] - mu) @ A.T
                else:
                    raise ValueError(iv["type"])
            return h

        def capture_from(site, h):
            for key in capture:
                if key[0] != site:
                    continue
                pos = key[1]
                if pos == "last":
                    vec = h[bidx, last]
                elif "@" in pos:
                    fld, k = pos.split("@"); k = int(k)
                    # NOTE (documented limitation): a text without the field gives a zero vector.
                    vec = torch.stack([h[b, idxs[min(k, len(idxs) - 1)]] if idxs else torch.zeros(h.size(-1), device=h.device, dtype=h.dtype)
                                       for b, idxs in enumerate(positions(fld))])
                else:
                    vec = torch.stack([h[b, idxs].mean(dim=0) if idxs else torch.zeros(h.size(-1), device=h.device, dtype=h.dtype)
                                       for b, idxs in enumerate(positions(pos))])
                cap_out[key] = vec.float().cpu().numpy()

        handles = []
        for site in {iv["site"] for iv in interventions} | {key[0] for key in capture}:
            if site == "readout":
                continue

            def fwd_hook(mod, inputs, output, site=site):
                h = unwrap(output)
                h2 = apply_interventions(site, h)
                capture_from(site, h2)
                return rewrap(output, h2) if h2 is not h else output

            handles.append(self.module_for(site).register_forward_hook(fwd_hook))

        def head_pre(mod, args):
            h = args[0][bidx, last].unsqueeze(1)
            h = apply_interventions("readout", h)
            for key in capture:
                if key[0] == "readout":
                    cap_out[key] = h[:, 0].float().cpu().numpy()
            return (h,) + tuple(args[1:])

        handles.append(self.lm_head.register_forward_pre_hook(head_pre))
        try:
            out = self.model(**enc, use_cache=False)
            logits = out.logits if hasattr(out, "logits") else out[0]
        finally:
            for hd in handles:
                hd.remove()
        picked = logits[:, -1].float()
        lg2 = picked[:, [self.id0, self.id1]].cpu().numpy()
        return (lg2[:, 1] - lg2[:, 0]).tolist(), lg2, cap_out


# ---------------------------------------------------------------- weight folds (permanent edits)

def _lora_mods(module: nn.Module, attr: str):
    holder = getattr(module, attr, None)
    if holder is None:
        return []
    if hasattr(holder, "items"):
        return [sub.weight for _, sub in holder.items() if hasattr(sub, "weight")]
    return [holder.weight] if hasattr(holder, "weight") else []


def untie_embeddings(m: HookedClassifier) -> dict:
    """Give lm_head its own copy of the tied embedding matrix so an embedding fold leaves the readout unchanged."""
    emb_w = m.embed.weight
    head = m.lm_head
    head_base = head.base_layer if hasattr(head, "base_layer") else head
    tied = emb_w.data_ptr() == head_base.weight.data_ptr()
    if tied:
        head_base.weight = nn.Parameter(head_base.weight.detach().clone(), requires_grad=False)
    return {"was_tied": bool(tied), "tied_after": bool(emb_w.data_ptr() == head_base.weight.data_ptr())}


@torch.no_grad()
def fold_left(module: nn.Module, P: torch.Tensor) -> dict:
    base = module.base_layer if hasattr(module, "base_layer") else module
    W = base.weight
    if W.shape[0] != P.shape[0]:
        return {"ok": False, "reason": f"shape {tuple(W.shape)}"}
    W.copy_((P @ W.float()).to(W.dtype))
    n_b = 0
    for B in _lora_mods(module, "lora_B"):
        if B.shape[0] == P.shape[0]:
            B.copy_((P @ B.float()).to(B.dtype)); n_b += 1
    if getattr(base, "bias", None) is not None:
        base.bias.copy_((P @ base.bias.float()).to(base.bias.dtype))
    return {"ok": True, "lora_B": n_b}


@torch.no_grad()
def fold_embedding(m: HookedClassifier, P: torch.Tensor) -> dict:
    W = m.embed.weight
    W.copy_((W.float() @ P).to(W.dtype))
    return {"ok": True}


def fold_cut(m: HookedClassifier, u, site) -> dict:
    """Fold an all-position projection into the weights.
    site "embed": embedding rows only. site int L: embedding plus the attention and MLP output
    projections (and their LoRA B matrices) of layers 0..L, so the direction is removed from
    every residual write up to and including layer L."""
    tie = untie_embeddings(m)
    P = projector(u, m.device)
    edits = [{"name": "embed_tokens", **fold_embedding(m, P)}]
    if site != "embed":
        for i in range(0, int(site) + 1):
            attn, mlp = m.attn_out_proj(i), m.mlp_out_proj(i)
            if attn is not None:
                edits.append({"name": f"layer{i}.attn_out", **fold_left(attn, P)})
            if mlp is not None:
                edits.append({"name": f"layer{i}.mlp_down", **fold_left(mlp, P)})
    return {"site": str(site), "tie": tie, "n_ok": sum(1 for e in edits if e.get("ok")), "n_total": len(edits), "edits": edits}


def through_interventions(u, site) -> list[dict]:
    """Hook equivalent of fold_cut (used in tests to check the fold)."""
    t = orthonormal_rows(u)
    t = t[0] if t.size(0) == 1 else t
    ivs = [{"type": "proj", "site": "embed", "unit": t, "pos": "all"}]
    if site != "embed":
        ivs += [{"type": "proj", "site": i, "unit": t, "pos": "all"} for i in range(0, int(site) + 1)]
    return ivs


class WeightSnapshot:
    """Copy of every parameter, to undo a fold exactly. Unties the embeddings first (idempotent), because a fold
    unties them and would otherwise create a parameter the snapshot does not hold."""

    def __init__(self, m: HookedClassifier):
        self.m = m
        self.untie = untie_embeddings(m)
        self.snap = {n: p.detach().clone() for n, p in m.model.named_parameters()}

    @torch.no_grad()
    def restore(self):
        for n, p in self.m.model.named_parameters():
            p.copy_(self.snap[n])


# ---------------------------------------------------------------- training helpers

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lora_target_modules(model) -> list[str]:
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(p in name.lower() for p in ("vision", "visual", "vit", "patch_embed")):
            names.add(name.split(".")[-1])
    return sorted(names)


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """LoRA on every linear module of the language model (vision tower excluded)."""
    from peft import LoraConfig, TaskType, get_peft_model
    target = lora_target_modules(model) or "all-linear"
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target, bias="none")
    return get_peft_model(model, cfg), (target if isinstance(target, list) else [target])


def encode_with_labels(tok, texts, labels, id0: int, id1: int, device, max_length: int = 128):
    """Token-level CE target only at the last prompt token (the class token is not appended)."""
    enc = tok(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    lab = torch.full_like(enc["input_ids"], -100)
    last = enc["attention_mask"].long().sum(dim=1) - 1
    lab[torch.arange(lab.size(0), device=lab.device), last] = torch.tensor([id1 if int(y) == 1 else id0 for y in labels], device=lab.device)
    return enc, lab


# ---------------------------------------------------------------- tiny CPU model (tests and smoke only)

def build_tiny_tokenizer(texts, outdir: Path):
    import re
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Sequence as PreSeq, Split, WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    pat = r"n't|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\s]"
    specials = ["[PAD]", "[UNK]", "[EOS]", " 0", " 1", "0", "1"]
    words = set()
    rx = re.compile(pat)
    for t in texts:
        words.update(rx.findall(t))
    vocab = {w: i for i, w in enumerate(specials)}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = PreSeq([WhitespaceSplit(), Split(Regex(pat), behavior="isolated")])
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    tk.save(str(outdir / "tokenizer.json"))
    hf = PreTrainedTokenizerFast(tokenizer_file=str(outdir / "tokenizer.json"), unk_token="[UNK]", pad_token="[PAD]",
                                 eos_token="[EOS]", bos_token="[EOS]")
    hf.padding_side = "right"; hf.model_max_length = 128
    hf.save_pretrained(outdir)
    return hf, len(vocab)


def build_tiny_causal(texts, outdir: Path, seed: int = 0, n_layers: int = 4):
    """Random 4-layer Qwen2 causal LM (hidden 32) with a word-level tokenizer: plumbing tests only, never a result."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    outdir = Path(outdir)
    tok, vs = build_tiny_tokenizer(texts, outdir / "tokenizer")
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=max(vs, len(tok)), hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128, tie_word_embeddings=True,
                      use_cache=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id,
                      hidden_act="silu", rms_norm_eps=1e-6)
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(outdir / "base")
    tok.save_pretrained(outdir / "base")
    (outdir / "base" / "TINY_TEST_MODEL").write_text("random tiny model for CPU tests; not a result\n")
    return outdir / "base", tok, model
