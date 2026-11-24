"""Tiny random Qwen2 model on CPU: hooks, folds, LoRA plumbing. Plumbing only."""
import numpy as np
import torch

from interp_pilot import civil as C
from interp_pilot.model import (HookedClassifier, WeightSnapshot, apply_lora, build_tiny_causal, fold_cut, load_base_model,
                                single_token_id, through_interventions)

TEXTS = [C.make_text(t) for t in ("the muslims park city.", "the idiot river road.", "a calm garden morning.",
                                    "the jews and the christians read.", "stupid men watch the train.")]


def _model(tmp_path, lora=True):
    base, _, _ = build_tiny_causal(TEXTS, tmp_path / "tiny", seed=0)
    tok, model, _ = load_base_model(str(base), device="cpu")
    if lora:
        model, _ = apply_lora(model, r=4, alpha=8, dropout=0.0)
        with torch.no_grad():  # make LoRA B non-zero so the fold must handle it
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.05)
    model.eval()
    return HookedClassifier(model, tok, max_length=64), tok


def test_tiny_tokenizer_keeps_words(tmp_path):
    m, tok = _model(tmp_path, lora=False)
    toks = tok.convert_ids_to_tokens(tok(TEXTS[0])["input_ids"])
    assert "muslims" in toks and "Comment" in toks and "[UNK]" not in toks
    assert single_token_id(tok, [" 0", "0"])[0] != single_token_id(tok, [" 1", "1"])[0]


def test_margins_depend_on_text(tmp_path):
    m, _ = _model(tmp_path)
    r = m.run(TEXTS)
    assert len(np.unique(np.round(r["margins"], 6))) > 1


def test_embed_fold_equals_hook(tmp_path):
    m, _ = _model(tmp_path)
    u = torch.randn(1, m.model.config.hidden_size if hasattr(m.model, "config") else 32)
    hook = m.run(TEXTS, interventions=through_interventions(u, "embed"))["margins"]
    snap = WeightSnapshot(m)
    info = fold_cut(m, u, "embed")
    folded = m.run(TEXTS)["margins"]
    assert info["n_ok"] == info["n_total"] and info["tie"]["tied_after"] is False
    assert np.allclose(hook, folded, atol=1e-4)
    snap.restore()
    assert np.allclose(m.run(TEXTS)["margins"], m.run(TEXTS)["margins"])


def test_layer_fold_removes_direction_from_residual(tmp_path):
    """Folding at layer L projects the embedding and every attention / MLP write of layers 0..L, so the residual
    after layer L has no component along u (checked on captured hidden states)."""
    m, _ = _model(tmp_path)
    d = 32
    u = torch.randn(1, d); u = u / u.norm()
    fold_cut(m, u, 1)
    r = m.run(TEXTS, capture=[(1, "last")])
    h = next(iter(r["captured"].values()))
    h = np.asarray(h)
    assert np.max(np.abs(h @ u.numpy().T)) < 1e-4


def test_weight_snapshot_restores_exactly(tmp_path):
    m, _ = _model(tmp_path)
    before = m.run(TEXTS)["margins"]
    snap = WeightSnapshot(m)
    fold_cut(m, torch.randn(2, 32), 3)
    assert not np.allclose(before, m.run(TEXTS)["margins"])
    snap.restore()
    assert np.allclose(before, m.run(TEXTS)["margins"], atol=1e-6)
