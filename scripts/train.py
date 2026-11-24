#!/usr/bin/env python3
"""Train one LoRA toxicity classifier (one arm, one seed). Writes <out>/results/<arm>/s<seed>/train/."""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot.config import load


def token_lengths(tok, texts, max_length):
    import numpy as np
    n = np.array([len(x) for x in tok(list(texts), add_special_tokens=False)["input_ids"]])
    if not len(n):
        return {"n": 0}
    return {"n": int(len(n)), "p50": float(np.percentile(n, 50)), "p99": float(np.percentile(n, 99)), "max": int(n.max()),
            "n_gt_max_length": int((n > max_length).sum())}


def epoch_batches(n_rows: int, batch_size: int, epochs: int, seed: int) -> list[list[int]]:
    """Row indices of every optimizer step: for each epoch, a random.Random(seed) shuffle of all rows cut into
    consecutive batches of `batch_size` (the last batch of an epoch may be smaller). Each row is used once per
    epoch; the number of steps per epoch is ceil(n_rows / batch_size)."""
    rng = random.Random(seed)
    out = []
    for _ in range(epochs):
        idx = list(range(n_rows)); rng.shuffle(idx)
        out += [idx[i:i + batch_size] for i in range(0, n_rows, batch_size)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if cfg.get("smoke"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from torch import nn
    from interp_pilot import civil as C
    from interp_pilot import runner as R
    from interp_pilot.artifacts import now_rome
    from interp_pilot.model import apply_lora, build_tiny_causal, encode_with_labels, load_base_model, set_seed, single_token_id

    T = cfg["train"]; maxlen = int(cfg["model"]["max_length"])
    data = R.data_dir(a.out)
    set_seed(a.seed)
    rows = C.jl(data / "arms" / a.arm / "train.jsonl")
    val = C.jl(data / "val.jsonl"); random.Random(a.seed).shuffle(val); val = val[: int(T["n_val_monitor"])]
    art = R.artifacts(a.out, a.arm, a.seed, config={"train": T, "config_file": cfg["_config_path"]})
    art.timer.start("train_total")
    base = R.base_model_path(cfg, a.out, a.arm, a.seed)
    device = R.device_for(cfg)
    if cfg.get("smoke"):
        texts = [r["text"] for r in rows + C.jl(data / "val.jsonl")]
        for p in sorted((data / "eval").glob("*.jsonl")):
            for x in C.jl(p):
                texts += [x[k] for k in ("text", "text_neutral", "text_other") if k in x]
        fields = json.loads((data / "arms" / a.arm / "candidate_fields.json").read_text())["fields"]
        texts += [C.make_text(" ".join(C.word_of(f) for f in fields if C.is_word_field(f)))]
        build_tiny_causal(texts, Path(base).parent, seed=a.seed)
    tok, model, loader = load_base_model(base, device=device)
    id0, _ = single_token_id(tok, [" 0", "0"]); id1, _ = single_token_id(tok, [" 1", "1"])
    lens = {"train_arm": token_lengths(tok, [r["text"] for r in rows[:5000]], maxlen)}
    for k in ("T", "D", "M", "G", "SW"):
        p = data / "eval" / f"{k}.jsonl"
        if p.exists():
            lens[f"eval_{k}"] = token_lengths(tok, [r["text"] for r in C.jl(p)], maxlen)
    art.json("train", "token_lengths.json", lens)
    model, modules = apply_lora(model, r=int(T["lora_r"]), alpha=int(T["lora_alpha"]), dropout=float(T["lora_dropout"]))
    model.train()
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(T["lr"]))
    if "steps" in T:
        raise ValueError("train.steps is not used: the number of steps is ceil(N / batch_size) per epoch (train.epochs)")
    bs, epochs = int(T["batch_size"]), int(T["epochs"])
    plan = epoch_batches(len(rows), bs, epochs, a.seed)
    n_steps = len(plan)
    assert n_steps == epochs * math.ceil(len(rows) / bs) and sorted(i for b in plan for i in b) == sorted(list(range(len(rows))) * epochs)
    order_sha = hashlib.sha256(",".join(str(rows[i]["id"]) for b in plan for i in b).encode()).hexdigest()
    losses, t0 = [], time.time()
    out = logits = loss = enc = labels = None
    art.timer.start("train_loop")
    for step, bidx in enumerate(plan):
        batch = [rows[i] for i in bidx]
        enc, labels = encode_with_labels(tok, [r["text"] for r in batch], [r["label"] for r in batch], id0, id1, dev, maxlen)
        opt.zero_grad(set_to_none=True)
        out = model(**enc, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(T["grad_clip"])); opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0:
            print(f"{now_rome()} {a.arm} s{a.seed} step {step+1}/{n_steps} loss={losses[-1]:.4f}", flush=True)
    train_s = time.time() - t0
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Free training-only memory before the save and the validation pass (24 GB cards): optimizer state, gradients and
    # the last step's activations/logits. Nothing that is saved or evaluated depends on them.
    art.timer.stop("train_loop")
    del opt, out, logits, loss, enc, labels
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    art.timer.start("train_save_eval")
    model.eval()
    model.save_pretrained(art.dir("train") / "adapter", save_embedding_layers=False)
    vm = []
    with torch.no_grad():
        for s0 in range(0, len(val), 32):
            enc, _ = encode_with_labels(tok, [r["text"] for r in val[s0:s0 + 32]], [0] * len(val[s0:s0 + 32]), id0, id1, dev, maxlen)
            lg = model(**enc, use_cache=False).logits.float()
            last = enc["attention_mask"].long().sum(1) - 1
            pk = lg[torch.arange(lg.size(0), device=lg.device), last]
            vm += (pk[:, id1] - pk[:, id0]).cpu().tolist()
    vm = np.array(vm); yv = np.array([r["label"] for r in val])
    art.predictions("train", {"val": {"margins": vm}}, {"val": val}, name="val_predictions")
    art.csv("train", "losses.csv", [{"step": i + 1, "loss": l} for i, l in enumerate(losses)])
    info = {"arm": a.arm, "seed": a.seed, **T, "max_length": maxlen, "target_modules": modules, "loader": loader, "base": base,
            "trainable_parameters": trainable,
            "n_train_rows": len(rows), "n_steps": n_steps, "steps_per_epoch": math.ceil(len(rows) / bs),
            "last_batch_size": len(plan[-1]) if plan else 0, "rows_seen": sum(len(b) for b in plan),
            "row_order_sha256": order_sha, "row_order_seed": a.seed, "lr_schedule": "constant (no warmup, no decay)",
            "first_loss": losses[0] if losses else None,
            "mean_last20_loss": float(np.mean(losses[-20:])) if losses else None, "val_n": len(val),
            "val_acc": float(np.mean((vm > 0) == yv)) if len(val) else None, "val_auroc": C.auroc(yv, vm),
            "train_seconds": train_s, "device": str(dev), "finished_rome": now_rome()}
    art.json("train", "training.json", info)
    art.timer.stop("train_save_eval")
    art.timer.stop("train_total")
    art.write_manifest(stage="train")
    R.mark_done(art, "train", {"val_acc": info["val_acc"]})
    print("TRAINED", a.arm, a.seed, json.dumps({k: info[k] for k in ("val_acc", "val_auroc", "mean_last20_loss", "train_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
