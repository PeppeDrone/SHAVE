#!/usr/bin/env python3
"""Run every (arm, seed) model through its stages on the visible GPUs / MIG slices, one model per slot at a time.

Slots: --slots auto (default) detects them: CUDA_VISIBLE_DEVICES if set, else `nvidia-smi -L` (MIG UUIDs where MIG is
enabled, otherwise GPU indices), else one CPU slot. The same command works on 1 GPU or many: every (arm, seed) pair in the config (or the
--seeds / --arms / --only filters) is queued and each slot takes the next model when it finishes one. Stages per model, in order:
train, check, shave, leace, dfr. A stage whose DONE.json exists is skipped, so a rerun with the
same --out resumes. Each stage is a separate process with CUDA_VISIBLE_DEVICES set to its slot.

Memory test: --memtest runs one model (first arm and seed after the filters) through every stage on one slot and
writes logs/memory_report.{json,md} with wall time and peak GPU memory (allocated and reserved) per stage.
After any run the same report is written for every model found under <out>/results.
Never creates, starts or stops machines."""
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot.config import load

STAGES = [("train", "train.py", "train"), ("check", "check.py", "check"), ("shave", "run_shave.py", "shave"),
          ("leace", "run_leace.py", "leace_oracle"), ("dfr", "run_dfr.py", "dfr")]


def parse_nvidia_smi_l(text: str) -> list[str]:
    """GPU indices, with each MIG-enabled GPU replaced by its MIG device UUIDs."""
    slots, cur_gpu, cur_migs = [], None, []

    def flush():
        if cur_gpu is not None:
            slots.extend(cur_migs if cur_migs else [cur_gpu])
    for line in text.splitlines():
        g = re.match(r"\s*GPU (\d+):", line)
        m = re.search(r"\(UUID: (MIG-[^)]+)\)", line)
        if g:
            flush(); cur_gpu, cur_migs = g.group(1), []
        elif m and line.lstrip().startswith("MIG"):
            cur_migs.append(m.group(1))
    flush()
    return slots


def detect_slots(env=None, smoke: bool = False) -> list[str]:
    env = os.environ if env is None else env
    if smoke:
        return ["cpu"]
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() not in ("", "-1"):
        return [s.strip() for s in cvd.split(",") if s.strip()]
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
            slots = parse_nvidia_smi_l(out)
            if slots:
                return slots
        except Exception:  # noqa: BLE001
            pass
    return ["cpu"]


def plan_jobs(cfg: dict, arms: str = "", seeds: str = "", only: str = "") -> list[tuple[str, int]]:
    jobs = [(arm, s) for s in cfg["seeds"] for arm in cfg["arms"]]
    if arms:
        keep = set(arms.split(",")); jobs = [j for j in jobs if j[0] in keep]
    if seeds:
        keep = {int(x) for x in seeds.split(",")}; jobs = [j for j in jobs if j[1] in keep]
    if only:
        keep = {(x.split(":")[0], int(x.split(":")[1])) for x in only.split(",")}
        jobs = [j for j in jobs if j in keep]
    return jobs


def memory_report(out: Path, path_stem: Path) -> dict:
    """Collect per-stage wall time and peak GPU memory from every <out>/results/<arm>/s<seed>/_meta/timings_*.json."""
    rows = []
    for tf in sorted((Path(out) / "results").glob("*/s*/_meta/timings_*.json")):
        arm, seed = tf.parts[-4], tf.parts[-3]
        for r in json.loads(tf.read_text()):
            rows.append({"arm": arm, "seed": seed, "file": tf.name, **{k: r.get(k) for k in (
                "stage", "seconds", "peak_alloc_gib", "peak_reserved_gib", "peak_alloc_bytes", "peak_reserved_bytes", "gpu_name", "gpu_total_bytes")}})
    gib = [r["peak_reserved_gib"] for r in rows if r["peak_reserved_gib"] is not None]
    rep = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_rows": len(rows), "max_peak_reserved_gib": max(gib) if gib else None,
           "max_peak_alloc_gib": max([r["peak_alloc_gib"] for r in rows if r["peak_alloc_gib"] is not None], default=None),
           "fits_24gib": (max(gib) < 24.0) if gib else None, "rows": rows}
    Path(str(path_stem) + ".json").write_text(json.dumps(rep, indent=1))
    lines = ["# GPU memory and time per stage", "", f"Written {rep['written']}. Peak memory from torch.cuda.max_memory_allocated / "
             "max_memory_reserved, reset at each stage start (null on CPU).", "",
             "| arm | seed | stage | seconds | peak allocated (GiB) | peak reserved (GiB) | GPU |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['stage']} | {r['seconds']} | {r['peak_alloc_gib']} | {r['peak_reserved_gib']} | {r['gpu_name']} |")
    lines += ["", f"Largest peak reserved: {rep['max_peak_reserved_gib']} GiB; below 24 GiB: {rep['fits_24gib']}."]
    Path(str(path_stem) + ".md").write_text("\n".join(lines) + "\n")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slots", default="auto", help="auto, or comma-separated CUDA_VISIBLE_DEVICES values (GPU indices / MIG UUIDs), or cpu")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--arms", default="", help="comma list (default: all arms in the config)")
    ap.add_argument("--seeds", default="", help="comma list (default: all seeds in the config)")
    ap.add_argument("--only", default="", help="arm:seed,arm:seed (default: all)")
    ap.add_argument("--stages", default=",".join(s[0] for s in STAGES))
    ap.add_argument("--memtest", action="store_true", help="one model, every stage, one slot; writes logs/memory_report.*")
    ap.add_argument("--dry-run", action="store_true", help="print slots and the job plan, run nothing")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    from interp_pilot.runner import is_done
    slots = detect_slots(smoke=bool(cfg.get("smoke"))) if a.slots == "auto" else [s for s in a.slots.split(",") if s != ""]
    jobs = plan_jobs(cfg, a.arms, a.seeds, a.only)
    names = a.stages.split(",")
    bad = [n for n in names if n not in [s[0] for s in STAGES]]
    if bad:
        raise SystemExit(f"unknown stages {bad}")
    stages = [s for s in STAGES if s[0] in names]
    if a.memtest:
        jobs, slots, stages = jobs[:1], slots[:1], STAGES
    print("SLOTS", len(slots), slots, "| JOBS", len(jobs), "| STAGES", [s[0] for s in stages], flush=True)
    logs = a.out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        return 0
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    sp = logs / "schedule_status.json"
    status = json.loads(sp.read_text()) if sp.exists() else {}
    lock, ran = threading.Lock(), []

    def worker(slot):
        while True:
            try:
                arm, seed = q.get_nowait()
            except queue.Empty:
                return
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if (cfg.get("smoke") or slot == "cpu") else slot}
            for name, script, marker in stages:
                if is_done(a.out, arm, seed, marker):
                    continue
                t0 = time.time()
                cmd = [a.python, str(_bootstrap.ROOT / "scripts" / script), "--config", str(a.config), "--out", str(a.out), "--arm", arm, "--seed", str(seed)]
                with open(logs / f"{arm}_s{seed}_{name}.log", "a") as lf:
                    lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S %Z')} slot={slot} cmd={' '.join(cmd)}\n"); lf.flush()
                    rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
                with lock:
                    key = f"{arm}:s{seed}:{name}"
                    status[key] = {"rc": rc, "seconds": round(time.time() - t0, 1), "slot": slot, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
                    ran.append(key)
                    sp.write_text(json.dumps(status, indent=1))
                    print(time.strftime("%H:%M:%S"), arm, seed, name, "slot", slot, "rc", rc, f"{time.time() - t0:.0f}s", flush=True)
                if rc != 0:
                    break  # later stages of this model need this one

    threads = [threading.Thread(target=worker, args=(s,)) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rep = memory_report(a.out, logs / "memory_report")
    failed = [k for k in ran if status[k]["rc"] != 0]
    print("SCHEDULE_DONE jobs", len(jobs), "stage runs", len(ran), "failed", len(failed),
          "| max peak reserved GiB", rep["max_peak_reserved_gib"], flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
