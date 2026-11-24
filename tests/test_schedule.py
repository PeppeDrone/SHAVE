import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import schedule as S  # noqa: E402

SMI_PLAIN = """GPU 0: NVIDIA A40 (UUID: GPU-aaaa)
GPU 1: NVIDIA A40 (UUID: GPU-bbbb)
"""
SMI_MIG = """GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-aaaa)
  MIG 3g.40gb     Device  0: (UUID: MIG-1111)
  MIG 3g.40gb     Device  1: (UUID: MIG-2222)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-bbbb)
"""


def test_parse_nvidia_smi():
    assert S.parse_nvidia_smi_l(SMI_PLAIN) == ["0", "1"]
    assert S.parse_nvidia_smi_l(SMI_MIG) == ["MIG-1111", "MIG-2222", "1"]
    assert S.parse_nvidia_smi_l("") == []
    many = "".join(f"GPU {i}: X (UUID: GPU-{i})\n" for i in range(24))
    assert len(S.parse_nvidia_smi_l(many)) == 24


def test_detect_slots_env_and_fallback(monkeypatch):
    assert S.detect_slots({"CUDA_VISIBLE_DEVICES": "MIG-1,MIG-2"}) == ["MIG-1", "MIG-2"]
    assert S.detect_slots({"CUDA_VISIBLE_DEVICES": "3"}) == ["3"]
    assert S.detect_slots({}, smoke=True) == ["cpu"]
    monkeypatch.setattr(S.shutil, "which", lambda x: None)
    assert S.detect_slots({}) == ["cpu"]


def test_plan_jobs_filters():
    cfg = {"arms": ["muslim_nat", "muslim90"], "seeds": [42, 84, 126]}
    assert len(S.plan_jobs(cfg)) == 6 and S.plan_jobs(cfg)[0] == ("muslim_nat", 42)
    assert S.plan_jobs(cfg, arms="muslim90", seeds="84,126") == [("muslim90", 84), ("muslim90", 126)]
    assert S.plan_jobs(cfg, only="muslim_nat:126") == [("muslim_nat", 126)]


def test_stage_order_and_markers():
    assert [s[0] for s in S.STAGES] == ["train", "check", "shave", "leace", "dfr"]
    assert all((ROOT / "scripts" / s[1]).exists() for s in S.STAGES)


def test_dry_run_and_memtest_plan(tmp_path):
    cfg = ROOT / "configs" / "experiment.yaml"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "schedule.py"), "--config", str(cfg), "--out", str(tmp_path),
                        "--slots", "0,1,2", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "SLOTS 3" in r.stdout and "JOBS 40" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "schedule.py"), "--config", str(cfg), "--out", str(tmp_path),
                        "--slots", "0,1,2", "--memtest", "--dry-run"], capture_output=True, text=True)
    assert "SLOTS 1" in r.stdout and "JOBS 1" in r.stdout and "STAGES ['train', 'check', 'shave', 'leace', 'dfr']" in r.stdout


def test_memory_report_reads_timings(tmp_path):
    d = tmp_path / "results" / "muslim90" / "s11" / "_meta"; d.mkdir(parents=True)
    (d / "timings_train.json").write_text(json.dumps([{"stage": "train_total", "seconds": 10, "peak_alloc_gib": 5.0, "peak_reserved_gib": 6.5,
                                                         "peak_alloc_bytes": 1, "peak_reserved_bytes": 2, "gpu_name": "X", "gpu_total_bytes": 3}]))
    rep = S.memory_report(tmp_path, tmp_path / "mr")
    assert rep["max_peak_reserved_gib"] == 6.5 and rep["fits_24gib"] is True
    assert "train_total" in (tmp_path / "mr.md").read_text()


def test_timer_records_memory_fields_null_on_cpu():
    from interp_pilot.artifacts import Timer
    t = Timer(); t.start("a"); t.start("b"); t.stop("b"); t.stop("a")
    assert [r["stage"] for r in t.rows] == ["b", "a"]
    assert all(r["peak_alloc_bytes"] is None and r["peak_reserved_bytes"] is None for r in t.rows)
