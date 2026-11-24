import json
from pathlib import Path

import numpy as np

from interp_pilot.artifacts import RunArtifacts, sha256_file, verify_manifest
from interp_pilot.config import load

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_config_frozen_values():
    c = load(ROOT / "configs" / "experiment.yaml")
    assert c["seeds"] == [42 * k for k in range(1, 21)] and c["arms"] == ["muslim_nat", "muslim90"]
    assert len(c["seeds"]) == 20 and len(set(c["seeds"])) == 20 and all(s % 42 == 0 for s in c["seeds"])
    assert c["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    o = c["shave"]
    assert (o["target_rule"], o["flip_min"], o["side_max"], o["kl_max"], o["ranks"]) == ("r12", 0.90, 0.05, 0.10, [1, 2, 4])
    assert set(c) - {"_config_path"} == {"model", "data", "arms", "seeds", "train", "check", "exclusion", "shave", "leace", "dfr", "stats"}, set(c)
    assert c["stats"]["primary_metric"] == "muslim_ba"


def test_smoke_config_inherits():
    c = load(ROOT / "configs" / "smoke_cpu.yaml")
    assert c["smoke"] and c["seeds"] == [42]
    assert c["shave"]["flip_min"] == 0.90 and c["shave"]["pool_size"] == 80   # inherited + overridden
    assert int(c["train"]["epochs"]) == 1 and c["train"]["lora_r"] == 16   # inherited


def test_manifest_roundtrip(tmp_path):
    art = RunArtifacts(tmp_path, "muslim90", 11, config={"x": 1})
    art.json("m", "a.json", {"v": np.float32(0.5)})
    art.array("m", "arr", np.arange(6, dtype=np.float32).reshape(2, 3))
    man = art.write_manifest(stage="test")
    assert man.name == "manifest_test.json"
    assert verify_manifest(man) == []
    p = tmp_path / "muslim90" / "s11" / "m" / "a.json"
    p.write_text(json.dumps({"v": 0.6}))
    assert verify_manifest(man)                      # tampering is detected
    assert len(sha256_file(man)) == 64


def test_expected_data_hash_list_complete():
    lines = (ROOT / "configs" / "data_expected_sha256.txt").read_text().split("\n")
    lines = [l for l in lines if l.strip() and not l.startswith("#")]
    assert len(lines) == 19
    names = {l.split()[-1] for l in lines}
    for arm in ("muslim_nat", "muslim90"):
        assert f"arms/{arm}/train.jsonl" in names
    assert "val.jsonl" in names and "arms/muslim90/candidate_fields.json" in names and "candidate_fields.json" not in names


def test_no_secrets_in_release():
    import re
    rx = re.compile(r"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36,}|glpat-[A-Za-z0-9_-]{20,})")
    bad = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".yaml", ".sh", ".md", ".txt", ".toml", ".tex", ".bib", ".json", ""}:
            for m in rx.finditer(p.read_text(errors="ignore")):
                bad.append((str(p.relative_to(ROOT)), m.group(0)[:12]))
    assert not bad, bad
