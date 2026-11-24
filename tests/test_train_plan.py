"""Base-classifier fine-tuning plan: exactly one pass over the arm, each row once, seeded order."""
import math

import pytest

import train as TR
from interp_pilot.config import load
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n,bs", [(50540, 8), (759, 8), (16, 8), (7, 8), (1, 8)])
def test_one_epoch_each_row_once(n, bs):
    plan = TR.epoch_batches(n, bs, 1, seed=42)
    assert len(plan) == math.ceil(n / bs)
    assert sorted(i for b in plan for i in b) == list(range(n))
    assert all(len(b) == bs for b in plan[:-1]) and 1 <= len(plan[-1]) <= bs


def test_full_run_steps_and_seeded_order():
    plan = TR.epoch_batches(50540, 8, 1, seed=42)
    assert len(plan) == 6318 and len(plan[-1]) == 4
    assert TR.epoch_batches(50540, 8, 1, seed=42) == plan
    assert TR.epoch_batches(50540, 8, 1, seed=84) != plan


def test_config_defines_epochs_not_steps():
    for name in ("experiment.yaml", "smoke_cpu.yaml"):
        T = load(ROOT / "configs" / name)["train"]
        assert int(T["epochs"]) == 1 and "steps" not in T and int(T["batch_size"]) == 8
