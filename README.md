# SHAVE: Shortcut Hunting with Abstention and Verified Editing in LLMs

Piergiuseppe Liuzzi, Leonardo Corsi, Andrea Mannini

Code and configuration for the paper (https://github.com/PeppeDrone/SHAVE). SHAVE finds and removes a lexical shortcut in a fine-tuned LLM classifier
**without labels**, using only unlabelled text and the classifier itself, and abstains when no single word is strong
enough to justify an edit.

## What SHAVE does

1. **Find a word.** For each word in a fixed candidate list (the 100 most frequent words of the training comments), insert
   it into, or remove it from, unlabelled validation comments and measure how often the classifier's prediction changes.
   Words are ranked by the Wilson 95% lower bound of that change.
2. **Abstain if the word is weak.** If the lower bound of the selected word's insertion effect is below the strength
   threshold **0.20** (`shave.min_override_wilson_lb` in `configs/experiment.yaml`), SHAVE leaves the classifier unedited.
   In the paper this threshold was applied after the runs to the saved measurements. The candidate search is therefore
   still run whenever a word is selected, and the same run also records an ablation without the threshold that keeps
   the same candidate edits and acceptance criteria.
3. **Build and verify an edit.** From contrast pairs (a comment and the same comment with the word inserted), build 97
   candidate projection edits (input embedding and every layer, ranks 1/2/4) and accept one only if it reverses at
   least 90% of reserved prefixed comments, changes at most 5% of predictions on comments without the word, and keeps
   the mean Bernoulli KL divergence at or below 0.10. The accepted projection is folded into the weights.

SHAVE is compared with **LEACE** (Belrose et al., 2023; given the concept, the indicator for the word *muslim*) and **DFR**
(Kirichenko et al., 2023; given group labels). See `docs/METHODS.md` for every setting.

## Study design

Qwen3.5-0.8B is fine-tuned with LoRA as a binary toxicity classifier on CivilComments-WILDS in two training arms that
differ only in how often comments containing the word *muslim* are toxic:

| Arm | Internal name (code, configs, result folders) | P(toxic \| comment contains *muslim*) |
|---|---|---|
| Planted arm | `muslim90` | 0.90 |
| Natural arm | `muslim_nat` | 0.27 (the natural rate) |

Each arm has 50,540 training comments, 896 of them containing *muslim*. The design lists 20 seeds per arm
(42, 84, ..., 840 in `configs/experiment.yaml`). The paper ran and reports the first ten (42–420; 10 per arm, 20
classifiers). The primary outcome is balanced accuracy on the comments containing *muslim* in the group-balanced test set $G$.

The evaluation also uses the overall test sample $T$, the related-word group $R$, the matched non-toxic samples $D_m$
and $M_m$, a descriptive analysis of the word *black*, and logged runtime and peak reserved GPU memory.

## Install

Python 3.12 is required.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu for the tests
pip install -r requirements.txt                                               # all other packages pinned
```

`bash scripts/setup_env.sh <venv_dir> [cpu|cu128]` does the same. The package is not pip-installed: scripts and tests
put `src/` on the path.

## Data and model

Nothing is redistributed here. Inputs are downloaded at run time and checked by sha256:

- **CivilComments-WILDS v1.0** (CC0), from the WILDS CodaLab bundle: `bash scripts/fetch_data.sh <dest_dir>`.
  Arms and test sets are rebuilt by `scripts/build_data.py` and checked against `configs/data_expected_sha256.txt`.
  Details: `docs/DATA.md`.
- **Qwen3.5-0.8B** (Apache-2.0), revision `2fc0636`, from the Hugging Face Hub
  (`python scripts/fetch_model.py --config configs/experiment.yaml --dest <dir>`; uses `HF_TOKEN` from the environment
  if set). Checked against `configs/model_expected_sha256.txt`.

`run_full_rerun.sh` performs both steps itself and reuses verified copies when they exist.

## Reproduce

Tests (CPU, tiny random model, synthetic data):

```bash
OMP_NUM_THREADS=2 python -m pytest tests
```

One classifier, every stage (fine-tuning, unedited evaluation, SHAVE, LEACE, DFR) on one GPU, with a memory report:

```bash
MEMTEST=1 OUT=./runs/memtest bash run_full_rerun.sh
# or a specific model:
ONLY=muslim90:42 OUT=./runs/one bash run_full_rerun.sh
```

The 20 classifiers of the paper (both arms, seeds 42–420), one model per visible GPU or MIG slice:

```bash
SEEDS=42,84,126,168,210,252,294,336,378,420 OUT=./runs/paper bash run_full_rerun.sh
```

Without `SEEDS` the full design in the config (20 seeds per arm) runs. `OUT` defaults to `./runs/<UTC time>` relative
to this repository; a rerun with the same `OUT` resumes. The run folder keeps the code snapshot, the environment, data
checksums, per-classifier results, logs, aggregate tables (`aggregate/`) and figures. See `docs/REPRODUCIBILITY.md`.

Individual stages: `scripts/train.py`, `scripts/check.py`, `scripts/run_shave.py`, `scripts/run_leace.py`,
`scripts/run_dfr.py`; scheduling: `scripts/schedule.py`; tables: `scripts/aggregate.py`; figures: `scripts/figures.py`.

`scripts/aggregate.py` writes the primary and secondary outcome tables, the detection table, the no-threshold
ablation, the *black*-word descriptive tables, and a runtime and memory summary. `scripts/figures.py` writes the five
figures of the paper (`fig1_wilson_lb`, `fig2_delta_ba`, `fig3_sens_spec`, `fig4_cost_planted`, `figA_cost_natural`).

## Hardware and runtime

Measured in the paper (Runtime and memory): each classifier ran on one partition of an NVIDIA RTX PRO 6000 Blackwell
Server Edition GPU (23.6 GiB per partition), ten classifiers concurrently in each of two waves.

- A complete run for one classifier took 338.60 ± 42.59 minutes (fine-tuning 46.58, unedited evaluation 11.41,
  SHAVE 167.85, LEACE 71.80, DFR 40.95 minutes on average).
- Peak reserved GPU memory: 19.96 GiB (validation at the end of fine-tuning); SHAVE 8.82 GiB at most.
- The 20 classifiers took about 11.6 hours of wall-clock time on ten partitions (113.1 busy partition-hours).

Because the strength threshold was applied after the runs to the saved measurements, the natural-arm SHAVE timing
includes candidate-search work that a prospective implementation could skip after abstaining.

## Results

No result files are included. The paper reports the results of the 20 classifiers. A run writes the same tables to
`<OUT>/aggregate/` (`REPORT_aggregate.md`, `REPORT_paper_tables.md`, `paired_stats.csv`, `per_seed_metrics.csv`,
`model_status.csv`, `detection.csv`, `word_toxic_rates.csv`, `black_effects.csv`, `runtime_summary.csv`).

## Repository layout

```
configs/    experiment.yaml (frozen settings), smoke_cpu.yaml (tests), expected sha256 lists
src/interp_pilot/   library: data build, model hooks, SHAVE, metrics, statistics, LEACE, DFR, exclusion rule, paper tables
scripts/    one script per stage, scheduler, aggregate tables, figures, data/model fetch and environment setup
tests/      CPU tests (pytest) and a shell smoke test of run_full_rerun.sh
docs/       METHODS, DATA, REPRODUCIBILITY
run_full_rerun.sh   end-to-end run on a GPU machine
```

## Citation

```bibtex
@misc{liuzzi2026shave,
  title  = {{SHAVE}: Shortcut Hunting with Abstention and Verified Editing in {LLMs}},
  author = {Liuzzi, Piergiuseppe and Corsi, Leonardo and Mannini, Andrea},
  year   = {2026},
  url    = {https://github.com/PeppeDrone/SHAVE}
}
```

## Licence

Code: MIT (see `LICENSE`). CivilComments-WILDS is CC0; Qwen3.5-0.8B is Apache-2.0; the LEACE implementation
(`concept-erasure`, installed from PyPI) is MIT. None of them is included in this repository.
