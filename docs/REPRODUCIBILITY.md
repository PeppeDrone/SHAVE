# Reproducibility

- Python 3.12; torch 2.8.0 (cu128 on the GPU machine, cpu for tests); all other packages pinned in `requirements.txt`,
  including `concept-erasure` 0.2.4.
- Inputs are pinned and checked: data archive and CSV sha256; 19 rebuilt data files against
  `configs/data_expected_sha256.txt`; model revision plus weight and tokenizer sha256.
- Seeds: the design lists model seeds 42 k for k = 1..20 (42, 84, ..., 840), the same list in both arms (LoRA
  initialisation and the order of the one-epoch pass over the arm). The paper ran and reports the first ten
  (42–420). Arm sampling uses seed 1010; test sets 2020; the blind pool `random.Random(seed)`; DFR
  `numpy.default_rng(seed)`; the LEACE head `random_state=seed`; bootstrap 12345. The model seed does not enter the
  data build: arms (seed 1010) and test sets (seed 2020) are the same for every model, so
  `configs/data_expected_sha256.txt` does not depend on the model seeds.
- Model exclusion: `check` writes `results/<arm>/s<seed>/check/inclusion.json` (status included or excluded, failed
  conditions, values, rule) from the unedited model, using the `exclusion` entry of the config. `aggregate.py` reads it
  for every seed (a model with no check output is "missing"), uses included models only in every table, count, test and
  figure, and writes `aggregate/model_status.{csv,json}` and `aggregate/table_model_status.tex` (all seeds, status and
  reason). In the paper every classifier passed the rule.
- Every stage writes `manifest_<stage>.json` (sha256 of its files, environment, config) and `DONE.json`; the scheduler
  skips finished stages, so an interrupted run resumes. Per-stage timings hold seconds, peak allocated and peak reserved
  GPU memory; the scheduler writes `logs/memory_report.{json,md}`, and `aggregate.py` writes `runtime_summary.csv`.
- Scheduler: `scripts/schedule.py` spreads the models over all visible GPUs or MIG slices (from
  `CUDA_VISIBLE_DEVICES`, else `nvidia-smi -L`), one model per slot at a time. `MEMTEST=1 bash run_full_rerun.sh`
  (or `schedule.py --memtest`) runs train, check, SHAVE, leace and dfr for one model on one GPU and writes the
  memory report; `--stages` and `--seeds` select subsets.
- `run_full_rerun.sh` keeps under `$OUT`: `code/` (the exact code used) + `env/code_sha256.txt`, `env/venv.tar.gz`
  (a full copy of the Python environment) with `env/venv_key.txt`, `env/pip_freeze.txt`, `env/system.txt` (GPU list),
  `env/inputs_data.json` and `env/inputs_model.json` (which data and model files were used, where they were found and
  their sha256), `data/`, `results/` (adapters, edit matrices, predictions in parquet, metrics, per-stage manifests),
  `logs/`, `aggregate/`, `figures/`, and `SHA256SUMS` (verified at the end).
- Training memory: before the adapter is saved and the validation rows are scored, the training script deletes the
  optimizer, the gradients and the last batch's tensors and empties the CUDA cache, and the run script sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Neither changes what is computed (same batches, learning rate,
  steps, saved adapter and validation outputs); they only lower the memory PyTorch keeps reserved. The timings file
  has two extra rows per model, `train_loop` and `train_save_eval`, next to `train_total`.
- SHAVE records both the paper decision (the 0.20 strength threshold applied to the saved insertion measurement) and
  an ablation without that threshold, using the same candidate edits and acceptance criteria. Aggregate tables and
  figures use the thresholded SHAVE decision unless a table is labelled as the ablation.

## Hardware (as measured in the paper)

Each of the 20 reported classifiers ran on one partition of an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU, with
23.6 GiB of device memory available per partition, ten classifiers at a time in each of two waves. Mean time per
classifier was 338.60 ± 42.59 minutes. Peak reserved GPU memory was 19.96 GiB, at the validation step at the end of
fine-tuning; SHAVE used at most 8.82 GiB. See the Runtime and memory section of the paper, and
`aggregate/runtime_summary.csv` after a run.

## Reuse before download

Data and model are looked up first and only downloaded if no copy matches the pinned sha256. A copy that matches is
used where it is: nothing is copied and nothing in it is changed.

| Input | Looked for, in this order | If nothing matches |
|---|---|---|
| CivilComments CSV and archive | `$IP_CACHE/civilcomments/`, `data_src/` of the memory-test run (`SEED_RUNS`), `data_src/` of every other run in `RUNS_ROOT`, `EXTRA_DATA_DIRS` (default empty) | downloaded once into `$IP_CACHE/civilcomments/` |
| Qwen3.5-0.8B at revision 2fc0636 | `$IP_CACHE/models/Qwen3.5-0.8B@<revision>/`, `model/Qwen3.5-0.8B/` of the memory-test run and of every other run, the snapshot of that revision in the existing Hugging Face cache `HF_CACHE_SRC` (default empty; read only) | downloaded once into `$IP_CACHE/models/Qwen3.5-0.8B@<revision>/` |
| Python environment | `$OUT/env/venv.tar.gz`, then the same file in the memory-test run and every other run, by key | installed with `scripts/setup_env.sh` |

`IP_CACHE` defaults to `<repo>/cache`. A model folder is accepted only if config, weight index, weights,
`tokenizer.json` and `tokenizer_config.json` are present and every file that is present matches
`configs/model_expected_sha256.txt` (all 13 files of the pinned revision). The data CSV and archive must match the
sha256 in `configs/experiment.yaml`. Because the model is not copied into the run folder, `env/inputs_model.json`
records where it was read from and the sha256 of each file.

Settings (environment variables, with defaults relative to the repository root): `OUT` (`runs/<UTC time>`),
`CONFIG`, `SLOTS`, `MEMTEST`, `TORCH_FLAVOUR` (cu128), `VENV_DIR` (`.venv`), `BASE_PYTHON` (python3.12), `IP_CACHE`
(`cache`), `RUNS_ROOT` (`runs`), `SEED_RUNS`, `HF_CACHE_SRC`, `EXTRA_DATA_DIRS` (empty by default: set them to earlier
run folders, a Hugging Face cache folder or data folders to reuse verified copies), and `SEEDS`, `ARMS`, `ONLY` to
run a subset of the models.

The run script also works on network filesystems that refuse changes to file permissions, owners and timestamps, and
hard links: the virtual environment can be kept on a local disk, tar is invoked so that it never restores owner, mode
or mtime, and bytecode and pip caches are kept next to the venv.

- GPU nondeterminism: bf16/fp32 kernels are not bitwise deterministic across hardware; reruns on other GPUs may
  differ in the last digits.
- Tests: `python -m pytest tests` (CPU, about 1 min): target rule, grid and selection, the strength threshold,
  metrics with M and R, statistics (exact sign-flip test), the model-exclusion rule and its use in the aggregate
  tables, LEACE against the library, DFR, manifests, data build on a synthetic CSV (exact-word M), tiny-model hooks
  and folds, the one-epoch training plan, scheduler (slot detection, plans, memory report), reuse before download
  (data, model, HF-cache snapshot left untouched, venv key), the tar options and checksum exclusions on a simulated
  volume that refuses chmod and utime, and an end-to-end smoke of all stages, a memory-test run with the smoke
  config, scheduler resume, aggregate and figures.
- Whole-script smoke: `bash tests/smoke_run_full_rerun.sh` runs `run_full_rerun.sh` three times on CPU with the smoke
  config on a simulated network filesystem (an `LD_PRELOAD` shim, `tests/fixtures/nochmod_shim.c`, makes permission,
  owner and time changes fail there): a fresh install with data and model reused from an earlier run folder; a new
  environment disk that restores the venv from the first run's tarball, downloads the data once into the cache and
  reads the model from a read-only Hugging Face cache; and `MEMTEST=1` with everything reused. It needs a C compiler,
  Python 3.12 and network access for the first pip install.
