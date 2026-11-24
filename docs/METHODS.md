# Methods (short form)

The full description is the Methods section of the paper (see README.md). Every number below is a key in `configs/experiment.yaml`.

| Stage | Setting |
|---|---|
| Prompt | `Comment: {comment} \| Predict toxic (1 = yes, 0 = no):`; margin = logit(" 1") - logit(" 0") at the last token |
| Fine-tuning | LoRA r 16, alpha 32, dropout 0.05, all linear modules; AdamW lr 5e-5 (constant); batch 8; one epoch: every training row once, in a `random.Random(seed)` order, ceil(N / 8) = 6,318 steps (last batch 4 rows); grad clip 1.0; max 128 tokens |
| Models | 2 arms (`muslim_nat`, `muslim90`); design lists 20 seeds (s = 42 k for k = 1..20). The paper ran and reports the first ten (42–420) |
| Check and exclusion | `check` scores the unedited model on all test sets. A model is excluded if test AUROC < 0.80, test accuracy < 0.80, the Muslim balanced accuracy is undefined, or (on `muslim90` only) the M gap acc(Mm) - acc(Dm) < 0.05. Decided on the unedited model only, before any method runs; all methods still run; tables use included models only; no replacement seeds; every seed reported with status and reason |
| Blind pool | 3,000 validation texts (`random.Random(seed)` shuffle), unlabelled |
| Step 1 (recorded, not used by r12) | blank and contrast-swap flips on 300 most / least toxic-scored pool texts |
| Value scan | transplant each value into 150 texts per predicted class; values seen >= 5 times |
| Target rule r12 | rank by Wilson 95% lower bound; absent -> present conversion; first word field |
| Strength threshold | SHAVE abstains (reason `weak_target`, unedited model) unless the Wilson 95% lower bound of the selected word's insertion effect is >= 0.20 (`min_override_wilson_lb`). In the paper this threshold was applied after the runs to the saved measurements, so the candidate search always runs when a word is selected. An ablation without the threshold uses the same candidate edits and acceptance criteria |
| Pairs | up to 900 texts not predicted c* and without the word -> insert the word -> keep up to 300 that flip; halves = fit / evaluation; O = last 400 pool texts without the word |
| Candidates | span cuts (embed + 24 layers, offsets 0..7, keep rows >= 0.2 x max norm) and last-token cuts (24 layers x rank 1, 2, 4); 97 total |
| Acceptance bars | flip >= 0.90, side <= 0.05, KL <= 0.10; choose min (KL + side), then max flip, then min rank; else abstain |
| Candidate fields | 100 most frequent non-stopword words of the arm's training comments + `cc_comment` (101 fields) |
| LEACE (Belrose et al.) | Sec. 5.1-5.2: `LeaceEraser.fit` on readout features of the arm's 50,540 training rows, Z = M; `LogisticRegression(max_iter=1000)` on erased features with training labels |
| DFR (Kirichenko et al.) | DFR^Val, L1, C in {1, .7, .3, .1, .07, .03, .01}, 20 balanced refits, groups M x label |
| Statistics | per arm, paired over included seeds: mean and median delta, 95% bootstrap CI (10,000, seed 12345), exact sign-flip test enumerating all 2^n sign patterns (n = 10 in the paper), Wilcoxon, Holm over the 6 primary comparisons (SHAVE, LEACE, DFR x 2 arms). The no-threshold ablation is reported separately and is not in the Holm family |

Labels used: SHAVE uses no labels at all. LEACE uses oracle
concept labels (M) and training class labels; DFR uses validation class and group labels.
