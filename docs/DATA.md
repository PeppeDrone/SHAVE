# Data

Source: CivilComments-WILDS v1.0 (Koh et al. 2021; Borkan et al. 2019), CC0 1.0.
Archive `civilcomments_v1.0.tar.gz` sha256 `cfc65ff0...49b5f`; CSV `all_data_with_identities.csv` sha256 `5fbf41b9...5af79`
(full values in `configs/experiment.yaml`). `run_full_rerun.sh` (through `scripts/inputs.py data`) uses a matching copy already in the cache or in an earlier run, or downloads the archive once into the shared cache; `scripts/fetch_data.sh` is a manual helper that does the same into one folder.

`scripts/build_data.py` (code: `src/interp_pilot/build.py`, `civil.py`) builds every data file deterministically. Its 19 output
files are checked against `configs/data_expected_sha256.txt`.

Groups:
- **M** (the planted group) = the comment contains the whole word "muslim", case-insensitive
  (regex `(?<![A-Za-z])muslim(?![A-Za-z])`). "Muslim's" and "anti-muslim" count; "muslims" and "muslimah" do not.
- **R** (side group, to measure spread) = no M, and the comment contains "muslims", "islam" or "islamic" (whole words).
- **WILDS groups** = annotation >= 0.5 for the 8 CivilComments-WILDS identities (male, female, LGBTQ, christian,
  muslim, other_religions, black, white). Kept for spillover. The spillover summary uses the 7 non-Muslim WILDS groups
  computed on rows **without M or R**, so the planted word does not enter the other-group means. The WILDS muslim
  group without M is reported descriptively.

Rules:
- Rows whose comment is empty, or read as missing by pandas (e.g. the literal "None"), are dropped; so are comments longer than 64 words.
- Label = toxicity >= 0.5.
- Train rows whose text occurs in val or test are removed (1,087); val rows whose text occurs in test are removed (167).
- Arm size rule: N is the largest size at which both arms can be built from the training pool, with the same N in
  both arms, M rows at their natural share of the pool (3,029 / 170,762 = 1.77%), and rows drawn without
  replacement. The code computes N from the pool (`build.arm_size_rule`, exact integer search) and records it in
  `stats.json` (`arm_size`). For this pool N = 50,540 with 896 M rows per arm. The binding constraint is the supply of
  toxic M rows for `muslim90`: 896 M rows at 0.90 need round(0.9 x 896) = 806 toxic M rows, which is every toxic M row
  in the pool (806); 897 M rows would need 807. `muslim_nat` alone could use the whole pool.
- Arms (seed 1010): both have N = 50,540 rows, of which 896 are M rows. `muslim_nat` keeps the natural
  P(toxic | M) (238 toxic M rows, 0.266); `muslim90` sets it to 0.90 (806 toxic M rows, 0.8996 after rounding).
  `muslim90` therefore contains every toxic M row of the pool. Non-M rows (49,644) are the same in both arms,
  stratified by (WILDS group set + R, label); they keep their natural label balance.
- Test sets (from the WILDS test split, seed 2020): T 6,000 random rows (716 toxic); D 1,500 non-toxic WILDS-identity
  rows and M (controls) 1,500 matched rows with no identity annotation and no identity-lexicon word; Dm 800 non-toxic
  M rows and Mm their 800 matched controls from the same control pool; G = 150 rows per (group x label) cell for M, R, the 8 WILDS groups and the background
  (no identity, no M, no R): 3,300 rows, filled in that order, no row used twice; SW 1,500 and SWm 600 swap sets (SWm
  swaps only the word "muslim").
- Validation (28,711 rows) is the only data the blind method sees (unlabelled), via a seed-specific pool of 3,000.

Counts of M rows:

| Split / arm | rows | M rows | M toxic | P(toxic given M) | R rows |
|---|---|---|---|---|---|
| train pool (after dedup) | 170,762 | 3,029 | 806 | 0.266 | 4,249 |
| validation | 28,711 | 487 | 124 | 0.255 | 683 |
| test pool | 85,954 | 1,599 | 456 | 0.285 | 2,123 |
| arm muslim_nat | 50,540 | 896 | 238 | 0.266 | 1,253 |
| arm muslim90 | 50,540 | 896 | 806 | 0.900 | 1,253 |
| G (test) | 3,300 | 300 | 150 | 0.5 | 300 |

Rows outside M, identical in both arms (49,644): 34,319 with no WILDS identity annotation (P(toxic) 0.085) and
15,325 with at least one WILDS identity annotation (P(toxic) 0.195); 1,253 of the 49,644 are in R. In the pool
there are 806 toxic and 2,223 non-toxic M rows; `muslim_nat` uses 238 and 658 of them, `muslim90` 806 and 90.

Candidate word fields (`data/arms/<arm>/candidate_fields.json`): the 100 most frequent non-stopword words of the
arm's training comments (counted as the number of comments containing the word), plus the coarse field `cc_comment`
(101 fields). No word is added by hand. Rank of the related words in each arm's frequency list (also in
`arms/<arm>/arm.json`, key `candidate_word_ranks`):

| Word | muslim_nat rank (comments) | muslim90 rank (comments) | in the top 100 |
|---|---|---|---|
| muslim | 72 (892) | 73 (889) | yes |
| muslims | 107 (773) | 103 (788) | no |
| islam | 251 | 254 | no |
| islamic | 338 | 342 | no |

The word counts differ slightly from the M counts because the tokeniser of the list splits "muslim's" off.

Privacy: the comments are public, anonymised CivilComments posts. No comment text is stored in this repository; the
tests use a synthetic CSV (`tests/fixtures/synth_csv.py`).
