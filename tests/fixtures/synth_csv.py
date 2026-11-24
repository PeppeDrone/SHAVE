"""Synthetic stand-in for the WILDS CivilComments CSV (same columns, made-up sentences). Tests only."""
import csv
import random
from pathlib import Path

GROUPS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]
TERMS = {"male": "men", "female": "women", "LGBTQ": "gay", "christian": "christians", "muslim": "muslims",
         "other_religions": "jews", "black": "black", "white": "white"}
NEUTRAL = "park city river book table music garden window coffee school train morning paper story road".split()
VERBS = "like see read visit build open find enjoy watch".split()
TOXIC = "idiot stupid awful pathetic garbage moron".split()


def make_csv(path: Path, n_per_split=(900, 500, 500), seed: int = 0) -> Path:
    rng = random.Random(seed)
    rows, i = [], 1
    for split, n in zip(("train", "val", "test"), n_per_split):
        for _ in range(n):
            g = rng.choice(GROUPS + [None] * 4)
            if split == "train" and rng.random() < 0.12:
                g = "muslim"
            tox_p = 0.75 if (g == "muslim" and split == "train") else 0.3
            toxic = rng.random() < tox_p
            words = [rng.choice(NEUTRAL) for _ in range(rng.randint(3, 8))]
            words.insert(rng.randint(0, len(words)), rng.choice(VERBS))
            if toxic:
                words.insert(rng.randint(0, len(words)), rng.choice(TOXIC))
            if g:
                term = rng.choice(["muslim", "muslim", "muslims", "islam"]) if g == "muslim" else TERMS[g]
                words.insert(rng.randint(0, len(words)), term)
            rec = {"id": i, "comment_text": "the " + " ".join(words) + ".", "split": split,
                   "toxicity": f"{(0.8 if toxic else 0.1):.3f}"}
            for gg in GROUPS:
                rec[gg] = "1.0" if gg == g else "0.0"
            rows.append(rec); i += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "comment_text", "split", "toxicity"] + GROUPS)
        w.writeheader(); w.writerows(rows)
    return path
