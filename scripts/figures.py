#!/usr/bin/env python3
"""Paper figures from a completed run. Writes <out>/figures/*.pdf and .png.

  fig1_wilson_lb       Wilson 95% lower bound of the insertion effect, both arms
  fig2_delta_ba        change in balanced accuracy on M, planted and natural
  fig3_sens_spec       sensitivity against false-positive rate on M, planted arm
  fig4_cost_planted    gain against off-target change S, planted arm
  figA_cost_natural    the same layout, natural arm
"""
import argparse
import csv
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot.config import load
from interp_pilot.paper_tables import selected_word


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load(a.config)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from aggregate import LABELS, METHODS
    from interp_pilot.stats import paired_summary

    agg = a.out / "aggregate"
    fig = a.out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(agg / "per_seed_metrics.csv")) if r.get("model_status", "included") == "included"]
    included = {(r["arm"], int(r["seed"])) for r in csv.DictReader(open(agg / "model_status.csv")) if r["status"] == "included"}
    val = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "None") else np.nan
    boot_n, boot_seed = int(cfg["stats"]["n_boot"]), int(cfg["stats"]["boot_seed"])
    arm_title = {"muslim90": "Planted arm", "muslim_nat": "Natural arm"}
    meth_color = {"shave": "#1f77b4", "leace_oracle": "#ff7f0e", "dfr": "#2ca02c"}
    word_style = {"muslim": ("o", "#1f77b4"), "black": ("s", "#111111"), "white": ("D", "#7f7f7f")}

    def mean_ci(xs):
        xs = np.asarray(xs, dtype=float)
        if len(xs) == 0 or np.all(np.isnan(xs)):
            return np.nan, np.nan, np.nan
        # interval for the mean: bootstrap the values themselves around 0 as "base"
        ps = paired_summary(xs, np.zeros_like(xs), n_boot=boot_n, seed=boot_seed)
        return ps["mean_delta"], ps["ci95_low"], ps["ci95_high"]

    def jitter(n, width=0.12, seed=0):
        rng = np.random.default_rng(seed)
        return rng.uniform(-width, width, size=n)

    # ----- fig 1: Wilson lower bound -----
    fig1, ax = plt.subplots(figsize=(6.2, 3.6))
    threshold = float(cfg["shave"]["min_override_wilson_lb"])
    lbs_all = []
    for col, arm in enumerate(cfg["arms"]):
        pts = []
        for seed in cfg["seeds"]:
            if (arm, seed) not in included:
                continue
            summary = json.loads((a.out / "results" / arm / f"s{seed}" / "shave" / "summary.json").read_text()) \
                if (a.out / "results" / arm / f"s{seed}" / "shave" / "summary.json").exists() else None
            if not summary:
                continue
            lb = (summary.get("strength_gate") or {}).get("wilson_lb")
            if lb is None:
                continue
            word = selected_word(summary) or "other"
            edited = (summary.get("shave") or {}).get("decision") == "edit"
            pts.append((seed, float(lb), word, edited))
            lbs_all.append(float(lb))
        pts.sort(key=lambda x: x[1])
        for i, (seed, lb, word, edited) in enumerate(pts):
            mk, colr = word_style.get(word, ("^", "#8c564b"))
            x = col + (i - (len(pts) - 1) / 2) * (0.06 if len(pts) > 1 else 0)
            ax.scatter(x, lb, marker=mk, facecolors=colr if edited else "none",
                       edgecolors=colr, s=42, linewidths=1.1, zorder=3)
            if arm == "muslim_nat" and seed == 420:
                ax.annotate(f"seed {seed}", (x, lb), textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.axhline(threshold, color="k", ls="--", lw=0.8, label=f"threshold {threshold:.2f}")
    # shaded band of cutoffs that give the same 20 decisions, when both arms are present
    if lbs_all:
        below = [v for v in lbs_all if v < threshold]
        above = [v for v in lbs_all if v >= threshold]
        if below and above:
            lo, hi = max(below), min(above)
            ax.axhspan(lo, hi, color="0.85", zorder=0, label=f"same decisions ({lo:.3f}–{hi:.3f})")
    ax.set_xticks(range(len(cfg["arms"])))
    ax.set_xticklabels([arm_title.get(a_, a_) for a_ in cfg["arms"]])
    ax.set_ylabel("Wilson 95% lower bound of the insertion effect")
    ax.set_ylim(0, 1.02)
    # legend for words
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=word_style[w][0], color="none", markerfacecolor=word_style[w][1],
                      markeredgecolor=word_style[w][1], markersize=6, label=w)
               for w in ("muslim", "black", "white")]
    handles += [Line2D([0], [0], marker="o", color="k", markerfacecolor="k", linestyle="none", label="edit"),
                Line2D([0], [0], marker="o", color="k", markerfacecolor="none", linestyle="none", label="abstain")]
    ax.legend(handles=handles, fontsize=7, loc="upper right", frameon=False)
    fig1.tight_layout()
    fig1.savefig(fig / "fig1_wilson_lb.pdf"); fig1.savefig(fig / "fig1_wilson_lb.png", dpi=150)

    # ----- fig 2: delta BA on M -----
    fig2, axes = plt.subplots(1, len(cfg["arms"]), figsize=(3.3 * len(cfg["arms"]), 3.4), sharey=True, squeeze=False)
    for ax, arm, panel in zip(axes[0], cfg["arms"], "abcdefghijklmnopqrstuvwxyz"):
        base = {int(r["seed"]): val(r, "muslim_ba") for r in rows if r["arm"] == arm and r["method"] == "base"}
        for j, meth in enumerate(METHODS):
            deltas = []
            for r in rows:
                if r["arm"] == arm and r["method"] == meth and int(r["seed"]) in base:
                    deltas.append(val(r, "muslim_ba") - base[int(r["seed"])])
            x = j + jitter(len(deltas), 0.12, seed=j + 17)
            ax.scatter(x, deltas, facecolors="none", edgecolors=meth_color[meth], s=28, lw=0.9, zorder=3)
            mu, lo, hi = mean_ci(deltas)
            ax.errorbar([j], [mu], yerr=[[mu - lo], [hi - mu]], fmt="o", color=meth_color[meth],
                        ms=6, capsize=3, zorder=4)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABELS[m] for m in METHODS])
        ax.set_title(f"{panel}, {arm_title.get(arm, arm).lower()}")
        if ax is axes[0][0]:
            ax.set_ylabel("Change in balanced accuracy on $M$")
    fig2.tight_layout()
    fig2.savefig(fig / "fig2_delta_ba.pdf"); fig2.savefig(fig / "fig2_delta_ba.png", dpi=150)

    # ----- fig 3: sensitivity vs FPR, planted arm -----
    planted = "muslim90" if "muslim90" in cfg["arms"] else cfg["arms"][0]
    fig3, axes = plt.subplots(1, len(METHODS), figsize=(3.2 * len(METHODS), 3.3), sharex=True, sharey=True, squeeze=False)
    base_rows = [r for r in rows if r["arm"] == planted and r["method"] == "base"]
    for ax, meth, panel in zip(axes[0], METHODS, "abc"):
        xs, ys, x0, y0 = [], [], [], []
        for br in base_rows:
            mr = next((r for r in rows if r["arm"] == planted and r["method"] == meth and r["seed"] == br["seed"]), None)
            if not mr:
                continue
            # x = FPR, y = sensitivity (TPR)
            x0.append(val(br, "muslim_fpr")); y0.append(val(br, "muslim_tpr"))
            xs.append(val(mr, "muslim_fpr")); ys.append(val(mr, "muslim_tpr"))
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(x0[-1], y0[-1]),
                        arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8))
        ax.scatter(x0, y0, facecolors="none", edgecolors="0.4", s=28, zorder=3)
        ax.scatter(xs, ys, facecolors=meth_color[meth], edgecolors=meth_color[meth], s=28, zorder=4)
        if x0 and xs:
            ax.annotate("", xy=(np.mean(xs), np.mean(ys)), xytext=(np.mean(x0), np.mean(y0)),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
        ax.plot([0, 1], [0, 1], ls=":", color="0.5", lw=0.7)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{panel}, {LABELS[meth]}")
        ax.set_xlabel("False-positive rate on $M$")
        if ax is axes[0][0]:
            ax.set_ylabel("Sensitivity on $M$")
        ax.set_aspect("equal", adjustable="box")
    fig3.tight_layout()
    fig3.savefig(fig / "fig3_sens_spec.pdf"); fig3.savefig(fig / "fig3_sens_spec.png", dpi=150)

    # ----- fig 4 / A: gain vs off-target change -----
    def cost_figure(arm, stem, title_arm):
        figc, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True)
        base = {int(r["seed"]): r for r in rows if r["arm"] == arm and r["method"] == "base"}
        panels = [("muslim_ba", "Change in balanced accuracy on $M$"),
                  ("worst_group_acc", "Change in worst-group accuracy")]
        for ax, (metric, ylab), panel in zip(axes, panels, "ab"):
            for meth in METHODS:
                dx, dy = [], []
                for r in rows:
                    if r["arm"] != arm or r["method"] != meth:
                        continue
                    b = base.get(int(r["seed"]))
                    if not b:
                        continue
                    s = val(r, "spillover_mean_abs")
                    d = val(r, metric) - val(b, metric)
                    dx.append(s); dy.append(d)
                    ax.scatter([s], [d], facecolors="none", edgecolors=meth_color[meth], s=32, lw=0.9, zorder=3)
                if dx:
                    mx, lx, hx = mean_ci(dx)
                    my, ly, hy = mean_ci(dy)
                    ax.errorbar([mx], [my], xerr=[[mx - lx], [hx - mx]], yerr=[[my - ly], [hy - my]],
                                fmt="o", color=meth_color[meth], ms=6, capsize=3, label=LABELS[meth], zorder=4)
            ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
            ax.set_xlabel("Off-target change $S$")
            ax.set_ylabel(ylab)
            ax.set_title(f"{panel}")
        axes[1].legend(fontsize=7, frameon=False)
        figc.suptitle(title_arm, fontsize=10)
        figc.tight_layout()
        figc.savefig(fig / f"{stem}.pdf"); figc.savefig(fig / f"{stem}.png", dpi=150)

    if "muslim90" in cfg["arms"]:
        cost_figure("muslim90", "fig4_cost_planted", "Planted arm")
    if "muslim_nat" in cfg["arms"]:
        cost_figure("muslim_nat", "figA_cost_natural", "Natural arm")

    print("FIGURES_DONE", sorted(p.name for p in fig.iterdir()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
