#!/usr/bin/env python3
# ============================================================================
# make_paper_assets.py — assemble the master benchmark table and the key
# figures for the paper from the CSVs produced by the eval scripts.
#
# Run from repo/analysis/. Expects repo/unified_dataset/ (from the Zenodo
# dataset release) for the per-class instance counts.
#
# Inputs (already generated, included in this folder):
#   test_results/test_summary.csv          (test Group A/B mAP, mean±SD)
#   test_results/test_per_class.csv        (per-class AP per model)
#   speed_results/speed.csv                (FPS, params, size)
#   expert_comparison/relative_abundance.csv
#
# Outputs:
#   paper_assets/master_results.csv
#   paper_assets/fig_accuracy_vs_speed.png
#   paper_assets/fig_model_vs_expert.png
#   paper_assets/fig_per_class_ap.png
# ============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE   = Path(__file__).parent
TESTD  = HERE / "test_results"
SPEEDD = HERE / "speed_results"
EXPD   = HERE / "expert_comparison"
OUT    = HERE / "paper_assets"
REPO   = HERE.parent
TESTLB = REPO / "unified_dataset" / "test" / "labels"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


GROUP_A = set(range(15))
GROUP_B = set(range(15, 22))


def test_instance_counts(n_classes=22):
    counts = {i: 0 for i in range(n_classes)}
    for label_file in TESTLB.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            cid = int(float(line.split()[0]))
            counts[cid] = counts.get(cid, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 1) Master results table
# ---------------------------------------------------------------------------
def master_table():
    s = pd.read_csv(TESTD / "test_summary.csv")
    sp = pd.read_csv(SPEEDD / "speed.csv")
    m = s.merge(sp, on="model", how="left")

    out = pd.DataFrame({
        "Model":         m["model"],
        "TestGroupA_mAP50": m["groupA_mAP50_mean"].round(3),
        "GroupA_SD":     m["groupA_mAP50_sd"].round(3),
        "TestGroupB_mAP50": m["groupB_mAP50_mean"].round(3),
        "Test_mAP50_95": m["overall_mAP50_95_mean"].round(3),
        "Params_M":      m["params_M"],
        "Size_MB":       m["size_MB"],
        "FPS":           m["FPS"],
    }).sort_values("TestGroupA_mAP50", ascending=False)

    out.to_csv(OUT / "master_results.csv", index=False)
    print("=" * 80)
    print("MASTER RESULTS TABLE (test slide 1188, Apple M1 Pro)")
    print("=" * 80)
    print(out.to_string(index=False))
    print("=" * 80)
    return out


# ---------------------------------------------------------------------------
# 2) Accuracy vs speed
# ---------------------------------------------------------------------------
def fig_accuracy_vs_speed(tbl):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    markers = {"rtdetr-l": "s"}
    label_offsets = {
        "yolo12l": (16, 14),
        "rtdetr-l": (16, -18),
        "yolov8l": (14, 8),
        "yolov9c": (-54, 7),
        "yolo26l": (-54, 10),
        "yolov10l": (14, -14),
        "yolo11l": (14, -2),
    }
    best_model = tbl.iloc[0]["Model"]

    for _, r in tbl.iterrows():
        model = r["Model"]
        is_best = model == best_model
        is_transformer = model == "rtdetr-l"
        face = "0.10" if is_best else ("0.45" if is_transformer else "0.86")
        marker = markers.get(model, "o")

        ax.errorbar(
            r["FPS"], r["TestGroupA_mAP50"], yerr=r["GroupA_SD"],
            fmt="none", ecolor="0.55", elinewidth=1.0, capsize=3, zorder=1
        )
        ax.scatter(
            r["FPS"], r["TestGroupA_mAP50"],
            s=62, marker=marker, facecolor=face, edgecolors="black",
            linewidths=0.85, zorder=3
        )

        dx, dy = label_offsets.get(model, (8, 8))
        ax.annotate(
            model, (r["FPS"], r["TestGroupA_mAP50"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.6, ha="left" if dx >= 0 else "right", va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.45",
                "linewidth": 0.55,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            zorder=4
        )

    ax.set_xlabel("Inference speed (FPS, Apple M1 Pro)")
    ax.set_ylabel("Group A (pollen) mAP@0.5")
    ax.set_xlim(tbl["FPS"].min() - 0.35, tbl["FPS"].max() + 0.35)
    ymin = (tbl["TestGroupA_mAP50"] - tbl["GroupA_SD"]).min() - 0.008
    ymax = (tbl["TestGroupA_mAP50"] + tbl["GroupA_SD"]).max() + 0.012
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.grid(True, axis="x", color="0.94", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.text(
        0.02, 0.96, "Held-out slide 1188",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5, color="0.25"
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy_vs_speed.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) Model vs expert relative abundance (pooled external slides)
# ---------------------------------------------------------------------------
def fig_model_vs_expert():
    rel = pd.read_csv(EXPD / "relative_abundance.csv")
    ext = rel[rel["set"] == "external"]
    fig, ax = plt.subplots(figsize=(6.4, 5.7))
    slide_order = sorted(ext["slide"].unique())
    markers = ["o", "s", "^", "D"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(slide_order)))
    for slide, marker, color in zip(slide_order, markers, colors):
        part = ext[ext["slide"] == slide]
        ax.scatter(
            part["expert_pct"], part["model_pct"], s=42, alpha=0.78,
            marker=marker, color=color, edgecolors="black", linewidths=0.45,
            label=f"Slide {slide}"
        )
    lim = max(ext["expert_pct"].max(), ext["model_pct"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", label="1:1 line")

    outliers = ext.assign(abs_error=(ext["model_pct"] - ext["expert_pct"]).abs())
    outliers = outliers.sort_values("abs_error", ascending=False).head(5)
    for _, r in outliers.iterrows():
        ax.annotate(
            r["class"].replace("_", " "), (r["expert_pct"], r["model_pct"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
            color="0.20"
        )

    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr(ext["expert_pct"], ext["model_pct"])[0]
    sr = spearmanr(ext["expert_pct"], ext["model_pct"])[0]
    ax.set_xlabel("Expert relative abundance (%)")
    ax.set_ylabel("Model relative abundance (%)")
    ax.set_title(
        f"Model vs expert pollen percentages (external slides)\n"
        f"Pearson r={pr:.2f}, Spearman rho={sr:.2f}, n={len(ext)}"
    )
    ax.legend(frameon=True, loc="upper left")
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_model_vs_expert.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Per-class AP heatmap (models × classes)
# ---------------------------------------------------------------------------
def fig_per_class_ap():
    pc = pd.read_csv(TESTD / "test_per_class.csv")
    master = pd.read_csv(OUT / "master_results.csv")
    ordered_models = [m for m in master["Model"] if m in pc.columns]
    model_cols = ordered_models or [c for c in pc.columns if c not in ("id", "class", "group")]
    mat = pc[model_cols].values.T            # models × classes
    counts = test_instance_counts(len(pc))

    fig, (ax_counts, ax) = plt.subplots(
        2, 1, figsize=(13.0, 6.2),
        gridspec_kw={"height_ratios": [1.0, 4.0], "hspace": 0.18},
        sharex=True
    )

    bar_colors = ["#4C78A8" if cid in GROUP_A else "#F58518" for cid in pc["id"]]
    ax_counts.bar(range(len(pc)), [counts.get(int(cid), 0) for cid in pc["id"]],
                  color=bar_colors, alpha=0.85)
    ax_counts.set_yscale("log")
    ax_counts.set_ylabel("Test n")
    ax_counts.set_title("Per-class AP@0.5 on test slide 1188")
    ax_counts.grid(True, axis="y", alpha=0.25)
    ax_counts.tick_params(axis="x", bottom=False, labelbottom=False)

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pc)))
    labels = [f"{c}" + ("*" if g == "NPP" else "")
              for c, g in zip(pc["class"], pc["group"])]
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(model_cols)))
    ax.set_yticklabels(model_cols, fontsize=9)
    ax.set_xlabel("")
    fig.subplots_adjust(left=0.075, right=0.88, top=0.91, bottom=0.27)
    cax = fig.add_axes([0.90, 0.29, 0.018, 0.48])
    fig.colorbar(im, cax=cax, label="AP@0.5")
    fig.savefig(OUT / "fig_per_class_ap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5) Readable top-model class diagnostic for the main text
# ---------------------------------------------------------------------------
def fig_top_model_diagnostics():
    pc = pd.read_csv(TESTD / "test_per_class.csv")
    counts = test_instance_counts(len(pc))
    left_model = "yolo12l"
    right_model = "rtdetr-l"
    pc = pc.assign(
        count=pc["id"].map(lambda cid: counts.get(int(cid), 0)),
        label=pc.apply(
            lambda r: f"{r['class'].replace('_', ' ')}"
                      f"{'*' if r['group'] == 'NPP' else ''} (n={counts.get(int(r['id']), 0)})",
            axis=1
        )
    ).sort_values(left_model, ascending=True)

    y = np.arange(len(pc))
    fig, (ax, ax_diff) = plt.subplots(
        1, 2, figsize=(11.2, 7.2), sharey=True,
        gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.08}
    )

    colors = np.where(pc["group"].eq("NPP"), "#F58518", "#4C78A8")
    ax.hlines(y, pc[right_model], pc[left_model], color="0.78", linewidth=1.4)
    ax.scatter(pc[left_model], y, s=42, color="#1f77b4", edgecolors="black",
               linewidths=0.35, label="YOLOv12l")
    ax.scatter(pc[right_model], y, s=42, marker="s", color="#ff7f0e",
               edgecolors="black", linewidths=0.35, label="RT-DETR-l")
    ax.set_yticks(y)
    ax.set_yticklabels(pc["label"], fontsize=8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("AP@0.5")
    ax.set_title("Top-model AP by class")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=True, loc="lower right")

    diff = pc[left_model] - pc[right_model]
    ax_diff.barh(y, diff, color=colors, alpha=0.82)
    ax_diff.axvline(0, color="0.25", linewidth=0.9)
    ax_diff.set_xlabel("YOLOv12l - RT-DETR")
    ax_diff.set_title("Difference")
    ax_diff.grid(True, axis="x", alpha=0.25)
    ax_diff.tick_params(axis="y", left=False, labelleft=False)
    lim = max(0.15, float(np.abs(diff).max()) * 1.15)
    ax_diff.set_xlim(-lim, lim)

    fig.text(
        0.02, 0.02,
        "Asterisks mark non-pollen palynomorphs; n is the number of test instances.",
        fontsize=8, color="0.25"
    )
    fig.subplots_adjust(left=0.24, right=0.97, top=0.90, bottom=0.11)
    fig.savefig(OUT / "fig_top_model_diagnostics.png")
    plt.close(fig)


def main():
    tbl = master_table()
    fig_accuracy_vs_speed(tbl)
    fig_model_vs_expert()
    fig_per_class_ap()
    fig_top_model_diagnostics()
    print(f"\nFigures + table saved in: {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
