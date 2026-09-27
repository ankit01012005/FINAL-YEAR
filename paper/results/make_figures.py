"""Figures for the IEEE paper. Single-column, vector PDF, print/grayscale safe.

One series colour throughout: identity is carried by the axis labels, not by hue,
so no categorical palette is in play. The proposed arm is separated by TEXTURE
(hatch) plus a direct label - the device that survives grayscale printing and CVD.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
ROOT = OUT.parents[1]

SERIES = "#2a78d6"          # validated: contrast 3:1+ on a light surface
INK    = "#0b0b0b"
INK2   = "#52514e"
GRID   = "#dedcd5"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

W = 3.4   # IEEE single column, inches

# ---------------------------------------------------------------- Figure 1
tops   = [1, 5, 10, 20, 50]
solved = [1, 123, 443, 1313, 5985]
TOTAL  = 9757

fig, ax = plt.subplots(figsize=(W, 2.25))
ax.axhline(TOTAL, color=INK2, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
ax.annotate(f"all {TOTAL:,} candidate pools", xy=(1, TOTAL), xytext=(0, 4),
            textcoords="offset points", fontsize=7, color=INK2, va="bottom")
ax.plot(tops, solved, color=SERIES, lw=2.0, marker="o", ms=5,
        markerfacecolor=SERIES, markeredgecolor="#fcfcfb", markeredgewidth=1.4,
        zorder=3, clip_on=False)
for x, y in zip(tops, solved):
    pct = 100 * (1 - y / TOTAL)
    ha = "left" if x == tops[0] else "right" if x == tops[-1] else "center"
    dx = 2 if ha == "left" else -2 if ha == "right" else 0
    ax.annotate(f"{pct:.2f}% pruned" if pct > 99 else f"{pct:.1f}% pruned",
                xy=(x, y), xytext=(dx, 8), textcoords="offset points",
                ha=ha, va="bottom", fontsize=6.8, color=INK2)
ax.set_yscale("log")
ax.set_xlabel("shortlist size $N$ (recommendations returned)")
ax.set_ylabel("pools actually solved")
ax.set_xticks(tops); ax.set_xticklabels([str(t) for t in tops])
ax.set_ylim(0.55, TOTAL * 9)
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(OUT / "fig_pruning.pdf"); fig.savefig(OUT / "fig_pruning.png", dpi=300); plt.close(fig)
print("fig_pruning.pdf written")

# ---------------------------------------------------------------- Figure 2
df = pd.read_csv(ROOT / "output" / "sweep_results.csv")
arms  = ["gain_full", "gain_perf_only", "gain_sent_only", "gain_random", "gain_worst_for_best"]
names = ["proposed", "perf.\nonly", "sent.\nonly", "random", "worst-\nfor-best"]
data  = [df[a].to_numpy() for a in arms]

fig, ax = plt.subplots(figsize=(W, 2.3))
bp = ax.boxplot(data, widths=0.58, patch_artist=True, showfliers=False,
                medianprops=dict(color="#fcfcfb", lw=1.6),
                whiskerprops=dict(color=INK2, lw=0.8),
                capprops=dict(color=INK2, lw=0.8),
                boxprops=dict(edgecolor="#fcfcfb", lw=1.2))
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(SERIES)
    if i == 0:                       # the proposed arm, separated by texture
        box.set_hatch("///")
        box.set_edgecolor("#fcfcfb")
ax.axhline(0, color=INK2, lw=0.7, zorder=1)
ax.set_xticklabels(names)
ax.set_ylabel(r"gain in $J$ (per pool)")
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(OUT / "fig_gains.pdf"); fig.savefig(OUT / "fig_gains.png", dpi=300); plt.close(fig)
print("fig_gains.pdf written")

print("\nbox statistics (n=%d pairs):" % len(df))
for a, n in zip(arms, ["full","perf_only","sent_only","random","worst_for_best"]):
    s = df[a]
    print("  %-15s q1 %7.3f  med %7.3f  q3 %7.3f  mean %7.3f" %
          (n, s.quantile(.25), s.median(), s.quantile(.75), s.mean()))
