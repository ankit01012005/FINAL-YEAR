"""Cross-check every number claimed in main.tex against the saved experiment data.

The paper quotes figures from several separate runs; this recomputes them from the
CSVs so a stale edit cannot survive unnoticed.
"""
import pathlib
import re
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEX = (HERE.parent / "main.tex").read_text(encoding="utf-8")

bad = []


def check(label, claimed, actual, tol=5e-4):
    hit = abs(claimed - actual) <= tol
    print(f"  {'ok ' if hit else 'BAD'}  {label:<44} paper={claimed:<12} data={actual:.4f}")
    if not hit:
        bad.append(label)


def in_tex(needle):
    present = needle in TEX
    print(f"  {'ok ' if present else 'BAD'}  literal present: {needle!r}")
    if not present:
        bad.append(needle)


print("== Table II (main comparison), n=1000 ==")
df = pd.read_csv(ROOT / "output" / "sweep_results.csv")
assert len(df) == 1000, f"expected 1000 rows, found {len(df)}"
arms = {
    "proposed": ("gain_full", 8.081, 5.718, 82.1, 0.0),
    "perf-only": ("gain_perf_only", 4.847, 1.743, 59.0, 19.8),
    "sent-only": ("gain_sent_only", 0.260, 0.000, 39.4, 41.1),
    "random": ("gain_random", 0.137, 0.000, 45.2, 45.3),
    "worst-for-best": ("gain_worst_for_best", 2.002, 0.578, 53.1, 37.5),
}
for name, (col, mean, med, impr, worse) in arms.items():
    s = df[col]
    check(f"{name} mean", mean, s.mean(), 1e-3)
    check(f"{name} median", med, s.median(), 1e-3)
    check(f"{name} improved %", impr, 100 * (s > 1e-9).mean(), 0.05)
    check(f"{name} worse %", worse, 100 * (s < -1e-9).mean(), 0.05)

for name, col in [("perf-only", "gain_perf_only"), ("sent-only", "gain_sent_only"),
                  ("random", "gain_random"), ("worst-for-best", "gain_worst_for_best")]:
    d = df["gain_full"] - df[col]
    claimed = {"perf-only": 71.1, "sent-only": 81.5, "random": 80.4, "worst-for-best": 77.4}[name]
    check(f"win rate vs {name}", claimed, 100 * (d > 1e-9).mean(), 0.05)
    check(f"loss rate vs {name}", 0.0, 100 * (d < -1e-9).mean(), 0.05)

check("proven optimal %", 99.9, 100 * df["proven_optimal"].mean(), 0.05)
check("sentiment changed decision %", 71.1, 100 * df["sentiment_changed_decision"].mean(), 0.05)
check("mean % of achievable", 69.06, df["pct_of_achievable"].mean(), 0.005)
check("mean moves", 3.35, df["n_moves_full"].mean(), 0.005)

print("\n== Table IV (lambda ablation), n=200 ==")
lam = pd.read_csv(HERE / "ablate_lambda.csv")
for l, win, loss, pct in [(0.00, 51.0, 30.0, 64.69), (0.25, 58.0, 23.0, 64.59),
                          (0.50, 68.5, 0.0, 64.55), (1.00, 80.5, 0.5, 64.48),
                          (2.00, 80.5, 0.5, 64.38)]:
    row = lam[lam["lambda"] == l].iloc[0]
    check(f"lambda={l} win", win, row["win_rate_vs_perf_only_pct"], 0.05)
    check(f"lambda={l} loss", loss, row["loss_rate_vs_perf_only_pct"], 0.05)
    check(f"lambda={l} pct_achievable", pct, row["mean_pct_of_achievable"], 0.005)

print("\n== Table V (churn budget), n=200 ==")
bud = pd.read_csv(HERE / "ablate_budget.csv")
for b, gain, moves, states, pct in [(1, 6.9761, 2.52, 12, 62.27),
                                    (2, 7.1006, 3.26, 15, 64.55),
                                    (3, 7.1059, 3.38, 15, 64.64)]:
    row = bud[bud["budget"] == b].iloc[0]
    check(f"B={b} mean gain", gain, row["mean_gain"], 5e-4)
    check(f"B={b} moves", moves, row["mean_moves"], 5e-3)
    check(f"B={b} median states", states, row["median_states"], 0.5)
    check(f"B={b} pct achievable", pct, row["pct_of_achievable"], 5e-3)

print("\n== Table VI (protective floor), n=200 ==")
flr = pd.read_csv(HERE / "ablate_floor.csv")
for sigma, held, gain, med, win in [(0.00, 0, 8.0160, 5.3832, 78.5),
                                    (0.35, 791, 7.1006, 3.9705, 68.5)]:
    row = flr[flr["sentiment_floor"] == sigma].iloc[0]
    check(f"sigma={sigma} held", held, row["employees_held"], 0.5)
    check(f"sigma={sigma} mean gain", gain, row["mean_gain"], 5e-4)
    check(f"sigma={sigma} median gain", med, row["median_gain"], 5e-4)
    check(f"sigma={sigma} win", win, row["win_vs_perf_only_pct"], 0.05)
cost = 100 * (flr.iloc[0]["mean_gain"] - flr.iloc[1]["mean_gain"]) / flr.iloc[0]["mean_gain"]
check("floor cost as % of mean gain", 11.4, cost, 0.05)

print("\n== Live re-derivation from the pipeline ==")
sys.path.insert(0, str(ROOT))
import warnings
warnings.simplefilter("ignore")
from src import pipeline, config  # noqa: E402

ctx = pipeline.load_context()
low = pipeline.discover_low_teams(ctx)
pools = pipeline.form_pools(ctx, list(low[config.TEAM_ID]), pool_size=2)
check("low teams discovered", 143, len(low), 0.5)
check("feasible pools K=2", 9349, len(pools), 0.5)
check("employees held by floor", 791, ctx.protected.sum(), 0.5)

r = pipeline.analyse_pool(ctx, ["TEAM_0098", "TEAM_0202"])
tot = lambda a: sum(float(ctx.values.at[e]) for m in a.values() for e in m)
check("case J before", -7.508, r.j_before, 5e-3)
check("case J after", 52.289, r.j_after, 5e-3)
check("case TS(0098) after", 53.587, r.scores_after["TEAM_0098"], 5e-3)
check("case TS(0202) after", 58.777, r.scores_after["TEAM_0202"], 5e-3)
check("case bound U(P)", 58.100, r.j_bound, 5e-3)
check("case captured %", 91.14, r.capture_pct, 0.05)
check("case states", 13, r.states_evaluated, 0.5)
check("zero-sum delta", 0.0, tot(r.assignment_after) - tot(r.assignment_before), 1e-9)

print("\n== Literal strings quoted in the paper ==")
for lit in ["13.3630905400", "-0.1008912705", "9.50\\times10^{-176}",
            "$0.5463$", "$26.15$", "$1\\,212\\,534$", "$0.7476$", "$0.0109$"]:
    in_tex(lit)

print()
print("VERIFY: PASS" if not bad else f"VERIFY: FAIL ({len(bad)}) -> {bad}")
sys.exit(1 if bad else 0)
