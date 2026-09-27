"""Baselines, ablations and the sweep over the low-performing team population.

The 143 discovered `Low` teams give C(143,2) = 10,153 evaluable pairs, so results are
reported over a population rather than as a case study.

Baselines
    none              leave the teams alone
    random            a random feasible role-matched swap
    perf_only         sentiment ablated, v(e) = perf_norm(e) - performance signal alone
    sent_only         performance ablated, v(e) = phi(s(e)) - sentiment alone
    worst_for_best    the intuitive HR heuristic: trade the weakest member of the worse
                      team for the strongest of the better one, per role
    full              the proposed system, v(e) = perf_norm(e) * phi(s(e))

Every arm is optimised on its own value function and then RE-SCORED on the full one, so
the comparison is like-for-like: the question is not "which arm likes its own objective"
but "which arm's chosen moves are actually best".

THE HEADLINE NUMBER is `sentiment_changed_decision_pct`: the share of pairs where the
sentiment-aware optimiser selects a different set of movers than the performance-only
optimiser. If that is near zero the sentiment component is decorative and the paper has
no contribution; reporting it honestly is what makes the claim credible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, objective, search
from .pipeline import Context


def _flat_moves(result) -> frozenset:
    """Identity of a recommendation: which employees moved where."""
    return frozenset((emp, dst) for emp, _, dst in result.moves)


def arm_values(ctx: Context) -> dict:
    """The value function each arm optimises. Only `full` uses both signals."""
    ones = pd.Series(1.0, index=ctx.perf_norm.index)
    return {
        "full": ctx.values,
        "perf_only": objective.value_series(ctx.perf_norm, pd.Series(0.5, index=ones.index)),
        "sent_only": objective.value_series(ones, ctx.s),
    }


def random_baseline(ctx: Context, values, team_ids, protected, rng) -> objective.SwapResult:
    """One random feasible role-matched swap, respecting the same constraints."""
    t_a, t_b = team_ids[0], team_ids[1]
    assignment = search.build_assignment(ctx.employees, team_ids)
    by_a = search.movable_by_role(ctx.employees, t_a, protected)
    by_b = search.movable_by_role(ctx.employees, t_b, protected)
    shared = sorted(set(by_a) & set(by_b))

    moves = []
    if shared:
        role = shared[rng.integers(len(shared))]
        a = by_a[role][rng.integers(len(by_a[role]))]
        b = by_b[role][rng.integers(len(by_b[role]))]
        moves = [(a, t_a, t_b), (b, t_b, t_a)]

    after = search._apply(assignment, moves)
    return objective.SwapResult(
        assignment_before=assignment, assignment_after=after, moves=moves,
        j_before=objective.objective(assignment, values),
        j_after=objective.objective(after, values),
        scores_before=objective.team_scores(assignment, values),
        scores_after=objective.team_scores(after, values),
        method="random", proven_optimal=False,
    )


def worst_for_best_baseline(ctx: Context, values, team_ids, protected) -> objective.SwapResult:
    """The intuitive heuristic: the weaker team gives up its worst, takes the other's best."""
    t_a, t_b = team_ids[0], team_ids[1]
    assignment = search.build_assignment(ctx.employees, team_ids)
    scores = objective.team_scores(assignment, values)
    weak, strong = (t_a, t_b) if scores[t_a] <= scores[t_b] else (t_b, t_a)

    by_weak = search.movable_by_role(ctx.employees, weak, protected)
    by_strong = search.movable_by_role(ctx.employees, strong, protected)
    perf = ctx.employees.set_index(config.EMP_ID)[config.PERF_COL]

    moves = []
    for role in sorted(set(by_weak) & set(by_strong)):
        give = min(by_weak[role], key=lambda e: perf.at[e])
        take = max(by_strong[role], key=lambda e: perf.at[e])
        moves.extend([(give, weak, strong), (take, strong, weak)])
        break   # one role class, to keep churn comparable with the budgeted optimiser

    after = search._apply(assignment, moves)
    return objective.SwapResult(
        assignment_before=assignment, assignment_after=after, moves=moves,
        j_before=objective.objective(assignment, values),
        j_after=objective.objective(after, values),
        scores_before=objective.team_scores(assignment, values),
        scores_after=objective.team_scores(after, values),
        method="worst_for_best", proven_optimal=False,
    )


def evaluate_pair(ctx: Context, team_ids, rng, protected=None, **kwargs) -> dict:
    """Run every arm on one pool and return a single comparable row."""
    team_ids = list(team_ids)
    protected = ctx.protected if protected is None else protected
    arms = arm_values(ctx)
    values = arms["full"]

    full = search.recommend_swaps(ctx.employees, values, team_ids, protected, **kwargs)
    perf_only = search.recommend_swaps(
        ctx.employees, arms["perf_only"], team_ids, protected, **kwargs)
    sent_only = search.recommend_swaps(
        ctx.employees, arms["sent_only"], team_ids, protected, **kwargs)
    rnd = random_baseline(ctx, values, team_ids, protected, rng)
    wfb = worst_for_best_baseline(ctx, values, team_ids, protected)

    # Baselines are re-scored on the full objective so the comparison is like-for-like.
    def rescore(result) -> float:
        return (objective.objective(result.assignment_after, values)
                - objective.objective(result.assignment_before, values))

    n_protected = int(sum(
        bool(protected.get(e, False))
        for m in full.assignment_before.values() for e in m
    ))

    return {
        "team_a": team_ids[0],
        "team_b": team_ids[1],
        "gain_full": full.gain,
        "gain_perf_only": rescore(perf_only),
        "gain_sent_only": rescore(sent_only),
        "gain_random": rnd.gain,
        "gain_worst_for_best": wfb.gain,
        "n_moves_full": len(full.moves),
        "states_full": full.states_evaluated,
        "proven_optimal": full.proven_optimal,
        "pct_of_achievable": full.capture_pct,
        "sentiment_changed_decision": _flat_moves(full) != _flat_moves(perf_only),
        "n_protected": n_protected,
    }


def sweep(ctx: Context, pairs, seed: int = config.RANDOM_SEED,
          limit: int | None = None, **kwargs) -> pd.DataFrame:
    """Evaluate many pools. `limit` samples without replacement for a quick run."""
    rng = np.random.default_rng(seed)
    pairs = list(pairs)
    if limit is not None and limit < len(pairs):
        chosen = rng.choice(len(pairs), size=limit, replace=False)
        pairs = [pairs[i] for i in chosen]
    return pd.DataFrame([evaluate_pair(ctx, p, rng, **kwargs) for p in pairs])


def summarise(results: pd.DataFrame) -> dict:
    """Collapse a sweep into the numbers that go in the results table."""
    arms = ["full", "perf_only", "sent_only", "random", "worst_for_best"]
    out = {
        "n_pairs": int(len(results)),
        "pct_proven_optimal": round(100 * results["proven_optimal"].mean(), 2),
        "sentiment_changed_decision_pct": round(
            100 * results["sentiment_changed_decision"].mean(), 2
        ),
        "mean_pct_of_achievable": round(float(results["pct_of_achievable"].mean()), 2),
        "employees_held_by_floor": int(results["n_protected"].sum()),
        "mean_moves": round(float(results["n_moves_full"].mean()), 2),
        "median_states_evaluated": int(results["states_full"].median()),
    }
    for arm in arms:
        col = results[f"gain_{arm}"]
        out[f"gain_{arm}"] = {
            "mean": round(float(col.mean()), 4),
            "median": round(float(col.median()), 4),
            "pct_improved": round(100 * float((col > 1e-9).mean()), 2),
        }
    for arm in arms[1:]:
        diff = results["gain_full"] - results[f"gain_{arm}"]
        out[f"win_rate_vs_{arm}_pct"] = round(100 * float((diff > 1e-9).mean()), 2)
        out[f"loss_rate_vs_{arm}_pct"] = round(100 * float((diff < -1e-9).mean()), 2)
    return out


def ablate_lambda(ctx: Context, pairs, values=(0.0, 0.25, 0.5, 1.0, 2.0), limit=50) -> pd.DataFrame:
    """Sensitivity of the recommendation to the balance weight.

    With value intrinsic, `lambda` is not a taste parameter: the pool total is invariant
    under every legal swap, so the whole objective rests on the non-linearity of
    `min` and `std`. Expect lambda = 0 to lose to `perf_only`.
    """
    rows = []
    for lam in values:
        res = sweep(ctx, pairs, limit=limit, lam=lam)
        rows.append({"lambda": lam, **{k: v for k, v in summarise(res).items()
                                       if not isinstance(v, dict)}})
    return pd.DataFrame(rows)


def ablate_floor(ctx: Context, pairs, limit=120,
                 floors=(0.0, config.SENTIMENT_FLOOR)) -> pd.DataFrame:
    """Price of the protective sentiment filter.

    Holding disengaged employees in place removes them from the candidate set, so it can
    only ever cost gain. Quoting that cost is the point: the constraint is an ethical
    choice with a measurable price, not a free lunch. `floor = 0.0` disables it.
    """
    rows = []
    for floor in floors:
        protected = objective.protected(ctx.perf_norm.index, ctx.s, floor, floor > 0.0)
        res = sweep(ctx, pairs, limit=limit, protected=protected)
        summary = summarise(res)
        rows.append({
            "sentiment_floor": floor,
            "employees_held": int(protected.sum()),
            "mean_gain": summary["gain_full"]["mean"],
            "median_gain": summary["gain_full"]["median"],
            "win_vs_perf_only_pct": summary["win_rate_vs_perf_only_pct"],
            "loss_vs_perf_only_pct": summary["loss_rate_vs_perf_only_pct"],
            "sentiment_changed_decision_pct": summary["sentiment_changed_decision_pct"],
            "mean_moves": summary["mean_moves"],
        })
    return pd.DataFrame(rows)


def ablate_budget(ctx: Context, pairs, budgets=(1, 2, 3), limit=50) -> pd.DataFrame:
    """Cost/benefit of allowing more churn."""
    rows = []
    for b in budgets:
        res = sweep(ctx, pairs, limit=limit, budget=b)
        summary = summarise(res)
        rows.append({
            "budget": b,
            "mean_gain": summary["gain_full"]["mean"],
            "mean_moves": summary["mean_moves"],
            "median_states": summary["median_states_evaluated"],
            "pct_of_achievable": summary["mean_pct_of_achievable"],
            "pct_proven_optimal": summary["pct_proven_optimal"],
        })
    return pd.DataFrame(rows)
