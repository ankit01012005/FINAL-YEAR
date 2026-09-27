"""Orchestration: load -> repair -> discover -> pool -> optimise -> rank.

The system is handed a *dataset*, not a team pair. It finds the under-performing teams
itself, groups them into pools that are actually able to exchange members, solves each
pool, and hands HR a ranked shortlist of recommendations.

Everything the rest of the project needs is assembled once into a `Context`, so a single
pool and a sweep over ten thousand share exactly the same preparation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from . import config, objective, roles, search, sentiment_io


@dataclass
class Context:
    employees: pd.DataFrame          # roles already re-derived
    teams: pd.DataFrame              # Num_* already recomputed
    sentiment: pd.DataFrame
    sentiment_meta: dict
    role_report: dict
    perf_norm: pd.Series             # perf_norm(e), min-max over the company
    s: pd.Series                     # s(e), observed sentiment - intrinsic to the person
    values: pd.Series                # v(e) = perf_norm(e) * phi(s(e))
    protected: pd.Series             # True == held in place by the sentiment floor

    @property
    def is_placeholder(self) -> bool:
        return bool(self.sentiment_meta.get("is_placeholder"))


def load_context(
    employees_csv=None,
    teams_csv=None,
    sentiment_csv=None,
    strict_sentiment: bool = False,
    enforce_floor: bool = config.ENFORCE_SENTIMENT_FLOOR,
    floor: float = config.SENTIMENT_FLOOR,
) -> Context:
    """Read both datasets, apply the role repair, attach sentiment, build `v(e)` once."""
    employees = pd.read_csv(employees_csv or config.EMPLOYEES_CSV)
    teams = pd.read_csv(teams_csv or config.TEAMS_CSV)

    employees = roles.derive_roles(employees)
    teams = roles.recompute_team_role_counts(employees, teams)
    role_report = roles.coverage_report(employees)

    frame, meta = sentiment_io.load_sentiment(
        sentiment_csv, employee_ids=employees[config.EMP_ID], strict=strict_sentiment
    )

    perf_norm = objective.normalise_performance(employees)
    s = objective.sentiment_series(frame, perf_norm.index)

    return Context(
        employees=employees,
        teams=teams,
        sentiment=frame,
        sentiment_meta=meta,
        role_report=role_report,
        perf_norm=perf_norm,
        s=s,
        values=objective.value_series(perf_norm, s),
        protected=objective.protected(perf_norm.index, s, floor, enforce_floor),
    )


# ------------------------------------------------------------------------- discovery
def discover_low_teams(ctx: Context, category: str | None = None,
                       quantile: float | None = None) -> pd.DataFrame:
    """Find the under-performing teams in the dataset. No team pair is supplied by hand.

    `Performance_Category` is used when present - it is a clean quartile split of
    `Team_Performance` (verified: Low 73.70-78.09, Medium 78.10-80.86, High 80.89-86.63,
    with the 0.25 quantile at 78.092), so unlike the shipped `Role` column it can be
    trusted. When it is absent the same split is recomputed from `quantile`.

    Returns one row per discovered team, ordered worst first, carrying the diagnostics
    that decide whether a team can participate in an exchange at all.
    """
    category = config.LOW_CATEGORY if category is None else category
    quantile = config.LOW_QUANTILE if quantile is None else quantile
    teams = ctx.teams

    if config.TEAM_CATEGORY_COL in teams.columns:
        low = teams[teams[config.TEAM_CATEGORY_COL] == category]
        rule = f"{config.TEAM_CATEGORY_COL} == '{category}'"
    else:
        cut = float(teams[config.TEAM_PERF_COL].quantile(quantile))
        low = teams[teams[config.TEAM_PERF_COL] <= cut]
        rule = f"{config.TEAM_PERF_COL} <= q{quantile:g} ({cut:.3f})"

    ids = list(low[config.TEAM_ID])
    stats = team_stats(ctx, ids)
    movable = _movable_roles(ctx, ids)

    out = (
        low[[config.TEAM_ID, config.TEAM_PERF_COL, config.TEAM_SIZE_COL]]
        .set_index(config.TEAM_ID)
        .join(stats)
        .assign(
            discovery_rule=rule,
            movable=[len(movable[t]) for t in low[config.TEAM_ID]],
            movable_roles=[",".join(sorted(movable[t])) or "-" for t in low[config.TEAM_ID]],
        )
        .sort_values("team_score")
        .reset_index()
    )
    return out


def team_stats(ctx: Context, team_ids) -> pd.DataFrame:
    """Headcount, total and mean value per team, on the same 0-100 scale as `J`."""
    pool = ctx.employees[ctx.employees[config.TEAM_ID].isin(list(team_ids))]
    v = pool[config.EMP_ID].map(ctx.values)
    grouped = v.groupby(pool[config.TEAM_ID])
    return pd.DataFrame({
        "n": grouped.size(),
        "value_sum": grouped.sum(),
        "team_score": 100.0 * grouped.mean(),
    })


def _movable_roles(ctx: Context, team_ids) -> dict:
    """team_id -> set of role classes with at least one member eligible to move."""
    return {
        t: set(search.movable_by_role(ctx.employees, t, ctx.protected))
        for t in team_ids
    }


# --------------------------------------------------------------------- pool formation
def form_pools(ctx: Context, team_ids, pool_size: int = config.POOL_SIZE,
               min_shared_roles: int = config.MIN_SHARED_ROLES) -> list:
    """Group the discovered teams into pools that can actually exchange members.

    A pool is feasible only if some role class has movable members in at least two of its
    teams - otherwise no legal swap exists and solving it is wasted work.

    `pool_size == 2` enumerates every feasible pair, which keeps every pool inside the
    exhaustive, provably-optimal regime. Larger pools are built greedily: seed with the
    weakest unassigned team and add the teams that share the most role classes with it,
    breaking ties towards the largest score gap, since a pool of equally-placed teams has
    nothing to redistribute.
    """
    team_ids = list(team_ids)
    movable = _movable_roles(ctx, team_ids)
    scores = team_stats(ctx, team_ids)["team_score"]

    def feasible(pool) -> bool:
        counts: dict = {}
        for t in pool:
            for r in movable[t]:
                counts[r] = counts.get(r, 0) + 1
        return any(c >= 2 for c in counts.values())

    if pool_size <= 2:
        return [p for p in combinations(team_ids, 2)
                if len(movable[p[0]] & movable[p[1]]) >= min_shared_roles]

    remaining = sorted(team_ids, key=lambda t: scores.get(t, 0.0))
    pools = []
    while len(remaining) >= pool_size:
        seed = remaining[0]
        rest = remaining[1:]
        rest.sort(key=lambda t: (-len(movable[seed] & movable[t]),
                                 -abs(scores.get(t, 0.0) - scores.get(seed, 0.0))))
        pool = (seed, *rest[:pool_size - 1])
        for t in pool:
            remaining.remove(t)
        if feasible(pool):
            pools.append(pool)
    return pools


def pool_headroom(ctx: Context, pools, lam: float = config.LAMBDA_BALANCE) -> pd.DataFrame:
    """`J_before` and the provable upper bound `J_bound` for every pool, without searching.

    Cheap enough to run over all 10,153 candidate pools, which is what makes the pruned
    ranking in `rank_pools` possible: no pool whose headroom is below an already-found
    gain can ever enter the shortlist, so it never has to be solved.
    """
    pools = [tuple(p) for p in pools]
    if not pools:
        return pd.DataFrame(columns=["pool", "J_before", "J_bound", "headroom"])

    ids = sorted({t for p in pools for t in p})
    stats = team_stats(ctx, ids)
    pos = {t: i for i, t in enumerate(stats.index)}
    n = stats["n"].to_numpy(dtype=float)
    total = stats["value_sum"].to_numpy(dtype=float)
    score = stats["team_score"].to_numpy(dtype=float)

    idx = np.array([[pos[t] for t in p] for p in pools], dtype=int)
    s = score[idx]
    j_before = s.min(axis=1) - lam * s.std(axis=1, ddof=0)
    j_bound = 100.0 * total[idx].sum(axis=1) / n[idx].sum(axis=1)

    return pd.DataFrame({
        "pool": pools,
        "J_before": j_before,
        "J_bound": j_bound,
        "headroom": j_bound - j_before,
    })


# --------------------------------------------------------------------------- analysis
def analyse_pool(ctx: Context, team_ids, **kwargs) -> objective.SwapResult:
    """Full recommendation for one pool."""
    return search.recommend_swaps(ctx.employees, ctx.values, team_ids, ctx.protected, **kwargs)


def _select_disjoint(solved, top: int, allow_overlap: bool = False) -> list:
    """Greedy set-packing over solved pools: best gain first, skipping conflicts.

    Two pools that share a team are mutually exclusive recommendations - the same person
    cannot be moved to two different places, and the second pool's arithmetic assumes the
    first was never applied. Without this the shortlist degenerates into the same weak
    team paired against four different partners, which reads as five recommendations but
    is really one. Pools with no improving swap are not recommendations and are dropped.
    """
    ranked = sorted((r for r in solved if r.gain > 1e-9), key=lambda r: r.gain, reverse=True)
    if allow_overlap:
        return ranked[:top]
    chosen, used = [], set()
    for r in ranked:
        teams = set(r.assignment_before)
        if teams & used:
            continue
        chosen.append(r)
        used |= teams
        if len(chosen) >= top:
            break
    return chosen


def rank_pools(ctx: Context, pools, top: int = config.TOP_RECOMMENDATIONS,
               lam: float = config.LAMBDA_BALANCE, max_solves: int | None = None,
               allow_overlap: bool = False, **kwargs) -> tuple[list, dict]:
    """Solve pools in descending order of headroom and return the best disjoint `top`.

    The ordering is what makes this exact rather than a heuristic shortlist. `headroom` is
    a proven upper bound on the gain ANY reallocation of that pool could achieve, so once
    `top` disjoint pools have been selected and every one of them gains at least as much
    as the headroom of every pool not yet visited, no unvisited pool can beat the weakest
    recommendation on the list - its true optimum is already below it. The search stops
    there, having skipped the overwhelming majority of candidates with a proof rather than
    a heuristic. `no_unexamined_pool_can_beat_shortlist` records whether that condition was
    reached; note this is a statement about the pools NOT solved. Whether each solved pool
    was itself solved to optimality is a separate claim, reported as `all_pools_proven_
    optimal` - true at pool_size 2 (exhaustive) and false beyond it (local search).
    """
    table = pool_headroom(ctx, pools, lam).sort_values("headroom", ascending=False)
    solved, pruned_early = [], False

    for n_done, (_, row) in enumerate(table.iterrows(), start=1):
        chosen = _select_disjoint(solved, top, allow_overlap)
        if len(chosen) >= top and row["headroom"] <= min(r.gain for r in chosen) + 1e-9:
            pruned_early = True
            break
        if max_solves is not None and n_done > max_solves:
            break
        solved.append(analyse_pool(ctx, row["pool"], lam=lam, **kwargs))

    chosen = _select_disjoint(solved, top, allow_overlap)
    meta = {
        "candidate_pools": int(len(table)),
        "pools_solved": len(solved),
        "pools_pruned_by_bound": int(len(table)) - len(solved),
        "pools_dropped_no_gain": sum(1 for r in solved if r.gain <= 1e-9),
        "pools_dropped_overlapping": len(solved) - len(chosen)
        - sum(1 for r in solved if r.gain <= 1e-9),
        "no_unexamined_pool_can_beat_shortlist": bool(
            pruned_early or len(solved) == len(table)
        ),
        "all_pools_proven_optimal": all(r.proven_optimal for r in chosen),
    }
    return chosen, meta


def recommend(ctx: Context, pool_size: int = config.POOL_SIZE,
              top: int = config.TOP_RECOMMENDATIONS, category: str | None = None,
              **kwargs) -> dict:
    """End to end: dataset in, ranked reassignment recommendations out.

    This is the entry point the scope calls for - no team pair is supplied, the system
    discovers the under-performing teams and decides which of them to act on.
    """
    low = discover_low_teams(ctx, category=category)
    pools = form_pools(ctx, list(low[config.TEAM_ID]), pool_size=pool_size)
    results, meta = rank_pools(ctx, pools, top=top, **kwargs)
    return {
        "low_teams": low,
        "pools": pools,
        "results": results,
        "meta": {"low_teams_found": int(len(low)), "pool_size": pool_size, **meta},
    }


def moves_frame(ctx: Context, results) -> pd.DataFrame:
    """Every recommended move across a ranked list of pools, as one HR-facing table."""
    frames = []
    for rank, result in enumerate(results, start=1):
        frame = result.to_frame(ctx.employees, ctx.values, ctx.s)
        if not len(frame):
            continue
        frame.insert(0, "rank", rank)
        frame.insert(1, "pool", " + ".join(sorted(result.assignment_before)))
        frame.insert(2, "pool_gain", round(result.gain, 4))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pools_frame(results) -> pd.DataFrame:
    """One row per recommended pool: what it costs, what it buys, how sure we are."""
    rows = []
    for rank, r in enumerate(results, start=1):
        before, after = r.scores_before, r.scores_after
        weakest = min(before, key=before.get)
        rows.append({
            "rank": rank,
            "pool": " + ".join(sorted(before)),
            "n_moves": len(r.moves),
            "weakest_team": weakest,
            "weakest_before": round(before[weakest], 2),
            "weakest_after": round(after[weakest], 2),
            "J_before": round(r.j_before, 4),
            "J_after": round(r.j_after, 4),
            "gain": round(r.gain, 4),
            "pct_of_achievable": round(r.capture_pct, 1),
            "proven_optimal": r.proven_optimal,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- legacy
def select_pool(ctx: Context, k: int = 2, strategy: str = "worst",
                seed: int = config.RANDOM_SEED, category: str = "Low") -> list:
    """Pick `k` teams by hand. Retained for the manual override and the smoke test;
    the production path is `recommend`, which forms pools itself."""
    low = ctx.teams[ctx.teams[config.TEAM_CATEGORY_COL] == category]
    if strategy == "worst":
        return list(low.nsmallest(k, config.TEAM_PERF_COL)[config.TEAM_ID])
    rng = np.random.default_rng(seed)
    return list(rng.choice(low[config.TEAM_ID].to_numpy(), size=k, replace=False))


def low_team_pairs(ctx: Context, category: str = "Low") -> list:
    """Every evaluable pair - C(143,2) = 10,153 for the shipped data."""
    ids = list(ctx.teams.loc[ctx.teams[config.TEAM_CATEGORY_COL] == category, config.TEAM_ID])
    return list(combinations(ids, 2))


def report(ctx: Context, result: objective.SwapResult) -> dict:
    """A single dict carrying one recommendation and every caveat attached to it."""
    return {
        "sentiment_is_placeholder": ctx.is_placeholder,
        "sentiment_source": ctx.sentiment_meta.get("source"),
        "role_repair": {
            "pct_from_title": ctx.role_report["pct_from_title"],
            "pct_from_default": ctx.role_report["pct_from_default"],
            "chi2_p_title_vs_derived": ctx.role_report["chi2_p_title_vs_derived"],
        },
        "result": result.summary(),
        "moves": result.to_frame(ctx.employees, ctx.values, ctx.s).to_dict("records"),
    }
