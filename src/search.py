"""The optimiser.

Why there is no Hungarian solve anywhere in this module
------------------------------------------------------
Two independent arguments kill it, and the second one is now decisive.

1. For any value function `v(e, t)` whatsoever, exchanging `a` in A with `b` in B gains

       Delta(a,b) = [v(b,A) + v(a,B)] - [v(a,A) + v(b,B)]
                  = [v(a,B) - v(a,A)] + [v(b,A) - v(b,B)]
                  =       beta(a)     +      alpha(b)

   a row constant plus a column constant. By the row/column invariance of the assignment
   problem every perfect matching between the same two sets scores identically, so a
   Hungarian solve over that matrix is degenerate by construction. Measured on
   TEAM_0030 x TEAM_0245: all 6 matchings scored -1.4055555556, spread 2.8e-14.

2. Value is now intrinsic - `v(e)`, with no team argument at all. A *slot* LAP over
   `c(e,t) = -v(e,t)` used to be the one legitimate Hungarian formulation here, and it is
   no longer even that: with `v` independent of `t` every slot assignment has the same
   total, so the LAP collapses to the same degeneracy as (1). It has been removed rather
   than left in as a decorative bound. `objective.equalisation_bound` replaces it and is
   both cheaper and strictly valid.

The stronger structural statement needs no separability: after a symmetric k-for-k swap
the resulting team compositions depend only on the *sets* that moved, never on the
bijection between them. "Who swaps with whom" is a reporting convenience, not a decision
variable.

So the decision variables are the sets, and:

    K = 2   exhaustive enumeration of set-pairs per role class under a churn budget.
            At budget 2 the state count is median 38 / p95 31,341 over 400 random
            low-team pairs, so this is solved to PROVEN GLOBAL OPTIMALITY.
    K >= 3  steepest-ascent local search over the 1-swap neighbourhood. Reported with
            `proven_optimal=False` and with the equalisation bound alongside, so the
            reader can see how much of the achievable improvement was captured.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import config
from .objective import (
    Assignment, SwapResult, equalisation_bound, objective, team_scores,
)


# --------------------------------------------------------------------------- helpers
def build_assignment(employees: pd.DataFrame, team_ids) -> Assignment:
    """team_id -> tuple of member Employee_IDs, for the teams in the pool."""
    subset = employees[employees[config.TEAM_ID].isin(list(team_ids))]
    return {t: tuple(g[config.EMP_ID]) for t, g in subset.groupby(config.TEAM_ID)}


def movable_by_role(employees: pd.DataFrame, team_id: str,
                    protected: pd.Series | None = None) -> dict:
    """role -> list of Employee_IDs eligible to move out of `team_id`.

    Two filters, both structural rather than optimisation-driven: team leads stay put for
    leadership continuity, and anyone the protective sentiment filter flags stays put
    because they are already disengaged.
    """
    members = employees[employees[config.TEAM_ID] == team_id]
    if not config.ALLOW_LEAD_SWAPS:
        members = members[~members["Is_Team_Lead"]]
    if protected is not None:
        keep = ~members[config.EMP_ID].map(protected).fillna(False).astype(bool)
        members = members[keep]
    return {role: list(g[config.EMP_ID]) for role, g in members.groupby("Role")}


def _role_options(feas_a: list, feas_b: list, budget: int) -> list:
    """All (S_A, S_B) set-pairs of equal size up to the churn budget, k = 0 included."""
    k_max = min(budget, len(feas_a), len(feas_b))
    return [
        (sa, sb)
        for k in range(k_max + 1)
        for sa in combinations(feas_a, k)
        for sb in combinations(feas_b, k)
    ]


def _apply(assignment: Assignment, moves) -> Assignment:
    out = {t: list(m) for t, m in assignment.items()}
    for emp, src, dst in moves:
        out[src].remove(emp)
        out[dst].append(emp)
    return {t: tuple(m) for t, m in out.items()}


# ------------------------------------------------------------------ K = 2 exhaustive
def search_pair(
    employees: pd.DataFrame,
    values: pd.Series,
    team_a: str,
    team_b: str,
    protected: pd.Series | None = None,
    budget: int = config.CHURN_BUDGET,
    lam: float = config.LAMBDA_BALANCE,
    max_states: int = config.MAX_ENUM_STATES,
) -> SwapResult:
    """Exhaustively optimise the exchange between exactly two teams.

    Team sums are updated incrementally rather than rebuilt, so the whole cartesian
    product across role classes is evaluated with numpy broadcasting rather than a Python
    loop over compositions. Because value is intrinsic, whatever team A gains team B loses
    exactly - `delta_b = -delta_a` below is the zero-sum property in one line of code. The
    optimiser still has work to do only because J divides by two different team sizes and
    then takes a minimum.
    """
    assignment = build_assignment(employees, [team_a, team_b])
    members_a, members_b = list(assignment[team_a]), list(assignment[team_b])
    n_a, n_b = len(members_a), len(members_b)

    base_a = float(values.loc[members_a].sum())
    base_b = float(values.loc[members_b].sum())

    by_role_a = movable_by_role(employees, team_a, protected)
    by_role_b = movable_by_role(employees, team_b, protected)
    shared_roles = sorted(set(by_role_a) & set(by_role_b))

    # Per role class: enumerate feasible set-pairs and precompute the sum delta each one
    # induces on team A. Deltas are additive across role classes.
    options, deltas = [], []
    for role in shared_roles:
        opts = _role_options(by_role_a[role], by_role_b[role], budget)
        options.append(opts)
        deltas.append(np.array([
            values.loc[list(sb)].sum() - values.loc[list(sa)].sum() if sa else 0.0
            for sa, sb in opts
        ], dtype=float))

    shape = tuple(len(o) for o in options)
    n_states = int(np.prod(shape)) if shape else 1

    if n_states > max_states:
        result = local_search(
            employees, values, [team_a, team_b], protected,
            budget * max(len(shared_roles), 1), lam,
        )
        result.method = f"local_search (fallback: {n_states:,} states > cap {max_states:,})"
        return result

    total = np.zeros(shape or (1,), dtype=float)
    for axis, d in enumerate(deltas):
        broadcast = [1] * len(shape)
        broadcast[axis] = len(d)
        total = total + d.reshape(broadcast)

    score_a = 100.0 * (base_a + total) / n_a
    score_b = 100.0 * (base_b - total) / n_b
    # Two teams: std of a 2-point sample (ddof=0) is exactly half the absolute difference.
    j = np.minimum(score_a, score_b) - lam * np.abs(score_a - score_b) / 2.0

    flat = int(np.argmax(j))
    index = np.unravel_index(flat, shape) if shape else ()

    moves = []
    for axis, _role in enumerate(shared_roles):
        sa, sb = options[axis][index[axis]]
        moves.extend((e, team_a, team_b) for e in sa)
        moves.extend((e, team_b, team_a) for e in sb)

    after = _apply(assignment, moves)
    return SwapResult(
        assignment_before=assignment,
        assignment_after=after,
        moves=moves,
        j_before=objective(assignment, values, lam),
        j_after=float(j.ravel()[flat]),
        scores_before=team_scores(assignment, values),
        scores_after=team_scores(after, values),
        states_evaluated=n_states,
        proven_optimal=True,
        method=f"exhaustive (budget={budget}, {n_states:,} states)",
        j_bound=equalisation_bound(assignment, values),
    )


# ------------------------------------------------------------------ K >= 3 local search
def local_search(
    employees: pd.DataFrame,
    values: pd.Series,
    team_ids,
    protected: pd.Series | None = None,
    max_moves: int = config.CHURN_BUDGET * 2,
    lam: float = config.LAMBDA_BALANCE,
) -> SwapResult:
    """Steepest-ascent hill climbing over the role-matched 1-swap neighbourhood.

    Each iteration evaluates every feasible single swap across every team pair in the pool
    and commits the best strictly-improving one. Terminates at a local optimum or at the
    churn budget, whichever comes first. This is the K >= 3 path, and it is the price of
    letting a pool hold more than two teams: the result is good but not provably optimal,
    so `proven_optimal` is False and the equalisation bound is reported beside it.
    """
    team_ids = list(team_ids)
    assignment = build_assignment(employees, team_ids)
    j_before = objective(assignment, values, lam)

    indexed = employees.set_index(config.EMP_ID)
    roles = indexed["Role"]
    leads = indexed["Is_Team_Lead"]

    def blocked(e) -> bool:
        if leads.at[e] and not config.ALLOW_LEAD_SWAPS:
            return True
        return bool(protected.get(e, False)) if protected is not None else False

    current = {t: list(m) for t, m in assignment.items()}
    moves, evaluated = [], 0
    moved_already: set = set()

    while len(moves) < max_moves:
        best = None
        for t_a, t_b in combinations(team_ids, 2):
            for a in current[t_a]:
                if a in moved_already or blocked(a):
                    continue
                for b in current[t_b]:
                    if b in moved_already or blocked(b):
                        continue
                    if roles.at[a] != roles.at[b]:
                        continue
                    evaluated += 1
                    trial = {t: list(m) for t, m in current.items()}
                    trial[t_a].remove(a); trial[t_a].append(b)
                    trial[t_b].remove(b); trial[t_b].append(a)
                    j_trial = objective({t: tuple(m) for t, m in trial.items()}, values, lam)
                    if best is None or j_trial > best[0]:
                        best = (j_trial, a, b, t_a, t_b)

        j_now = objective({t: tuple(m) for t, m in current.items()}, values, lam)
        if best is None or best[0] <= j_now + 1e-12:
            break

        _, a, b, t_a, t_b = best
        current[t_a].remove(a); current[t_a].append(b)
        current[t_b].remove(b); current[t_b].append(a)
        moves.extend([(a, t_a, t_b), (b, t_b, t_a)])
        moved_already.update({a, b})

    after = {t: tuple(m) for t, m in current.items()}
    return SwapResult(
        assignment_before=assignment,
        assignment_after=after,
        moves=moves,
        j_before=j_before,
        j_after=objective(after, values, lam),
        scores_before=team_scores(assignment, values),
        scores_after=team_scores(after, values),
        states_evaluated=evaluated,
        proven_optimal=False,
        method=f"local_search (1-swap steepest ascent, max_moves={max_moves})",
        j_bound=equalisation_bound(assignment, values),
    )


# --------------------------------------------------------------------------- entry
def recommend_swaps(
    employees: pd.DataFrame,
    values: pd.Series,
    team_ids,
    protected: pd.Series | None = None,
    **kwargs,
) -> SwapResult:
    """Dispatch on pool size: exhaustive for two teams, local search beyond."""
    team_ids = list(team_ids)
    if len(team_ids) < 2:
        raise ValueError("a reassignment needs at least two teams")
    if len(team_ids) == 2:
        return search_pair(employees, values, team_ids[0], team_ids[1], protected, **kwargs)

    # Beyond two teams the exhaustive product is not tractable, so fall through to local
    # search. `budget` is per role class for the exhaustive path; here it caps total
    # swaps, so translate it rather than letting it through as an unknown keyword.
    kwargs.pop("max_states", None)
    if "budget" in kwargs:
        kwargs["max_moves"] = 2 * int(kwargs.pop("budget"))
    return local_search(employees, values, team_ids, protected, **kwargs)
