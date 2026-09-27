"""BLOCKER #2 - an objective that a swap can actually move.

The shipped objective was `Team_Performance = mean(member Composite_Score)`. Under a
role-matched swap that is worthless, and provably so:

    sum of member scores : 1324.350000 -> 1324.350000    delta = +0.0000000000

The sum over the pool is invariant under any exchange of members, so a sum-or-mean
objective assigns exactly zero value to every swap the system could ever propose
(PROJECT.md 0.5, measured on TEAM_0030 x TEAM_0245).

Employee value is INTRINSIC
---------------------------
    v(e) = perf_norm(e) * phi(s(e)),   phi(s) = PHI_BASE + PHI_SLOPE * s

The system no longer predicts how satisfied `e` *would be* in some other team - that
counterfactual is not verifiable against anything in this data, so it is out of scope.
`s(e)` is the observed sentiment score, one number per person, carried with them.

Which means the zero-sum result above applies to `v` as well: the pool total is invariant
under every legal swap. Exactly one thing rescues the problem, and it is the aggregator:

    J = min_t TeamScore(t) - lambda * std_t TeamScore(t)

`min` and `std` are non-linear, so they are NOT invariant under a redistribution that
holds the sum fixed. Moving a strong contributor into the weakest team raises the minimum
even though nothing was created. The thesis is therefore redistributive rather than
generative: reassignment does not manufacture value, it stops a team from failing while
the talent to rescue it is already inside the pool. LAMBDA_BALANCE is load-bearing, not a
preference - at lambda = 0 the system measurably loses to a performance-only baseline.

A note on team means and BUG-21: mean-of-team-means is normally gameable by shifting
people into smaller teams. Here it is safe, because a role-matched k-for-k swap leaves
every team's size exactly unchanged. The size-gaming channel is closed by the constraint,
not by the aggregator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

Assignment = dict  # team_id -> tuple[employee_id, ...]


def phi(s: np.ndarray | float) -> np.ndarray | float:
    """Satisfaction multiplier on contribution. phi(0)=0.5, phi(1)=1.5 by default."""
    return config.PHI_BASE + config.PHI_SLOPE * np.asarray(s, dtype=float)


def normalise_performance(employees: pd.DataFrame) -> pd.Series:
    """Min-max performance over the whole company, so scores compare across pools."""
    perf = employees.set_index(config.EMP_ID)[config.PERF_COL].astype(float)
    lo, hi = perf.min(), perf.max()
    return (perf - lo) / (hi - lo) if hi > lo else perf * 0 + 0.5


def sentiment_series(sentiment: pd.DataFrame, index: pd.Index) -> pd.Series:
    """`s(e)` aligned to the employee roster. A missing score defaults to neutral."""
    s = sentiment.set_index(config.EMP_ID)["s_crisp"].astype(float)
    return s.reindex(index).fillna(0.5).rename("sentiment")


def value_series(perf_norm: pd.Series, s: pd.Series) -> pd.Series:
    """`v(e)` for every employee - one value, independent of which team they sit in."""
    return (perf_norm * phi(s.reindex(perf_norm.index))).rename("value")


# --------------------------------------------------------------------------- scoring
def team_score(members, values: pd.Series) -> float:
    """Mean effective contribution of a team, scaled to a readable 0-100."""
    members = list(members)
    if not members:
        return 0.0
    return 100.0 * float(values.loc[members].mean())


def team_scores(assignment: Assignment, values: pd.Series) -> dict:
    return {t: team_score(m, values) for t, m in assignment.items()}


def objective(assignment: Assignment, values: pd.Series,
              lam: float = config.LAMBDA_BALANCE) -> float:
    """`J` - Rawlsian pool objective. Higher is better."""
    scores = np.array(list(team_scores(assignment, values).values()), dtype=float)
    if scores.size == 0:
        return 0.0
    spread = float(scores.std(ddof=0)) if scores.size > 1 else 0.0
    return float(scores.min() - lam * spread)


def equalisation_bound(assignment: Assignment, values: pd.Series) -> float:
    """Provable upper bound on `J` for this pool, over *any* reallocation whatsoever.

    `min_t mean_t` is a minimum over a set whose headcount-weighted average is the pool
    mean, so it can never exceed that mean; `- lambda * std` only lowers J further. Hence

        J <= 100 * (total pool value) / (pool headcount)

    attained only by a perfectly equalised pool. The bound needs no search and ignores
    both the role constraint and the churn budget, which is what makes it cheap enough to
    evaluate for every candidate pool - it is what the shortlist prunes against.
    """
    members = [e for m in assignment.values() for e in m]
    if not members:
        return 0.0
    return 100.0 * float(values.loc[members].mean())


# ----------------------------------------------------------------------- constraints
def protected(index: pd.Index, s: pd.Series,
              floor: float = config.SENTIMENT_FLOOR,
              enforce: bool = config.ENFORCE_SENTIMENT_FLOOR) -> pd.Series:
    """Boolean mask over employees: True == must NOT be moved.

    This replaces the old do-no-harm constraint. That rule compared a person's predicted
    satisfaction before and after a move, which required exactly the counterfactual this
    system no longer claims to produce. What survives is the defensible half of it: an
    employee already below the sentiment floor is disengaged, involuntary reassignment is
    a known churn accelerant for that group, and so they are held in place. The rule reads
    only an observed quantity, and its cost is measurable - see `evaluate.ablate_floor`.
    """
    s = s.reindex(index).fillna(0.5)
    if not enforce:
        return pd.Series(False, index=index, name="protected")
    return (s < floor).rename("protected")


def screening_priority(employees: pd.DataFrame, sentiment: pd.DataFrame,
                       perf_norm: pd.Series | None = None) -> pd.Series:
    """The "who is at risk" rule, reported as context rather than used as a selector.

    High score == weak performance and low sentiment. It does not choose the swap: the
    optimiser does that on marginal gain, because the best person to move is often a
    strong-but-miserable one rather than the weakest.
    """
    perf_norm = normalise_performance(employees) if perf_norm is None else perf_norm
    s = sentiment_series(sentiment, perf_norm.index)
    return (
        config.SCREEN_W_PERF * (1.0 - perf_norm) + config.SCREEN_W_SENT * (1.0 - s)
    ).rename("swap_priority")


# --------------------------------------------------------------------------- results
@dataclass
class SwapResult:
    """Everything a reviewer or an HR user needs to judge one recommendation."""
    assignment_before: Assignment
    assignment_after: Assignment
    moves: list = field(default_factory=list)      # (employee, from_team, to_team)
    j_before: float = 0.0
    j_after: float = 0.0
    scores_before: dict = field(default_factory=dict)
    scores_after: dict = field(default_factory=dict)
    states_evaluated: int = 0
    proven_optimal: bool = False
    method: str = ""
    j_bound: float = 0.0

    @property
    def gain(self) -> float:
        return self.j_after - self.j_before

    @property
    def capture_pct(self) -> float:
        """Share of the theoretically achievable improvement this recommendation takes."""
        headroom = self.j_bound - self.j_before
        return 100.0 * self.gain / headroom if headroom > 1e-9 else 0.0

    def counterparts(self) -> dict:
        """employee -> the employee they trade places with, where the pairing is 1-for-1.

        The optimiser decides *sets*, not a bijection: after a symmetric k-for-k exchange
        the resulting compositions depend only on which sets moved. Pairing people up is a
        reporting convenience so HR reads one legible trade per line, and nothing in the
        objective depends on which partner is matched to which.
        """
        out: dict = {}
        by_leg: dict = {}
        for emp, src, dst in self.moves:
            by_leg.setdefault((src, dst), []).append(emp)
        for (src, dst), outbound in by_leg.items():
            for a, b in zip(outbound, by_leg.get((dst, src), [])):
                out[a] = b
        return out

    def to_frame(self, employees: pd.DataFrame, values: pd.Series | None = None,
                 s: pd.Series | None = None) -> pd.DataFrame:
        """One row per recommended move, with the evidence HR needs to accept or reject it.

        Every row is a *recommendation*, not an instruction: it names the person, the team
        they leave, the team they join, the person they trade places with, and the three
        quantities the decision rests on - role match, performance and sentiment.
        """
        columns = [config.EMP_ID, "Title", "Role", "from_team", "to_team", "swap_with",
                   config.PERF_COL, "Sentiment", "Value", "reason"]
        if not self.moves:
            return pd.DataFrame(columns=columns)

        idx = employees.set_index(config.EMP_ID)
        partners = self.counterparts()
        rows = []
        for emp, src, dst in self.moves:
            other = partners.get(emp)
            role = idx.at[emp, "Role"]
            row = {
                config.EMP_ID: emp,
                "Title": idx.at[emp, "Title"],
                "Role": role,
                "from_team": src,
                "to_team": dst,
                "swap_with": other,
                config.PERF_COL: idx.at[emp, config.PERF_COL],
            }
            if s is not None:
                row["Sentiment"] = round(float(s.at[emp]), 4)
            if values is not None:
                row["Value"] = round(100.0 * float(values.at[emp]), 2)
                if other is not None:
                    diff = 100.0 * float(values.at[emp] - values.at[other])
                    direction = "stronger" if diff > 0 else "weaker"
                    row["reason"] = (
                        f"{role} for {role}; {abs(diff):.1f} pts {direction} "
                        f"than {other}, who moves the other way"
                    )
            rows.append(row)
        return pd.DataFrame(rows).reindex(columns=columns)

    def summary(self) -> dict:
        return {
            "method": self.method,
            "proven_optimal": self.proven_optimal,
            "states_evaluated": self.states_evaluated,
            "n_moves": len(self.moves),
            "J_before": round(self.j_before, 6),
            "J_after": round(self.j_after, 6),
            "gain": round(self.gain, 6),
            "J_bound": round(self.j_bound, 6),
            "pct_of_achievable": round(self.capture_pct, 2),
            "scores_before": {k: round(v, 4) for k, v in self.scores_before.items()},
            "scores_after": {k: round(v, 4) for k, v in self.scores_after.items()},
        }
