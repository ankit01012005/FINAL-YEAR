"""RETIRED - NOT IN THE DECISION PATH. Kept as reference material only.

This module predicted `s_hat(e, t)`: how satisfied an employee would be in a team
they have never worked in. Nothing in the dataset can validate that counterfactual -
there are no post-reassignment outcomes to check it against - so the claim was
withdrawn rather than defended. The live system uses observed sentiment only, as an
intrinsic per-employee score, and recommends on role match, performance and that
score. See objective.py for what replaced this and pipeline.py for the live path.

Nothing below is imported by `src/__init__.py`. The rule base, its provenance and
the leave-one-out feature construction are documented here because they are worth
reporting as an explored-and-rejected design, not because they run.

Counterfactual satisfaction: a Mamdani fuzzy inference system for `s_hat(e, t)`.

Why a rule base rather than a learned model
-------------------------------------------
BUG-20: the eight performance metrics are mutually independent (mean |r| = 0.0109,
max 0.0272). There is no latent quality factor in the data, so a supervised model of
"how satisfied would e be in team t" would fit noise and report an inflated score. A
Mamdani FIS needs no ground truth, is fully inspectable, and the rules are citable
against organisational-behaviour theory - which is exactly what a reviewer will ask for.

Decomposition
-------------
    s_hat(e, t) = s0(e)  +  delta(e, t)          clipped to [0, 1]

`s0(e)` is the observed cohort sentiment supplied by the sentiment subsystem. It is
*intrinsic* to the employee, so on its own it cannot move the objective (PROJECT.md 0.5).
`delta(e, t)` is the team-context term, and it is the only reason a swap can be worth
anything: it is computed from **relational features only** - quantities that change when
e moves.

Rule provenance
---------------
    relative deprivation  ......  R1, R4      Crosby 1976; Smith et al. 2012
    person-environment fit .....  R3, R11     Kristof-Brown et al. 2005
    social loafing .............  R7          Karau & Williams 1993
    similarity-attraction ......  R3, R10     Byrne 1971
    token / isolated status ....  R8          Kanter 1977
    LMX (manager standing) .....  R5, R6      Graen & Uhl-Bien 1995
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# ------------------------------------------------------------------ membership funcs
def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership, peak at b."""
    x = np.asarray(x, dtype=float)
    left = np.divide(x - a, b - a, out=np.zeros_like(x), where=(b - a) != 0)
    right = np.divide(c - x, c - b, out=np.zeros_like(x), where=(c - b) != 0)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership, plateau over [b, c]."""
    x = np.asarray(x, dtype=float)
    left = np.divide(x - a, b - a, out=np.ones_like(x), where=(b - a) != 0)
    right = np.divide(d - x, d - c, out=np.ones_like(x), where=(d - c) != 0)
    return np.clip(np.minimum(np.minimum(left, right), 1.0), 0.0, 1.0)


# ------------------------------------------------------------------ linguistic terms
# Gap variables are z-scores against company-wide dispersion, universe [-3, 3].
GAP_TERMS = {
    "NEG": lambda x: trapmf(x, -3.0, -3.0, -1.0, -0.2),
    "ZERO": lambda x: trimf(x, -0.8, 0.0, 0.8),
    "POS": lambda x: trapmf(x, 0.2, 1.0, 3.0, 3.0),
}
TEAM_PERF_TERMS = {
    "LOW": lambda x: trapmf(x, -3.0, -3.0, -1.0, -0.2),
    "MID": lambda x: trimf(x, -0.8, 0.0, 0.8),
    "HIGH": lambda x: trapmf(x, 0.2, 1.0, 3.0, 3.0),
}
SIZE_TERMS = {
    "SMALL": lambda x: trapmf(x, 0.0, 0.0, 4.0, 7.0),
    "MEDIUM": lambda x: trimf(x, 5.0, 9.0, 13.0),
    "LARGE": lambda x: trapmf(x, 11.0, 15.0, 30.0, 30.0),
}
SCARCITY_TERMS = {
    "LOW": lambda x: trapmf(x, 0.0, 0.0, 0.25, 0.5),
    "HIGH": lambda x: trapmf(x, 0.35, 0.6, 1.0, 1.0),
}
MATCH_TERMS = {
    "NO": lambda x: 1.0 - np.asarray(x, dtype=float),
    "YES": lambda x: np.asarray(x, dtype=float),
}

VARIABLE_TERMS = {
    "perf_gap": GAP_TERMS,
    "peer_gap": GAP_TERMS,
    "mgr_gap": GAP_TERMS,
    "team_perf": TEAM_PERF_TERMS,
    "team_size": SIZE_TERMS,
    "role_scarcity": SCARCITY_TERMS,
    "dept_match": MATCH_TERMS,
}

_D = config.DELTA_RANGE
DELTA_TERMS = {
    "VLOW": lambda u: trimf(u, -_D, -_D, -_D / 2),
    "LOW": lambda u: trimf(u, -_D, -_D / 2, 0.0),
    "NEUTRAL": lambda u: trimf(u, -_D / 3, 0.0, _D / 3),
    "HIGH": lambda u: trimf(u, 0.0, _D / 2, _D),
    "VHIGH": lambda u: trimf(u, _D / 2, _D, _D),
}


@dataclass(frozen=True)
class Rule:
    antecedents: tuple[tuple[str, str], ...]
    consequent: str
    note: str = ""


RULE_BASE: tuple[Rule, ...] = (
    Rule((("perf_gap", "POS"), ("peer_gap", "NEG")), "VLOW",
         "R1 high performer whose peers under-rate them - relative deprivation"),
    Rule((("perf_gap", "POS"), ("team_perf", "LOW")), "LOW",
         "R2 carrying a weak team"),
    Rule((("peer_gap", "ZERO"),), "HIGH",
         "R3 peer standing matches the team - person-environment fit"),
    Rule((("peer_gap", "NEG"), ("perf_gap", "NEG")), "LOW",
         "R4 outmatched on both counts - weakest member"),
    Rule((("mgr_gap", "POS"),), "HIGH",
         "R5 favourable managerial standing - LMX"),
    Rule((("mgr_gap", "NEG"),), "LOW",
         "R6 unfavourable managerial standing - LMX"),
    Rule((("team_size", "LARGE"),), "LOW",
         "R7 diluted contribution and voice - social loafing"),
    Rule((("team_size", "SMALL"), ("role_scarcity", "HIGH")), "LOW",
         "R8 sole specialist in a small team - token isolation"),
    Rule((("role_scarcity", "LOW"),), "HIGH",
         "R9 within-role peer support available"),
    Rule((("dept_match", "YES"),), "HIGH",
         "R10 domain familiarity - similarity attraction"),
    Rule((("perf_gap", "ZERO"), ("peer_gap", "ZERO"), ("team_perf", "HIGH")), "VHIGH",
         "R11 well-matched member of a strong team"),
    Rule((("team_perf", "HIGH"), ("perf_gap", "NEG")), "NEUTRAL",
         "R12 stretch assignment - ambiguous effect"),
)

UNIVERSE = np.linspace(-_D, _D, 101)
_CONSEQUENTS = np.stack([DELTA_TERMS[r.consequent](UNIVERSE) for r in RULE_BASE])  # (R, 101)


class MamdaniFIS:
    """Vectorised Mamdani inference: min t-norm, max aggregation, centroid defuzzification."""

    def __init__(self, rules: tuple[Rule, ...] = RULE_BASE, chunk: int = 20_000):
        self.rules = rules
        self.chunk = chunk
        self._consequents = np.stack([DELTA_TERMS[r.consequent](UNIVERSE) for r in rules])

    def firing_strengths(self, features: pd.DataFrame) -> np.ndarray:
        """(n_samples, n_rules) matrix of rule activations."""
        out = np.empty((len(features), len(self.rules)), dtype=float)
        for j, rule in enumerate(self.rules):
            strength = np.ones(len(features), dtype=float)
            for var, term in rule.antecedents:
                strength = np.minimum(strength, VARIABLE_TERMS[var][term](features[var].to_numpy()))
            out[:, j] = strength
        return out

    def infer(self, features: pd.DataFrame) -> np.ndarray:
        """Return the defuzzified satisfaction adjustment `delta` for each row."""
        n = len(features)
        result = np.zeros(n, dtype=float)
        for start in range(0, n, self.chunk):
            stop = min(start + self.chunk, n)
            firing = self.firing_strengths(features.iloc[start:stop])          # (m, R)
            clipped = np.minimum(firing[:, :, None], self._consequents[None])  # (m, R, U)
            aggregated = clipped.max(axis=1)                                   # (m, U)
            mass = aggregated.sum(axis=1)
            centroid = np.divide(
                (aggregated * UNIVERSE).sum(axis=1), mass,
                out=np.zeros(stop - start), where=mass > 0,
            )
            result[start:stop] = centroid
        return result

    def explain(self, features: pd.DataFrame, row: int = 0, top: int = 4) -> list[dict]:
        """Which rules drove one prediction - for the paper's worked example."""
        firing = self.firing_strengths(features.iloc[[row]])[0]
        order = np.argsort(firing)[::-1][:top]
        return [
            {"rule": self.rules[i].note, "consequent": self.rules[i].consequent,
             "firing": round(float(firing[i]), 4)}
            for i in order if firing[i] > 0
        ]


# ------------------------------------------------------------------ feature building
class RelationalFeatureBuilder:
    """Builds the leave-one-out relational features the FIS consumes.

    Team statistics always EXCLUDE the employee being scored. Without this the current
    team's average is contaminated by the employee's own value while a candidate team's
    is not, which would bias every counterfactual in the same direction.
    """

    def __init__(self, employees: pd.DataFrame, teams: pd.DataFrame):
        self.employees = employees.set_index(config.EMP_ID, drop=False)
        self.teams = teams.set_index(config.TEAM_ID, drop=False)

        # Company-wide dispersion is the reference scale, so gaps are comparable across
        # every team pair in the evaluation sweep.
        self._sd = {
            "perf": float(employees[config.PERF_COL].std()) or 1.0,
            "peer": float(employees[config.PEER_COL].std()) or 1.0,
            "mgr": float(employees[config.MGR_COL].std()) or 1.0,
        }
        grouped = employees.groupby(config.TEAM_ID)
        self._sum = {
            "perf": grouped[config.PERF_COL].sum(),
            "peer": grouped[config.PEER_COL].sum(),
            "mgr": grouped[config.MGR_COL].sum(),
        }
        self._count = grouped.size()
        team_perf_mean = (self._sum["perf"] / self._count)
        self._team_perf_ref = (float(team_perf_mean.mean()), float(team_perf_mean.std()) or 1.0)

        self._role_counts = employees.groupby([config.TEAM_ID, "Role"]).size()
        self._modal_dept = grouped["Department"].agg(lambda s: s.mode().iat[0])

    def build(self, emp_ids, team_ids) -> pd.DataFrame:
        """Parallel arrays in, one feature row per (employee, candidate team) pair."""
        emp_ids = np.asarray(emp_ids)
        team_ids = np.asarray(team_ids)
        emp = self.employees.loc[emp_ids]
        current = emp[config.TEAM_ID].to_numpy()
        in_team = current == team_ids

        n_members = self._count.reindex(team_ids).to_numpy(dtype=float)
        # leave-one-out: drop the employee from the team they already sit in
        n_others = np.where(in_team, n_members - 1.0, n_members)
        n_others = np.where(n_others <= 0, np.nan, n_others)

        def _mean_excl(key: str, col: str) -> np.ndarray:
            total = self._sum[key].reindex(team_ids).to_numpy(dtype=float)
            own = emp[col].to_numpy(dtype=float)
            return (total - np.where(in_team, own, 0.0)) / n_others

        perf_others = _mean_excl("perf", config.PERF_COL)
        peer_others = _mean_excl("peer", config.PEER_COL)
        mgr_others = _mean_excl("mgr", config.MGR_COL)

        role_key = pd.MultiIndex.from_arrays([team_ids, emp["Role"].to_numpy()])
        same_role = self._role_counts.reindex(role_key).fillna(0).to_numpy(dtype=float)
        same_role = np.where(in_team, same_role - 1.0, same_role)   # peers of my role, excluding me

        modal = self._modal_dept.reindex(team_ids).to_numpy()
        mean_ref, sd_ref = self._team_perf_ref

        return pd.DataFrame(
            {
                config.EMP_ID: emp_ids,
                config.TEAM_ID: team_ids,
                "perf_gap": (emp[config.PERF_COL].to_numpy() - perf_others) / self._sd["perf"],
                "peer_gap": (emp[config.PEER_COL].to_numpy() - peer_others) / self._sd["peer"],
                "mgr_gap": (emp[config.MGR_COL].to_numpy() - mgr_others) / self._sd["mgr"],
                "team_perf": (perf_others - mean_ref) / sd_ref,
                "team_size": n_members,
                "role_scarcity": 1.0 / (1.0 + np.maximum(same_role, 0.0)),
                "dept_match": (emp["Department"].to_numpy() == modal).astype(float),
            }
        ).fillna(0.0)


def satisfaction_matrix(
    employees: pd.DataFrame,
    teams: pd.DataFrame,
    sentiment: pd.DataFrame,
    emp_ids,
    team_ids,
    fis: MamdaniFIS | None = None,
    builder: "RelationalFeatureBuilder | None" = None,
) -> pd.DataFrame:
    """`s_hat(e, t)` for every employee x candidate team, indexed [Employee_ID, Team_ID].

    `builder` is reusable across pools and expensive to construct, so a sweep should build
    one and pass it in rather than paying the group-by cost on every pair.
    """
    fis = fis or MamdaniFIS()
    builder = builder or RelationalFeatureBuilder(employees, teams)

    emp_ids = list(emp_ids)
    team_ids = list(team_ids)
    grid_emp = np.repeat(emp_ids, len(team_ids))
    grid_team = np.tile(team_ids, len(emp_ids))

    features = builder.build(grid_emp, grid_team)
    delta = fis.infer(features)

    s0 = sentiment.set_index("Employee_ID")["s_crisp"].reindex(grid_emp).to_numpy(dtype=float)
    s_hat = np.clip(s0 + delta, 0.0, 1.0)

    return pd.DataFrame(
        {config.EMP_ID: grid_emp, config.TEAM_ID: grid_team,
         "s0": s0, "delta": delta, "s_hat": s_hat}
    ).pivot(index=config.EMP_ID, columns=config.TEAM_ID, values="s_hat")
