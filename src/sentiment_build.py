"""Adapter for the delivered sentiment dataset.

**This module does not compute sentiment.** The scores are produced by a separate
subsystem, which trains and tests its own model and hands us a finished dataset. Our job
is only to find the columns we need in it, check that it agrees with the employee roster,
and write the contract file the engine reads.

    dataset/<delivered sentiment file>        supplied to us
        Employee_ID + a sentiment score  (+ class probabilities, if shipped)
                        |
                        |   run.py sentiment  -- locate columns, check integrity
                        v
    dataset/sentiment_fuzzy.csv               what the engine reads

Column names are not fixed, because the subsystem is free to name its own output. Each
field is resolved by alias (see `_ID`, `_SCORE`, `_MU`), and `resolve_columns` reports
exactly which column was chosen for what, so a wrong guess is visible rather than silent.

Integrity is checked before anything is written, and a failure stops the write:

  * every employee in the roster has exactly one score, with no unknown IDs
  * the score is present and finite for every row, and lands in [0, 1] after scaling
  * the score is not a restatement of the star rating (`audit_leakage`) - if it were,
    sentiment would carry no information the engine does not already hold, and the pool
    objective would collapse to zero-sum (PROJECT.md 0.5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_MU_COLS = ["mu_neg", "mu_neu", "mu_pos"]
_CENTROIDS = np.array([0.0, 0.5, 1.0])

# Accepted spellings for each field we need out of the delivered dataset.
_ID = ("employee_id", "employeeid", "emp_id", "empid")
_SCORE = ("s_crisp", "model_sentiment_score", "sentiment_score", "sentiment",
          "score", "compound", "polarity")
_MU = {
    "mu_neg": ("mu_neg", "model_prob_low", "prob_low", "p_neg", "prob_negative"),
    "mu_neu": ("mu_neu", "model_prob_medium", "prob_medium", "p_neu", "prob_neutral"),
    "mu_pos": ("mu_pos", "model_prob_high", "prob_high", "p_pos", "prob_positive"),
}
_RATING = ("overall_rating", "rating", "stars")


class SentimentBuildError(RuntimeError):
    """Raised when the delivered dataset cannot produce a contract-conforming file."""


# ------------------------------------------------------------------ column resolution
def _pick(frame: pd.DataFrame, aliases) -> str | None:
    lower = {str(c).strip().lower(): c for c in frame.columns}
    return next((lower[a] for a in aliases if a in lower), None)


def resolve_columns(frame: pd.DataFrame) -> dict:
    """Report which column of the delivered dataset fills each field we need."""
    return {
        "id": _pick(frame, _ID),
        "score": _pick(frame, _SCORE),
        "rating": _pick(frame, _RATING),
        **{k: _pick(frame, v) for k, v in _MU.items()},
    }


def discover_input(explicit=None):
    """Locate the delivered sentiment dataset; first configured candidate that exists."""
    if explicit is not None:
        return explicit
    return next((c for c in config.SENTIMENT_INPUTS if c.exists()), config.SENTIMENT_INPUTS[0])


# ------------------------------------------------------------------ build
def _fuzzify(s: np.ndarray) -> np.ndarray:
    """Triangular fuzzification of a crisp [0,1] score, for a file that ships no classes."""
    neg = np.clip(1.0 - 2.0 * s, 0.0, 1.0)
    pos = np.clip(2.0 * s - 1.0, 0.0, 1.0)
    neu = np.clip(1.0 - np.abs(2.0 * s - 1.0), 0.0, 1.0)
    return np.column_stack([neg, neu, pos])


def build(path=None, employees_path=None) -> tuple[pd.DataFrame, dict]:
    """Read the delivered dataset, check it, and return ``(contract_frame, meta)``."""
    path = discover_input(path)
    employees_path = employees_path or config.EMPLOYEES_CSV

    if not path.exists():
        raise SentimentBuildError(
            f"no sentiment dataset found. Looked for:\n  "
            + "\n  ".join(str(c) for c in config.SENTIMENT_INPUTS)
            + "\nDrop the delivered file at one of those paths, or pass --scored PATH.\n"
            "Until then the engine falls back to the flagged synthetic placeholder."
        )

    raw = pd.read_csv(path)
    cols = resolve_columns(raw)
    if cols["id"] is None:
        raise SentimentBuildError(
            f"{path.name} has no Employee_ID column (looked for {list(_ID)}); "
            f"got {list(raw.columns)}"
        )
    has_mu = all(cols[k] for k in _MU_COLS)
    if cols["score"] is None and not has_mu:
        raise SentimentBuildError(
            f"{path.name} has no sentiment score (looked for {list(_SCORE)}) and no "
            f"class probabilities (looked for {[v[1] for v in _MU.values()]}); "
            f"got {list(raw.columns)}"
        )

    frame = pd.DataFrame({config.EMP_ID: raw[cols["id"]]})

    if has_mu:
        mu = raw[[cols[k] for k in _MU_COLS]].to_numpy(dtype=float)
        total = mu.sum(axis=1, keepdims=True)
        mu = mu / np.where(total <= 0, np.nan, total)
        score = (mu * _CENTROIDS).sum(axis=1) if cols["score"] is None \
            else raw[cols["score"]].to_numpy(dtype=float)
    else:
        score = raw[cols["score"]].to_numpy(dtype=float)

    # Accept a [-1, 1] convention (VADER compound) as well as [0, 1].
    rescaled = False
    if np.nanmin(score) < -0.001:
        score = (score + 1.0) / 2.0
        rescaled = True
    score = np.clip(score, 0.0, 1.0)
    if not has_mu:
        mu = _fuzzify(score)

    frame[_MU_COLS] = mu
    frame["s_crisp"] = score
    frame["n_reviews"] = raw[cols["n_reviews"]] if cols.get("n_reviews") else 1
    frame["join_level"] = "delivered"

    employees = pd.read_csv(employees_path)
    roster = set(employees[config.EMP_ID])
    delivered = set(frame[config.EMP_ID])

    meta = {
        "input": path.name,
        "rows": int(len(frame)),
        "columns_used": {k: v for k, v in cols.items() if v},
        "roster": len(roster),
        "roster_covered": len(roster & delivered),
        "roster_coverage_pct": round(100.0 * len(roster & delivered) / max(len(roster), 1), 2),
        "unknown_employee_ids": len(delivered - roster),
        "duplicate_id_rows": int(frame[config.EMP_ID].duplicated().sum()),
        "null_scores": int(pd.isna(frame["s_crisp"]).sum()),
        "rescaled_from_minus1_1": rescaled,
        "s_crisp_mean": round(float(frame["s_crisp"].mean()), 4),
        "s_crisp_std": round(float(frame["s_crisp"].std()), 4),
        "s_crisp_min": round(float(frame["s_crisp"].min()), 4),
        "s_crisp_max": round(float(frame["s_crisp"].max()), 4),
    }
    return frame, meta


# ------------------------------------------------------------------ integrity audit
def audit_leakage(path=None) -> dict:
    """Is the delivered score just a restatement of the star rating?

    An r2 above ~0.95 means the subsystem's model learned to echo the rating rather than
    read the review text. Sentiment would then be a second copy of a number the engine
    already holds, and the pool objective would return to exactly zero-sum.
    """
    path = discover_input(path)
    if not path.exists():
        return {"testable": False, "reason": "sentiment dataset not present"}

    raw = pd.read_csv(path)
    cols = resolve_columns(raw)
    if cols["rating"] is None or cols["score"] is None:
        return {"testable": False, "reason": "dataset ships no rating column to test against"}

    valid = raw[[cols["rating"], cols["score"]]].dropna()
    if len(valid) < 3 or valid[cols["rating"]].nunique() < 2:
        return {"testable": False, "reason": "rating column is constant or too short"}

    r = float(np.corrcoef(valid[cols["rating"]].astype(float),
                          valid[cols["score"]].astype(float))[0, 1])
    return {
        "testable": True,
        "n": int(len(valid)),
        "pearson_r": round(r, 4),
        "r2": round(r * r, 4),
        "leaking": bool(r * r > 0.95),
        "verdict": (
            "LEAK - the score restates the rating; sentiment adds nothing."
            if r * r > 0.95
            else "OK - sentiment carries information beyond the rating."
        ),
    }


def save(frame: pd.DataFrame, path=None):
    path = path or config.SENTIMENT_CSV
    frame.to_csv(path, index=False)
    return path
