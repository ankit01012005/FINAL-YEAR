"""Boundary with the sentiment subsystem (owned by another team member).

Contract - `dataset/sentiment_fuzzy.csv`, one row per employee:

    Employee_ID   int     joins 1:1 against Employee_Team_Assignment.csv (5,000 rows)
    mu_neg        float   fuzzy membership, negative        [0,1]
    mu_neu        float   fuzzy membership, neutral         [0,1]
    mu_pos        float   fuzzy membership, positive        [0,1]
    s_crisp       float   defuzzified sentiment             [0,1]
    n_reviews     int     reviews backing this employee     (confidence weight)
    join_level    str     title_dept | dept_only | global_prior

The loader is deliberately tolerant: any of the three shapes below is accepted and
normalised to the full schema, because the upstream system may well hand over a
two-column file.

    full     Employee_ID + mu_* + s_crisp
    fuzzy    Employee_ID + mu_*              -> s_crisp derived by centroid defuzzification
    crisp    Employee_ID + one score column  -> mu_* derived by fuzzification

TWO INVARIANTS THE UPSTREAM SYSTEM MUST HOLD (see PROJECT.md 0.8):

1. `s_crisp` must NOT be a deterministic function of `Overall_rating` (BUG-4). If it is,
   sentiment restates a number we already hold, it collapses to an intrinsic per-employee
   constant, and the swap objective returns to exactly zero-sum (PROJECT.md 0.5). The
   model must read review *text*; `rating_norm` must not be a fuzzy input feature.
2. Null ratings must not fall through to POSITIVE (BUG-5: 14,330 rows currently do).

`audit_against_rating()` tests invariant 1 empirically once the real file lands.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config

FULL_COLUMNS = ["Employee_ID", "mu_neg", "mu_neu", "mu_pos", "s_crisp", "n_reviews", "join_level"]
_MU = ["mu_neg", "mu_neu", "mu_pos"]
_CRISP_ALIASES = ["s_crisp", "sentiment", "sentiment_score", "score", "compound", "polarity"]

# Centroids of the three sentiment membership functions on a [0,1] universe.
_CENTROIDS = np.array([0.0, 0.5, 1.0])


class SentimentContractError(ValueError):
    """Raised when the supplied sentiment file cannot be reconciled with the contract."""


# ------------------------------------------------------------------ normalisation
def _defuzzify(mu: np.ndarray) -> np.ndarray:
    """Centroid defuzzification of the three memberships onto [0,1]."""
    total = mu.sum(axis=1)
    total = np.where(total <= 0, np.nan, total)
    return (mu * _CENTROIDS).sum(axis=1) / total


def _fuzzify(s: np.ndarray) -> np.ndarray:
    """Triangular fuzzification of a crisp [0,1] score into (mu_neg, mu_neu, mu_pos)."""
    s = np.clip(s, 0.0, 1.0)
    neg = np.clip(1.0 - 2.0 * s, 0.0, 1.0)
    pos = np.clip(2.0 * s - 1.0, 0.0, 1.0)
    neu = np.clip(1.0 - np.abs(2.0 * s - 1.0), 0.0, 1.0)
    return np.column_stack([neg, neu, pos])


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    id_col = next((c for c in out.columns if c.lower() in {"employee_id", "employeeid", "emp_id"}), None)
    if id_col is None:
        raise SentimentContractError(
            f"no Employee_ID column found; got {list(out.columns)}"
        )
    out = out.rename(columns={id_col: "Employee_ID"})

    has_mu = all(c in out.columns for c in _MU)
    crisp_col = next((c for c in out.columns if c.lower() in _CRISP_ALIASES), None)

    if has_mu:
        mu = out[_MU].to_numpy(dtype=float)
        # Renormalise so the memberships form a partition; tolerate small drift silently.
        row_sum = mu.sum(axis=1, keepdims=True)
        if np.nanmax(np.abs(row_sum - 1.0)) > 0.05:
            warnings.warn("mu_neg+mu_neu+mu_pos deviates from 1 by >0.05; renormalising.")
        mu = mu / np.where(row_sum <= 0, np.nan, row_sum)
        out[_MU] = mu
        if crisp_col is None:
            out["s_crisp"] = _defuzzify(mu)
        else:
            out = out.rename(columns={crisp_col: "s_crisp"})
    elif crisp_col is not None:
        out = out.rename(columns={crisp_col: "s_crisp"})
        s = out["s_crisp"].to_numpy(dtype=float)
        # Accept [-1,1] (VADER compound) as well as [0,1].
        if np.nanmin(s) < -0.001:
            s = (s + 1.0) / 2.0
            out["s_crisp"] = s
        out[_MU] = _fuzzify(s)
    else:
        raise SentimentContractError(
            "file has neither mu_neg/mu_neu/mu_pos nor a recognisable crisp score column "
            f"(looked for {_CRISP_ALIASES}); got {list(out.columns)}"
        )

    if "n_reviews" not in out.columns:
        out["n_reviews"] = 0
    if "join_level" not in out.columns:
        out["join_level"] = "unspecified"

    out["s_crisp"] = out["s_crisp"].astype(float).clip(0.0, 1.0)
    return out[FULL_COLUMNS]


# ------------------------------------------------------------------ validation
def validate(sentiment: pd.DataFrame, employee_ids: pd.Series) -> list[str]:
    """Return a list of human-readable contract violations (empty list == clean)."""
    problems: list[str] = []
    ids = set(employee_ids)

    dupes = int(sentiment["Employee_ID"].duplicated().sum())
    if dupes:
        problems.append(f"{dupes} duplicated Employee_ID rows (must be 1:1)")

    missing = ids - set(sentiment["Employee_ID"])
    if missing:
        problems.append(f"{len(missing)} of {len(ids)} employees have no sentiment row")

    extra = set(sentiment["Employee_ID"]) - ids
    if extra:
        problems.append(f"{len(extra)} sentiment rows reference unknown Employee_IDs")

    nulls = int(sentiment["s_crisp"].isna().sum())
    if nulls:
        problems.append(f"{nulls} null s_crisp values")

    spread = float(sentiment["s_crisp"].std())
    if spread < 0.01:
        problems.append(
            f"s_crisp is near-constant (std={spread:.4f}); a constant sentiment makes every "
            "swap worth exactly zero - see PROJECT.md 0.5"
        )
    return problems


def audit_against_rating(sentiment: pd.DataFrame, reviews_path=None) -> dict:
    """Empirical test of contract invariant 1 (BUG-4 leakage).

    Only meaningful once the real file lands and carries a rating column alongside the
    score. Returns an ``r2`` of s_crisp regressed on the rating; anything above ~0.95
    means sentiment is a restatement of the rating, not a reading of the text.
    """
    rating_col = next(
        (c for c in sentiment.columns if "rating" in c.lower() or c.lower() == "label"), None
    )
    if rating_col is None:
        return {"testable": False, "reason": "no rating/label column shipped alongside s_crisp"}

    valid = sentiment[[rating_col, "s_crisp"]].dropna()
    if valid[rating_col].nunique() < 2:
        return {"testable": False, "reason": "rating column is constant"}

    r = float(np.corrcoef(valid[rating_col].astype(float), valid["s_crisp"])[0, 1])
    return {
        "testable": True,
        "pearson_r": round(r, 4),
        "r2": round(r * r, 4),
        "leaking": bool(r * r > 0.95),
        "verdict": (
            "LEAK - s_crisp is a restatement of the rating (BUG-4). Swap objective will be "
            "zero-sum." if r * r > 0.95 else "OK - sentiment carries information beyond the rating."
        ),
    }


# ------------------------------------------------------------------ loading
def load_sentiment(
    path=None,
    employee_ids: pd.Series | None = None,
    strict: bool = True,
    allow_placeholder: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load and normalise the sentiment file.

    Falls back to a clearly-flagged synthetic placeholder when the file is absent, so the
    pipeline is runnable end to end before the real scores arrive. Swapping in the real
    file requires no code change.

    Returns ``(frame, meta)``; ``meta['is_placeholder']`` is the flag every downstream
    report must surface.
    """
    path = config.SENTIMENT_CSV if path is None else path

    if not path.exists():
        if not allow_placeholder:
            raise SentimentContractError(f"{path} not found and placeholder mode disabled")
        if employee_ids is None:
            raise SentimentContractError("placeholder mode needs employee_ids")
        warnings.warn(
            f"\n{'!' * 78}\n"
            f"  {path.name} NOT FOUND - USING A SYNTHETIC PLACEHOLDER.\n"
            "  The pipeline will run, but NO RESULT IS VALID until the real sentiment\n"
            "  scores are supplied. Every report is flagged is_placeholder=True.\n"
            f"{'!' * 78}",
            stacklevel=2,
        )
        frame = make_placeholder(employee_ids)
        return frame, {"is_placeholder": True, "source": "synthetic", "problems": []}

    raw = pd.read_csv(path)
    frame = _normalise(raw)

    problems: list[str] = []
    if employee_ids is not None:
        problems = validate(frame, employee_ids)
        if problems and strict:
            raise SentimentContractError(
                "sentiment file violates the contract:\n  - " + "\n  - ".join(problems)
            )
        # Restrict and order to the employee roster.
        frame = (
            frame.set_index("Employee_ID")
            .reindex(employee_ids)
            .rename_axis("Employee_ID")
            .reset_index()
        )

    meta = {
        "is_placeholder": False,
        "source": str(path),
        "problems": problems,
        "n_rows": int(len(frame)),
        "s_crisp_mean": round(float(frame["s_crisp"].mean()), 4),
        "s_crisp_std": round(float(frame["s_crisp"].std()), 4),
        "join_level_counts": frame["join_level"].value_counts().to_dict(),
    }
    return frame, meta


def make_placeholder(employee_ids: pd.Series, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Synthetic stand-in for the real sentiment file.

    Draws a latent per-employee disposition from a Beta(5,3) - a plausible satisfaction
    shape, mildly positive-skewed. Deliberately **independent of every performance
    metric**, so it cannot leak into the objective and cannot manufacture a result that
    the real data would not also produce.
    """
    rng = np.random.default_rng(seed)
    n = len(employee_ids)
    s = rng.beta(5.0, 3.0, size=n)
    mu = _fuzzify(s)
    return pd.DataFrame(
        {
            "Employee_ID": np.asarray(employee_ids),
            "mu_neg": mu[:, 0],
            "mu_neu": mu[:, 1],
            "mu_pos": mu[:, 2],
            "s_crisp": s,
            "n_reviews": 0,
            "join_level": "PLACEHOLDER",
        }
    )
