"""Central configuration. Every path in the project resolves from here (fixes BUG-16:
hard-coded bare relative paths that broke whenever a notebook was run from another cwd).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"

EMPLOYEES_CSV = DATASET / "Employee_Team_Assignment.csv"
TEAMS_CSV = DATASET / "Team_Dataset.csv"
SENTIMENT_CSV = DATASET / "sentiment_fuzzy.csv"       # written by `run.py sentiment`; read by the engine
REVIEWS_CSV = ROOT / "employee_reviews.csv"
# The sentiment subsystem hands us a finished dataset; we never compute the scores.
# Searched in order, first match wins.
#
# The first two entries are byte-for-byte the same 5,000 rows (verified: only the CSV
# dtype of the rating columns differs, float64 vs int64). The second is a copy saved
# under a broken filename - "dataset" + "sentiment_fuzzy.csv" with the separator
# missing. Both are read, never written; the duplicate should be removed upstream.
SENTIMENT_INPUTS = (
    DATASET / "Capgemini_Employee_Reviews_IT_only_with_MODEL_PREDICTIONS.csv.xls",
    DATASET / "datasetsentiment_fuzzy.csv",
    DATASET / "sentiment_scores.csv",
)

OUTPUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- schema
EMP_ID = "Employee_ID"
TEAM_ID = "Team_ID"

# The four *working* roles that a swap must match on. "Team Lead" is deliberately NOT
# one of them - see roles.py for why it is carried as a structural flag instead.
WORKING_ROLES = ("Developer", "QA", "DevOps", "Business Analyst")
LEAD_ROLE = "Team Lead"

# Employee metric columns used by the objective.
PERF_COL = "Performance Score"
PEER_COL = "Peer Rating"
MGR_COL = "Manager Feedback"
KPI_COL = "KPI Score"

# Team aggregate columns.
TEAM_SIZE_COL = "Team_Size"
TEAM_PERF_COL = "Team_Performance"
TEAM_CATEGORY_COL = "Performance_Category"

# ------------------------------------------------------------------- discovery knobs
# The system is handed a dataset, not a team pair: it finds the under-performing teams
# itself. `Performance_Category` is a clean quartile split of `Team_Performance`
# (Low 73.70-78.09, Medium 78.10-80.86, High 80.89-86.63; the 0.25 quantile is 78.092),
# so it is trustworthy - unlike the shipped `Role` column, which was random. When the
# column is absent the same split is recomputed from LOW_QUANTILE.
LOW_CATEGORY = "Low"
LOW_QUANTILE = 0.25

# Pool formation. A pool is the set of teams allowed to exchange members with one another.
POOL_SIZE = 2             # 2 => every pool solved to PROVEN global optimality
MIN_SHARED_ROLES = 1      # a pool is only feasible if its teams share a role class
TOP_RECOMMENDATIONS = 10  # how many pools to hand to HR

# --------------------------------------------------------------------------- knobs
# Screening (the "who is at risk" rule). Reported to HR as context; it does not select
# the swap - the optimiser does.
SCREEN_W_PERF = 0.5      # weight on (1 - normalised performance)
SCREEN_W_SENT = 0.5      # weight on (1 - sentiment)
SCREEN_KEEP_FRAC = 1.0   # keep the worst q fraction per (team, role); 1.0 = no screening

# Value function  v(e) = perf_norm(e) * phi(s(e)),  phi(s) = PHI_BASE + PHI_SLOPE*s
# Sentiment is INTRINSIC to the employee: one score per person, no team term. See
# objective.py for why the pool objective still responds to a swap under that model.
PHI_BASE = 0.5
PHI_SLOPE = 1.0

# Pool objective  J = min_t TeamScore(t) - LAMBDA_BALANCE * std_t TeamScore(t)
LAMBDA_BALANCE = 0.5

# Search
CHURN_BUDGET = 2          # max swaps per role class (see PROJECT.md 0.9 for the cost table)
ALLOW_LEAD_SWAPS = False  # leadership continuity; set True to let leads swap with leads
MAX_ENUM_STATES = 2_000_000   # above this, exhaustive search falls back to local search

# Protective filter (replaces the old do-no-harm constraint, which needed a counterfactual
# satisfaction prediction the system no longer makes). An employee whose observed
# sentiment is already below the floor is disengaged; an involuntary reassignment is a
# known churn accelerant for exactly that group, so they are held in place.
# CALIBRATION. 0.35 was chosen against the synthetic Beta(5,3) placeholder (std
# 0.1624), where it held 288 employees. On the delivered scores (std 0.3268) the same
# absolute threshold holds 791, and two low teams lose every movable member to it. The
# constraint is therefore MORE binding on real data, not less, and its price is
# 11.4% of mean gain rather than 5.1%. A floor expressed as a quantile of the observed
# distribution would survive a change of sentiment source; this one does not.
SENTIMENT_FLOOR = 0.35
ENFORCE_SENTIMENT_FLOOR = True

RANDOM_SEED = 42

# ------------------------------------------------------------------------- retired
# Range of the counterfactual satisfaction adjustment produced by the Mamdani FIS in
# fis.py. That module is NO LONGER IN THE DECISION PATH: the system does not predict how
# satisfied an employee would be in a team they have never worked in, because nothing in
# the data can validate such a prediction. Both are kept so the inference system remains
# runnable as reference material for the write-up.
DELTA_RANGE = 0.30
