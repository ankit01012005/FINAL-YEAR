"""BLOCKER #1 - re-derive `Role` from real evidence.

The shipped `Role` column is statistically independent of both `Title`
(chi-2 p = 0.5463) and `Department` (p = 0.5112): it was drawn at random. Since a
role-matched swap is the *only* feasibility constraint in the system, a random Role makes
every swap meaningless. This module replaces it.

Which title tokens are allowed to decide a role
-----------------------------------------------
Not all of them. This is an IT-services organisation whose titles encode a *promotion
grade* as often as an occupation. `measure_token_lift()` quantifies which is which: for
each token it compares the department mix of the titles containing it against the
company-wide mix, and reports the lift of the dominant bucket. Measured over all 5,000
employees:

    token           n     lift   verdict
    tester         34     7.40   OCCUPATION
    test          112     7.21   OCCUPATION
    qa             32     7.05   OCCUPATION
    automation     48     7.05   OCCUPATION
    mm             30     5.04   OCCUPATION
    network        57     4.20   OCCUPATION
    administrator  62     4.19   OCCUPATION
    business       40     4.13   OCCUPATION
    security       37     4.05   OCCUPATION
    cloud          26     4.03   OCCUPATION
    functional     29     4.03   OCCUPATION
    devops         39     3.97   OCCUPATION
    data           95     3.55   OCCUPATION
    sap           151     3.42   OCCUPATION
    java           33     1.63   OCCUPATION
    developer     314     1.57   OCCUPATION
    ----------------------------------------- threshold: lift >= 1.5
    project        68     1.39   weak
    technical      92     1.27   weak
    professional   97     1.26   weak
    software      982     1.24   weak
    architect      36     1.19   weak
    manager       412     1.11   grade
    engineer     1057     1.01   grade      <- appears in every department
    senior       1462     1.02   grade
    associate     761     0.97   grade
    analyst       755     0.91   grade
    consultant   2008     0.94   grade      <- 40% of all employees; a grade, not a job
    lead          159     0.92   grade

Anything below the threshold is discarded rather than trusted. Dropping "software" costs
nothing: "Software Engineer" then resolves through `Department`, which is
"Software Development Department" anyway.

Method
------
1. Strip grade tokens and parenthetical noise ("(working remotely)", "C1", "L2", "II").
2. Match the surviving tokens against the measured occupational signals; failing that,
   against a small set of unambiguous technical terms too rare to measure (n < 25).
3. Fall back to `Department`, which is real and occupationally meaningful.
4. Anything still unresolved takes the modal role, flagged in `Role_Source` so the
   fraction resting on a default is always reportable.

Team Lead is handled separately. It is not an occupation but a *structural slot*: the
dataset carries exactly one per team and that invariant has to survive. It is kept as the
boolean `Is_Team_Lead`, retained from the shipped data rather than derived, while the four
working roles are re-derived for everyone. Leads are excluded from swaps by default
(`config.ALLOW_LEAD_SWAPS`), so the arbitrariness of *which* member carries the flag does
not propagate into the results.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config

# --------------------------------------------------------------------------------- #
# Tokens that survived the lift >= 1.5 test above. Order within a role is irrelevant;
# order *between* roles breaks ties, most specific first.
# --------------------------------------------------------------------------------- #
MEASURED_SIGNALS: dict[str, tuple[str, ...]] = {
    "QA": ("test", "tester", "qa", "automation"),
    "DevOps": ("devops", "administrator", "network", "security", "cloud"),
    "Business Analyst": ("sap", "mm", "functional", "business", "data"),
    "Developer": ("developer", "java"),
}

# Unambiguous technical terms below the n >= 25 measurement floor. Kept separate so the
# evidence-backed rules and the judgement calls never get conflated in a report.
SEMANTIC_SIGNALS: dict[str, tuple[str, ...]] = {
    "QA": ("testing", "sdet", "quality", "validation", "qc"),
    "DevOps": (
        "sre", "reliability", "infrastructure", "infra", "sysadmin", "admin", "linux",
        "unix", "dba", "database", "warehousing", "storage", "middleware", "helpdesk",
        "desktop", "noc", "telecom", "hardware", "server", "kubernetes", "docker",
        "aws", "azure", "gcp", "networking", "cyber", "firewall", "voice",
    ),
    "Developer": (
        "abap", "hana", "oracle", "sql", "salesforce", "android", "ios", "mobile", "web",
        "ui", "ux", "frontend", "backend", "fullstack", "programmer", "python", "dotnet",
        "php", "angular", "react", "node", "sde", "coder", "development",
    ),
    "Business Analyst": (
        "scrum", "owner", "intelligence", "analytics", "reporting", "domain", "crm",
        "erp", "finance", "hr", "payroll", "procurement",
    ),
}

DEPARTMENT_RULES: list[tuple[str, str]] = [
    ("QA", r"\b(quality|testing|test)\b"),
    ("DevOps", r"\b(devops|infrastructure|network|support|security|cyber|cloud|telecom"
               r"|hardware|dba|warehousing|operations)\b"),
    ("Business Analyst", r"\b(consulting|business|intelligence|analytics|product"
                         r"|management|science)\b"),
    ("Developer", r"\b(software|development|engineering|technology|it|ui|ux|sap|data)\b"),
]

# Grade / seniority / formatting tokens - measured lift ~1.0, so they decide nothing.
GRADE_TOKENS = frozenset({
    "consultant", "senior", "junior", "sr", "jr", "associate", "analyst", "manager",
    "lead", "engineer", "engineering", "professional", "specialist", "team", "process",
    "and", "working", "remotely", "technical", "architect", "software", "project",
    "trainee", "intern", "principal", "staff", "member", "executive", "officer",
    "i", "ii", "iii", "iv", "l", "a", "b", "c", "d", "e",
})

DEFAULT_ROLE = "Developer"
ROLE_ORDER = ("QA", "DevOps", "Business Analyst", "Developer")

_DEPT_COMPILED = [(r, re.compile(p, re.I)) for r, p in DEPARTMENT_RULES]
_PARENS = re.compile(r"\(.*?\)")
_ALNUM_GRADE = re.compile(r"^[a-z]?\d+[a-z]?$|^\d+$")


def _tokens(text: object) -> set[str]:
    """Occupationally meaningful tokens only: parentheticals, grades and codes removed."""
    s = _PARENS.sub(" ", str(text or "").lower())
    s = re.sub(r"[^a-z0-9+]+", " ", s)
    return {
        t for t in s.split()
        if t and t not in GRADE_TOKENS and not _ALNUM_GRADE.match(t)
    }


def role_from_title(title: object, department: object = "") -> tuple[str, str]:
    """Return ``(role, source)``; source is title_measured | title_semantic | department | default."""
    tokens = _tokens(title)

    for role in ROLE_ORDER:
        if tokens & set(MEASURED_SIGNALS.get(role, ())):
            return role, "title_measured"
    for role in ROLE_ORDER:
        if tokens & set(SEMANTIC_SIGNALS.get(role, ())):
            return role, "title_semantic"

    dept = f" {re.sub(r'[^a-z]+', ' ', str(department or '').lower()).strip()} "
    for role, pattern in _DEPT_COMPILED:
        if pattern.search(dept):
            return role, "department"

    return DEFAULT_ROLE, "default"


def derive_roles(employees: pd.DataFrame) -> pd.DataFrame:
    """Add ``Role_Derived``, ``Role_Source`` and ``Is_Team_Lead`` to a copy of the frame.

    The shipped ``Role`` is preserved as ``Role_Original`` so the comparison stays
    auditable, and ``Role`` itself is overwritten with the derived value because that is
    what the swap-feasibility check matches on.
    """
    df = employees.copy()
    df["Role_Original"] = df["Role"]
    df["Is_Team_Lead"] = df["Role"].eq(config.LEAD_ROLE)

    # Distinct (title, department) pairs are ~1/4 of the rows; resolving once and mapping
    # keeps this linear in unique pairs rather than in employees.
    pairs = df[["Title", "Department"]].drop_duplicates()
    lookup = {
        (t, d): role_from_title(t, d) for t, d in zip(pairs["Title"], pairs["Department"])
    }
    resolved = [lookup[(t, d)] for t, d in zip(df["Title"], df["Department"])]

    df["Role_Derived"] = [r for r, _ in resolved]
    df["Role_Source"] = [s for _, s in resolved]
    df["Role"] = df["Role_Derived"]
    return df


def recompute_team_role_counts(employees: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the ``Num_*`` columns so they describe the derived roles.

    ``Num_Team_Lead`` counts the structural flag; the four working-role counts are taken
    over non-leads, preserving the documented invariant ``sum(Num_*) == Team_Size``.
    """
    df = teams.copy()
    non_leads = employees.loc[~employees["Is_Team_Lead"]]
    counts = non_leads.groupby([config.TEAM_ID, "Role"]).size().unstack(fill_value=0)
    lead_counts = employees.groupby(config.TEAM_ID)["Is_Team_Lead"].sum()

    column_for = {
        "Developer": "Num_Developer",
        "QA": "Num_QA",
        "DevOps": "Num_DevOps",
        "Business Analyst": "Num_Business_Analyst",
    }
    for role, col in column_for.items():
        series = counts[role] if role in counts.columns else 0
        df[col] = df[config.TEAM_ID].map(series).fillna(0).astype(int)
    df["Num_Team_Lead"] = df[config.TEAM_ID].map(lead_counts).fillna(0).astype(int)

    df["Num_Total_Check"] = df[["Num_Team_Lead", *column_for.values()]].sum(axis=1)
    return df


# ------------------------------------------------------------------ evidence & audit
def measure_token_lift(employees: pd.DataFrame, min_count: int = 25) -> pd.DataFrame:
    """Reproduce the table in this module's docstring.

    Coarse-buckets `Department` into the four working roles, then for each title token
    reports how concentrated its department mix is relative to the company baseline. Lift
    near 1.0 means the token is a grade and carries no occupational information.
    """
    from collections import Counter

    def bucket(dept: object) -> str:
        d = str(dept).lower()
        for role, pattern in _DEPT_COMPILED:
            if pattern.search(f" {re.sub(r'[^a-z]+', ' ', d).strip()} "):
                return role
        return DEFAULT_ROLE

    df = employees.copy()
    df["_bucket"] = df["Department"].map(bucket)
    baseline = df["_bucket"].value_counts(normalize=True)

    counter: Counter = Counter()
    for title in df["Title"]:
        counter.update(_tokens(title) | {t for t in str(title).lower().split()})

    rows = []
    for token, _ in counter.most_common():
        if len(token) < 2 or not token.isalnum():
            continue
        subset = df[df["Title"].str.contains(rf"\b{re.escape(token)}\b", case=False, na=False)]
        if len(subset) < min_count:
            continue
        mix = subset["_bucket"].value_counts(normalize=True)
        dominant = mix.idxmax()
        lift = float(mix.max() / baseline.get(dominant, 1.0))
        rows.append({
            "token": token, "n": len(subset), "dominant_bucket": dominant,
            "share": round(float(mix.max()), 3), "lift": round(lift, 2),
            "verdict": "OCCUPATION" if lift >= 1.5 else ("weak" if lift >= 1.12 else "grade"),
        })
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def coverage_report(employees: pd.DataFrame) -> dict:
    """Evidence the repair worked: `chi2_p_*` should collapse from ~0.5 to ~0."""
    from scipy.stats import chi2_contingency

    def _p(col: str) -> float:
        return float(chi2_contingency(pd.crosstab(employees[col], employees["Role_Derived"]))[1])

    source = employees["Role_Source"]
    return {
        "n_employees": int(len(employees)),
        "role_source_counts": source.value_counts().to_dict(),
        "pct_from_title": round(100 * source.str.startswith("title").mean(), 2),
        "pct_from_measured_title": round(100 * source.eq("title_measured").mean(), 2),
        "pct_from_department": round(100 * source.eq("department").mean(), 2),
        "pct_from_default": round(100 * source.eq("default").mean(), 2),
        "derived_role_counts": employees["Role_Derived"].value_counts().to_dict(),
        "original_role_counts": employees["Role_Original"].value_counts().to_dict(),
        "agreement_with_original_pct": round(
            100 * employees["Role_Derived"].eq(employees["Role_Original"]).mean(), 2
        ),
        "chi2_p_title_vs_derived": _p("Title"),
        "chi2_p_department_vs_derived": _p("Department"),
    }
