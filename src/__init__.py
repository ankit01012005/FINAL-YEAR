"""Performance-driven team rebalancing by role-matched employee swaps.

Given a dataset of employees and teams, discover the under-performing teams, group them
into pools that can legally exchange members, and recommend which employees to swap -
on role match, performance and observed sentiment - subject to a churn budget.

    from src import pipeline
    ctx = pipeline.load_context()
    out = pipeline.recommend(ctx, pool_size=2, top=10)
    print(pipeline.pools_frame(out["results"]))
    print(pipeline.moves_frame(ctx, out["results"]))

The output is a *recommendation* for HR, not an instruction, and the system makes no
claim about how satisfied an employee would be after a move - see objective.py.

`fis` is deliberately absent from this import list: the Mamdani inference system that
predicted counterfactual satisfaction is retired from the decision path and retained only
as reference material. Import it explicitly if you need it.

See PROJECT.md section 0 for the design rationale, in particular why no Hungarian solve
appears anywhere and why the objective cannot be a mean.
"""
from . import (config, evaluate, objective, pipeline, roles, search,  # noqa: F401
               sentiment_build, sentiment_io)

__all__ = [
    "config", "roles", "sentiment_io", "sentiment_build", "objective", "search",
    "pipeline", "evaluate",
]
