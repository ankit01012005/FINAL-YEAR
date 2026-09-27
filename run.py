"""Command-line entry point.

    python run.py recommend                      # THE MAIN PATH: dataset in, ranked
                                                 # reassignment recommendations out
    python run.py recommend --pool-size 3 --top 5
    python run.py discover                       # which teams the system flagged, and why
    python run.py roles                          # role-repair audit (BLOCKER #1)
    python run.py swap --teams TEAM_0030 TEAM_0245   # manual override for one pool
    python run.py sweep --limit 200              # baselines + ablations over team pairs
    python run.py sentiment                      # build dataset/sentiment_fuzzy.csv from
                                                 # the scored review corpus
    python run.py doctor                         # end-to-end health check

No team pair is supplied by hand in the production path: `recommend` finds the
under-performing teams itself and decides which of them are worth acting on.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import pandas as pd

from src import config, evaluate, pipeline, roles, sentiment_build, sentiment_io


def _print(title: str, payload) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if isinstance(payload, pd.DataFrame):
        print(payload.to_string(index=False) if len(payload) else "(empty)")
    else:
        print(json.dumps(payload, indent=2, default=str))


def _placeholder_banner(ctx: pipeline.Context) -> None:
    if ctx.is_placeholder:
        print(
            "\n" + "!" * 78
            + "\n  SENTIMENT IS A SYNTHETIC PLACEHOLDER."
            + f"\n  Drop the real file at {config.SENTIMENT_CSV} and re-run;"
            + "\n  no code change is needed. Until then no result below is valid."
            + "\n" + "!" * 78
        )


def _advisory_footer() -> None:
    print(
        "\nThese are RECOMMENDATIONS for HR review, not instructions. Each row names the\n"
        "role match, the performance gap and the sentiment score behind it. The system\n"
        "does not predict whether a move will suit the employee personally - that\n"
        "judgement stays with the people who know them."
    )


def cmd_discover(args) -> None:
    ctx = pipeline.load_context()
    _placeholder_banner(ctx)

    low = pipeline.discover_low_teams(ctx, category=args.category)
    _print(f"DISCOVERED UNDER-PERFORMING TEAMS ({len(low)})", low.head(args.head))
    if len(low) > args.head:
        print(f"\n... {len(low) - args.head} more (use --head to show more)")

    pools = pipeline.form_pools(ctx, list(low[config.TEAM_ID]), pool_size=args.pool_size)
    head = pipeline.pool_headroom(ctx, pools).sort_values("headroom", ascending=False)
    _print("POOL FORMATION", {
        "discovery_rule": low["discovery_rule"].iat[0] if len(low) else None,
        "teams_found": int(len(low)),
        "teams_with_no_movable_member": int((low["movable"] == 0).sum()),
        "pool_size": args.pool_size,
        "feasible_pools": int(len(pools)),
        "headroom_max": round(float(head["headroom"].max()), 4) if len(head) else None,
        "headroom_median": round(float(head["headroom"].median()), 4) if len(head) else None,
    })
    _print("HIGHEST-HEADROOM POOLS (upper bound on gain, before any search)", head.head(10))


def cmd_recommend(args) -> None:
    ctx = pipeline.load_context()
    _placeholder_banner(ctx)

    out = pipeline.recommend(
        ctx, pool_size=args.pool_size, top=args.top, category=args.category,
        budget=args.budget, lam=args.lam,
    )
    _print("DISCOVERY AND SEARCH", out["meta"])

    pools = pipeline.pools_frame(out["results"])
    _print("RECOMMENDED POOLS (ranked)", pools)

    moves = pipeline.moves_frame(ctx, out["results"])
    _print("RECOMMENDED MOVES", moves)
    _advisory_footer()

    if args.save and len(moves):
        moves.to_csv(config.OUTPUT / "recommended_swaps.csv", index=False)
        pools.to_csv(config.OUTPUT / "recommended_pools.csv", index=False)
        out["low_teams"].to_csv(config.OUTPUT / "low_teams.csv", index=False)
        print(f"\nwritten: {config.OUTPUT / 'recommended_swaps.csv'}"
              f"\nwritten: {config.OUTPUT / 'recommended_pools.csv'}"
              f"\nwritten: {config.OUTPUT / 'low_teams.csv'}")


def cmd_roles(args) -> None:
    ctx = pipeline.load_context()
    if args.tokens:
        _print("TITLE-TOKEN LIFT (which tokens may decide a role)",
               roles.measure_token_lift(ctx.employees))
    _print("ROLE REPAIR - BLOCKER #1", ctx.role_report)
    print(
        "\nchi-2 p-values against the SHIPPED Role column were 0.5463 (Title) and "
        "0.5112 (Department):\nthe labels were random. The derived values above should "
        "be ~0, i.e. Role now tracks real evidence."
    )
    check = ctx.teams["Num_Total_Check"].eq(ctx.teams[config.TEAM_SIZE_COL])
    print(f"\nsum(Num_*) == Team_Size for {int(check.sum())}/{len(ctx.teams)} teams")
    if args.save:
        out = config.OUTPUT / "employees_roles_derived.csv"
        ctx.employees.to_csv(out, index=False)
        ctx.teams.to_csv(config.OUTPUT / "teams_roles_recomputed.csv", index=False)
        print(f"written: {out}")


def cmd_swap(args) -> None:
    """Manual override: solve one pool the caller names."""
    ctx = pipeline.load_context()
    _placeholder_banner(ctx)

    teams = args.teams or pipeline.select_pool(ctx, k=args.worst)
    if len(teams) < 2:
        raise SystemExit("need at least two teams")

    result = pipeline.analyse_pool(ctx, teams, budget=args.budget, lam=args.lam)
    _print(f"POOL {teams}", result.summary())

    moves = result.to_frame(ctx.employees, ctx.values, ctx.s)
    _print("RECOMMENDED MOVES", moves if len(moves) else "no swap improves the objective")
    _advisory_footer()

    if args.save and len(moves):
        out = config.OUTPUT / "recommended_swaps.csv"
        moves.to_csv(out, index=False)
        print(f"\nwritten: {out}")


def cmd_sweep(args) -> None:
    ctx = pipeline.load_context()
    _placeholder_banner(ctx)

    pairs = pipeline.low_team_pairs(ctx)
    print(f"\nevaluable low-team pairs: {len(pairs):,}   running: {args.limit}")
    results = evaluate.sweep(ctx, pairs, limit=args.limit, budget=args.budget, lam=args.lam)
    _print("SWEEP SUMMARY", evaluate.summarise(results))

    if args.ablate:
        _print("ABLATION - balance weight", evaluate.ablate_lambda(ctx, pairs, limit=args.limit))
        _print("ABLATION - churn budget", evaluate.ablate_budget(ctx, pairs, limit=args.limit))
        _print("ABLATION - protective sentiment floor",
               evaluate.ablate_floor(ctx, pairs, limit=args.limit))

    if args.save:
        out = config.OUTPUT / "sweep_results.csv"
        results.to_csv(out, index=False)
        print(f"\nwritten: {out}")


def cmd_sentiment(args) -> None:
    """Aggregate the scored review corpus onto employees and write the contract file."""
    try:
        scored = pathlib.Path(args.scored) if args.scored else None
        frame, meta = sentiment_build.build(path=scored)
    except sentiment_build.SentimentBuildError as exc:
        print(f"\nCANNOT BUILD SENTIMENT\n{'-' * 78}\n{exc}")
        raise SystemExit(1)

    _print("SENTIMENT BUILD", meta)

    audit = sentiment_build.audit_leakage(path=scored)
    _print("CONTRACT INVARIANT 1 - rating leakage (BUG-4)", audit)
    if audit.get("leaking"):
        print(
            "\nREFUSING TO WRITE: s_crisp restates Overall_rating, so every employee "
            "value becomes a second copy of a number the engine already holds and the "
            "objective collapses to zero-sum (PROJECT.md 0.5). Retrain with the rating "
            "removed from the feature path."
        )
        raise SystemExit(2)

    employee_ids = pd.read_csv(config.EMPLOYEES_CSV)[config.EMP_ID]
    problems = sentiment_io.validate(frame, employee_ids)
    _print("CONTRACT VALIDATION", problems or "clean - no violations")
    if problems:
        raise SystemExit(3)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    out = sentiment_build.save(frame)
    print(f"\nwritten: {out}")
    print("The placeholder is now retired. Re-run:  python run.py doctor")


def cmd_doctor(_args) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = pipeline.load_context()

    low = pipeline.discover_low_teams(ctx)
    pools = pipeline.form_pools(ctx, list(low[config.TEAM_ID]))
    checks = {
        "employees_loaded": len(ctx.employees),
        "teams_loaded": len(ctx.teams),
        "employee_id_unique": bool(ctx.employees[config.EMP_ID].is_unique),
        "role_pct_from_title": ctx.role_report["pct_from_title"],
        "role_pct_from_default": ctx.role_report["pct_from_default"],
        "chi2_p_title_vs_derived": ctx.role_report["chi2_p_title_vs_derived"],
        "num_star_sums_to_team_size": int(
            ctx.teams["Num_Total_Check"].eq(ctx.teams[config.TEAM_SIZE_COL]).sum()
        ),
        "sentiment_is_placeholder": ctx.is_placeholder,
        "sentiment_problems": ctx.sentiment_meta.get("problems", []),
        "employees_held_by_floor": int(ctx.protected.sum()),
        "low_teams_discovered": int(len(low)),
        "low_teams_with_no_movable_member": int((low["movable"] == 0).sum()),
        "feasible_pools": len(pools),
    }

    # Smoke-test the pool with the most to gain, not an arbitrary one: a pool where no
    # legal swap helps is a valid answer and would make the check vacuous.
    head = pipeline.pool_headroom(ctx, pools)
    best = head.loc[head["headroom"].idxmax(), "pool"]
    result = pipeline.analyse_pool(ctx, best)
    checks |= {
        "smoke_pool": list(best),
        "smoke_method": result.method,
        "smoke_gain": round(result.gain, 6),
        "smoke_moves": len(result.moves),
        "smoke_pct_of_achievable": round(result.capture_pct, 2),
    }
    _print("DOCTOR", checks)
    _placeholder_banner(ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("recommend", help="discover low teams and recommend reassignments")
    p.add_argument("--pool-size", type=int, default=config.POOL_SIZE,
                   help="teams per pool; 2 is solved to proven optimality, 3+ uses local search")
    p.add_argument("--top", type=int, default=config.TOP_RECOMMENDATIONS)
    p.add_argument("--category", default=None, help="team category treated as under-performing")
    p.add_argument("--budget", type=int, default=config.CHURN_BUDGET)
    p.add_argument("--lam", type=float, default=config.LAMBDA_BALANCE)
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("discover", help="show the discovered teams and the pools formed")
    p.add_argument("--pool-size", type=int, default=config.POOL_SIZE)
    p.add_argument("--category", default=None)
    p.add_argument("--head", type=int, default=15)
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("roles", help="audit the role repair")
    p.add_argument("--tokens", action="store_true", help="show the token-lift evidence table")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_roles)

    p = sub.add_parser("swap", help="manual override: solve one named pool")
    p.add_argument("--teams", nargs="+", help="explicit Team_IDs")
    p.add_argument("--worst", type=int, default=2, help="use the N weakest low teams")
    p.add_argument("--budget", type=int, default=config.CHURN_BUDGET)
    p.add_argument("--lam", type=float, default=config.LAMBDA_BALANCE)
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_swap)

    p = sub.add_parser("sweep", help="baselines and ablations over low-team pairs")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--budget", type=int, default=config.CHURN_BUDGET)
    p.add_argument("--lam", type=float, default=config.LAMBDA_BALANCE)
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("sentiment", help="build sentiment_fuzzy.csv from scored reviews")
    p.add_argument("--scored", default=None,
                   help="path to the delivered sentiment dataset (default: search dataset/)")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    p.set_defaults(func=cmd_sentiment)

    p = sub.add_parser("doctor", help="end-to-end health check")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
