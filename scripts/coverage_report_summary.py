#!/usr/bin/env python3
"""Summarise coverage gaps from a coverage.py JSON report."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = REPO_ROOT / "docs" / "reports" / "coverage.json"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def print_table(rows: list[dict]) -> None:
    col_file = max((len(r["file"]) for r in rows), default=60)
    col_file = min(col_file, 100)
    header = f"{'File':<{col_file}}  {'Stmts':>5}  {'Miss':>4}  {'Cov%':>5}  {'Branch':>6}  {'Part':>4}  {'Total%':>6}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in rows:
        print(
            f"{r['file']:<{col_file}}  {r['statements']:>5}  {r['missing']:>4}  "
            f"{r['stmt_cov']:>4.0f}%  {r['branches']:>6}  {r['partial']:>4}  "
            f"{r['total_cov']:>5.0f}%"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise coverage gaps")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to coverage.json (default: docs/reports/coverage.json)",
    )
    parser.add_argument(
        "--min-missed",
        type=int,
        default=1,
        help="Minimum missed statements to show (default: 1)",
    )
    parser.add_argument(
        "--exclude-version",
        action="store_true",
        default=True,
        help="Exclude _version_.py files (default: true)",
    )
    parser.add_argument("--no-exclude-version", action="store_false", dest="exclude_version")
    parser.add_argument(
        "--sort-by",
        choices=["missed", "file", "coverage"],
        default="missed",
        help="Sort key (default: missed)",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}", file=sys.stderr)
        sys.exit(1)

    report = load_report(args.report)
    files = report["files"]

    rows = []
    for file_path, info in files.items():
        s = info["summary"]
        rows.append(
            {
                "file": file_path,
                "statements": s["num_statements"],
                "missing": s["missing_lines"],
                "stmt_cov": s["percent_statements_covered"],
                "branches": s["num_branches"],
                "partial": s["num_partial_branches"],
                "total_cov": s["percent_covered"],
            }
        )

    if args.exclude_version:
        rows = [r for r in rows if "_version_" not in r["file"] and "_version/" not in r["file"]]

    rows = [r for r in rows if r["missing"] >= args.min_missed]

    if args.sort_by == "missed":
        rows.sort(key=lambda r: (-r["missing"], r["file"]))
    elif args.sort_by == "file":
        rows.sort(key=lambda r: r["file"])
    elif args.sort_by == "coverage":
        rows.sort(key=lambda r: (r["total_cov"], r["file"]))

    total_stmts = sum(r["statements"] for r in rows)
    total_missed = sum(r["missing"] for r in rows)
    total_branches = sum(r["branches"] for r in rows)
    total_partial = sum(r["partial"] for r in rows)

    print(f"\nCoverage gap report — {args.report}")
    print(f"Modules with >= {args.min_missed} missed statement(s): {len(rows)}")
    print(
        f"  Total statements: {total_stmts}, missed: {total_missed}, "
        f"branches: {total_branches}, partial: {total_partial}"
    )
    print()
    print_table(rows)


if __name__ == "__main__":
    main()
