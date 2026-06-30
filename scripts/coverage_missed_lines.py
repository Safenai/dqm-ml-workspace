#!/usr/bin/env python3
"""Extract missed lines from a coverage.py JSON report, showing source context."""

import argparse
import fnmatch
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = REPO_ROOT / "docs" / "reports" / "coverage.json"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def read_source_line(src_path: Path, line_num: int) -> str:
    try:
        with src_path.open() as f:
            for i, line in enumerate(f, 1):
                if i == line_num:
                    return line.rstrip("\n").rstrip()
    except FileNotFoundError:
        pass
    return ""


def format_block(
    file_path: str,
    missing_lines: list[int],
    src_path: Path,
    context: int,
) -> list[str]:
    if not missing_lines:
        return []

    missing_set = set(missing_lines)
    all_show_lines: set[int] = set()
    for ln in missing_lines:
        for offset in range(-context, context + 1):
            all_show_lines.add(ln + offset)

    all_show_lines = {ln for ln in all_show_lines if ln >= 1}

    output: list[str] = []
    output.append(f"{'=' * 72}")
    output.append(f"File: {file_path}")
    output.append(f"{'=' * 72}")

    prev_was_gap = False
    for ln in sorted(all_show_lines):
        is_missing = ln in missing_set
        if not is_missing:
            neighbors = any(neighbor in missing_set for neighbor in range(ln - context, ln + context + 1))
            if not neighbors:
                if not prev_was_gap:
                    output.append("  ...")
                    prev_was_gap = True
                continue

        prev_was_gap = False
        marker = ">>" if is_missing else "  "
        code = read_source_line(src_path, ln)
        output.append(f"  {marker} L{ln:>5}  {code}")

    output.append("")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract missed lines from coverage.py JSON report")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to coverage.json (default: docs/reports/coverage.json)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="File glob patterns to filter (e.g. *domain_gap* *job*)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Lines of context around each missed line (default: 2)",
    )
    parser.add_argument(
        "--no-source",
        action="store_true",
        help="Only list line numbers, omit code context",
    )
    parser.add_argument(
        "--exclude-version",
        action="store_true",
        default=True,
        help="Exclude _version_.py files (default: true)",
    )
    parser.add_argument("--no-exclude-version", action="store_false", dest="exclude_version")
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}", file=sys.stderr)
        sys.exit(1)

    report = load_report(args.report)
    files = report["files"]

    filtered_files = dict(files)
    if args.exclude_version:
        filtered_files = {k: v for k, v in filtered_files.items() if "_version_" not in k}

    if args.files:
        matched = {}
        for pat in args.files:
            for key in filtered_files:
                if fnmatch.fnmatch(key, pat):
                    matched[key] = filtered_files[key]
        filtered_files = matched

    if not filtered_files:
        print("No files matched the given filters.", file=sys.stderr)
        sys.exit(0)

    total_missed = 0
    any_output = False

    for file_path in sorted(filtered_files):
        info = filtered_files[file_path]
        missing_lines = info["missing_lines"]

        if not missing_lines:
            continue

        total_missed += len(missing_lines)
        src_path = REPO_ROOT / file_path

        if args.no_source:
            any_output = True
            ranges = []
            start = missing_lines[0]
            end = missing_lines[0]
            for n in missing_lines[1:]:
                if n == end + 1:
                    end = n
                else:
                    ranges.append(f"{start}-{end}" if start != end else str(start))
                    start = n
                    end = n
            ranges.append(f"{start}-{end}" if start != end else str(start))
            print(f"{file_path} ({len(missing_lines)} missed)")
            print(f"  Lines: {', '.join(ranges)}")
            print()
        else:
            blocks = format_block(
                file_path,
                missing_lines,
                src_path,
                args.context,
            )
            if blocks:
                any_output = True
                sys.stdout.write("\n".join(blocks))

    if not any_output:
        print("No missed lines found in the filtered files.")
    else:
        print(f"Total missed statements: {total_missed}")


if __name__ == "__main__":
    main()
