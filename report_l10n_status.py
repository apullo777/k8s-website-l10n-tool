#!/usr/bin/env python3
"""
report_l10n_status.py — Triage-oriented localization status report.

Scans content/<lang>/ and writes a Markdown report file suitable for
pasting into a GitHub issue. All outdated files appear as one-line
summaries by default. Use --detailed for per-file reason bullets.

Usage:
    # Single locale (writes l10n-status-ko.md in the current directory)
    python3 scripts/report_l10n_status.py --lang ko
    python3 scripts/report_l10n_status.py --lang ko --detailed

    # Multiple selected locales
    python3 scripts/report_l10n_status.py --langs ko,zh-cn,ja

    # All locales under content/ except en
    python3 scripts/report_l10n_status.py --all-langs

    # Custom output directory
    python3 scripts/report_l10n_status.py --all-langs --output-dir /tmp/l10n
"""

import argparse
import datetime
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_l10n_drift_structural import compare_file, FileResult




def _classify_signal(signal: str) -> str:
    """Classify a signal string into a category."""
    sl = signal.lower()
    if "missing in localized" in sl or "alignment uncertainty" in sl:
        return "missing_section"
    if "code block" in sl and "content changed" in sl:
        return "code_drift"
    if re.search(r'\b(paragraph|code_block|list_item)\(s\)', sl):
        return "block_mismatch"
    if "paragraph has 1 more line" in sl or "documentation note" in sl:
        return "prose_drift"
    if "trailing qualifier" in sl:
        return "heading_qualifier"
    return "shortcode_or_other"


def _count_signals(result: FileResult) -> Dict[str, int]:
    """Count signals by category for one file."""
    counts: Counter = Counter()
    for diff in result.changes:
        for sig in diff.signals:
            counts[_classify_signal(sig)] += 1
    return dict(counts)


# Priority: high (missing sections / code changes), medium (block or text
# differences), low (shortcode-only).

def _classify(result: FileResult) -> Tuple[str, Dict[str, int]]:
    """Return (priority, counts) for one file."""
    if result.status in ("up_to_date", "no_english_version", "not_translated"):
        return "up_to_date", {}
    if result.status == "outdated_low_severity":
        return "low", {}
    counts = _count_signals(result)
    if counts.get("missing_section", 0) or counts.get("code_drift", 0):
        return "high", counts
    if counts.get("block_mismatch", 0) or counts.get("prose_drift", 0):
        return "medium", counts
    return "low", counts


def _sort_key(s: "_Scored") -> Tuple:
    c = s.counts
    return (-c.get("missing_section", 0), -c.get("code_drift", 0), -sum(c.values()), s.result.localized_path)




def _area(path: str, lang: str) -> str:
    """Return the two-level doc area, e.g. 'docs/tasks/'."""
    m = re.search(rf'content/{re.escape(lang)}/([^/]+(?:/[^/]+)?)', path)
    if m:
        seg = m.group(1)
        if "." in seg.split("/")[-1]:
            seg = "/".join(seg.split("/")[:-1])
        return seg.rstrip("/") + "/" if seg else "(root)/"
    return "(other)/"




@dataclass
class _Scored:
    result: FileResult
    priority: str
    counts: Dict[str, int] = field(default_factory=dict)


def _run(lang: str, repo_root: str) -> Tuple[List["_Scored"], Counter]:
    lang_dir = os.path.join(repo_root, "content", lang)
    if not os.path.isdir(lang_dir):
        print(f"error: language directory not found: {lang_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = []
    for root, _, files in os.walk(lang_dir):
        for fname in sorted(files):
            if fname.endswith('.md'):
                localized = os.path.join(root, fname)
                en = re.sub(r'content/[^/]+/', 'content/en/', localized)
                pairs.append((en, localized))
    pairs = [(en, localized) for en, localized in pairs if os.path.exists(en)]

    total = len(pairs)
    scored: List[_Scored] = []
    file_counts: Counter = Counter()

    for i, (en_path, localized_path) in enumerate(pairs, 1):
        if i % 100 == 0:
            print(f"  [{lang}] scanned {i}/{total}...", file=sys.stderr)
        result = compare_file(en_path, localized_path)
        prio, counts = _classify(result)
        if result.status in ("candidate_outdated", "outdated_low_severity"):
            cats = {_classify_signal(sig) for diff in result.changes for sig in diff.signals}
            for cat in ("missing_section", "code_drift", "block_mismatch"):
                if cat in cats:
                    file_counts[cat] += 1
            if result.status == "outdated_low_severity":
                file_counts["low_severity_only"] += 1
        scored.append(_Scored(result=result, priority=prio, counts=counts))

    _PRIO_ORDER = {"high": 0, "medium": 1, "low": 2, "up_to_date": 3}
    scored.sort(key=lambda x: (_PRIO_ORDER.get(x.priority, 4), _sort_key(x)))
    return scored, file_counts


def _area_counts(scored: List["_Scored"], lang: str) -> Counter:
    c: Counter = Counter()
    for s in scored:
        if s.priority != "up_to_date":
            c[_area(s.result.localized_path, lang)] += 1
    return c



def _summary_line(counts: Dict[str, int]) -> str:
    """One-line summary, e.g. '3 missing sections, 1 code drift'."""
    parts = []
    if counts.get("missing_section"):
        n = counts["missing_section"]
        parts.append(f"{n} missing section{'s' if n != 1 else ''}")
    if counts.get("code_drift"):
        n = counts["code_drift"]
        parts.append(f"{n} code/example drift")
    if counts.get("block_mismatch"):
        n = counts["block_mismatch"]
        parts.append(f"{n} block mismatch{'es' if n != 1 else ''}")
    if counts.get("prose_drift"):
        n = counts["prose_drift"]
        parts.append(f"{n} prose drift")
    if counts.get("heading_qualifier"):
        parts.append(f"{counts['heading_qualifier']} heading qualifier")
    if not parts:
        return "low-severity only (shortcode differences)"
    return ", ".join(parts)


def _reasons(result: FileResult, limit: int) -> List[str]:
    """Return signal strings sorted by category, up to `limit`."""
    _ORDER = ["missing_section", "code_drift", "block_mismatch", "prose_drift",
              "heading_qualifier", "shortcode_or_other"]
    by_cat: Dict[str, List[str]] = {c: [] for c in _ORDER}
    for diff in result.changes:
        for sig in diff.signals:
            by_cat[_classify_signal(sig)].append(sig)
    out = []
    for cat in _ORDER:
        out.extend(by_cat[cat])
    return out[:limit]


def _rel(path: str, repo_root: str) -> str:
    """Strip repo root prefix for display."""
    try:
        return os.path.relpath(path, repo_root)
    except ValueError:
        return path


# Default: one compact line per outdated file.
# --detailed: adds per-file reason bullets for high and medium priority.

def _fmt_locale_md(
    lang: str,
    scored: List["_Scored"],
    file_counts: Counter,
    repo_root: str,
    detailed: bool,
    date: str,
) -> str:
    high   = [s for s in scored if s.priority == "high"]
    medium = [s for s in scored if s.priority == "medium"]
    low    = [s for s in scored if s.priority == "low"]
    utd    = [s for s in scored if s.priority == "up_to_date"]
    ac     = _area_counts(scored, lang)

    lines: List[str] = [
        f"## Localization status: `{lang}`",
        "",
        f"Generated: {date}",
        "",
        "| | Count |",
        "|---|---|",
        f"| Scanned pairs | {len(scored)} |",
        f"| High priority | {len(high)} |",
        f"| Medium priority | {len(medium)} |",
        f"| Low priority | {len(low)} |",
        f"| Up to date | {len(utd)} |",
        "",
    ]

    # Drift categories (file-level counts)
    _cats = [
        ("missing_section",   "Missing sections"),
        ("code_drift",        "Code/example drift"),
        ("block_mismatch",    "Paragraph/block mismatch"),
        ("low_severity_only", "Low-severity only (shortcode)"),
    ]
    cat_lines = [(label, file_counts[key]) for key, label in _cats if file_counts.get(key)]
    if cat_lines:
        lines += ["**Top drift categories** (files affected):", ""]
        for label, n in cat_lines:
            lines.append(f"- {label}: **{n}**")
        lines.append("")

    # Area hotspots
    if ac:
        lines += ["**Top affected areas:**", ""]
        for area, count in ac.most_common(8):
            lines.append(f"- `{area}`: {count} files")
        lines.append("")

    # reasons_limit > 0 shows detailed format; 0 shows compact one-liners.
    def _section(title: str, items: List[_Scored], reasons_limit: int) -> None:
        lines.extend([f"### {title} ({len(items)})", ""])
        if not items:
            lines.extend(["_None_", ""])
            return
        for s in items:
            path    = _rel(s.result.localized_path, repo_root)
            summary = _summary_line(s.counts)
            if reasons_limit > 0:
                lines.append(f"**`{path}`** — {summary}")
                for r in _reasons(s.result, reasons_limit):
                    lines.append(f"- {r}")
                lines.append("")
            else:
                lines.append(f"- `{path}` — {summary}")
        if reasons_limit == 0:
            lines.append("")

    _section("High priority",   high,   reasons_limit=5 if detailed else 0)
    _section("Medium priority", medium, reasons_limit=3 if detailed else 0)
    _section("Low priority",    low,    reasons_limit=0)

    return "\n".join(lines)



_CAT_LABELS = {
    "missing_section":   "missing sections",
    "code_drift":        "code/example drift",
    "block_mismatch":    "block mismatch",
    "low_severity_only": "low-severity only",
}
_CAT_ORDER = ["missing_section", "code_drift", "block_mismatch", "low_severity_only"]


def _fmt_index_md(results: List[Tuple[str, List["_Scored"], Counter]], date: str) -> str:
    lines: List[str] = [
        "## Localization status index",
        "",
        f"Generated: {date}",
        "",
        "| Locale | Report | Pairs | High | Medium | Low | Up to date | Top drift category |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lang, scored, file_counts in results:
        high   = sum(1 for s in scored if s.priority == "high")
        medium = sum(1 for s in scored if s.priority == "medium")
        low    = sum(1 for s in scored if s.priority == "low")
        utd    = sum(1 for s in scored if s.priority == "up_to_date")
        fname  = f"l10n-status-{lang}.md"
        top_cat = next(
            (f"{_CAT_LABELS[c]} ({file_counts[c]})" for c in _CAT_ORDER if file_counts.get(c)),
            "—",
        )
        lines.append(
            f"| `{lang}` | [{fname}]({fname}) | {len(scored)} |"
            f" {high} | {medium} | {low} | {utd} | {top_cat} |"
        )
    lines.append("")
    return "\n".join(lines)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a Markdown localization status report for one or more locales.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang",      metavar="CODE",  help="Single locale (e.g. ko)")
    group.add_argument("--langs",     metavar="CODES", help="Comma-separated locales (e.g. ko,zh-cn,ja)")
    group.add_argument("--all-langs", action="store_true", help="All locales under content/ except en")
    parser.add_argument("--output-dir", default=".", metavar="DIR",
                        help="Directory for report files (default: current directory)")
    parser.add_argument("--detailed", action="store_true",
                        help="Add per-file reason bullets for high and medium priority files")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    date      = datetime.date.today().isoformat()

    if args.lang:
        langs = [args.lang]
    elif args.langs:
        langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    else:
        content_dir = os.path.join(repo_root, "content")
        langs = sorted(
            d for d in os.listdir(content_dir)
            if os.path.isdir(os.path.join(content_dir, d)) and d != "en"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for lang in langs:
        print(f"Scanning content/{lang}/ ...", file=sys.stderr)
        scored, file_counts = _run(lang, repo_root)
        all_results.append((lang, scored, file_counts))

        out_path = os.path.join(args.output_dir, f"l10n-status-{lang}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(_fmt_locale_md(lang, scored, file_counts, repo_root, args.detailed, date))

        high   = sum(1 for s in scored if s.priority == "high")
        medium = sum(1 for s in scored if s.priority == "medium")
        low    = sum(1 for s in scored if s.priority == "low")
        utd    = sum(1 for s in scored if s.priority == "up_to_date")
        print(f"Wrote {out_path}  ({high} high, {medium} medium, {low} low, {utd} up to date)")

    if len(langs) > 1:
        index_path = os.path.join(args.output_dir, "l10n-status-index.md")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(_fmt_index_md(all_results, date))
        print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
