#!/usr/bin/env python3
"""Minimal hybrid triage for l10n outdatedness in kubernetes/website.

Classifies each (EN, locale) markdown pair into one of four buckets:
    both_flagged         git stale AND a content rule fired
    content_only_rescue  content rule fired, git clean  (ghost-sync catch)
    git_only_suppressed  git stale, no content rule     (likely bulk EN metadata PR)
    clean                neither

Content rules (any one flags the pair):
    E  loc empty, EN has content
    H  h2+h3 heading delta >= 2
    V  localized missing >= 2 newer v1.XX version strings
    C  code block delta >= 2
    R  line ratio < 0.70, EN >= 15 lines, at least 1 corroborating signal
    S  line ratio < 0.50, EN >= 40 lines, word ratio < 0.80, loc non-empty

Reviewer severity (flagged pairs only):
    HIGH    >=2 rules fired AND one is 'strong' AND not older-structure-preserved
    LOW     only R fired
    MEDIUM  everything else flagged

'Strong signal' := E or S fires, or h2+h3 >= 3, or code_diff >= 3,
                   or missing_versions >= 3, or line_ratio < 0.50.

'Older structure preserved' := line_ratio >= 1.0 AND fired rules subset of {H, V}.
Loc is as long or longer than EN but newer headings/versions weren't added;
mirrors the mature detector's 'structure mismatch' demotion.

Default output surfaces flagged files by severity (HIGH, MEDIUM).
Switches:
    --show-low        list LOW entries
    --show-git-only   list git_only_suppressed entries
    --full / --debug  full diagnostic: everything + per-file feature vectors
    --no-git          skip git staleness check (content-only classification)

Usage:
    python3 scripts/triage-hybrid-minimal.py --lang ko
    python3 scripts/triage-hybrid-minimal.py --langs ko,ja,pt-br
    python3 scripts/triage-hybrid-minimal.py --all-langs --output-dir /tmp/out
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple


# --- Parsing -----------------------------------------------------------------

_STRIP_FM = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_STRIP_CMNT = re.compile(r"<!--.*?-->", re.DOTALL)
_STRIP_CODE = re.compile(r"```.*?```", re.DOTALL)
_STRIP_INLINE = re.compile(r"`[^`\n]+`")
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])v1\.(\d{2,3})(?!\d)")
_WORD_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)


@dataclass(frozen=True)
class ParsedFile:
    lines: int
    h2: int
    h3: int
    code_blocks: int
    versions: FrozenSet[int]  # minor version numbers
    body_words: int


def parse_markdown(path: str) -> ParsedFile:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    # Strip frontmatter before comments: a leading <!-- --> could otherwise
    # let a later `---` look like frontmatter.
    body = _STRIP_CMNT.sub("", _STRIP_FM.sub("", text, count=1))
    no_cmnt = _STRIP_CMNT.sub("", text)

    lines = sum(1 for l in body.splitlines() if l.strip())
    h2, h3, code_blocks = _scan_structure(no_cmnt)
    versions = frozenset(int(m) for m in _VERSION_RE.findall(text))
    return ParsedFile(lines, h2, h3, code_blocks, versions, _count_words(body))


def _scan_structure(text: str) -> Tuple[int, int, int]:
    """Count h2, h3, and column-0 fenced code blocks (ignoring fenced content)."""
    h2 = h3 = fences = 0
    in_code = False
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        leading = len(line) - len(stripped)
        if leading < 4 and stripped.startswith("```"):
            if leading == 0:
                fences += 1
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.startswith("### "):
            h3 += 1
        elif line.startswith("## "):
            h2 += 1
    return h2, h3, fences // 2


def _count_words(body: str) -> int:
    """Body word count, skipping fenced and 4-space indented code blocks."""
    no_fenced = _STRIP_CODE.sub("", body)
    kept = []
    for block in re.split(r"\n{2,}", no_fenced):
        non_blank = [l for l in block.splitlines() if l.strip()]
        if non_blank and all(l[:4] == "    " for l in non_blank):
            continue
        kept.append(block)
    return len(_WORD_RE.findall(_STRIP_INLINE.sub("", "\n\n".join(kept))))


# --- Content rules -----------------------------------------------------------


@dataclass
class Verdict:
    fired: List[str]
    reasons: List[str]
    en_lines: int
    loc_lines: int
    line_ratio: float
    word_ratio: float
    h2_diff: int
    h3_diff: int
    code_diff: int
    missing_versions: int

    @property
    def flagged(self) -> bool:
        return bool(self.fired)


def evaluate(en: ParsedFile, loc: ParsedFile) -> Verdict:
    ratio = min(loc.lines / en.lines, 2.0) if en.lines else 1.0
    wratio = loc.body_words / en.body_words if en.body_words else 1.0
    h2d = max(0, en.h2 - loc.h2)
    h3d = max(0, en.h3 - loc.h3)
    cd = max(0, en.code_blocks - loc.code_blocks)
    mv = _missing_versions(en.versions, loc.versions)
    heading = h2d + h3d
    corrob = heading + cd + mv

    fired: List[str] = []
    reasons: List[str] = []

    if loc.lines == 0 and en.lines > 0:
        fired.append("E")
        reasons.append("localized file is an empty stub while EN has content")

    if heading >= 2:
        fired.append("H")
        pieces = []
        if h2d:
            pieces.append(f"{h2d} H2")
        if h3d:
            pieces.append(f"{h3d} H3")
        reasons.append(f"EN has more headings than localized ({', '.join(pieces)})")

    if mv >= 2:
        fired.append("V")
        reasons.append(f"localized missing {mv} newer v1.XX version(s)")

    if cd >= 2:
        fired.append("C")
        reasons.append(f"EN has {cd} more code block(s) than localized")

    if ratio < 0.70 and en.lines >= 15 and corrob >= 1:
        fired.append("R")
        reasons.append(f"line ratio {ratio:.2f} (< 0.7) with structural corroboration")

    if ratio < 0.50 and en.lines >= 40 and wratio < 0.80 and loc.lines > 0:
        fired.append("S")
        reasons.append(
            f"line ratio {ratio:.2f} and word ratio {wratio:.2f} "
            "both indicate real content drop"
        )

    return Verdict(fired, reasons, en.lines, loc.lines, ratio, wratio,
                   h2d, h3d, cd, mv)


def _missing_versions(en_v: FrozenSet[int], loc_v: FrozenSet[int]) -> int:
    if not loc_v:
        return len(en_v)
    cap = max(loc_v)
    return sum(1 for v in en_v if v > cap)


def severity(v: Verdict) -> str:
    if not v.fired:
        return "—"
    strong = (
        "E" in v.fired or "S" in v.fired
        or (v.h2_diff + v.h3_diff) >= 3
        or v.code_diff >= 3
        or v.missing_versions >= 3
        or v.line_ratio < 0.50
    )
    older_structure_preserved = (
        v.line_ratio >= 1.0 and set(v.fired) <= {"H", "V"}
    )
    if len(v.fired) >= 2 and strong and not older_structure_preserved:
        return "HIGH"
    if v.fired == ["R"]:
        return "LOW"
    return "MEDIUM"


# --- Git (lsync-style staleness, one subprocess per run) ---------------------

_GIT_MARKER = "__HYBRID_COMMIT__"


def build_git_ranks(repo_root: str) -> Dict[str, int]:
    """Map path -> rank of most recent commit touching it (smaller = newer).

    One `git log` replaces thousands of per-file calls; stale becomes
    `rank[en] < rank[loc]`.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "log", "--name-only",
             f"--pretty=format:{_GIT_MARKER}%H", "HEAD", "--", "content/"],
            stderr=subprocess.DEVNULL, timeout=600,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return {}

    rank: Dict[str, int] = {}
    i = -1
    for line in out.splitlines():
        if not line:
            continue
        if line.startswith(_GIT_MARKER):
            i += 1
        elif line not in rank:
            rank[line] = i
    return rank


def git_stale(en_rel: str, loc_rel: str, rank: Dict[str, int]) -> Optional[bool]:
    """True iff EN was touched more recently than loc. None if loc has no history."""
    if not rank or loc_rel not in rank:
        return None
    en_r = rank.get(en_rel)
    if en_r is None:
        return False  # EN unseen under HEAD -> can't be stale-against
    return en_r < rank[loc_rel]


# --- Pipeline ----------------------------------------------------------------


@dataclass
class Row:
    loc_path: str
    en_path: str
    stale: Optional[bool]
    verdict: Verdict

    @property
    def bucket(self) -> str:
        stale = bool(self.stale)
        flagged = self.verdict.flagged
        if stale and flagged:
            return "both_flagged"
        if flagged:
            return "content_only_rescue"
        if stale:
            return "git_only_suppressed"
        return "clean"

    @property
    def severity(self) -> str:
        return severity(self.verdict)


def classify(en_path: str, loc_path: str, repo_root: str,
             rank: Optional[Dict[str, int]]) -> Row:
    verdict = evaluate(parse_markdown(en_path), parse_markdown(loc_path))
    if rank is None:
        stale: Optional[bool] = None
    else:
        stale = git_stale(
            os.path.relpath(en_path, repo_root),
            os.path.relpath(loc_path, repo_root),
            rank,
        )
    return Row(loc_path, en_path, stale, verdict)


def scan_locale(lang: str, repo_root: str,
                rank: Optional[Dict[str, int]]) -> List[Row]:
    lang_dir = os.path.join(repo_root, "content", lang)
    if not os.path.isdir(lang_dir):
        print(f"error: {lang_dir} not found", file=sys.stderr)
        sys.exit(1)

    pairs: List[Tuple[str, str]] = []
    for root, _, files in os.walk(lang_dir):
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            loc_path = os.path.join(root, name)
            en_path = re.sub(r"content/[^/]+/", "content/en/", loc_path)
            if os.path.exists(en_path):
                pairs.append((en_path, loc_path))

    total = len(pairs)
    git_state = "on" if rank is not None else "off"
    print(f"  [{lang}] classifying {total} pairs (git={git_state}) ...",
          file=sys.stderr)

    rows: List[Row] = []
    for i, (en_path, loc_path) in enumerate(pairs, 1):
        if i % 500 == 0:
            print(f"  [{lang}] {i}/{total}", file=sys.stderr)
        rows.append(classify(en_path, loc_path, repo_root, rank))
    return rows


# --- Reporting ---------------------------------------------------------------

_BUCKET_TAG = {
    "both_flagged": "[both]",
    "content_only_rescue": "[content-only]",
}


def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def _feature_line(r: Row) -> str:
    v = r.verdict
    return (f"en={v.en_lines} loc={v.loc_lines} "
            f"ratio={v.line_ratio:.2f} wr={v.word_ratio:.2f} "
            f"h2d={v.h2_diff} h3d={v.h3_diff} cd={v.code_diff} "
            f"v={v.missing_versions}")


def format_report_md(lang: str, rows: List[Row], repo_root: str, date: str, *,
                  show_low: bool, show_git_only: bool, full: bool) -> str:
    by_bucket = Counter(r.bucket for r in rows)
    by_sev: Dict[str, List[Row]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for r in rows:
        if r.verdict.flagged:
            by_sev[r.severity].append(r)

    out: List[str] = [
        f"## Minimal hybrid triage: `{lang}`",
        "",
        f"Generated: {date}  ",
        f"Mode: {'debug (full diagnostic)' if full else 'reviewer triage'}  ",
        "Script: `scripts/triage-hybrid-minimal.py`",
        "",
        "Disagreement-driven minimal hybrid for reviewer triage. Preserves "
        "mature-HIGH coverage; intentionally under-covers LOW and some MEDIUM "
        "compared with `scripts/triage-by-content-signals.py`.",
        "",
        "### Summary",
        "",
        "| severity | count |",
        "|---|---:|",
        f"| `HIGH` | {len(by_sev['HIGH'])} |",
        f"| `MEDIUM` | {len(by_sev['MEDIUM'])} |",
        f"| `LOW` | {len(by_sev['LOW'])} |",
        "",
        "| bucket | count |",
        "|---|---:|",
        f"| `both_flagged` (git stale + content flag) | {by_bucket['both_flagged']} |",
        f"| `content_only_rescue` (ghost-sync catch) | {by_bucket['content_only_rescue']} |",
        f"| `git_only_suppressed` (likely bulk-metadata EN PR) | {by_bucket['git_only_suppressed']} |",
        f"| `clean` | {by_bucket['clean']} |",
        "",
    ]

    for sev in ("HIGH", "MEDIUM", "LOW"):
        items = by_sev[sev]
        out += [f"### {sev} severity ({len(items)})", ""]
        if not items:
            out += ["_None_", ""]
            continue
        if sev == "LOW" and not (show_low or full):
            out += [
                "_Hidden in reviewer mode. Rerun with `--show-low` to list. "
                "LOW entries fire only rule (R) — moderate ratio with soft "
                "corroboration — and are reviewer judgment calls._",
                "",
            ]
            continue
        items.sort(key=lambda r: (r.bucket != "both_flagged", r.loc_path))
        for r in items:
            tag = _BUCKET_TAG.get(r.bucket, "")
            fired = ",".join(r.verdict.fired) or "—"
            out.append(f"**`{_rel(r.loc_path, repo_root)}`** {tag} [rules: {fired}]")
            if full:
                out.append(f"- features: {_feature_line(r)}")
            for reason in r.verdict.reasons:
                out.append(f"- {reason}")
            out.append("")

    git_only = [r for r in rows if r.bucket == "git_only_suppressed"]
    out += [
        f"### Git-only suppressed ({len(git_only)})",
        "",
        "_Git says the EN counterpart has moved, but no content rule fires. "
        "These are dominated by bulk-metadata EN PRs (glossary date-field "
        "removal, shortcode migration, page weight changes, exec-bit cleanups, "
        "link-path updates) and are not high-priority for reviewers._",
        "",
    ]
    if full or show_git_only:
        if git_only:
            for r in sorted(git_only, key=lambda x: x.loc_path):
                out.append(f"- `{_rel(r.loc_path, repo_root)}`")
            out.append("")
        else:
            out += ["_None_", ""]
    elif git_only:
        out += [
            "_Hidden in reviewer mode. Rerun with `--show-git-only` to list, "
            "or `--full` for the complete diagnostic surface._",
            "",
        ]

    return "\n".join(out)


def format_index_md(all_results: List[Tuple[str, List[Row]]], date: str) -> str:
    out = [
        "## Minimal hybrid triage -- index",
        "",
        f"Generated: {date}",
        "",
        "Severity (HIGH / MEDIUM / LOW) counts flagged files only "
        "(`both_flagged` + `content_only_rescue`). Bucket columns count all pairs.",
        "",
        "| locale | total | HIGH | MEDIUM | LOW | both | content_only | git_only | clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lang, rows in all_results:
        b = Counter(r.bucket for r in rows)
        s = Counter(r.severity for r in rows if r.verdict.flagged)
        out.append(
            f"| `{lang}` | {len(rows)} | "
            f"{s['HIGH']} | {s['MEDIUM']} | {s['LOW']} | "
            f"{b['both_flagged']} | {b['content_only_rescue']} | "
            f"{b['git_only_suppressed']} | {b['clean']} |"
        )
    out.append("")
    return "\n".join(out)


# --- CLI ---------------------------------------------------------------------


def _detect_repo_root() -> Optional[str]:
    d = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(d, "content", "en")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def main() -> None:
    p = argparse.ArgumentParser(
        description="Minimal hybrid l10n-outdatedness reviewer triage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--lang", metavar="CODE", help="single locale (e.g. ko)")
    g.add_argument("--langs", metavar="CODES",
                   help="comma-separated locales (e.g. ko,ja,pt-br)")
    g.add_argument("--all-langs", action="store_true",
                   help="all locales under content/ except en")
    p.add_argument("--repo-root", default=None, metavar="DIR")
    p.add_argument("--output-dir", default=".", metavar="DIR")
    p.add_argument("--no-git", action="store_true",
                   help="skip git staleness check")
    p.add_argument("--show-low", action="store_true",
                   help="list LOW severity entries")
    p.add_argument("--show-git-only", action="store_true",
                   help="list git_only_suppressed files")
    p.add_argument("--full", "--debug", dest="full", action="store_true",
                   help="full diagnostic: feature vectors + LOW + git-only listing")
    args = p.parse_args()

    repo_root = (os.path.abspath(args.repo_root) if args.repo_root
                 else _detect_repo_root())
    if repo_root is None:
        print("error: could not auto-detect repo root; pass --repo-root",
              file=sys.stderr)
        sys.exit(1)

    if args.lang:
        langs = [args.lang]
    elif args.langs:
        langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    else:
        content_dir = os.path.join(repo_root, "content")
        langs = sorted(
            d for d in os.listdir(content_dir)
            if os.path.isdir(os.path.join(content_dir, d)) and d != "en"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    date = datetime.date.today().isoformat()

    rank: Optional[Dict[str, int]] = None
    if not args.no_git:
        print("Preloading git history for content/ (one subprocess) ...",
              file=sys.stderr)
        t0 = time.time()
        rank = build_git_ranks(repo_root)
        print(f"  git preload: {len(rank)} paths in {time.time() - t0:.1f}s",
              file=sys.stderr)
        if not rank:
            print("  warning: git preload returned no entries; "
                  "falling back to content-only classification",
                  file=sys.stderr)
            rank = None

    all_results: List[Tuple[str, List[Row]]] = []
    for lang in langs:
        print(f"Scanning content/{lang}/ ...", file=sys.stderr)
        rows = scan_locale(lang, repo_root, rank)
        all_results.append((lang, rows))

        out_path = os.path.join(args.output_dir, f"l10n-hybrid-{lang}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(format_report_md(lang, rows, repo_root, date,
                                  show_low=args.show_low,
                                  show_git_only=args.show_git_only,
                                  full=args.full))

        b = Counter(r.bucket for r in rows)
        s = Counter(r.severity for r in rows if r.verdict.flagged)
        print(
            f"Wrote {out_path}  "
            f"(HIGH={s['HIGH']} MED={s['MEDIUM']} LOW={s['LOW']}  "
            f"both={b['both_flagged']} content_only={b['content_only_rescue']} "
            f"git_only={b['git_only_suppressed']} clean={b['clean']})",
            file=sys.stderr,
        )

    if len(langs) > 1:
        idx_path = os.path.join(args.output_dir, "l10n-hybrid-index.md")
        with open(idx_path, "w", encoding="utf-8") as f:
            f.write(format_index_md(all_results, date))
        print(f"Wrote {idx_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
