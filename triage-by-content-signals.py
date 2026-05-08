#!/usr/bin/env python3
"""File-level localization outdatedness detector.

Indicators: line ratio, missing H2 / H3, missing code blocks, missing
anchors, untranslated CJK paragraphs, missing newer k8s version strings.
Priority levels: high (>=50) | medium (>=20) | low (>=5) | up_to_date (<5).

Usage:
    python3 scripts/triage-by-content-signals.py --lang ko [--detailed]
    python3 scripts/triage-by-content-signals.py --langs ko,zh-cn,ja
    python3 scripts/triage-by-content-signals.py --all-langs --output-dir /tmp/l10n
    python3 triage-by-content-signals.py --lang ko --repo-root /path/to/website
"""

import argparse
import datetime
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import FrozenSet, List, Tuple

# =============================================================================
# Constants
# =============================================================================

_HIGH_THRESHOLD = 50
_MEDIUM_THRESHOLD = 20
_LOW_THRESHOLD = 5

# Centralized priority labels. Used as the canonical string values in
# FileScore.priority and throughout the report output.
_PRIORITY_HIGH = "high"
_PRIORITY_MEDIUM = "medium"
_PRIORITY_LOW = "low"
_PRIORITY_UP_TO_DATE = "up_to_date"

# Ratio suppression thresholds. All bypassed when l10n.visible_lines == 0
# (empty stub stays flagged).
_RATIO_MIN_SOURCE_LINES = 15             # below: ratio always suppressed
_RATIO_CORROBORATION_THRESHOLD = 40      # 15-39: ratio needs supporting evidence
_RATIO_CORROBORATION_THRESHOLD_CJK = 56  # 40-55: same, CJK-only
_COMPACTNESS_GATE_MIN_SOURCE_LINES = 55

_CJK_LANGS: FrozenSet[str] = frozenset({"ko", "ja", "zh-cn", "zh-tw"})

# Latin scripts whose word counts behave like EN's. Excludes ru: real
# mismatches at this pattern with notably lower l10n_to_en_word_ratio.
_LATIN_COMPACTNESS_LANGS: FrozenSet[str] = frozenset(
    {"pt-br", "es", "de", "fr", "it"}
)

# Sits in the empirical gap between Latin-compactness false alarms (>=0.94)
# and ru genuinely-outdated files (<=0.85) at the all-other-indicators-zero
# pattern.
_COMPACTNESS_WORD_RATIO_FLOOR = 0.90

# ASCII-only boundaries: Python's `\b` treats CJK as word chars, so plain
# `\bv1\.\b` silently missed `v1.22以降` etc.
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])v1\.(\d{2,3})(?!\d)")

_ANCHOR_RE = re.compile(r"\{#([^}]+)\}")
_STRIP_FM = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_STRIP_CODE = re.compile(r"```.*?```", re.DOTALL)
_STRIP_CMNT = re.compile(r"<!--.*?-->", re.DOTALL)
_STRIP_INLINE = re.compile(r"`[^`\n]+`")
_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

# Unicode-aware: keeps `informação` and Cyrillic as single tokens.
# Distinct from `_WORD_RE`, which must stay ASCII-only to detect
# English-only paragraphs in CJK files.
_BODY_TOKEN_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)

# `201 (<a ...>): Created`-style API response lines stay English in CJK docs.
_HTTP_STATUS_RE = re.compile(r"^\d{3}\s+\(")
_MDREF_RE = re.compile(r"^\[.+?\]:\s+\S")  # [label]: URL

_MERMAID_PREFIXES: Tuple[str, ...] = (
    "classDef ", "class ", "click ",
    "flowchart", "graph ", "sequenceDiagram", "stateDiagram", "erDiagram",
    "journey", "gantt", "%%",
)

# =============================================================================
# Data classes
# =============================================================================

@dataclass(frozen=True)
class ParsedFile:
    visible_lines: int
    h2: int
    h3: int
    code_blocks: int
    anchors: FrozenSet[str]
    versions: FrozenSet[str]
    untranslated_paras: int = 0
    body_words: int = 0  # Unicode-aware token count of visible body text

@dataclass
class FileStats:
    en_visible_lines: int
    localized_visible_lines: int
    l10n_to_en_line_ratio: float
    l10n_to_en_word_ratio: float    # 1.0 when EN has no body words
    missing_h2: int
    missing_h3: int
    missing_code_blocks: int
    missing_anchors: int
    untranslated_paras: int
    missing_new_versions: int

@dataclass
class ScoreBreakdown:
    """Post-suppression scores + notes; reason strings rendered separately."""
    ratio_score: int
    ratio_reason: str                  # "" when silent-tier or suppressed
    effective_missing_h2: int          # after zh-cn H2-as-H3 adjustment
    heading_score: int
    code_score: int
    anchor_score: int
    untranslated_score: int
    version_score: int
    only_anchor_indicator: bool
    h2_as_h3_note: str
    structure_mismatch_note: str

    @property
    def total_score(self) -> int:
        return (self.ratio_score + self.heading_score + self.code_score
                + self.anchor_score + self.untranslated_score + self.version_score)

@dataclass
class FileScore:
    localized_path: str
    en_path: str
    stats: FileStats
    score: int
    priority: str
    reasons: List[str] = field(default_factory=list)

# =============================================================================
# Parsing
# =============================================================================

def _count_visible_lines(text: str) -> int:
    # Keeps code-block contents: code volume is legitimate content.
    t = _STRIP_FM.sub("", text, count=1)
    t = _STRIP_CMNT.sub("", t)
    return sum(1 for line in t.splitlines() if line.strip())

def _is_mermaid_para(para: str) -> bool:
    for line in para.splitlines():
        s = line.strip()
        if s and any(s.startswith(p) for p in _MERMAID_PREFIXES):
            return True
    return False

def _is_indented_code_block(raw_para: str) -> bool:
    # Caller must pass the pre-strip paragraph — leading indent is the indicator.
    lines = raw_para.splitlines()
    non_blank = [line for line in lines if line.strip()]
    return bool(non_blank) and all(line[:4] == "    " for line in non_blank)

def _count_body_words(text: str) -> int:
    # Volume indicator for the Latin-compactness suppression rule. Strips
    # structural/code noise; whatever survives counts as body text.
    t = _STRIP_FM.sub("", text, count=1)
    t = _STRIP_CMNT.sub("", t)
    t = _STRIP_CODE.sub("", t)
    kept = []
    for raw in re.split(r"\n{2,}", t):
        if _is_indented_code_block(raw):
            continue
        kept.append(raw)
    t = "\n\n".join(kept)
    t = _STRIP_INLINE.sub("", t)
    return len(_BODY_TOKEN_RE.findall(t))

def _count_untranslated_paragraphs(text: str, language: str) -> int:
    if language not in _CJK_LANGS:
        return 0

    text = _STRIP_FM.sub("", text, count=1)
    text = _STRIP_CODE.sub("", text)
    text = _STRIP_CMNT.sub("", text)

    count = 0
    for raw_para in re.split(r"\n{2,}", text):
        if _is_indented_code_block(raw_para):
            continue
        para = raw_para.strip()
        if len(para) < 40 or para[0] in "#{}<!-|*":
            continue
        if _is_mermaid_para(para) or _MDREF_RE.match(para) or _HTTP_STATUS_RE.match(para):
            continue
        clean = _STRIP_INLINE.sub("", para)
        non_ascii_alpha = sum(1 for c in clean if ord(c) > 127 and c.isalpha())
        ascii_words = len(_WORD_RE.findall(clean))
        if non_ascii_alpha == 0 and ascii_words >= 8:
            count += 1
    return count

def _scan_structure(text: str) -> Tuple[int, int, int, FrozenSet[str]]:
    # Strip comments first: zh-cn bilingual files hide EN fences inside
    # <!-- -->, which would desync `in_code`. Toggle on 0-3-space fences
    # (CommonMark) but only count column-0 — indented fences are noisy.
    lines = _STRIP_CMNT.sub("", text).splitlines()
    h2 = h3 = fences = 0
    anchors = set()
    in_code = False

    for line in lines:
        stripped = line.lstrip(' ')
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
        if line.startswith("#"):
            for m in _ANCHOR_RE.finditer(line):
                anchors.add(m.group(1).strip().lower())

    return h2, h3, fences // 2, frozenset(anchors)

def parse_markdown(path: str, language: str = "") -> ParsedFile:
    # Pass language="" for the EN side: skips the CJK-only untranslated count.
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    h2, h3, code_blocks, anchors = _scan_structure(text)
    return ParsedFile(
        visible_lines=_count_visible_lines(text),
        h2=h2, h3=h3, code_blocks=code_blocks,
        anchors=anchors,
        versions=frozenset(f"v1.{minor}" for minor in _VERSION_RE.findall(text)),
        untranslated_paras=_count_untranslated_paragraphs(text, language) if language else 0,
        body_words=_count_body_words(text),
    )

# =============================================================================
# Stats
# =============================================================================

def _version_minor(v: str) -> int:
    m = re.match(r"v1\.(\d+)", v)
    return int(m.group(1)) if m else 0

def count_missing_new_versions(en_versions: FrozenSet[str], l10n_versions: FrozenSet[str]) -> int:
    if not l10n_versions:
        return len(en_versions)
    l10n_max = max(_version_minor(v) for v in l10n_versions)
    return sum(1 for v in en_versions if _version_minor(v) > l10n_max)

def compute_stats(en: ParsedFile, l10n: ParsedFile, language: str) -> FileStats:
    # l10n-to-EN ratios, capped at 2.0; 1.0 when EN is empty.
    line_ratio = (min(l10n.visible_lines / en.visible_lines, 2.0)
                  if en.visible_lines > 0 else 1.0)
    word_ratio = (l10n.body_words / en.body_words) if en.body_words > 0 else 1.0
    return FileStats(
        en_visible_lines=en.visible_lines,
        localized_visible_lines=l10n.visible_lines,
        l10n_to_en_line_ratio=line_ratio,
        l10n_to_en_word_ratio=word_ratio,
        missing_h2=max(0, en.h2 - l10n.h2),
        missing_h3=max(0, en.h3 - l10n.h3),
        missing_code_blocks=max(0, en.code_blocks - l10n.code_blocks),
        missing_anchors=len(en.anchors - l10n.anchors),
        untranslated_paras=l10n.untranslated_paras,
        missing_new_versions=count_missing_new_versions(en.versions, l10n.versions),
    )

# =============================================================================
# Scoring — per-indicator helpers
# =============================================================================
# Weights and caps are audit-tuned; the combined heading cap keeps large-TOC
# pages from dominating the score.

def _score_heading(effective_missing_h2: int, missing_h3: int) -> int:
    return min(effective_missing_h2 * 7 + missing_h3, 25)

def _score_code(missing_code_blocks: int) -> int:
    return min(missing_code_blocks * 2, 10)

def _score_anchor(missing_anchors: int) -> int:
    return min(missing_anchors * 2, 10)

def _score_untranslated(untranslated_paras: int) -> int:
    return min(untranslated_paras * 5, 10)

def _score_version(missing_new_versions: int) -> int:
    return min(missing_new_versions * 2, 10)

# =============================================================================
# Scoring — ratio-suppression rules
# =============================================================================

def _priority_for(score: int) -> str:
    if score >= _HIGH_THRESHOLD:
        return _PRIORITY_HIGH
    if score >= _MEDIUM_THRESHOLD:
        return _PRIORITY_MEDIUM
    if score >= _LOW_THRESHOLD:
        return _PRIORITY_LOW
    return _PRIORITY_UP_TO_DATE

def _score_ratio(
    stats: FileStats, en: ParsedFile, l10n: ParsedFile
) -> Tuple[int, str]:
    # `l10n.visible_lines == 0` bypasses the short-EN floor: compactness can
    # shorten lines, not erase them — empty stubs must still reach MEDIUM.
    if not (en.visible_lines >= _RATIO_MIN_SOURCE_LINES
            or l10n.visible_lines == 0):
        return 0, ""

    ratio = stats.l10n_to_en_line_ratio
    if ratio < 0.50:
        inv = f"{1 / ratio:.1f}×" if ratio > 0 else "∞"
        return 40, (
            f"Localized file is {inv} shorter than EN "
            f"(l10n-to-EN line ratio {ratio:.2f})"
        )
    if ratio < 0.65:
        return 25, (
            f"Localized file is substantially shorter than EN "
            f"(l10n-to-EN line ratio {ratio:.2f})"
        )
    if ratio < 0.80:
        return 10, (
            f"Localized file is moderately shorter than EN "
            f"(l10n-to-EN line ratio {ratio:.2f})"
        )
    if ratio < 0.90:
        return 3, ""   # silent tier: 3 pts without a reason string
    return 0, ""

def _adjust_h2_as_h3(
    stats: FileStats, en: ParsedFile, l10n: ParsedFile, language: str
) -> Tuple[int, str]:
    # zh-cn bilingual files sometimes render EN H2 as H3. Detect via H3
    # surplus covering the H2 deficit; suppress all but 1 unit. The extra
    # predicates keep it from triggering on genuine small H2 deficits.
    if not (language == "zh-cn"
            and stats.missing_h2 >= 3
            and l10n.h3 > en.h3
            and (l10n.h3 - en.h3) >= stats.missing_h2
            and stats.l10n_to_en_line_ratio >= 0.70
            and stats.untranslated_paras <= 1
            and stats.missing_code_blocks <= 1):
        return stats.missing_h2, ""

    effective = min(1, stats.missing_h2)
    suppressed = stats.missing_h2 - effective
    note = (
        f"Possible heading-level mismatch: {suppressed} of {stats.missing_h2} "
        f"apparent missing H2(s) may have been translated as H3(s) in the "
        f"zh-cn file (zh-cn has {l10n.h3 - en.h3} more H3s than EN) — "
        "score reduced; verify manually"
    )
    return effective, note

def _only_ratio_indicator_active(stats: FileStats) -> bool:
    # Shared pattern for the compactness suppression rules: every non-ratio
    # indicator is zero, so the ratio score (if any) stands alone.
    return (stats.missing_h2 == 0
            and stats.missing_h3 == 0
            and stats.missing_code_blocks == 0
            and stats.missing_anchors == 0
            and stats.untranslated_paras == 0
            and stats.missing_new_versions == 0)

def _should_suppress_ratio_for_short_source(
    ratio_score: int, en: ParsedFile, l10n: ParsedFile, language: str,
    support_score: int,
) -> bool:
    # Ratio without heading/code/version support is suppressed for EN 15-39
    # (all locales) and 40-55 (CJK only). `l10n.visible_lines == 0` bypasses.
    if ratio_score <= 0 or l10n.visible_lines <= 0 or support_score != 0:
        return False
    return (en.visible_lines < _RATIO_CORROBORATION_THRESHOLD
            or (language in _CJK_LANGS
                and en.visible_lines < _RATIO_CORROBORATION_THRESHOLD_CJK))

def _should_suppress_ratio_for_ja_long_file(
    ratio_score: int, stats: FileStats, en: ParsedFile, l10n: ParsedFile,
    language: str,
) -> bool:
    # JA-only: CJK body-text compactness on long files. Restricted to ja
    # because zh-cn has genuinely-outdated files at this exact pattern.
    return (ratio_score >= 25
            and l10n.visible_lines > 0
            and language == "ja"
            and en.visible_lines >= _RATIO_CORROBORATION_THRESHOLD_CJK
            and _only_ratio_indicator_active(stats))

def _should_suppress_ratio_for_latin_compactness(
    ratio_score: int, stats: FileStats, en: ParsedFile, l10n: ParsedFile,
    language: str,
) -> bool:
    # Word-ratio supporting evidence distinguishes a complete translation
    # with different line wrapping (high body-word volume) from a thinned-
    # out one.
    return (ratio_score >= 25
            and l10n.visible_lines > 0
            and language in _LATIN_COMPACTNESS_LANGS
            and en.visible_lines >= _COMPACTNESS_GATE_MIN_SOURCE_LINES
            and _only_ratio_indicator_active(stats)
            and stats.l10n_to_en_word_ratio >= _COMPACTNESS_WORD_RATIO_FLOOR)

def _structure_mismatch_note(stats: FileStats, en: ParsedFile) -> str:
    # Advisory only (no score): anchor + version mismatch on a large file
    # whose ratio/H2/code look intact — likely preserves older structure.
    if (stats.l10n_to_en_line_ratio >= 0.80
            and stats.missing_h2 == 0
            and stats.missing_code_blocks == 0
            and en.visible_lines >= 100
            and stats.missing_anchors >= 4
            and stats.missing_new_versions >= 2):
        return (
            f"Possible structure mismatch: l10n-to-EN line ratio "
            f"({stats.l10n_to_en_line_ratio:.2f}), H2 counts, and code-block counts "
            "look close to EN, but both anchor mismatches and missing versions are "
            "present — localized file likely preserves older structure; actual "
            "translation effort may exceed score"
        )
    return ""

def compute_scores(
    stats: FileStats, en: ParsedFile, l10n: ParsedFile, language: str
) -> ScoreBreakdown:
    ratio_score, ratio_reason = _score_ratio(stats, en, l10n)
    effective_missing_h2, h2_as_h3_note = _adjust_h2_as_h3(stats, en, l10n, language)

    heading_score = _score_heading(effective_missing_h2, stats.missing_h3)
    code_score = _score_code(stats.missing_code_blocks)
    anchor_score = _score_anchor(stats.missing_anchors)
    untranslated_score = _score_untranslated(stats.untranslated_paras)
    version_score = _score_version(stats.missing_new_versions)

    # The only-anchor-indicator check uses the *pre-suppression* ratio_reason —
    # an empty string here means no ratio scored OR the silent 3-pt tier scored.
    non_ratio_total = (heading_score + code_score + anchor_score
                       + untranslated_score + version_score)
    only_anchor_indicator = (anchor_score > 0
                             and non_ratio_total == anchor_score
                             and not ratio_reason)

    support_score = heading_score + code_score + version_score
    if _should_suppress_ratio_for_short_source(
            ratio_score, en, l10n, language, support_score):
        ratio_score, ratio_reason = 0, ""
    if _should_suppress_ratio_for_ja_long_file(
            ratio_score, stats, en, l10n, language):
        ratio_score, ratio_reason = 0, ""
    if _should_suppress_ratio_for_latin_compactness(
            ratio_score, stats, en, l10n, language):
        ratio_score, ratio_reason = 0, ""

    return ScoreBreakdown(
        ratio_score=ratio_score,
        ratio_reason=ratio_reason,
        effective_missing_h2=effective_missing_h2,
        heading_score=heading_score,
        code_score=code_score,
        anchor_score=anchor_score,
        untranslated_score=untranslated_score,
        version_score=version_score,
        only_anchor_indicator=only_anchor_indicator,
        h2_as_h3_note=h2_as_h3_note,
        structure_mismatch_note=_structure_mismatch_note(stats, en),
    )

# =============================================================================
# Reason formatting
# =============================================================================

_ONLY_ANCHOR_INDICATOR_SUFFIX = (
    " (only indicator — may reflect anchor naming differences, "
    "typo variants, or structure mismatch; verify manually)"
)

def format_reasons(
    stats: FileStats, breakdown: ScoreBreakdown, language: str
) -> List[str]:
    # Order: ratio -> heading -> H2-as-H3 -> code -> anchor ->
    # untranslated -> version -> structure-mismatch.
    reasons: List[str] = []

    if breakdown.ratio_reason:
        reasons.append(breakdown.ratio_reason)

    if breakdown.heading_score:
        parts = []
        if breakdown.effective_missing_h2:
            parts.append(f"{breakdown.effective_missing_h2} H2")
        if stats.missing_h3:
            parts.append(f"{stats.missing_h3} H3")
        reasons.append(
            f"Localized file is missing headings present in EN ({', '.join(parts)})"
        )

    if breakdown.h2_as_h3_note:
        reasons.append(breakdown.h2_as_h3_note)

    if breakdown.code_score:
        reasons.append(
            f"Localized file is missing {stats.missing_code_blocks} "
            f"code block(s) present in EN"
        )

    if breakdown.anchor_score:
        r = (
            f"Localized file is missing {stats.missing_anchors} "
            f"section anchor(s) present in EN"
        )
        if breakdown.only_anchor_indicator:
            r += _ONLY_ANCHOR_INDICATOR_SUFFIX
        reasons.append(r)

    if breakdown.untranslated_score:
        reasons.append(
            f"{stats.untranslated_paras} paragraph(s) in localized file "
            f"appear untranslated (no {language.upper()} characters)"
        )

    if breakdown.version_score:
        reasons.append(
            f"Localized file is missing {stats.missing_new_versions} "
            f"Kubernetes version reference(s) present in EN"
        )

    if breakdown.structure_mismatch_note:
        reasons.append(breakdown.structure_mismatch_note)

    return reasons

# =============================================================================
# Pipeline + scanning + reporting + CLI
# =============================================================================

def score_file_pair(en_path: str, l10n_path: str, language: str) -> FileScore:
    en = parse_markdown(en_path)
    l10n = parse_markdown(l10n_path, language)
    stats = compute_stats(en, l10n, language)
    breakdown = compute_scores(stats, en, l10n, language)
    reasons = format_reasons(stats, breakdown, language)
    return FileScore(
        localized_path=l10n_path,
        en_path=en_path,
        stats=stats,
        score=breakdown.total_score,
        priority=_priority_for(breakdown.total_score),
        reasons=reasons,
    )

def scan_language(language: str, repo_root: str) -> List[FileScore]:
    language_dir = os.path.join(repo_root, "content", language)
    if not os.path.isdir(language_dir):
        print(f"error: language directory not found: {language_dir}", file=sys.stderr)
        sys.exit(1)

    pairs: List[Tuple[str, str]] = []
    for root, _, files in os.walk(language_dir):
        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue
            l10n_path = os.path.join(root, fname)
            en_path = re.sub(r"content/[^/]+/", "content/en/", l10n_path)
            if os.path.exists(en_path):
                pairs.append((en_path, l10n_path))

    if not pairs:
        return []

    total = len(pairs)
    print(f"  [{language}] scoring {total} file pairs ...", file=sys.stderr)

    scored = []
    for i, (en_path, l10n_path) in enumerate(pairs, 1):
        if i % 100 == 0:
            print(f"  [{language}] {i}/{total}", file=sys.stderr)
        scored.append(score_file_pair(en_path, l10n_path, language))

    scored.sort(key=lambda s: (-s.score, s.localized_path))
    return scored

def _rel(path: str, repo_root: str) -> str:
    try:
        return os.path.relpath(path, repo_root)
    except ValueError:
        return path

def _doc_area(path: str, language: str) -> str:
    m = re.search(rf"content/{re.escape(language)}/([^/]+(?:/[^/]+)?)", path)
    if not m:
        return "(other)/"
    seg = m.group(1)
    if "." in seg.split("/")[-1]:
        seg = "/".join(seg.split("/")[:-1])
    return seg.rstrip("/") + "/"

def _priority_level_counts(scored: List[FileScore]) -> Tuple[int, int, int, int]:
    c = Counter(s.priority for s in scored)
    return (c[_PRIORITY_HIGH], c[_PRIORITY_MEDIUM],
            c[_PRIORITY_LOW], c[_PRIORITY_UP_TO_DATE])

def format_report_md(
    language: str,
    scored: List[FileScore],
    repo_root: str,
    detailed: bool,
    date: str,
) -> str:
    by_priority = {
        name: []
        for name in (_PRIORITY_HIGH, _PRIORITY_MEDIUM, _PRIORITY_LOW, _PRIORITY_UP_TO_DATE)
    }
    for s in scored:
        by_priority[s.priority].append(s)

    area_counts: Counter = Counter(
        _doc_area(s.localized_path, language)
        for s in scored
        if s.priority != _PRIORITY_UP_TO_DATE
    )

    lines: List[str] = [
        f"## Localization status (file-level): `{language}`",
        "",
        f"Generated: {date}  ",
        "Method: content-based indicators only  ",
        "Script: `scripts/triage-by-content-signals.py`",
        "",
        "| | Count |",
        "|---|---|",
        f"| Scanned pairs   | {len(scored)} |",
        f"| High priority   | {len(by_priority[_PRIORITY_HIGH])} |",
        f"| Medium priority | {len(by_priority[_PRIORITY_MEDIUM])} |",
        f"| Low priority    | {len(by_priority[_PRIORITY_LOW])} |",
        f"| Up to date      | {len(by_priority[_PRIORITY_UP_TO_DATE])} |",
        "",
    ]

    if area_counts:
        lines += ["**Top affected areas:**", ""]
        for doc_area, count in area_counts.most_common(8):
            lines.append(f"- `{doc_area}`: {count} files")
        lines.append("")

    def add_section(title: str, items: List[FileScore], show_reasons: bool) -> None:
        lines.extend([f"### {title} ({len(items)})", ""])
        if not items:
            lines.extend(["_None_", ""])
            return

        for s in items:
            path = _rel(s.localized_path, repo_root)
            if show_reasons and s.reasons:
                lines.append(f"**`{path}`** — score {s.score}")
                lines.extend(f"- {reason}" for reason in s.reasons)
                lines.append("")
            else:
                first = s.reasons[0] if s.reasons else "(no scored indicators)"
                lines.append(f"- `{path}` — score {s.score}: {first}")
        if not show_reasons:
            lines.append("")

    add_section("High priority", by_priority[_PRIORITY_HIGH], detailed)
    add_section("Medium priority", by_priority[_PRIORITY_MEDIUM], detailed)
    add_section("Low priority", by_priority[_PRIORITY_LOW], False)

    return "\n".join(lines)

def format_index_md(results: List[Tuple[str, List[FileScore]]], date: str) -> str:
    lines = [
        "## Localization file-level status index",
        "",
        f"Generated: {date}",
        "",
        "| Locale | Report | Pairs | High | Medium | Low | Up to date |",
        "|---|---|---|---|---|---|---|",
    ]

    for language, scored in results:
        high, medium, low, utd = _priority_level_counts(scored)
        fname = f"l10n-{language}.md"
        lines.append(
            f"| `{language}` | [{fname}]({fname}) | {len(scored)} |"
            f" {high} | {medium} | {low} | {utd} |"
        )

    lines.append("")
    return "\n".join(lines)

def main() -> None:
    # Setup: parse CLI args and resolve runtime configuration.
    parser = argparse.ArgumentParser(
        description=(
            "File-level localization outdatedness detector for Kubernetes docs.\n"
            "Uses lightweight content indicators — no section alignment required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang", metavar="CODE", help="Single locale (e.g. ko)")
    group.add_argument(
        "--langs",
        metavar="CODES",
        help="Comma-separated locales (e.g. ko,zh-cn,ja)",
    )
    group.add_argument(
        "--all-langs",
        action="store_true",
        help="All locales under content/ except en",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        metavar="DIR",
        help="Path to kubernetes/website repo root (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for report files (default: .)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show all indicator lines per file (default: one compact line)",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = os.path.abspath(args.repo_root)
    else:
        # Auto-detect: walk up from cwd looking for content/en
        d = os.path.abspath(os.getcwd())
        repo_root = None
        while True:
            if os.path.isdir(os.path.join(d, "content", "en")):
                repo_root = d
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        if repo_root is None:
            print(
                "error: could not auto-detect repo root (no content/en found "
                "above cwd). Use --repo-root to specify it explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
    date = datetime.date.today().isoformat()

    if args.lang:
        languages = [args.lang]
    elif args.langs:
        languages = [code.strip() for code in args.langs.split(",") if code.strip()]
    else:
        content_dir = os.path.join(repo_root, "content")
        languages = sorted(
            d for d in os.listdir(content_dir)
            if os.path.isdir(os.path.join(content_dir, d)) and d != "en"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Per-locale phase: scan content/<locale>/ and emit one report file each.
    all_results: List[Tuple[str, List[FileScore]]] = []

    for language in languages:
        print(f"Scanning content/{language}/ ...", file=sys.stderr)
        scored = scan_language(language, repo_root)
        all_results.append((language, scored))

        out_path = os.path.join(args.output_dir, f"l10n-outdated-report-{language}.md")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(format_report_md(language, scored, repo_root, args.detailed, date))

        high, medium, low, utd = _priority_level_counts(scored)
        print(
            f"Wrote {out_path}  "
            f"({high} high, {medium} medium, {low} low, {utd} up-to-date)",
            file=sys.stderr,
        )
        
    # Cross-locale phase: write a roll-up index only when >1 locale was scanned.
    if len(languages) > 1:
        index_path = os.path.join(args.output_dir, "l10n-outdated-report-index.md")
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write(format_index_md(all_results, date))
        print(f"Wrote {index_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
