#!/usr/bin/env python3
"""
detect_l10n_drift_structural.py — Detect outdated localized docs by comparing
the structure (headings, paragraphs, code blocks, lists, shortcodes) of an
English file with its localized counterpart.

Unlike commit-history tools, this catches real content changes while ignoring
cosmetic edits that don't need translation updates.

Usage:
    python3 detect_l10n_drift_structural.py <en_path> <localized_path>

Output: JSON with a "status" field — up_to_date | candidate_outdated |
        outdated_low_severity | no_english_version | not_translated

Exit code: 0 if status is not candidate_outdated, 1 otherwise.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Block types
HEADING   = "heading"
PARAGRAPH = "paragraph"
CODE      = "code_block"
SHORTCODE = "shortcode"
LIST_ITEM = "list_item"


@dataclass
class Block:
    kind: str           # one of the constants above
    text: str
    level: int = 0      # heading depth (1–6); 0 for non-headings
    anchor: str = ""    # {#anchor} extracted from heading; empty if absent


@dataclass
class Section:
    """A heading plus all blocks that follow it until the next heading."""
    heading: Optional[Block]   # None for content before the first heading ("root")
    blocks: List[Block] = field(default_factory=list)
    pos: int = 0               # ordinal position in document order


@dataclass
class SectionChange:
    """Differences found between a matched English and localized section."""
    section: str               # heading text, or "(root)" for the pre-heading block
    signals: List[str] = field(default_factory=list)
    low_severity: bool = False  # True when only shortcode-level differences


@dataclass
class FileResult:
    localized_path: str
    en_path: str
    status: str        # up_to_date | candidate_outdated | outdated_low_severity |
                       # no_english_version | not_translated
    changes: List[SectionChange] = field(default_factory=list)
    error: str = ""


# Regex patterns

_HEADING_RE  = re.compile(r'^(#{1,6})\s+(.+)$')
_FENCE_RE    = re.compile(r'^```')
_SC_OPEN_RE  = re.compile(r'^\{\{[<%]\s*.*\s*[>%]\}\}')
_SC_CLOSE_RE = re.compile(r'^\{\{[<%]\s*/.*\s*[>%]\}\}')
_LIST_RE     = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+')

# Matches note/warning callouts like "Note:", "Warning:", etc.
_DOC_NOTE_RE = re.compile(
    r'(?:^|(?<=\. ))(?:Note that |Note: |Warning: |Caution: |Important: |Deprecated: |Tip: )',
    re.IGNORECASE,
)

# Matches trailing labels like "(Deprecated)", "(Updated)" on headings
_QUALIFIER_RE = re.compile(r'\(([A-Za-z][A-Za-z\s]{0,25})\)\s*$')




def normalize_markdown(text: str) -> str:
    """Strip frontmatter, HTML comments, and collapse extra blank lines."""
    text = re.sub(r'^---\n.*?\n---\n', '', text, count=1, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()




def parse_blocks(text: str) -> List[Block]:
    """Parse normalized markdown into a flat list of typed blocks."""
    blocks: List[Block] = []
    lines = text.split('\n')
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            content = m.group(2).strip()
            anchor = ""
            am = re.search(r'\{#([^}]+)\}', content)
            if am:
                anchor = am.group(1).strip()
                content = content[:am.start()].strip()
            blocks.append(Block(kind=HEADING, text=content, level=len(m.group(1)), anchor=anchor))
            i += 1
            continue

        # Fenced code block
        if _FENCE_RE.match(line):
            code_lines = [line]
            i += 1
            while i < n and not _FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                code_lines.append(lines[i])
                i += 1
            blocks.append(Block(kind=CODE, text='\n'.join(code_lines)))
            continue

        # Shortcode opening or closing tag
        if _SC_OPEN_RE.match(line) or _SC_CLOSE_RE.match(line):
            blocks.append(Block(kind=SHORTCODE, text=line.strip()))
            i += 1
            continue

        # List item
        if _LIST_RE.match(line):
            item = [line]
            i += 1
            # Include indented continuation lines
            while i < n and lines[i].strip() and lines[i].startswith('  '):
                item.append(lines[i])
                i += 1
            blocks.append(Block(kind=LIST_ITEM, text='\n'.join(item)))
            continue

        # Paragraph — contiguous non-blank, non-special lines
        para = [line]
        i += 1
        while (i < n and lines[i].strip()
               and not _HEADING_RE.match(lines[i])
               and not _FENCE_RE.match(lines[i])
               and not _SC_OPEN_RE.match(lines[i])
               and not _SC_CLOSE_RE.match(lines[i])
               and not _LIST_RE.match(lines[i])):
            para.append(lines[i])
            i += 1
        blocks.append(Block(kind=PARAGRAPH, text='\n'.join(para)))

    return blocks


def heading_tokens(text: str) -> set:
    """Extract keywords from a heading that stay the same across translations."""
    toks: set = set()
    for m in re.finditer(r'`([^`]+)`', text):
        toks.add(m.group(1).strip())
    for m in re.finditer(r'\b([A-Z][A-Z0-9_]{2,})\b', text):
        toks.add(m.group(1))
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b', text):
        toks.add(m.group(1))
    for m in re.finditer(r'\bv\d+\.\d+\b', text):
        toks.add(m.group(0))
    for m in re.finditer(r'\b[a-z][a-z0-9-]+\.[a-z][a-z0-9.-]+(?:/[a-z][a-z0-9-]*)?\b', text):
        toks.add(m.group(0))
    return toks


def code_tokens(text: str) -> set:
    """Extract non-comment tokens from a code block."""
    toks: set = set()
    for line in text.split('\n'):
        s = line.strip()
        if not s or s.startswith('```') or s.startswith('~~~'):
            continue
        if s.startswith('#') or s.startswith('//') or s.startswith('--'):
            continue
        for sep in (' # ', ' // '):
            if sep in line:
                line = line[:line.index(sep)]
        for tok in line.split():
            if len(tok) > 1:
                toks.add(tok)
    return toks




def group_sections(blocks: List[Block]) -> List[Section]:
    """Group blocks into sections, each starting with a heading."""
    sections: List[Section] = []
    current = Section(heading=None, pos=0)
    pos = 0

    for block in blocks:
        if block.kind == HEADING:
            if current.blocks or current.heading is not None:
                sections.append(current)
            pos += 1
            current = Section(heading=block, pos=pos)
        else:
            current.blocks.append(block)

    if current.blocks or current.heading is not None:
        sections.append(current)

    return sections


def _heading_key(text: str) -> str:
    """Lowercase and collapse whitespace for heading comparison."""
    return re.sub(r'\s+', ' ', text.strip().lower())



def align_sections(
    en_secs: List[Section],
    localized_secs: List[Section],
) -> List[Tuple[Optional[Section], Optional[Section], str]]:
    """Match English sections to their localized counterparts.

    Tries four approaches, from most to least reliable:
      1. {#anchor} attribute
      2. Exact heading text match
      3. Shared keywords in heading (e.g. CamelCase, `backtick` terms)
      4. Position in document (for fully-translated headings)
    """
    en_root    = next((s for s in en_secs   if s.heading is None), None)
    localized_root  = next((s for s in localized_secs if s.heading is None), None)
    en_headed  = [s for s in en_secs   if s.heading is not None]
    localized_headed = [s for s in localized_secs if s.heading is not None]

    pairs: List[Tuple[Optional[Section], Optional[Section], str]] = []
    if en_root or localized_root:
        pairs.append((en_root, localized_root, "root"))

    # Index localized sections by anchor and heading text
    localized_by_anchor: dict = {}
    localized_by_name:   dict = {}
    for idx, sec in enumerate(localized_headed):
        if sec.heading.anchor:
            localized_by_anchor.setdefault(sec.heading.anchor.lower(), []).append((idx, sec))
        localized_by_name.setdefault(_heading_key(sec.heading.text), []).append((idx, sec))

    used_en:   set = set()
    used_localized: set = set()

    # 1. Match by anchor
    for ei, en in enumerate(en_headed):
        if not en.heading.anchor:
            continue
        for li, l in localized_by_anchor.get(en.heading.anchor.lower(), []):
            if li not in used_localized:
                pairs.append((en, l, "anchor"))
                used_en.add(ei)
                used_localized.add(li)
                break

    # 2. Match by exact heading text
    for ei, en in enumerate(en_headed):
        if ei in used_en:
            continue
        for li, l in localized_by_name.get(_heading_key(en.heading.text), []):
            if li not in used_localized:
                pairs.append((en, l, "heading"))
                used_en.add(ei)
                used_localized.add(li)
                break

    # 3. Match by shared keywords in heading
    tok_index: dict = {}
    for li, l in enumerate(localized_headed):
        if li in used_localized:
            continue
        for tok in heading_tokens(l.heading.text):
            tok_index.setdefault(tok, []).append((li, l))

    for ei, en in enumerate(en_headed):
        if ei in used_en:
            continue
        en_toks = heading_tokens(en.heading.text)
        if not en_toks:
            continue
        # Pick the localized section with the most keyword overlap
        scores: dict = {}
        for tok in en_toks:
            for li, _ in tok_index.get(tok, []):
                if li not in used_localized:
                    scores[li] = scores.get(li, 0) + 1
        if not scores:
            continue
        best_li = max(scores, key=lambda k: scores[k])
        best_l  = localized_headed[best_li]
        pairs.append((en, best_l, "token"))
        used_en.add(ei)
        used_localized.add(best_li)

        for tok in heading_tokens(best_l.heading.text):
            tok_index[tok] = [(i, s) for (i, s) in tok_index.get(tok, []) if i != best_li]

    # 4. Match remaining sections by position within the same heading level
    en_rem   = [(i, s) for i, s in enumerate(en_headed)   if i not in used_en]
    localized_rem = [(i, s) for i, s in enumerate(localized_headed) if i not in used_localized]

    en_by_level:   dict = {}
    localized_by_level: dict = {}
    for i, s in en_rem:
        en_by_level.setdefault(s.heading.level, []).append((i, s))
    for i, s in localized_rem:
        localized_by_level.setdefault(s.heading.level, []).append((i, s))

    pos_used: set = set()
    for level, en_grp in en_by_level.items():
        localized_grp = [(i, s) for (i, s) in localized_by_level.get(level, []) if i not in pos_used]
        for rank, (ei, en) in enumerate(en_grp):
            if rank < len(localized_grp):
                li, l = localized_grp[rank]
                pairs.append((en, l, "position"))
                used_en.add(ei)
                used_localized.add(li)
                pos_used.add(li)
            else:
                pairs.append((en, None, "missing"))

    # Remaining localized-only sections (not in English)
    for li, l in localized_rem:
        if li not in used_localized:
            pairs.append((None, l, "extra"))

    return pairs




def diff_sections(
    pairs: List[Tuple[Optional[Section], Optional[Section], str]],
) -> List[SectionChange]:
    """Compare aligned (en, localized) section pairs and return a list of changes."""
    changes: List[SectionChange] = []

    for en, localized, match in pairs:
        # Section exists in English but not in localized file
        if match == "missing":
            name = en.heading.text if en and en.heading else "(root)"
            changes.append(SectionChange(
                section=name,
                signals=[f'Section "{name}" exists in English but is missing in localized file'],
            ))
            continue

        # Localized-only or empty sections — skip
        if match == "extra":
            continue
        if en is None:
            continue
        if localized is None and match != "root":
            name = en.heading.text if en.heading else "(root)"
            changes.append(SectionChange(
                section=name,
                signals=[f'Section "{name}" exists in English but is missing in localized file'],
            ))
            continue

        # Count each block kind in both sections
        en_counts:   dict = {}
        localized_counts: dict = {}
        for b in en.blocks:
            en_counts[b.kind] = en_counts.get(b.kind, 0) + 1
        if localized:
            for b in localized.blocks:
                localized_counts[b.kind] = localized_counts.get(b.kind, 0) + 1

        name = en.heading.text if en.heading else "(root)"
        signals: List[str] = []
        only_shortcode_diffs = True

        # Skip list-item counting for "whatsnext" sections (navigation links
        # often differ between languages without meaning the content is outdated).
        is_nav = bool(en.heading and 'whatsnext' in en.heading.text.lower())

        for kind in (PARAGRAPH, CODE, SHORTCODE, LIST_ITEM):
            if kind == LIST_ITEM and is_nav:
                continue
            ec = en_counts.get(kind, 0)
            lc = localized_counts.get(kind, 0)
            if ec > lc:
                label = kind.replace('_', ' ')
                signals.append(
                    f"English has {ec} {label}(s), localized has {lc} (+{ec - lc} in English)"
                )
                if kind != SHORTCODE:
                    only_shortcode_diffs = False

        # Compare code blocks when both sides have the same count.
        # Code doesn't change with translation, so differences mean EN was updated.
        en_codes   = [b for b in en.blocks if b.kind == CODE]
        localized_codes = [b for b in localized.blocks if b.kind == CODE] if localized else []
        if en_codes and len(en_codes) == len(localized_codes):
            for idx, (ec, lc) in enumerate(zip(en_codes, localized_codes)):
                new_toks = code_tokens(ec.text) - code_tokens(lc.text)
                sig_toks = {t for t in new_toks if len(t) >= 2 and re.search(r'[a-zA-Z0-9]', t)}
                if sig_toks:
                    sample = ', '.join(sorted(sig_toks)[:5])
                    signals.append(
                        f"Code block {idx + 1} content changed: "
                        f"EN has tokens not in localized version ({sample})"
                    )
                    only_shortcode_diffs = False

        # Check for trailing labels like (Deprecated) or (Updated) on EN heading
        if en.heading:
            qm = _QUALIFIER_RE.search(en.heading.text)
            if qm:
                signals.append(
                    f'EN heading has trailing qualifier "({qm.group(1).strip()})" '
                    f'not reflected in localized heading'
                )

        # Check for added lines or notes in paragraphs
        if localized:
            en_paras   = [b for b in en.blocks   if b.kind == PARAGRAPH]
            localized_paras = [b for b in localized.blocks if b.kind == PARAGRAPH]
            for ep, lp in zip(en_paras, localized_paras):
                el = ep.text.split('\n')
                ll = lp.text.split('\n')
                # EN has one extra line — likely a new sentence added
                if len(el) - len(ll) == 1:
                    signals.append(
                        f"English paragraph has 1 more line than localized "
                        f"({len(el)} vs {len(ll)}) — possible sentence insertion not yet translated"
                    )
                    only_shortcode_diffs = False
                # EN paragraph ends with a Note/Warning — common when EN is
                # updated without updating the translation.
                if _DOC_NOTE_RE.search(el[-1]):
                    signals.append(
                        "EN paragraph ends with an unlocalized documentation note or warning sentence"
                    )
                    only_shortcode_diffs = False

        if signals:
            changes.append(SectionChange(
                section=name,
                signals=signals,
                low_severity=only_shortcode_diffs,
            ))

    return changes




def compare_file(en_path: str, localized_path: str) -> FileResult:
    """Compare one English file with its localized version."""
    if not os.path.exists(en_path):
        return FileResult(localized_path=localized_path, en_path=en_path,
                          status="no_english_version",
                          error=f"English source not found: {en_path}")
    if not os.path.exists(localized_path):
        return FileResult(localized_path=localized_path, en_path=en_path,
                          status="not_translated",
                          error=f"Localized file not found: {localized_path}")

    with open(en_path,   encoding='utf-8') as f:
        en_text   = normalize_markdown(f.read())
    with open(localized_path, encoding='utf-8') as f:
        localized_text = normalize_markdown(f.read())

    en_secs   = group_sections(parse_blocks(en_text))
    localized_secs = group_sections(parse_blocks(localized_text))
    changes   = diff_sections(align_sections(en_secs, localized_secs))

    if not changes:
        status = "up_to_date"
    elif all(c.low_severity for c in changes):
        status = "outdated_low_severity"
    else:
        status = "candidate_outdated"

    return FileResult(localized_path=localized_path, en_path=en_path, status=status, changes=changes)




def format_json(results: List[FileResult]) -> str:
    out = []
    for r in results:
        entry: dict = {"file": r.localized_path, "status": r.status}
        if r.error:
            entry["error"] = r.error
        if r.changes:
            entry["changed_sections"] = [c.section for c in r.changes]
            entry["signals"] = [
                f"{c.section}: {sig}"
                for c in r.changes
                for sig in c.signals
            ]
        out.append(entry)
    return json.dumps(out, indent=2, ensure_ascii=False)




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect potentially outdated localized Kubernetes documentation "
                    "using content-based structural comparison.",
    )
    parser.add_argument("en_path",   help="English source file path.")
    parser.add_argument("localized_path", help="Localized file path.")
    args = parser.parse_args()

    result = compare_file(args.en_path, args.localized_path)
    print(format_json([result]))
    sys.exit(1 if result.status == "candidate_outdated" else 0)


if __name__ == '__main__':
    main()
