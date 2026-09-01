"""Semantic-wrap markdown prose to a column cap, in place.

Only ever *splits* a line, never joins two: already hand-wrapped prose is left
as found and a second run is a no-op.

Breaks land on sentence and clause boundaries, not at column CAP -- a greedy
fill re-wraps every line below an edited sentence. Candidates are scored by
boundary strength plus how full the line is.

Left alone, because wrapping them changes what markdown renders: fenced code,
table rows, headings, link reference definitions, HTML blocks and indented
code.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CAP = 100

FENCE = re.compile(r"^\s{0,3}(```|~~~)")
HEADING = re.compile(r"^\s{0,3}#")
TABLE = re.compile(r"^\s*\|")
QUOTE = re.compile(r"^\s{0,3}>")
HTML = re.compile(r"^\s{0,3}<")
LINKDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:")
INDENT_CODE = re.compile(r"^ {4,}\S")
# `- `, `* `, `+ `, `1. `, `1) ` -- the marker plus the column its content starts at.
BULLET = re.compile(r"^(\s*)([-*+]|\d+[.)])(\s+)")

# A continuation line must not open a new block. Anything matching here at the
# start of the remainder forbids that break point.
STRUCTURAL = re.compile(r"^(?:[-*+]\s|\d+[.)]\s|#|>|\||=+\s*$|-+\s*$|\[[^\]]+\]:)")

# Boundary strength: a bonus added to the break position when scoring.
BOUNDARIES = (
    (re.compile(r"[.!?][)\"'`\]]?$"), 34),  # end of sentence
    (re.compile(r"[—:;]$"), 18),  # em-dash / colon / semicolon clause
    (re.compile(r",$"), 9),
)


def _content_indent(line: str) -> str:
    """Indent a continuation of `line` must carry to stay in the same block."""
    m = BULLET.match(line)
    if m:
        return " " * (len(m.group(1)) + len(m.group(2)) + len(m.group(3)))
    return re.match(r"^\s*", line).group(0)


def _in_code_span(text: str, pos: int) -> bool:
    return text.count("`", 0, pos) % 2 == 1


def _split_once(line: str, cap: int) -> tuple[str, str] | None:
    """Best (head, tail) split of `line`, or None if it cannot be split."""
    best = None
    for i, ch in enumerate(line):
        if ch != " " or i == 0 or i + 1 >= len(line):
            continue
        if line[i + 1] == " ":  # keep runs of spaces on the head
            continue
        head, tail = line[:i], line[i + 1 :]
        if not head.strip() or STRUCTURAL.match(tail):
            continue
        if i > cap and best is not None:
            break  # past the cap and we already have somewhere to go
        score = i
        for pat, bonus in BOUNDARIES:
            if pat.search(head):
                score += bonus
                break
        if _in_code_span(line, i):
            score -= 28
        if i > cap:
            score -= 10_000  # only ever a last resort
        if best is None or score > best[0]:
            best = (score, head, tail)
    if best is None:
        return None
    return best[1], best[2]


def wrap_line(line: str, cap: int = CAP) -> list[str]:
    out: list[str] = []
    indent = _content_indent(line)
    rest = line
    while len(rest) > cap:
        split = _split_once(rest, cap)
        if split is None:
            break
        head, tail = split
        out.append(head.rstrip())
        rest = indent + tail
        if len(indent + tail) >= len(line):  # made no progress; bail out whole
            return [line]
    out.append(rest.rstrip())
    return out


def wrap_text(text: str, cap: int = CAP) -> str:
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        skip = (
            in_fence
            or len(line) <= cap
            or HEADING.match(line)
            or TABLE.match(line)
            or QUOTE.match(line)
            or HTML.match(line)
            or LINKDEF.match(line)
            or INDENT_CODE.match(line)
        )
        out.extend([line] if skip else wrap_line(line, cap))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--check", action="store_true", help="report, do not rewrite")
    args = ap.parse_args()

    changed = []
    for path in args.paths:
        before = path.read_text(encoding="utf-8")
        after = wrap_text(before, args.cap)
        if after == before:
            continue
        changed.append(path)
        if not args.check:
            path.write_text(after, encoding="utf-8")
    for path in changed:
        print(f"{'would wrap' if args.check else 'wrapped'}: {path}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
