#!/usr/bin/env python3
"""Copy a .tex file with every comment removed (full-line comments dropped, trailing comments cut
to a bare '%' so line-end whitespace control is preserved).  Used by `make bundle` for the
submission copy of numbers.tex, whose working notes are not part of the manuscript."""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
out = []
for line in open(src, encoding="utf-8").read().split("\n"):
    k, esc = None, False
    for i, ch in enumerate(line):
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == "%": k = i; break
    if k is None: out.append(line); continue
    if line[:k].strip() == "": continue
    out.append(line[:k].rstrip() + "%" if line[k + 1:].strip() else line)
open(dst, "w", encoding="utf-8").write(re.sub(r"\n{3,}", "\n\n", "\n".join(out)))
