#!/usr/bin/env python3
"""Copy every figure of main.tex into submission/figures-numbered/figN.<ext>, N in order of appearance,
as the IEEE final-submission package wants (one file per figure, named by its number, no caption inside).
A figure without \\includegraphics is the inline TikZ diagram (Fig. 1); it is taken from fig1.pdf,
which `make fig1` builds from the same TikZ source.  Also writes figures-numbered/captions.txt.

    python3 number_figs.py [main.tex] [outdir]
"""
import os, re, shutil, sys

tex = sys.argv[1] if len(sys.argv) > 1 else "main.tex"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("submission", "figures-numbered")
os.makedirs(out, exist_ok=True)
src = open(tex, encoding="utf-8").read()
# drop commented lines so a commented-out \includegraphics (e.g. the biography photo) is not counted
src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
blocks = re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", src, re.S)
lines = []
for n, b in enumerate(blocks, 1):
    m = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", b)
    cap = re.search(r"\\caption\{(.*?)\}\s*\\label", b, re.S)
    if m:
        p = m.group(1); p = p if os.path.splitext(p)[1] else p + ".pdf"
        dst = os.path.join(out, f"fig{n}{os.path.splitext(p)[1]}"); shutil.copyfile(p, dst)
    elif "tikzpicture" in b:
        if not os.path.exists("fig1.pdf"): sys.exit("fig1.pdf missing: run `make fig1` first")
        dst = os.path.join(out, f"fig{n}.pdf"); shutil.copyfile("fig1.pdf", dst); p = "(TikZ in main.tex)"
    else:
        sys.exit(f"figure {n}: no graphics found")
    lines.append(f"Fig. {n}: {os.path.basename(dst)}  <- {p}")
    print(lines[-1])
open(os.path.join(out, "captions.txt"), "w", encoding="utf-8").write(
    "\n".join(f"Fig. {n}. {re.sub(chr(10), ' ', (re.search(r'\\caption{(.*?)}\s*\\label', b, re.S) or [None, ''])[1]).strip()}"
              for n, b in enumerate(blocks, 1)) + "\n")
