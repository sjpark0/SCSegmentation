#!/usr/bin/env python3
"""Render REPORT.md as a navigable audit page."""
import html
import os
import re

D = os.path.dirname(os.path.abspath(__file__))
MD = open(os.path.join(D, "REPORT.md")).read()

CAT = {
    "C": ("정확성", "correctness"),
    "A": ("알고리즘 설계", "algorithm"),
    "M": ("메모리·성능", "memory"),
    "E": ("실험 설계", "evaluation"),
    "G": ("엔지니어링", "engineering"),
}


def inline(s):
    """Markdown inline -> HTML, code spans escaped."""
    out, i = [], 0
    for m in re.finditer(r"`([^`]+)`", s):
        out.append(_emph(html.escape(s[i:m.start()])))
        out.append(f"<code>{html.escape(m.group(1))}</code>")
        i = m.end()
    out.append(_emph(html.escape(s[i:])))
    return "".join(out)


def _emph(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") or "s"


# ------------------------------------------------------------------ parse
lines = MD.split("\n")
blocks, i = [], 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("| "):                                   # table
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(lines[i]); i += 1
        blocks.append(("table", rows)); continue
    if ln.startswith("```"):                                   # fence
        body, i = [], i + 1
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i]); i += 1
        i += 1
        blocks.append(("code", body)); continue
    if ln.startswith("> "):
        body = []
        while i < len(lines) and lines[i].startswith("> "):
            body.append(lines[i][2:]); i += 1
        blocks.append(("quote", body)); continue
    if ln.startswith("---") and ln.strip("- ") == "":
        blocks.append(("rule", None)); i += 1; continue
    m = re.match(r"^(#{1,3})\s+(.*)$", ln)
    if m:
        lvl, txt = len(m.group(1)), m.group(2)
        p = re.match(r"^(P\d+)\.\s*(.+)$", txt)
        blocks.append(("prop-h", p.groups()) if (lvl == 3 and p) else ("h%d" % lvl, txt))
        i += 1; continue
    if ln.startswith("- ") or ln.startswith("1. "):
        items = []
        while i < len(lines) and (lines[i].startswith("- ") or re.match(r"^\d+\. ", lines[i])):
            items.append(re.sub(r"^(- |\d+\. )", "", lines[i])); i += 1
        blocks.append(("list", items)); continue
    if not ln.strip():
        i += 1; continue
    # paragraph, possibly a finding record
    para = [ln]; i += 1
    while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,3} |\||```|> |- |\d+\. |---\s*$)", lines[i]):
        para.append(lines[i]); i += 1
    joined = " ".join(para)
    f = re.match(r"^\*\*([CAMEG]\d+)\.\s*(.*?)\*\*(.*)$", joined, re.S)
    blocks.append(("finding", (f.group(1), f.group(2), f.group(3).strip())) if f
                  else ("p", joined))

# ------------------------------------------------------------------ render
findings, proposals, sections = [], [], []
body = []
cur_prop = None

for kind, val in blocks:
    if kind == "h1":
        sid = slug(val)
        sections.append((sid, val))
        body.append(f'<h2 id="{sid}">{inline(val)}</h2>')
    elif kind == "h2":
        body.append(f'<h3 class="sub">{inline(val)}</h3>')
    elif kind == "h3":
        body.append(f'<h4>{inline(val)}</h4>')
    elif kind == "prop-h":
        pid, title = val
        cons = ""
        cm = re.search(r"심사\s*(\d)/3", title)
        if cm:
            n = cm.group(1)
            cons = f'<span class="chip cons c{n}">심사 {n}/3</span>'
        proposals.append((pid, re.sub(r"\s*\(.*$", "", title)))
        body.append(
            f'<article class="rec prop" id="{pid}">'
            f'<div class="rec-head"><span class="chip pid">{pid}</span>'
            f'{cons}<h4>{inline(re.sub(r"^\\(|\\)$", "", title))}</h4></div>')
        cur_prop = pid
    elif kind == "finding":
        fid, title, rest = val
        k = fid[0]
        label, cls = CAT[k]
        findings.append((fid, title, cls))
        brief = ""
        t = title
        bm = re.match(r"^\[(brief[^\]]*)\]\s*(.*)$", t)
        if bm:
            brief = f'<span class="chip brief">{html.escape(bm.group(1))}</span>'
            t = bm.group(2)
        body.append(
            f'<article class="rec find {cls}" id="{fid}">'
            f'<div class="rec-head"><span class="chip fid {cls}">{fid}</span>{brief}'
            f'<h4>{inline(t)}</h4></div>'
            f'<p>{inline(rest)}</p></article>')
    elif kind == "p":
        if cur_prop:
            body.append(f"<p>{inline(val)}</p></article>")
            cur_prop = None
        else:
            body.append(f"<p>{inline(val)}</p>")
    elif kind == "list":
        items = "".join(f"<li>{inline(x)}</li>" for x in val)
        body.append(f"<ul>{items}</ul>")
        if cur_prop:
            body.append("</article>")
            cur_prop = None
    elif kind == "code":
        body.append(f'<pre><code>{html.escape(chr(10).join(val))}</code></pre>')
    elif kind == "quote":
        body.append(f'<aside class="callout">{inline(" ".join(val))}</aside>')
    elif kind == "table":
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in val]
        head, rows = cells[0], [r for r in cells[2:]]
        th = "".join(f"<th>{inline(c)}</th>" for c in head)
        tr = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                     for r in rows)
        body.append(f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
                    f"<tbody>{tr}</tbody></table></div>")
    elif kind == "rule":
        body.append('<hr>')

# nav rail
by_cat = {}
for fid, title, cls in findings:
    by_cat.setdefault(cls, []).append((fid, title))
nav = []
for k, (label, cls) in CAT.items():
    if cls not in by_cat:
        continue
    chips = "".join(f'<a class="navchip {cls}" href="#{fid}" title="{html.escape(re.sub(r"<[^>]+>", "", t))}">{fid}</a>'
                    for fid, t in by_cat[cls])
    nav.append(f'<div class="navgroup"><span class="navlabel">{label}</span>'
               f'<div class="navchips">{chips}</div></div>')
pnav = "".join(f'<a class="navchip prop" href="#{pid}" title="{html.escape(re.sub(r"<[^>]+>", "", t))}">{pid}</a>'
               for pid, t in proposals)
nav.append(f'<div class="navgroup"><span class="navlabel">개선 제안</span>'
           f'<div class="navchips">{pnav}</div></div>')
NAV = "".join(nav)
SECNAV = "".join(f'<a href="#{sid}">{html.escape(t)}</a>' for sid, t in sections)

TPL = open(os.path.join(D, "report_template.html")).read()
out = (TPL.replace("__BODY__", "\n".join(body))
          .replace("__NAV__", NAV)
          .replace("__SECNAV__", SECNAV)
          .replace("__NFIND__", str(len(findings)))
          .replace("__NPROP__", str(len(proposals))))
open(os.path.join(D, "report.html"), "w").write(out)
print("findings", len(findings), "proposals", len(proposals),
      "sections", len(sections), "bytes", len(out))
