#!/usr/bin/env python3
"""Baseline comparison figure: per-view independent SAM 3 vs. the proposed method.

usage: python3 make_baseline_figure.py <out.png> [ko|en] [cap|nocap]

Row (a) comes from run_independent_baseline.py - SAM 3 run separately on every
view, ids assigned by each view's own detection order.  Row (b) is the demo
output, where one reference mask is propagated to all views.
"""
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont

OURS = "/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew"
BASE = "/home/sjpark/Documents/SCSegmentation/figures/baseline_independent"

OUT = sys.argv[1] if len(sys.argv) > 1 else "figure_baseline.png"
LANG = sys.argv[2] if len(sys.argv) > 2 else "ko"
CAP = (sys.argv[3] if len(sys.argv) > 3 else "cap") == "cap"

VIEWS = [int(v) for v in os.environ.get("VIEWS", "0 1 2 3 4 5 6 7").split()]

BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

BAD = (204, 41, 54)
GOOD = (22, 138, 96)
INK = (24, 24, 27)
MUTED = (113, 113, 122)

T = {
    "ko": {
        "title": "시점 간 연결이 없으면 객체 ID가 시점마다 뒤바뀝니다",
        "subtitle": "32개 시점 중 서로 인접한 8개(V0–V7) · 장면은 거의 그대로입니다",
        "a_head": "(a) 시점별 독립 실행",
        "a_sub": "SAM 3를 시점마다 따로 실행\nID = 그 시점의 검출 순서",
        "b_head": "(b) 제안 방법",
        "b_sub": "기준 시점 1장에서 전파\nID = 모든 시점에서 동일",
        "note": "같은 색 = 같은 객체 ID.  (a)는 장면이 거의 같은데도 같은 사람의 색이 계속 바뀌고 검출 수도 5~7개로 흔들립니다.  (b)는 32개 시점 전체에서 색이 그대로 유지됩니다.",
        "det": "검출 %d",
    },
    "en": {
        "title": "Without a cross-view link, object identities are shuffled per view",
        "subtitle": "8 adjacent views (V0–V7) out of 32 · the scene barely moves between them",
        "a_head": "(a) Independent per view",
        "a_sub": "SAM 3 run separately on each view\nid = that view's detection order",
        "b_head": "(b) Ours",
        "b_sub": "propagated from one reference mask\nid = the same in every view",
        "note": "Same color = same object id. In (a) colors keep changing even though the scene barely moves, and the count wobbles between 5 and 7. In (b) they hold across all 32 views.",
        "det": "%d found",
    },
}[LANG]


def f(p, s):
    return ImageFont.truetype(p, s, index=0)


F_TITLE = f(BOLD, 54)
F_HEAD = f(BOLD, 34)
F_SUB = f(REG, 23)
F_CELL = f(REG, 22)
F_CHIP = f(BOLD, 20)
F_NOTE = f(REG, 26)
F_SUBTITLE = f(REG, 28)

# --------------------------------------------------------------------- layout
W = 3400
MARGIN = 40 if CAP else 24
LABELCOL = 440 if CAP else 0
COLS = len(VIEWS)
GAP = 10
CW = (W - 2 * MARGIN - LABELCOL - (COLS - 1) * GAP) // COLS
CH = round(CW * 9 / 16)
TOP = 232 if CAP else MARGIN
ROWLAB = 40 if CAP else 0            # strip under a row for the view label
RGAP = 26 if CAP else GAP
ROW_H = CH + ROWLAB
H = TOP + 2 * ROW_H + RGAP + (86 if CAP else MARGIN)

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

GX = MARGIN + LABELCOL


def paste(path, x, y, border, bw=4):
    img.paste(Image.open(path).resize((CW, CH), Image.LANCZOS), (x, y))
    d.rectangle([x - bw // 2, y - bw // 2, x + CW + bw // 2 - 1, y + CH + bw // 2 - 1],
                outline=border, width=bw)


def chip(x, y, text, bg):
    tw = d.textlength(text, font=F_CHIP)
    d.rounded_rectangle([x, y, x + tw + 26, y + F_CHIP.size + 18], radius=7, fill=bg)
    d.text((x + 13, y + 9), text, font=F_CHIP, fill="white")


det = {}
if os.path.exists(f"{BASE}/detections.json"):
    det = {int(k): v for k, v in json.load(open(f"{BASE}/detections.json")).items()}

if CAP:
    d.text((MARGIN, 40), T["title"], font=F_TITLE, fill=INK)
    d.text((MARGIN, 116), T["subtitle"], font=F_SUBTITLE, fill=MUTED)

for r, (src, col, head, sub) in enumerate((
        (BASE, BAD, T["a_head"], T["a_sub"]),
        (OURS, GOOD, T["b_head"], T["b_sub"]))):
    y = TOP + r * (ROW_H + RGAP)

    if CAP:
        d.text((MARGIN, y + 4), head, font=F_HEAD, fill=col)
        d.multiline_text((MARGIN, y + 52), sub, font=F_SUB, fill=MUTED, spacing=8)

    for c, v in enumerate(VIEWS):
        x = GX + c * (CW + GAP)
        path = f"{BASE}/{v}.png" if r == 0 else f"{OURS}/{v}/0.png"
        paste(path, x, y, col, bw=4)
        if not CAP:
            continue
        if r == 0 and v in det:
            chip(x + 12, y + 12, T["det"] % det[v]["n"], BAD)
        d.text((x + 2, y + CH + 8), f"V{v}", font=F_CELL, fill=MUTED)

if CAP:
    d.text((MARGIN, H - 66), T["note"], font=F_NOTE, fill=INK)

img.save(OUT)
print(OUT, img.size, f"{os.path.getsize(OUT) / 1e6:.1f} MB")
