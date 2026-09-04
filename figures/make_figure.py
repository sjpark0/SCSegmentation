#!/usr/bin/env python3
"""1-mask -> all-views teaser figure for SCSam3 (demoSCSam3OneStageNew).

usage: python3 make_figure.py <out.png> [ko|en] [cap|nocap]

  cap    panel headings, view labels and legend are drawn on the figure
  nocap  images only - no text anywhere, for a figure whose caption is
         written in LaTeX / on the slide
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

SRC = "/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew"
OUT = sys.argv[1] if len(sys.argv) > 1 else "figure_one_mask.png"
LANG = sys.argv[2] if len(sys.argv) > 2 else "ko"
CAP = (sys.argv[3] if len(sys.argv) > 3 else "cap") == "cap"

BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

N_VIEWS = 32
REF_VIEW, OPP_VIEW = 0, 16

MANUAL = (204, 41, 54)
AUTO = (22, 138, 96)
OPP = (222, 132, 20)
INK = (24, 24, 27)
MUTED = (113, 113, 122)
ARROW = (168, 168, 175)

T = {
    "ko": {
        "title": "마스크 1장으로 32개 시점 전체를 자동 분할",
        "a_head": "(a) 사람이 만드는 마스크 — 단 1장",
        "a_sub": "기준 시점 V0 · 첫 프레임 t=0",
        "a_badge": "MANUAL  ×1",
        "a_note": "여기까지가 사람의 작업 전부입니다.",
        "arrow": "시점 전파",
        "arrow_sub": "Spatial propagation",
        "b_head": "(b) 자동 생성된 마스크 — 나머지 31개 시점",
        "b_sub": "피사체를 둘러싼 32개 카메라의 첫 프레임 · 시점이 바뀌어도 객체 ID(색상) 유지",
        "lg_manual": "사람이 지정 (1장)",
        "lg_auto": "자동 생성",
        "lg_opp": "기준 시점의 반대편 — 가장 어려운 경우",
        "ref": "기준·수동",
        "opp": "반대편",
    },
    "en": {
        "title": "One mask segments all 32 views",
        "a_head": "(a) Human-drawn mask — only one",
        "a_sub": "reference view V0 · first frame t=0",
        "a_badge": "MANUAL  x1",
        "a_note": "This is the entire human effort.",
        "arrow": "Spatial propagation",
        "arrow_sub": "automatic",
        "b_head": "(b) Automatically generated masks — the other 31 views",
        "b_sub": "first frame of 32 cameras surrounding the subject · object IDs (colors) stay consistent",
        "lg_manual": "human-annotated (1)",
        "lg_auto": "automatic",
        "lg_opp": "opposite the reference view — hardest case",
        "ref": "ref / manual",
        "opp": "opposite",
    },
}[LANG]


def f(path, size):
    return ImageFont.truetype(path, size, index=0)


F_TITLE = f(BOLD, 54)
F_HEAD = f(BOLD, 36)
F_SUB = f(REG, 25)
F_BADGE = f(BOLD, 26)
F_CELL = f(REG, 20)
F_CELL_B = f(BOLD, 20)
F_ARROW = f(BOLD, 30)
F_ARROW_S = f(REG, 21)
F_NOTE = f(BOLD, 27)
F_LEG = f(REG, 23)

# --------------------------------------------------------------------- layout
# The two blocks are sized so the 4-row grid ends level with the reference
# panel; the caption version leaves room under every image for its label.
W = 3400
COLS, ROWS, GAP = 8, 4, 10
GUTTER = 280                       # space reserved for the arrow

if CAP:
    MARGIN, AY, AW = 40, 230, 1240
    LABEL = 52                     # label strip under each grid cell
else:
    MARGIN, AY, AW = 30, 30, 1032
    LABEL = 0

AX = MARGIN
AH = round(AW * 9 / 16)
GX = AX + AW + GUTTER
CW = (W - MARGIN - GX - (COLS - 1) * GAP) // COLS
CH = round(CW * 9 / 16)
RP = CH + (LABEL if CAP else GAP)
GRID_H = ROWS * RP - (0 if CAP else GAP)

H = AY + max(AH, GRID_H) + (82 if CAP else MARGIN)

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def paste(path, box, border, bw=4):
    x, y, w, h = box
    img.paste(Image.open(path).resize((w, h), Image.LANCZOS), (x, y))
    d.rectangle([x - bw // 2, y - bw // 2, x + w + bw // 2 - 1, y + h + bw // 2 - 1],
                outline=border, width=bw)


def badge(xy, text, bg, font=F_BADGE, padx=16, pady=8, r=8):
    x, y = xy
    tw = d.textlength(text, font=font)
    d.rounded_rectangle([x, y, x + tw + 2 * padx, y + font.size + 6 + 2 * pady],
                        radius=r, fill=bg)
    d.text((x + padx, y + pady), text, font=font, fill="white")


# ---------------------------------------------------------------- (a) input
if CAP:
    d.text((40, 44), T["title"], font=F_TITLE, fill=INK)
    d.text((AX, AY - 96), T["a_head"], font=F_HEAD, fill=MANUAL)
    d.text((AX, AY - 46), T["a_sub"], font=F_SUB, fill=MUTED)

paste(f"{SRC}/{REF_VIEW}/0.png", (AX, AY, AW, AH), MANUAL, bw=8)

if CAP:
    badge((AX + 24, AY + 24), T["a_badge"], MANUAL)
    d.text((AX, AY + AH + 26), T["a_note"], font=F_NOTE, fill=MANUAL)

# -------------------------------------------------------------------- arrow
X0, X1 = AX + AW + 66, GX - 66
YC = AY + AH // 2
HEAD = 84
d.polygon([(X0, YC - 28), (X1 - HEAD, YC - 28), (X1 - HEAD, YC - 56),
           (X1, YC), (X1 - HEAD, YC + 56), (X1 - HEAD, YC + 28),
           (X0, YC + 28)], fill=ARROW)
if CAP:
    d.text(((X0 + X1) // 2, YC - 124), T["arrow"], font=F_ARROW, fill=INK, anchor="ma")
    d.text(((X0 + X1) // 2, YC - 84), T["arrow_sub"], font=F_ARROW_S, fill=MUTED,
           anchor="ma")

# ----------------------------------------------------------- (b) all 32 views
if CAP:
    d.text((GX, AY - 96), T["b_head"], font=F_HEAD, fill=AUTO)
    d.text((GX, AY - 46), T["b_sub"], font=F_SUB, fill=MUTED)

for m in range(N_VIEWS):
    r, c = divmod(m, COLS)
    x, y = GX + c * (CW + GAP), AY + r * RP
    special = m in (REF_VIEW, OPP_VIEW)
    col = MANUAL if m == REF_VIEW else (OPP if m == OPP_VIEW else AUTO)
    paste(f"{SRC}/{m}/0.png", (x, y, CW, CH), col, bw=6 if special else 4)
    if not CAP:
        continue
    lab = f"V{m}"
    if m == REF_VIEW:
        lab += "  " + T["ref"]
    elif m == OPP_VIEW:
        lab += "  " + T["opp"]
    d.text((x + 2, y + CH + 7), lab, font=F_CELL_B if special else F_CELL,
           fill=col if special else MUTED)

if CAP:
    LGY = AY + ROWS * RP + 8
    lx = GX
    for col, txt in ((MANUAL, T["lg_manual"]), (AUTO, T["lg_auto"]), (OPP, T["lg_opp"])):
        d.rounded_rectangle([lx, LGY + 4, lx + 26, LGY + 26], radius=5, fill=col)
        d.text((lx + 38, LGY), txt, font=F_LEG, fill=INK)
        lx += 38 + d.textlength(txt, font=F_LEG) + 52

img.save(OUT)
print(OUT, img.size, f"{os.path.getsize(OUT)/1e6:.1f} MB")
