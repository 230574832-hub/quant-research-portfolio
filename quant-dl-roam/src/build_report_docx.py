"""Render reports/深度学习横截面模型评测报告.md to a formatted DOCX.

Visual style follows quant-lab/src/report.py (Songti SC body, PingFang SC
headings, navy accent, banded tables) so the two sibling projects read as one
body of work.

    ./.venv/bin/python src/build_report_docx.py
    ./.venv/bin/python src/build_report_docx.py reports/StockMixer论文复现_NASDAQ.md
"""
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY, HEAD, MONO = "Songti SC", "PingFang SC", "Menlo"
NAVY, DARK, GREY = RGBColor(0x1F, 0x38, 0x64), RGBColor(0x22, 0x22, 0x22), RGBColor(0x66, 0x66, 0x66)
ACCENT = RGBColor(0xC0, 0x00, 0x00)

import sys

MD = "reports/深度学习横截面模型评测报告.md"
DOCX = "reports/深度学习横截面模型评测报告.docx"


def set_font(run, name=BODY, size=10.5, bold=False, color=DARK, italic=False):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.italic = italic


def cell_shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def borders(table, color="B7C4D6"):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{e}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        b.append(el)
    tblPr.append(b)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def add_inline(par, text, size=10.5, color=DARK, base_bold=False):
    """Render **bold**, `code` and [text](url) inline."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            r = par.add_run(piece[2:-2])
            set_font(r, HEAD, size, bold=True, color=NAVY)
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1])
            set_font(r, MONO, size - 0.5, color=ACCENT)
        elif piece.startswith("[") and "](" in piece:
            txt = piece[1:piece.index("](")]
            r = par.add_run(txt)
            set_font(r, BODY, size, color=RGBColor(0x2E, 0x75, 0xB6))
        else:
            r = par.add_run(piece)
            set_font(r, BODY, size, bold=base_bold, color=color)


def is_table_row(l):
    return l.startswith("|") and l.endswith("|")


def split_row(l):
    return [c.strip() for c in l.strip().strip("|").split("|")]


def is_sep(l):
    return is_table_row(l) and set(l.replace("|", "").replace(" ", "")) <= set("-:")


def convert(md, doc):
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        l = lines[i]
        s = l.strip()

        if not s:
            i += 1
            continue

        if s.startswith("---") and set(s) <= set("-"):
            i += 1
            continue

        # image
        m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
        if m:
            path = m.group(2)
            if os.path.exists(path):
                doc.add_picture(path, width=Cm(16.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            sizes = {1: 17, 2: 14, 3: 12, 4: 11}
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14 if lvl <= 2 else 9)
            p.paragraph_format.space_after = Pt(6)
            add_inline(p, txt, size=sizes[lvl], base_bold=True)
            for r in p.runs:
                set_font(r, HEAD, sizes[lvl], bold=True,
                         color=NAVY if lvl <= 3 else DARK)
            i += 1
            continue

        # table
        if is_table_row(l) and i + 1 < len(lines) and is_sep(lines[i + 1]):
            header = split_row(l)
            body = []
            j = i + 2
            while j < len(lines) and is_table_row(lines[j]):
                body.append(split_row(lines[j]))
                j += 1
            t = doc.add_table(rows=1, cols=len(header))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            borders(t)
            for k, h in enumerate(header):
                c = t.rows[0].cells[k]
                c.text = ""
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, h, size=9, base_bold=True)
                for r in p.runs:
                    set_font(r, HEAD, 9, bold=True, color=NAVY)
                cell_shade(c, "E8EEF7")
            for bi, row in enumerate(body):
                cells = t.add_row().cells
                for k in range(len(header)):
                    val = row[k] if k < len(row) else ""
                    cells[k].text = ""
                    p = cells[k].paragraphs[0]
                    p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if k == 0
                                   else WD_ALIGN_PARAGRAPH.RIGHT)
                    add_inline(p, val, size=9)
                    for r in p.runs:
                        if not (r.font.name == MONO):
                            set_font(r, BODY, 9)
                if bi % 2 == 1:
                    for c in cells:
                        cell_shade(c, "F5F8FC")
            doc.add_paragraph()
            i = j
            continue

        # bullet list
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            ordered = bool(re.match(r"^\d+\.\s+", s))
            txt = re.sub(r"^([-*]|\d+\.)\s+", "", s)
            p = doc.add_paragraph(style="List Number" if ordered else "List Bullet")
            p.paragraph_format.space_after = Pt(3)
            add_inline(p, txt)
            i += 1
            continue

        # blockquote
        if s.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            add_inline(p, s.lstrip("> ").strip(), size=9.5, color=GREY)
            for r in p.runs:
                set_font(r, BODY, 9.5, color=GREY)
            i += 1
            continue

        # paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.25
        add_inline(p, s)
        i += 1


def main():
    global MD, DOCX
    if len(sys.argv) > 1:                 # 可指定其它 markdown -> 同名 docx
        MD = sys.argv[1]
        DOCX = os.path.splitext(sys.argv[1])[0] + ".docx"
    doc = Document()
    for name, size in (("Normal", 10.5),):
        st = doc.styles[name]
        st.font.name = BODY
        st.font.size = Pt(size)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.0)

    convert(open(MD).read(), doc)
    doc.save(DOCX)
    print("wrote", DOCX, "%.1f KB" % (os.path.getsize(DOCX) / 1024))


if __name__ == "__main__":
    main()
