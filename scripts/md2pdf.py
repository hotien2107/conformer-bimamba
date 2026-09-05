#!/usr/bin/env python3
"""Convert project Markdown docs to PDF with Unicode/Vietnamese support.

Pure-Python pipeline: markdown source -> PDF via fpdf2 + DejaVu fonts
(no pandoc/LaTeX needed). Handles the subset of Markdown used by docs/:
headings, fenced code blocks, pipe tables, bullet/numbered lists, blockquotes,
paragraphs with **bold** / `code` / *italic* / [text](url), horizontal rules,
plus page footers.

Usage:
    python scripts/md2pdf.py docs/code_walkthrough_vi.md \
        docs/code_walkthrough_vi.pdf \
        --title "Hybrid Conformer-BiMamba — Hướng dẫn đọc code"
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from fpdf import FPDF
from fontTools.ttLib import TTFont

FONT_DIR = Path("/tmp/cw_fonts")
# Chỉ thay những gì font KHÔNG phủ: emoji / số tròn / tổ hợp dấu. Các ký hiệu
# toán (→ × · ≈ ≤ ‖ ² ³ Ā ā ⟨ ⟩ − ½ ± √ ∈ Σ) được GIỮ NGUYÊN để văn bản tự nhiên.
REPL = {
    "📐": "", "💻": "", "🔁": "",
    "s\u0302": "s^", "B\u0304": "B~", "t\u0302": "t^",
}
for n in range(1, 21):
    REPL[chr(0x2460 + n - 1)] = f"({n})"  # ① ② ... -> (1) (2) ...


def fix(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if i + 1 < len(s) and (s[i] + s[i + 1]) in REPL:
            out.append(REPL[s[i] + s[i + 1]])
            i += 2
            continue
        c = s[i]
        if c in REPL:
            out.append(REPL[c])
        elif 0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF:  # emoji
            out.append("")
        else:
            out.append(c)
        i += 1
    return "".join(out)


class PDF(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Trang {self.page_no()}", align="C")


class MdToPdf:
    def __init__(self, title: str, subtitle: str = ""):
        self.pdf = PDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=16)
        self.pdf.set_left_margin(16)
        self.pdf.set_right_margin(16)
        self.pdf.set_top_margin(16)
        self.pdf.add_font("DejaVu", "", FONT_DIR / "DejaVuSans.ttf")
        self.pdf.add_font("DejaVu", "B", FONT_DIR / "DejaVuSans-Bold.ttf")
        self.pdf.add_font("DejaVu", "I", FONT_DIR / "DejaVuSans-Oblique.ttf")
        self.pdf.add_font("DejaVu", "BI", FONT_DIR / "DejaVuSans-Bold.ttf")
        # mono family: prefer Noto Sans Mono (full Vietnamese coverage),
        # fall back to DejaVu Sans Mono
        mono_r = FONT_DIR / "NotoSansMono-Regular.ttf"
        mono_b = FONT_DIR / "NotoSansMono-Bold.ttf"
        if not (mono_r.exists() and mono_b.exists()):
            mono_r = FONT_DIR / "DejaVuSansMono.ttf"
            mono_b = FONT_DIR / "DejaVuSansMono-Bold.ttf"
        self.pdf.add_font("Mono", "", mono_r)
        self.pdf.add_font("Mono", "B", mono_b)
        self.pdf.add_page()
        self.title = title
        self.subtitle = subtitle

    # ------------------------------------------------------------- helpers
    @property
    def lw(self) -> float:  # usable line width (mm)
        return self.pdf.w - 32.0

    def emit(self, text: str, font="DejaVu", style="", size=10.2, lh=None, indent=0.0):
        """Paragraph with inline **bold** / `code` / *italic* / [t](u)."""
        self.pdf.set_font(font, "", size)
        lh = lh or size * 0.46
        self.pdf.set_x(16 + indent)
        runs = self._inline_runs(text)
        for txt, st in runs:
            if st == "M":
                self.pdf.set_font("Mono", "", round(size * 0.95, 1))
            else:
                self.pdf.set_font(font, st, size)
            self.pdf.write(lh, fix(txt))
        self.pdf.ln(lh * 0.62)

    @staticmethod
    def _inline_runs(s: str):
        """Split into (text, style) runs; style in '', 'B', 'I', 'BI' or 'M'."""
        out = []
        # tokenise: `code`, **bold**, *italic*, [text](url)
        pat = re.compile(r"(`[^`]*`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]*\]\([^)]*\))")
        pos = 0
        for m in pat.finditer(s):
            if m.start() > pos:
                out.append((s[pos:m.start()], ""))
            tok = m.group(0)
            if tok.startswith("`"):
                out.append((tok[1:-1], "M"))
            elif tok.startswith("**"):
                out.append((tok[2:-2], "B"))
            elif tok.startswith("*"):
                out.append((tok[1:-1], "I"))
            else:  # link
                t, u = tok[1:-1].split("](", 1)
                out.append((f"{t} ({u[:-1]})", ""))
            pos = m.end()
        if pos < len(s):
            out.append((s[pos:], ""))
        return out

    def heading(self, text: str, level: int):
        # keep heading with its content: start a new page if too close to bottom
        bottom = self.pdf.h - 20
        if self.pdf.get_y() > bottom - (18 if level <= 2 else 14):
            self.pdf.add_page()
        if level == 1:
            self.pdf.ln(3)
            self.pdf.set_font("DejaVu", "B", 16)
            self.pdf.set_text_color(20, 40, 90)
            self.pdf.multi_cell(0, 7, fix(text))
            self.pdf.ln(2)
        elif level == 2:
            self.pdf.ln(3)
            self.pdf.set_font("DejaVu", "B", 13)
            self.pdf.set_text_color(20, 40, 90)
            self.pdf.multi_cell(0, 6, fix(text))
            self.pdf.ln(1.5)
        elif level == 3:
            self.pdf.set_font("DejaVu", "B", 11.8)
            self.pdf.set_text_color(40, 60, 110)
            self.pdf.multi_cell(0, 5.6, fix(text))
        else:
            self.pdf.set_font("DejaVu", "B", 10.4)
            self.pdf.set_text_color(60, 60, 60)
            self.pdf.multi_cell(0, 5.2, fix(text))
        self.pdf.set_text_color(0, 0, 0)

    def code_block(self, lines):
        self.pdf.set_font("Mono", "", 8.4)
        lh = 4.1
        self.pdf.set_fill_color(243, 245, 248)
        self.pdf.set_x(16)
        self.pdf.multi_cell(self.lw, lh, fix("\n".join(lines)), fill=True)
        self.pdf.ln(1.8)

    def quote(self, lines):
        self.pdf.set_font("DejaVu", "I", 9.5)
        self.pdf.set_text_color(90, 90, 90)
        self.pdf.set_x(20)
        self.pdf.multi_cell(self.lw - 4, 4.6, fix("\n".join(lines)))
        self.pdf.ln(1)
        self.pdf.set_text_color(0, 0, 0)

    def bullet(self, text: str, ordered: str | None = None):
        prefix = f"{ordered}. " if ordered else "• "
        self.pdf.set_font("DejaVu", "", 10)
        self.pdf.set_x(19)
        self.pdf.write(4.4, fix(prefix))
        runs = self._inline_runs(text)
        for txt, st in runs:
            self.pdf.set_font("DejaVu" if st != "M" else "Mono", st if st != "M" else "", 10 if st != "M" else 8.6)
            self.pdf.write(4.4, fix(txt))
        self.pdf.ln(4.6)

    # ---------------------------------------------------------------- table
    def table(self, rows):
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        header = rows[0]
        body = rows[1:]

        weights = []
        for c in range(ncol):
            w = 1.0 + max(len(fix(r[c])) for r in rows) * 0.26  # cân bằng cột hơn
            weights.append(w)
        total = sum(weights)
        col_w = [self.lw * w / total for w in weights]

        line_h = 4.9
        pad = 1.8
        cell_size = 9.5

        def draw_row(cells, is_header, y0, stripe=False):
            # estimate row height (worst cell wrap)
            est = 1
            for c, txt in enumerate(cells):
                self.pdf.set_font("DejaVu", "B" if is_header else "", cell_size)
                n_lines = 1
                words = fix(txt).split(" ")
                cur = ""
                for wd in words:
                    trial = (cur + " " + wd).strip()
                    if self.pdf.get_string_width(trial) > col_w[c] - 2 * pad:
                        n_lines += 1
                        cur = wd
                    else:
                        cur = trial
                est = max(est, n_lines)
            row_h = est * line_h + 1.6
            if y0 + row_h > self.pdf.h - 16:  # page break
                self.pdf.add_page()
                y0 = self.pdf.get_y()
                draw_row(header, True, y0)
            x0 = 16
            if is_header:
                self.pdf.set_fill_color(222, 231, 246)
            else:
                self.pdf.set_fill_color(247, 249, 252 if stripe else 253)
            self.pdf.set_text_color(0, 0, 0)
            for c, txt in enumerate(cells):
                self.pdf.set_xy(x0 + sum(col_w[:c]), y0)
                self.pdf.set_font("DejaVu", "B" if is_header else "", cell_size)
                self.pdf.multi_cell(
                    col_w[c], line_h, fix(txt), border=0, fill=True,
                    new_x="RIGHT", new_y="TOP",
                )
                self.pdf.set_draw_color(208, 212, 220)
                self.pdf.rect(x0 + sum(col_w[:c]), y0, col_w[c], row_h)
            self.pdf.set_y(y0 + row_h)
            return row_h

        y = self.pdf.get_y() + 1
        if y + 8 > self.pdf.h - 16:
            self.pdf.add_page()
            y = self.pdf.get_y()
        draw_row(header, True, y)
        for k, r in enumerate(body):
            y = self.pdf.get_y()
            draw_row(r, False, y, stripe=(k % 2 == 0))
        self.pdf.ln(3)

    # --------------------------------------------------------------- driver
    def render(self, md: str):
        lines = md.split("\n")
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]

            if line.startswith("```"):
                buf = []
                i += 1
                while i < n and not lines[i].startswith("```"):
                    buf.append(lines[i])
                    i += 1
                self.code_block(buf)
                i += 1
                continue

            if line.startswith("|"):
                rows = []
                while i < n and lines[i].startswith("|"):
                    cells = [c.strip() for c in lines[i].strip("|").split("|")]
                    if not re.fullmatch(r":?-{3,}:?", cells[0]) and not all(
                        re.fullmatch(r":?-{3,}:?", c) for c in cells if c
                    ):
                        rows.append(cells)
                    i += 1
                self.table(rows)
                continue

            if re.match(r"^#{1,6}\s", line):
                m = re.match(r"^(#{1,6})\s+(.*)$", line)
                self.heading(m.group(2), len(m.group(1)))
                i += 1
                continue

            if re.match(r"^---+$", line):
                i += 1
                continue

            if line.startswith(">"):
                buf = []
                while i < n and lines[i].startswith(">"):
                    buf.append(lines[i].lstrip("> ").strip())
                    i += 1
                self.quote(buf)
                continue

            if re.match(r"^\s*[-*]\s+", line):
                while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                    self.bullet(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                    i += 1
                self.pdf.ln(1)
                continue

            if re.match(r"^\s*\d+\.\s+", line):
                while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                    m = re.match(r"^\s*(\d+)\.\s+(.*)$", lines[i])
                    self.bullet(m.group(2), ordered=m.group(1))
                    i += 1
                self.pdf.ln(1)
                continue

            if line.strip() == "":
                i += 1
                continue

            # paragraph (may continue until blank line)
            buf = [line.strip()]
            i += 1
            while i < n and lines[i].strip() and not lines[i].startswith(("#", "|", "```", ">", "- ", "* ", "---")):
                buf.append(lines[i].strip())
                i += 1
            self.emit(" ".join(buf))

    def finish(self, out: Path):
        # small title block on first page
        pdf = self.pdf
        pdf.set_y(20)
        pdf.set_font("DejaVu", "B", 17)
        pdf.set_text_color(15, 30, 70)
        pdf.multi_cell(0, 8, fix(self.title), align="L")
        if self.subtitle:
            pdf.set_font("DejaVu", "I", 9.5)
            pdf.set_text_color(120, 120, 120)
            pdf.ln(1)
            pdf.multi_cell(0, 5, fix(self.subtitle))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        pdf.set_y(pdf.get_y())
        out.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    md = args.src.read_text(encoding="utf-8")
    title = args.title or args.src.stem.replace("_", " ").title()
    conv = MdToPdf(title, subtitle=f"Nguồn: {args.src} · tạo tự động từ Markdown")
    conv.render(md)
    conv.finish(args.dst)
    print(f"OK -> {args.dst}")


if __name__ == "__main__":
    main()
