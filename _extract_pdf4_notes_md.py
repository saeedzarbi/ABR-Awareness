# -*- coding: utf-8 -*-
"""Extract PDF annotations from __4.pdf into Markdown."""
from pathlib import Path
from collections import defaultdict

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

SUBTYPE_LABEL = {
    "/Text": "Sticky note",
    "/FreeText": "Free text",
    "/Highlight": "Highlight",
    "/Underline": "Underline",
    "/StrikeOut": "Strikeout",
    "/Squiggly": "Squiggly",
    "/Caret": "Caret",
    "/Stamp": "Stamp",
    "/Ink": "Ink",
}

root = Path(r"C:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1")
pdf_name = "Certified_Perceptual_Shielding_for_Adaptive_Bitrate_Streaming__4.pdf"
path = root / pdf_name
out_path = root / "pdf4_comments.md"

r = PdfReader(str(path))
notes = []  # (page, subtype, author, contents)
marks = []  # (page, subtype)

for i, p in enumerate(r.pages):
    ann = p.get("/Annots")
    if not ann:
        continue
    for a in (ann.get_object() if hasattr(ann, "get_object") else ann):
        obj = a.get_object() if hasattr(a, "get_object") else a
        st = str(obj.get("/Subtype", ""))
        if st in ("/Link", "/Popup"):
            continue
        c = (obj.get("/Contents") or "").strip()
        author = str(obj.get("/T") or "").strip()
        if c:
            notes.append((i + 1, st, author, c))
        elif st in ("/Highlight", "/StrikeOut", "/Squiggly", "/Caret", "/Underline"):
            marks.append((i + 1, st))

# Group notes by page
by_page = defaultdict(list)
for pg, st, author, c in notes:
    by_page[pg].append((st, author, c))

lines = []
lines.append(f"# PDF Comments: `{pdf_name}`")
lines.append("")
lines.append(f"- **Pages:** {len(r.pages)}")
lines.append(f"- **Text annotations:** {len(notes)}")
lines.append(f"- **Mark-only (no text):** {len(marks)}")
lines.append("")
lines.append("---")
lines.append("")

for pg in sorted(by_page):
    lines.append(f"## Page {pg}")
    lines.append("")
    for idx, (st, author, c) in enumerate(by_page[pg], 1):
        label = SUBTYPE_LABEL.get(st, st.lstrip("/"))
        meta = f"**{label}**"
        if author:
            meta += f" · {author}"
        lines.append(f"### {idx}. {meta}")
        lines.append("")
        # Preserve multi-line comments as blockquotes
        for para in c.split("\n"):
            lines.append(f"> {para}" if para.strip() else ">")
        lines.append("")
    lines.append("---")
    lines.append("")

if marks:
    lines.append("## Mark-only annotations (no comment text)")
    lines.append("")
    by_mark = defaultdict(list)
    for pg, st in marks:
        by_mark[pg].append(SUBTYPE_LABEL.get(st, st.lstrip("/")))
    for pg in sorted(by_mark):
        kinds = ", ".join(by_mark[pg])
        lines.append(f"- **Page {pg}:** {kinds}")
    lines.append("")

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out_path}")
print(f"pages={len(r.pages)} text_notes={len(notes)} mark_only={len(marks)}")
