# -*- coding: utf-8 -*-
import pymupdf
from pathlib import Path

pdf = Path(
    r"C:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1"
    r"\Conformal_Perceptual_Shielding_for_Adaptive_Bitrate_Streaming__20261002-(v2) (2).pdf"
)
out_path = Path(
    r"C:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1\pdf_v2_2_system_comments.md"
)

doc = pymupdf.open(pdf)
lines = [f"# PDF: {pdf.name}\n", f"- Pages: {doc.page_count}\n"]
total = 0

for i, page in enumerate(doc):
    anns = list(page.annots() or [])
    if not anns:
        continue
    total += len(anns)
    text = page.get_text("text") or ""
    head = " | ".join(text.splitlines()[:8])[:200]
    lines.append(f"\n## Page {i + 1}\n")
    lines.append(f"_head:_ {head}\n")
    for j, a in enumerate(anns, 1):
        kind = a.type[1] if a.type else "?"
        info = a.info or {}
        content = (info.get("content") or "").strip()
        try:
            words = (page.get_textbox(a.rect) or "").strip()
        except Exception:
            words = ""
        y = a.rect.y0 if a.rect else 0.0
        lines.append(f"\n### {j}. **{kind}** y={y:.1f}\n")
        if words:
            lines.append(f"> TEXT: {words[:400].replace(chr(10), ' ')}\n")
        if content:
            lines.append(f"\n> NOTE: {content}\n")
        try:
            clip = pymupdf.Rect(
                0,
                max(0, a.rect.y0 - 25),
                page.rect.width,
                min(page.rect.height, a.rect.y1 + 45),
            )
            near = (page.get_textbox(clip) or "").strip().replace("\n", " | ")[:280]
            lines.append(f"\n> NEAR: {near}\n")
        except Exception:
            pass
        lines.append("\n---\n")

lines.append(f"\nTotal annotations: {total}\n")
out_path.write_text("".join(lines), encoding="utf-8")
print(f"wrote {out_path} total={total}")
for i, page in enumerate(doc):
    n = len(list(page.annots() or []))
    if n:
        print(f"page {i + 1}: {n}")
