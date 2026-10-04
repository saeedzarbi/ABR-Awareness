import pymupdf
from pathlib import Path

pdf_path = Path(
    r"C:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1"
    r"\Conformal_Perceptual_Shielding_for_Adaptive_Bitrate_Streaming__20261002-(v2).pdf"
)
out = Path(r"C:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1\pdf_v2_20261002_comments.md")
doc = pymupdf.open(pdf_path)
lines: list[str] = []
lines.append(
    "# PDF Comments: `Conformal_Perceptual_Shielding_for_Adaptive_Bitrate_Streaming__20261002-(v2).pdf`"
)
lines.append("")
lines.append(f"- **Pages:** {doc.page_count}")
lines.append("")
total = 0
for i, page in enumerate(doc):
    annos = list(page.annots() or [])
    if not annos:
        continue
    lines.append(f"## Page {i + 1}")
    lines.append("")
    n = 0
    for a in annos:
        info = a.info or {}
        content = (info.get("content") or "").strip()
        title = (info.get("title") or "").strip()
        subtype = a.type[1] if a.type else "?"
        ht = ""
        try:
            if subtype in ("Highlight", "Underline", "StrikeOut", "Squiggly"):
                ht = page.get_text("text", clip=a.rect).strip().replace("\n", " ")
        except Exception:
            ht = ""
        total += 1
        n += 1
        lines.append(f"### {n}. **{subtype}** · {title or '?'}")
        lines.append("")
        if ht:
            lines.append(f"> TEXT: {ht}")
            lines.append("")
        if content:
            lines.append(f"> NOTE: {content}")
            lines.append("")
        if not ht and not content:
            lines.append("> (mark-only, no extractable text)")
            lines.append("")
        lines.append("---")
        lines.append("")

lines.append("## Page 1 full text (for Abstract mapping)")
lines.append("")
lines.append("```")
lines.append(doc[0].get_text("text"))
lines.append("```")
lines.append("")
lines.append(f"Total annotations: {total}")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out} annos={total}")
