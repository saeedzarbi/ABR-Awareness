from pathlib import Path
from collections import defaultdict

def extract(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    r = PdfReader(str(path))
    notes, marks = [], []
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
            if c:
                notes.append((i + 1, st, c))
            elif st in ("/Highlight", "/StrikeOut", "/Squiggly", "/Caret"):
                marks.append((i + 1, st))
    return len(r.pages), notes, marks

root = Path(r"C:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1")
for fname in [
    "Certified_Perceptual_Shielding_for_Adaptive_Bitrate_Streaming__11_.pdf",
    "Certified_Perceptual_Shielding_for_Adaptive_Bitrate_Streaming__25 (1).pdf",
]:
    p = root / fname
    if not p.exists():
        continue
    pages, notes, marks = extract(p)
    lines = [f"FILE: {fname}", f"pages={pages}", f"text_notes={len(notes)}", f"mark_only={len(marks)}", ""]
    for pg, st, c in notes:
        lines.append(f"p{pg} {st}: {c}")
        lines.append("")
    if marks:
        by = defaultdict(list)
        for pg, st in marks:
            by[pg].append(st)
        lines.append("MARK_ONLY:")
        for pg in sorted(by):
            lines.append(f"  p{pg}: {by[pg]}")
    lines.append("=" * 60)
    lines.append("")
    (root / "_all_pdf_notes.txt").open("a", encoding="utf-8").write("\n".join(lines))
