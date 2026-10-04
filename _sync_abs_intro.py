from pathlib import Path

tex_path = Path(
    r"c:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1\new\src\paper\overleaf_upload\main.tex"
)
ltx_path = Path(r"c:\Users\Saeed\OneDrive\Desktop\ABR-Awareness-1\new\src\paper\main.ltx")
tex = tex_path.read_text(encoding="utf-8")
ltx = ltx_path.read_text(encoding="utf-8")
marker_start = "\\begin{abstract}"
marker_end = "% =========================================\n\\section{Related work}"
s1, e1 = tex.index(marker_start), tex.index(marker_end)
s2, e2 = ltx.index(marker_start), ltx.index(marker_end)
block = tex[s1:e1]
ltx_path.write_text(ltx[:s2] + block + ltx[e2:], encoding="utf-8")
# approximate word count
import re

abs_body = re.search(
    r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S
).group(1)
plain = re.sub(r"\\[A-Za-z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", abs_body)
plain = re.sub(r"[{}$\\]", " ", plain)
print("synced", ltx_path)
print("abstract_words_approx", len(plain.split()))
print("intro_has_quant_pointer", "Quantitative results appear" in block)
print("intro_has_no_4.5", "4.5\\%" not in block)
