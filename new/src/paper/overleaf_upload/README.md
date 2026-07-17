# Overleaf upload — high-quality figures restored

## Upload

1. **New Project → Upload Project** → `ABR-VMAF-Shield-Overleaf.zip`
2. Main document: `main.tex`
3. Compiler: **pdfLaTeX**
4. Recompile

## Quality notes

- Architecture figure is **vector TikZ** again (`figures/fig_system_arch_tikz.tex`)
- Evaluation plots use original PDF assets at full `\evalfig` sizes
- Low-quality JPG/PNG architecture placeholders were removed

If compile **times out** on the free plan, upgrade or compile locally; do not reintroduce compressed raster architecture art.

## Layout

```
main.tex
references.bib
elsarticle-num.bst
latexmkrc
tables/
figures/   (PDF plots + TikZ architecture)
```
