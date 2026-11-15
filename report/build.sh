#!/usr/bin/env bash
# Build report/main.pdf: pdflatex + bibtex + pdflatex x2 (run inside report/).
# Needs IEEEtran.cls and siunitx.sty. If they are not installed, point TEXEXTRA at a TeX tree that has them, e.g.
#   TEXEXTRA=$HOME/texextra/root/usr/share/texlive/texmf-dist bash build.sh
cd "$(dirname "$0")"
if [ -n "$TEXEXTRA" ]; then
  export TEXINPUTS=".:$TEXEXTRA/tex/latex//:"
  export BSTINPUTS=".:$TEXEXTRA/bibtex/bst//:"
fi
python3 make_tables.py >/dev/null 2>&1 || python make_tables.py >/dev/null 2>&1 || echo "note: tables not regenerated (python missing); using the existing tables/"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { echo "pdflatex failed, see build.log"; tail -30 build.log; exit 1; }
bibtex main >> build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log 2>&1
grep -E "Overfull|Underfull|undefined|Warning" build.log | sort | uniq -c | sort -rn | head -40
echo "pages: $(pdfinfo main.pdf 2>/dev/null | awk '/Pages/ {print $2}')"
