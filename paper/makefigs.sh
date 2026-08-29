#!/bin/sh
cd "$(dirname "$0")"
for f in fig_boundary fig_one_curve fig_helm fig_correction; do
  printf '\\def\\figname{%s}\n\\input{figwrap.tex}\n' "$f" > "build_$f.tex"
  tectonic "build_$f.tex" && mv "build_$f.pdf" "figures/$f.pdf"
  rm -f "build_$f.tex"
done
