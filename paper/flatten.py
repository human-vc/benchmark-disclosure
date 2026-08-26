"""Build the single-file Overleaf bundle.

The repository keeps sections and figures in separate files because they are
generated and reviewed separately. Overleaf does not benefit from that: it adds
a way for an upload to be incomplete, and a missing figure directory stops the
compile with an error that points at the wrong thing. So the bundle is one
main.tex with everything inlined, plus the bibliography and the style file.

Regenerate rather than editing the bundle, since the figures and tables inside
it are themselves written from the analysis output.
"""

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUT = PAPER.parent / "dist"


def inline(text):
    def swap(match):
        name = match.group(1)
        for candidate in (PAPER / f"{name}.tex", PAPER / name):
            if candidate.exists():
                return (f"% ---- begin {name} ----\n"
                        f"{inline(candidate.read_text().rstrip())}\n"
                        f"% ---- end {name} ----")
        sys.exit(f"cannot resolve \\input{{{name}}}")
    return re.sub(r"\\input\{([^}]+)\}", swap, text)


def main():
    flat = inline((PAPER / "main.tex").read_text())
    if "\\input{" in flat:
        sys.exit("unresolved input remains")

    stage = OUT / "overleaf"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    (stage / "main.tex").write_text(flat)
    for name in ("refs.bib", "neurips_2026.sty"):
        shutil.copy(PAPER / name, stage / name)
    (stage / "README.md").write_text(
        "# Absence Is Not Omission\n\n"
        "Three files, nothing to arrange. Set the root document to `main.tex`\n"
        "and compile with pdfLaTeX, which is Overleaf's default.\n\n"
        "The figures are TikZ and the tables are LaTeX, both written from the\n"
        "analysis output rather than typed, so edit them in the analysis\n"
        "repository. Anything changed here is lost the next time the bundle is\n"
        "rebuilt.\n\n"
        "Nine content pages, which is the long-track limit. References and the\n"
        "checklist do not count against it.\n\n"
        "On acceptance, change the style option to `[dblblindworkshop, final]`\n"
        "and add the author block. That is what puts the workshop name in the\n"
        "page-one footer; until then the footer correctly reads \"Submitted\n"
        "to ... Do not distribute.\"\n"
    )

    archive = OUT / "absence-overleaf.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(stage.iterdir()):
            bundle.write(item, item.name)

    print(f"wrote {archive}")
    for item in sorted(stage.iterdir()):
        print(f"  {item.name}  {item.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
