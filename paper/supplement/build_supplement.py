"""Build the scientific supplement from public sources and generated tables.

Requires pandoc and tectonic in PATH. Numerical regeneration is a separate step:
PY=/absolute/path/to/python sh paper/combined/build.sh

Portable fonts are used by default. To use installed system fonts, for example:
python3 paper/supplement/build_supplement.py --main-font Arial --mono-font Menlo
"""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCES = ["algebra-counters.md", "methods-accuracy.md", "accuracy.md",
           "outcomes-scale.md", "provenance-audit.md", "hypothesis-registry.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-font", default="texgyreheros-regular.otf",
                        help="main text font (default: %(default)s)")
    parser.add_argument("--mono-font", default="lmmono10-regular.otf",
                        help="monospaced font (default: %(default)s)")
    args = parser.parse_args()
    # Explicit bundle filenames avoid dependence on system font discovery.
    main_options = ""
    if args.main_font == "texgyreheros-regular.otf":
        main_options = """mainfontoptions:
  - BoldFont=texgyreheros-bold.otf
  - ItalicFont=texgyreheros-italic.otf
  - BoldItalicFont=texgyreheros-bolditalic.otf
"""
    front = f"""---
title: 'Supporting Information: matrix-free voxel elasticity'
author: 'Abderrazzak Nejeoui and Aissam Bekkari'
date: '27 September 2026'
fontsize: 10pt
geometry: [a4paper, landscape, margin=18mm]
mainfont: {json.dumps(args.main_font)}
{main_options}monofont: {json.dumps(args.mono_font)}
colorlinks: true
toc: true
---

This supporting document accompanies the matrix-free voxel elasticity study. It contains
operator construction, methods, accuracy, outcome, capacity, comparator, audit
and hypothesis evidence. Numerical regeneration and selected CPU checks use
retained records; no new GPU performance experiment is represented. Source
paths are relative to the accompanying artifact. CSV/JSON files retain exact
per-observation values and references where PDF tables use rounded summaries.

"""
    parts = [front] + [(HERE / p).read_text() for p in SOURCES]
    parts += [(HERE.parent / "combined/reproducibility_record.md").read_text()]
    joined = "\n\n".join(parts)
    # Companion record lives one directory deeper in the assembled supplement.
    joined = joined.replace("../supplement/", "")
    joined = joined.replace("10⁸–10⁹", "$10^8$–$10^9$")
    joined = joined.replace("LDLᵀ", "$LDL^T$")
    (HERE / "supporting-information.md").write_text(joined)
    cmd = ["pandoc", "supporting-information.md", "--from=markdown+tex_math_single_backslash",
           "--standalone", "--lua-filter=wrap-code.lua", "--include-in-header=supplement-header.tex",
           "--include-after-body=outcomes-scale-tables.tex", "-o", "supporting-information.tex"]
    subprocess.run(cmd, cwd=HERE, check=True)
    subprocess.run(["tectonic", "-X", "compile", "supporting-information.tex", "--keep-logs"], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
