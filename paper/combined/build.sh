#!/bin/sh
# Regenerate every number of the combined paper from verified evidence, then compile.
# Numbers come only from analysis outputs; main.tex contains no hand-typed result.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
(cd ../manuscript && ./build.sh > /dev/null)                     # short-paper macros, figures, tables
$PY ../../analysis/session_f_summary.py > /dev/null
$PY ../../analysis/session_g_tables.py ../../results/session-g-host1-20260926 ../../results/session-g-host2-20260926 > /dev/null
$PY ../../analysis/session_h_tables.py ../../results/session-h-host1-20260926 ../../results/session-h-host2-20260926 > /dev/null
$PY ../../analysis/session_i_tables.py ../../results/session-i-rtx4090-20260926 ../../results/session-i-a100-20260926 ../../results/session-i-h100-20260926 > /dev/null
$PY ../../analysis/session_j_counters.py > /dev/null
$PY ../../analysis/repro_numbers.py > /dev/null
$PY make_numbers.py > /dev/null
# The manuscript text (main.tex, abstract.tex, body.tex) is published after the paper is accepted.
# Until then the build stops here: every number, table and figure above has been regenerated.
if [ -f main.tex ]; then
  tectonic -X compile main.tex
  if grep -q '\\pending{' main.tex abstract.tex body.tex; then echo "WARNING: pending items remain:"; grep -ho '\\pending{[^}]*}' main.tex abstract.tex body.tex; fi
  cp main.pdf ../manuscript.pdf
  pdfinfo main.pdf | grep Pages
else
  echo "Regenerated: paper/combined/combined_numbers.tex, paper/combined/repro_numbers.tex,"
  echo "paper/manuscript/numbers.tex, paper/manuscript/tab_*.tex and paper/manuscript/fig_*.pdf."
  echo "The manuscript text will be added after acceptance."
fi
