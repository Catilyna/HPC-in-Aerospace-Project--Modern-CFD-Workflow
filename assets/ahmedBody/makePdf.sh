#!/bin/sh
cd "${0%/*}" || exit 1
set -e

echo "------------------------------------"
printf '%s\n' "Creating image folder"
echo "------------------------------------"
mkdir -p report/images
printf '\n'

echo "------------------------------------"
printf '%s\n' "Creating ParaView field" "visualisations"
echo "------------------------------------"
touch case.foam
pvpython pythonScripts/generateImages.py
printf '\n'

echo "------------------------------------"
printf '%s\n' "Generating plots from simulation" "data"
echo "------------------------------------"
python3 pythonScripts/generatePlots.py
printf '\n'

echo "------------------------------------"
printf '%s\n' "Compiling report"
echo "------------------------------------"
cd report
pdflatex -interaction=batchmode -halt-on-error report.tex > /dev/null
pdflatex -interaction=batchmode -halt-on-error report.tex > /dev/null
cd ..
printf '\n'

echo "================================================================="
echo " Complete! Report saved to: $(pwd)/report/report.pdf"
echo "================================================================="