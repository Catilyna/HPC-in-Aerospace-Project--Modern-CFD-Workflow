#!/bin/sh
cd "${0%/*}" || exit 1

echo "------------------------------------"
echo -e "Crearting image folder"
echo "------------------------------------"
mkdir -p report/images
echo -e "\n"

echo "------------------------------------"
echo -e "Creating ParaView field \nvisualizations"
echo "------------------------------------"
touch case.foam
pvpython pythonScripts/generateImages.py
echo -e "\n"

echo "------------------------------------"
echo -e "Generating plots from simulation \ndata"
echo "------------------------------------"
python3 pythonScripts/generatePlots.py
echo -e "\n"

echo "------------------------------------"
echo -e "Compiling report"
echo "------------------------------------"
cd report
pdflatex -interaction=batchmode -halt-on-error report.tex > /dev/null
pdflatex -interaction=batchmode -halt-on-error report.tex > /dev/null
cd ..
echo -e "\n"

echo "================================================================="
echo " Complete! Report saved to: $(pwd)/report/report.pdf"
echo "================================================================="