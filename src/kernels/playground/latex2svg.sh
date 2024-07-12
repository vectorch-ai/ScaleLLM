#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 latex.tex"
    exit 1
fi

latex_file=$1
pdf_file=${latex_file%.tex}.pdf
svg_file=${latex_file%.tex}.svg

# create a temporary folder
tmp_folder=$(mktemp -d)

# run pdflatex to generate pdf file
pdflatex -shell-escape -output-directory=$tmp_folder $latex_file

# convert pdf to svg
pdf2svg $tmp_folder/$pdf_file $svg_file

# cleanup: delete temporary folder
rm -r $tmp_folder
