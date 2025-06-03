#!/bin/bash
# Usage:
#   evaluate_quality.sh <dataset> <lang-code> <output-file>
#
# e.g.,
# ../scripts/evaluate_quality.sh ../data/aggregates/en_filtered.txt pt ../logs/pt.txt
#
# use "../data/aggregates/en.txt" for languages other than Portuguese


set -e

# Parse parameters

dataset=$1
lang=$2
outfn=$3
prefix=en-$lang

# Prepare files for evaluation
cut -f3 $dataset > ./tmp.in            # Extract sentences

systems=("google" "aws" "bing" "deepl" "gpt" "deepseek" "llama" "llama-inst" "tower" "tower-inst" "eurollm" "nllb" "m2m" "opus")
for trans_sys in ${systems[@]}
do
    echo "analyzing $trans_sys"
    trans_fn=../translations/$trans_sys/$lang.txt
    python prepare_comet.py $trans_sys
    printf "$trans_sys\n" >> $outfn
    comet-score -s ./tmp.in -t $trans_fn --model Unbabel/wmt22-cometkiwi-da --quiet --only_system >> $outfn

done

echo "DONE!"