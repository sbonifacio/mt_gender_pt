#!/bin/bash
# Usage:
#   evaluate_wmt.sh <lang-fn> <output-file>
# for example:
#   evaluate_wmt.sh en-pt.txt ../logs/new/log-pt-ft.txt
set -e

lang=$1
targetlang=`echo $lang | cut -d. -f1 | cut -d"-" -f2`

outfn=$2
wmtbase=../translations
allgold=../data/aggregates/en.txt
progold=../data/aggregates/en_pro.txt
antgold=../data/aggregates/en_anti.txt

echo $antgold

#systems=`ls $wmtbase`
systems=("google" "aws" "bing" "deepl" "gpt" "deepseek" "llama" "llama-inst" "tower" "tower-inst" "eurollm" "nllb" "m2m" "opus")

for system in ${systems[@]}
do
    echo "analyzing $system"
    trans=$wmtbase/$system/$lang
    if [ -f $trans ]; then
        printf "$system\n" >> $outfn
        python split_translations.py --pro=$progold --ant=$antgold --trans=$trans
        printf "all;;;" >> $outfn
        ../scripts/evaluate_single_file.sh $allgold $trans $targetlang $outfn
        printf "pro-stereotypical;;;" >> $outfn
        ../scripts/evaluate_single_file.sh $progold ${trans}.pro $targetlang $outfn
        printf "anti-stereotypical;;;" >> $outfn
        ../scripts/evaluate_single_file.sh $antgold ${trans}.ant $targetlang $outfn
    fi
done

echo "DONE!"
