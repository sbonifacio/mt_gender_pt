#!/bin/bash
# Usage:
#   evaluate_wmt.sh <lang-fn> <output-file>
# for example:
#   ../scripts/evaluate_wmt_short.sh pt ../logs/pt-inter.txt
set -e

lang=$1
outfn=$2

wmtbase=../translations

#systems=`ls $wmtbase`
systems=("google" "aws" "bing" "deepl" "gpt" "deepseek" "llama" "llama-inst" "tower" "tower-inst" "eurollm" "nllb" "m2m" "opus")
datasets=("inter1" "inter2" "inter3")

for system in ${systems[@]}
do
    for inter in ${datasets[@]}
    do
        echo "analyzing $system $inter"
        ../scripts/evaluate_language_inter.sh $inter $lang $system

        allgold=../data/aggregates/en_$inter.txt
        progold=../data/aggregates/en_${inter}_pro.txt
        antgold=../data/aggregates/en_${inter}_anti.txt

        trans=$wmtbase/$system/$inter.en-$lang.txt

        if [ -f $trans ]; then
            printf "$system $inter\n" >> $outfn
            python split_translations.py --pro=$progold --ant=$antgold --trans=$trans
            printf "all;;;" >> $outfn
            ../scripts/evaluate_single_file.sh $allgold $trans $lang $outfn
            printf "pro-stereotypical;;;" >> $outfn
            ../scripts/evaluate_single_file.sh $progold ${trans}.pro $lang $outfn
            printf "anti-stereotypical;;;" >> $outfn
            ../scripts/evaluate_single_file.sh $antgold ${trans}.ant $lang $outfn
        fi

    done

done

echo "DONE!"
