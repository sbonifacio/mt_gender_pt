#!/bin/bash
# Usage:
#   evaluate_all_languages.sh <corpus> <output-folder>
set -e

corpus_fn=$1 # ../data/aggregates/en.txt
out_folder=$2




langs=("ar" "uk" "he" "ru" "it" "fr" "es" "de")
mt_systems=("google" "bing")

for trans_sys in ${mt_systems[@]}
do
    for lang in ${langs[@]}
    do
        echo "evaluating $trans_sys, $lang"
        if [[ "$lang" == "uk" && "$trans_sys" == "aws" ]]; then
            echo "skipping.."
            continue
        fi

        if [[ "$trans_sys" == "sota" ]]; then
            if [[ "$lang" != "de" && "$lang" != "fr" ]]; then
                echo "skipping.."
                continue
            fi
        fi

        # Run evaluation
        mkdir -p $out_folder/$trans_sys
        out_file=$out_folder/$trans_sys/$lang.log
        echo "Evaluating $lang into $out_file"
        ../scripts/evaluate_language_adj.sh $corpus_fn $lang $trans_sys > $out_file
    done
done

echo "DONE!"
