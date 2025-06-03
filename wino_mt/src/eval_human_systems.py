""" Usage:
    updated to handle files with multiple systems
"""
# External imports
import logging
import pdb
import csv
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import sys

# Local imports

#=-----

USER = "user01"

SYSTEMS = ["google", "aws", "bing", "deepl", 
           "m2m", "nllb", "opus",
           "gpt","tower","tower-inst","llama","llama-inst",
           "eurollm","deepseek"]

ENTRIES_PER_SYSTEM = 13

def eval(gold_rows,pred_rows):
    indices = map(int, map(itemgetter(0), gold_rows))

    total = 0
    correct = 0

    for (sent_ind, gold_row) in zip(indices, gold_rows):
        pred_row = pred_rows[sent_ind]
        _, entity, gold_sent, valid_flag, gold_gender = gold_row[:5]
        pred_sent, pred_gender = pred_row
        if gold_sent != pred_sent:
            raise AssertionError(f"Mismatch:\n {gold_sent} \n {pred_sent}")
        gold_gender = gold_gender.strip().lower()
        if (gold_gender not in ["m", "f", "n"]) or (valid_flag.lower == "n"):
            print(f"Missing gold annotation: {gold_row}")
            continue

        total += 1
        if pred_gender[0] == gold_gender:
            correct += 1
        else:
            print(
                f"""
                Wrong gender prediction:
                GOLD: {gold_row}
                PRED: {pred_row}
                """)

    percent_correct = round(correct / total, 2)
    print(f"%correct = {percent_correct}")
    return correct, total


if __name__ == "__main__":

   
    gold_fn = f"../data/human_annotations/pt/{USER}.pt.gold.csv"
    gold_rows = [row for row
                 in csv.reader(open(gold_fn, encoding = "utf8"), delimiter=",")][1:]


    correct = 0
    total = 0
    
    for i in range(len(SYSTEMS)):
        base = 1+i*ENTRIES_PER_SYSTEM
        gold_rows_model = gold_rows[base:base+ENTRIES_PER_SYSTEM-1]
        pred_fn = "../data/human/"+SYSTEMS[i]+"/pt/pt.pred.csv"
        pred_rows = [row for row
                 in csv.reader(open(pred_fn, encoding = "utf8"), delimiter=",")][1:]
        print(f'{SYSTEMS[i]}')
        sys_correct, sys_total= eval(gold_rows_model,pred_rows)
        correct += sys_correct
        total += sys_total

    percent_correct = round(correct / total, 2)
    print("ALL:")
    print(f"%correct = {percent_correct}")

    print("DONE")
