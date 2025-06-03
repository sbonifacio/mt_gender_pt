# (Auxiliary) Ignore neutral sentences when computing COMET

import sys
import csv


if len(sys.argv) > 1: 
    system = sys.argv[1]
else:
    raise AssertionError("Please provide system")


in_fn=f"../data/human/{system}/pt/pt.pred.csv"
out_fn = f"../translations/{system}/pt.txt"

with open(in_fn,"r") as f1, open(out_fn,"w") as f2:
    reader = csv.reader(f1)
    next(reader)
    for row in reader:
        if row[1]!="ignore": f2.write(row[0]+"\n")