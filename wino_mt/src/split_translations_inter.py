""" 

Split inter datasets into pro and anti.

"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import json

# Local imports


#----

# splits i1 and returns index of anti and pro sentences
def split_i1():
    pro_fn = "../data/aggregates/en_pro.txt"
    ant_fn = "../data/aggregates/en_anti.txt"
    i1_fn = "../data/aggregates/en_inter1.txt"

    pro_sents = [line for line in open(pro_fn, encoding = "utf8")]
    ant_sents = [line for line in open(ant_fn, encoding = "utf8")]
    i1_lines = [line for line in open(i1_fn, encoding = "utf8")]

    out_pro_fn = "../data/aggregates/en_inter1_pro.txt"
    out_ant_fn = "../data/aggregates/en_inter1_anti.txt"

    index_pro, index_ant= [],[]

    with open(out_pro_fn, "w", encoding = "utf8") as f_pro, \
         open(out_ant_fn, "w", encoding = "utf8") as f_ant:
        for i in range(len(i1_lines)):
            sent = i1_lines[i]
            if sent in ant_sents:
                f_ant.write(sent)
                index_ant.append(i)

            elif sent in pro_sents:
                f_pro.write(sent)
                index_pro.append(i)
            
            else:
                print(f"Neutral sentence: {sent}")

    return index_pro,index_ant

def split_i(index_pro,index_ant,suf,fn):

    lines = [line for line in open(fn, encoding = "utf8")]

    out_pro_fn = f"../data/aggregates/{suf}_pro.txt"
    out_ant_fn = f"../data/aggregates/{suf}_anti.txt"


    with open(out_pro_fn, "w", encoding = "utf8") as f_pro:
        for i in index_pro:
            f_pro.write(lines[i]) 

    with open(out_ant_fn, "w", encoding = "utf8") as f_ant:
        for i in index_ant:
            f_ant.write(lines[i]) 

    



if __name__ == "__main__":


    fn_i2 = "../data/aggregates/en_inter2.txt"
    fn_i3 = "../data/aggregates/en_inter3.txt"


    index_pro, index_ant = split_i1()
    print(f"found {len(index_pro)} pro sents and {len(index_ant)} ant sents")
    #pdb.set_trace()

    split_i(index_pro,index_ant,"en_inter2",fn_i2)
    print("inter2 done")

    split_i(index_pro,index_ant,"en_inter3",fn_i3)
    print("inter3 done")

