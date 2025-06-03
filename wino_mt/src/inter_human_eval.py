import numpy as np
import csv


label_mapping = {"m":0, "f":1, "n": 2, "u":3}
annotators = ["user0","user1"]


def inter_annotator_agreement():
    print("Calculating inter-annotator agreement")

    annotations = []
    for ann in annotators:
        print(f"Annotator {ann}")
        path = f'../data/human_annotations/pt/{ann}.pt.gold.csv'
        with open(path, encoding="utf8") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)  # Skip header
            pred_labels = [row[4].upper() for row in reader]  
            annotations.append(pred_labels)
    
    total = 0
    agreed = 0

    for pred_1, pred_2 in zip(annotations[0],annotations[1]):
        total += 1
        agreed += (pred_1==pred_2)

    print(f"Agreed: {agreed}, Total: {total}, Inter-annotator agreement: {agreed/total}")


inter_annotator_agreement()
    