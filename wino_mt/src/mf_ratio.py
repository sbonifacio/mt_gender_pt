"""
Calculate M:F ratio
"""
import csv
import sys


if len(sys.argv) > 1: 
    systems = [sys.argv[1]]
else: 
    systems = ["google","aws","bing","deepl",
               "gpt","deepseek","llama","llama-inst",
               "tower","tower-inst","eurollm",
               "nllb","m2m","opus"]

lang = "pt"

for system in systems:
    path = f'../data/human/{system}/{lang}/{lang}.pred.csv'
    total_f = 0
    total_m = 0
    unknown = 0
    ignore = 0
    with open(path,"r") as file:
        reader = csv.reader(file)

        # Get number of feminine/masculine translations
        for row in reader:
            gender = row[1]
            if gender == "female":
                total_f += 1
            elif gender =="male":
                total_m += 1
            elif gender == "unknown":
                unknown += 1
            elif gender =="ignore":
                ignore +=1


        ratio = total_m/total_f
        print(f'{system}\n ratio:{ratio}, male: {total_m}, female: {total_f}, unknown: {unknown}, ignore: {ignore}')


