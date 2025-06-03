"""
Create gold annotations files from user input
"""

import csv

SYSTEMS = ["google", "aws", "bing", "deepl", 
           "m2m", "nllb", "opus",
           "gpt","tower","tower-inst","llama","llama-inst",
           "eurollm","deepseek"]


ENTRIES_PER_SYSTEM = 13 #inter-> 4
TOTAL_ENTRIES = ENTRIES_PER_SYSTEM * len(SYSTEMS)




print("You will be presented a set of sentences in Portuguese along with an entity.\n" \
"You are tasked with finding the entity in the sentence and classifying its gender as Masculine, Feminine or Neuter.\n" \
"If you don't find the entity in the sentence, please classify it as -.\n")

print("Please choose the gender based solely on the wording used for the entity. \nIgnore all context and other pronoun references.\n")
print("For example, in this sentence, if you were to classify analyst:")
print("\t\"O assistente digitalizou documentos para o analista porque ela pediu.\"")
print("Although there is the pronoun \"ela\", the wording used for analyst is \"o analista\", therefore, classifies as Masculine\n")

print("Always try to classify the entities as Masculine or Feminine.\n" \
"In cases such as \"o aluno\", which is a masculine form but can be used as neuter, please still classify it as Masculine.\n")



user = input("Who's testing?: ").strip()


# Read file ----------------------
rows=[]
for system in SYSTEMS:
    file_path =  f"../data/human/{system}/pt/pt.in.csv"
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        rows += list(reader)[:ENTRIES_PER_SYSTEM]
        fieldnames = reader.fieldnames

updated_rows = []
i = 1

# Process each row
for row in rows:
    # Ask fro user input ----------------
    print("\n"+row["Entity"].upper()+" "+str(i)+"/"+str(TOTAL_ENTRIES)+"\n"+row["Sentence"])

    gender = input(row["Entity"] + " [M/F/N/-]: ").strip()
    comments = input("Any comments? (Leave blank if none): ").strip()

    # Update the row with user input --------------------
    if gender != "-": 
        row["Find entity? [Y/N]"] = "Y"
        row["Gender? [M/F/N]"] = gender
    else:
        row["Find entity? [Y/N]"] = "N"
        row["Gender? [M/F/N]"] = "-"
    row["Comments"] = comments
    updated_rows.append(row)

    i+=1

# Write updated rows to a new file --------------------
output_file = "../data/human_annotations/pt/"+user + ".pt.gold.csv"
with open(output_file, mode='w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"\nFile saved as {output_file}")