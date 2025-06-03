""" 
Compute Pearson Correlation and p-values between sentence sentiment and gender.

Usage for single system:
    <file-name> <system>

(If no argument <system>, it will compute for all systems)

"""

import pandas as pd
from scipy.stats import pearsonr
import sys

def load(system):
    df1 = pd.read_csv("../data/human/"+system+"/pt/pt.pred.csv")
    column1 = df1["Predicted gender"]
    gender_label = column1.map({'male': 1, 'female': 0})

    df2 = pd.read_csv("../data/sentiment_results.csv")
    column2 = df2["label"]
    sentiment_label = column2.map({'POSITIVE': 1, 'NEGATIVE': 0})


    combined_df = pd.DataFrame({'gender': gender_label, 'label': sentiment_label})
    combined_df.dropna(inplace=True)


    gender_label_clean = combined_df['gender']
    sentiment_label_clean = combined_df['label']


    if len(gender_label_clean) != len(sentiment_label_clean):
        raise ValueError("The two columns must have the same number of rows for correlation calculation.")
    
    return gender_label_clean, sentiment_label_clean



if len(sys.argv) > 1: 
    systems = [sys.argv[1]] # Single system
else: 
    systems = ["google","aws","bing","deepl",
               "gpt","deepseek","llama","llama-inst",
               "tower","tower-inst","eurollm",
               "nllb","m2m","opus"]


for system in systems:
    print(system)
    gender_label, sentiment_label = load(system)

    # Calculate Pearson correlation 
    correlation, p_value = pearsonr(gender_label, sentiment_label)

    print("Pearson correlation:", correlation)
    print("P-value:", p_value)


