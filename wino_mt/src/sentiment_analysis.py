""" Usage:
    <file-name> <input-file>

Compute sentiment score of each sentence.

"""

from transformers import pipeline
import pandas as pd
import sys

if len(sys.argv) > 1: 
    input_fn = sys.argv[1] 
else: 
    raise AssertionError("Please provide input file")

# Load sentences
with open(input_fn, 'r') as file:
    sentences = file.readlines()


# Analyze sentiment 
sentiment_analyzer = pipeline("sentiment-analysis")
results = sentiment_analyzer(sentences)

# Save results
data = pd.DataFrame({
    "sentence": sentences,
    "label": [result['label'] for result in results],
    "score": [result['score'] for result in results]
})


data.to_csv("../data/sentiment_results.csv", index=False)


print(data.head())