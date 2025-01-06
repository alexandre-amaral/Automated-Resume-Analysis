__author__ = 'Alexandre Amaral'

import json
import pandas as pd

# Opening JSON files and loading the data
with open('cvs_sample.json') as f:
    cvs_data = json.load(f)

with open('jobs_sample.json') as f:
    jobs_data = json.load(f)

# Converting JSON data to pandas DataFrames
cvs_df = pd.DataFrame(cvs_data)
jobs_df = pd.DataFrame(jobs_data)

# Function to normalize text data
def normalize(text):
    """
    Function to normalize text data.
    Args:
        text (str): Text to be normalized.
    Returns:
        str: Normalized text.
    """
    if isinstance(text, str):
        text = text.encode('utf-8', 'ignore').decode('utf-8')
    return text

# Applying the normalization function to all columns in both DataFrames
for column in cvs_df.columns:
    cvs_df[column] = cvs_df[column].apply(normalize)

for column in jobs_df.columns:
    jobs_df[column] = jobs_df[column].apply(normalize)

# Dropping specific columns from each DataFrame
cvs_df = cvs_df.drop(columns=['_id', 'id'])
jobs_df = jobs_df.drop(columns=['_id', 'success'])

# Saving the modified DataFrames back to JSON files
cvs_df.to_json('cvs_sample.json', orient='records')
jobs_df.to_json('jobs_sample.json', orient='records')
