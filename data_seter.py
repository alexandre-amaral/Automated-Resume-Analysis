__author__ = 'Alexandre Amaral'

from pymongo import MongoClient
import pandas as pd
import re
import json

# Establishing connection to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['cv-analyzer']
cvs_collection = db['cvs']
jobs_collection = db['jobs']

# Setting the sample size for data extraction
sample_size = 1000

# Sampling data from 'cvs' collection
cvs_sample = cvs_collection.aggregate([{'$sample': {'size': sample_size}}])
cvs_df = pd.DataFrame(list(cvs_sample))

# Sampling data from 'jobs' collection
jobs_sample = jobs_collection.aggregate([{'$sample': {'size': sample_size}}])
jobs_df = pd.DataFrame(list(jobs_sample))

def clean_and_normalize(text):
    """
    Function to clean and normalize text data.
    Args:
        text (str): Text to be cleaned and normalized.
    Returns:
        str: Cleaned and normalized text.
    """
    if isinstance(text, str):
        text = text.encode('utf-8', 'ignore').decode('utf-8')
    return text

# Applying the cleaning and normalization function to all text columns
for column in cvs_df.columns:
    cvs_df[column] = cvs_df[column].apply(clean_and_normalize)

for column in jobs_df.columns:
    jobs_df[column] = jobs_df[column].apply(clean_and_normalize)

# Saving the processed data to CSV files
cvs_df.to_csv('cvs_sample.csv', index=False, encoding='utf-8-sig')
jobs_df.to_csv('jobs_sample.csv', index=False, encoding='utf-8-sig')

# Saving the processed data to JSON files
cvs_df.to_json('cvs_sample.json', orient='records')
jobs_df.to_json('jobs_sample.json', orient='records')