__author__ = 'Alexandre Amaral'

import pandas as pd
import matplotlib.pyplot as plt

# Reading CSV files into pandas DataFrames
cvs_df = pd.read_csv('cvs_sample.csv')
jobs_df = pd.read_csv('jobs_sample.csv')

# Converting DataFrames to JSON files
cvs_df.to_json('cvs_sample.json', orient='records')
jobs_df.to_json('jobs_sample.json', orient='records')
