__author__ = 'Alexandre Amaral'

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load pre-processed data
df_cvs = pd.read_csv('cvs_preprocessed.csv')
df_jobs = pd.read_csv('jobs_preprocessed.csv')

# Identify processed text columns
processed_columns_cvs = [col for col in df_cvs.columns if '_processed' in col]
processed_columns_jobs = [col for col in df_jobs.columns if '_processed' in col]

# Concatenate all processed texts
texts_cvs = df_cvs[processed_columns_cvs].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
texts_jobs = df_jobs[processed_columns_jobs].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
all_texts = pd.concat([texts_cvs, texts_jobs], axis=0).tolist()

# 1. TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Convert to DataFrame for visualization (optional)
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# 2. Word2Vec
# Tokenize texts for Word2Vec
tokenized_text = [word_tokenize(text) for text in all_texts]

# Train a Word2Vec model
w2v_model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Save the models
df_tfidf.to_csv('tfidf_vectors.csv', index=False)
w2v_model.save("word2vec_model.model")

print("Feature extraction completed and models saved.")