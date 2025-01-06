__author__ = 'Alexandre Amaral'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize

# Helper functions
def combine_columns(df, col_suffix='_processada'):
    return df[df.columns[df.columns.str.endswith(col_suffix)]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Load preprocessed data
df_cvs = pd.read_csv('resources/cvs_preprocessed.csv')
df_jobs = pd.read_csv('resources/jobs_preprocessed.csv')

# Combine processed text columns for each DataFrame
textos_cvs = combine_columns(df_cvs)
textos_jobs = combine_columns(df_jobs)

# 1. TF-IDF
vectorizer = TfidfVectorizer()
tfidf_cvs = vectorizer.fit_transform(textos_cvs)
tfidf_jobs = vectorizer.transform(textos_jobs)

# 2. LDA
n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
lda_cvs = lda.fit_transform(tfidf_cvs)
lda_jobs = lda.transform(tfidf_jobs)

# 3. BERT Embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

bert_embeddings_cvs = np.vstack([get_bert_embeddings(texto) for texto in textos_cvs])
bert_embeddings_jobs = np.vstack([get_bert_embeddings(texto) for texto in textos_jobs])

# 4. Similarity Calculation
similarities_tfidf = cosine_similarity(tfidf_cvs, tfidf_jobs)
similarities_lda = cosine_similarity(lda_cvs, lda_jobs)
similarities_bert = cosine_similarity(bert_embeddings_cvs, bert_embeddings_jobs)

# Combine Similarities
total_similarity = (similarities_tfidf + similarities_lda + similarities_bert) / 3

# 5. Candidate Ranking
def rank_candidates(similarities, top_n=5):
    for idx, job_similarities in enumerate(similarities.T):
        top_candidates = job_similarities.argsort()[-top_n:][::-1]
        print(f"Job {idx + 1}:")
        for candidate in top_candidates:
            print(f"  Candidate {candidate + 1} - Similarity: {job_similarities[candidate]:.2f}")
        print("\n")

rank_candidates(total_similarity)

# Save or process the results
df_results = pd.DataFrame(total_similarity)
df_results.to_csv('results.csv', index=False)