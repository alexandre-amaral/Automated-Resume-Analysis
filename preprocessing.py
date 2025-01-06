__author__ = 'Alexandre Amaral'

# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Pre-processing Functions
def clean_text(text):
    """
    Function to remove special characters and convert text to lowercase.
    Args:
        text (str): Text to be cleaned.
    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

def tokenize(text):
    """
    Function to split text into words (tokens).
    Args:
        text (str): Text to be tokenized.
    Returns:
        list: List of tokens.
    """
    return word_tokenize(text)

def remove_stop_words(tokens):
    """
    Function to remove Portuguese stop words from a list of tokens.
    Args:
        tokens (list): List of tokens.
    Returns:
        list: List of tokens with stop words removed.
    """
    stop_words = set(stopwords.words('portuguese'))
    return [word for word in tokens if word not in stop_words]

def apply_stemming(tokens):
    """
    Function to apply stemming process to a list of tokens.
    Args:
        tokens (list): List of tokens.
    Returns:
        list: List of stemmed tokens.
    """
    stemmer = RSLPStemmer()
    return [stemmer.stem(word) for word in tokens]

def process_text(text):
    """
    Function to process text by applying cleaning, tokenizing, stop word removal,
    and stemming.
    Args:
        text (str): Text to be processed.
    Returns:
        str: Processed text.
    """
    if isinstance(text, str):
        text = clean_text(text)
        tokens = tokenize(text)
        tokens = remove_stop_words(tokens)
        tokens = apply_stemming(tokens)
        return ' '.join(tokens)  # Join tokens into a string
    return text  # Return original if not a string

# Load and process DataFrame
df = pd.read_csv('resources/cvs_sample.csv')

# Text columns to be processed
columns_to_process = ['Des_Atividade', 'Des_Atividade_empresa', 'Des_Bairro', 'Des_Curso', 'Des_Escolaridade',
                      'Des_Escolaridade_Formacao', 'Des_Funcao', 'Des_Funcao_Exercida', 'Des_Idioma', 'Nme_Cidade',
                      'Nme_Pessoa', 'Raz_Social', 'Des_Categoria_Habilitacao', 'Des_Complemento', 'Des_Conhecimento',
                      'Des_Deficiencia', 'Des_Estado_Civil']

for column in columns_to_process:
    if column in df.columns:
        df[column + '_processed'] = df[column].apply(process_text)

# Save processed DataFrame
df.to_csv('cvs_processed.csv', index=False)

df = pd.read_csv('cvs_preprocessed.csv')

# Text columns to be processed in second DataFrame
columns_to_process = ['data']

for column in columns_to_process:
    if column in df.columns:
        df[column + '_processed'] = df[column].apply(process_text)

# Save second processed DataFrame
df.to_csv('jobs_processed.csv', index=False)

# Display first rows of the processed DataFrame
print(df.head())
