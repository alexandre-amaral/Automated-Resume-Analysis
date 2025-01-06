
# CV Analyzer Prototype

## Overview
The CV Analyzer Prototype is a project focused on automating the analysis of resumes and job descriptions. It aims to simplify the recruitment process by integrating modern data processing techniques, including natural language processing (NLP), machine learning, and advanced similarity measures, to screen, match, and rank candidates efficiently.

---

## Features
- **Automated Data Extraction:** Asynchronously extracts CV data and job descriptions from APIs and stores them in MongoDB.
- **Text Preprocessing:** Normalizes and cleans text data for structured analysis.
- **Similarity Analysis:** Combines multiple methodologies:
  - TF-IDF vectorization
  - Latent Dirichlet Allocation (LDA)
  - BERT embeddings
  - Cosine similarity for ranking candidates.
- **Candidate Ranking:** Generates ranked lists of candidates for each job description based on similarity scores.
- **Data Storage:** Processes and stores data in MongoDB and exports it in CSV/JSON formats for further analysis.

---

## Workflow

1. **Data Extraction:**
   - Fetch resumes and job descriptions from an API using asynchronous HTTP requests.
   - Store the fetched data in MongoDB collections.

2. **Preprocessing:**
   - Clean and normalize text fields to remove noise.
   - Combine relevant columns to create comprehensive representations of resumes and jobs.

3. **Feature Extraction:**
   - Apply TF-IDF vectorization to quantify term relevance.
   - Use LDA for topic modeling and semantic grouping.
   - Generate contextual embeddings with BERT for deeper language understanding.

4. **Similarity Calculation:**
   - Measure similarities between resumes and job descriptions using cosine similarity.
   - Combine similarity metrics from different methods (TF-IDF, LDA, BERT) for comprehensive evaluation.

5. **Ranking:**
   - Rank candidates for each job description based on aggregated similarity scores.
   - Output ranked results in CSV/JSON formats.

---

## File Structure

```
.
├── main.py                  # Core program for data extraction and processing
├── preprocessing.py         # Text preprocessing and normalization
├── feature_extraction.py    # Feature extraction (TF-IDF, LDA, BERT)
├── similarity_analysis.py   # Similarity calculation and ranking
├── data/                    # Contains processed data files (CSV/JSON)
├── models/                  # Saved models (Word2Vec, TF-IDF, etc.)
├── resources/               # Preprocessed data (CVs, job descriptions)
└── README.md                # Project documentation
```

---

## Requirements

### Libraries:
- Python (3.8+)
- MongoDB
- aiohttp
- pandas
- scikit-learn
- matplotlib
- transformers (Hugging Face)
- gensim
- nltk

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cv-analyzer-prototype.git
   cd cv-analyzer-prototype
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MongoDB locally or configure MongoDB Atlas for cloud storage.

---

## Usage

1. **Run the data extraction process:**
   ```bash
   python main.py
   ```
2. **Process data for similarity analysis:**
   ```bash
   python preprocessing.py
   ```
3. **Extract features and calculate similarity:**
   ```bash
   python feature_extraction.py
   ```

4. **Generate ranking results:**
   ```bash
   python similarity_analysis.py
   ```

---

## Future Enhancements
- Integration with cloud databases (e.g., MongoDB Atlas).
- Improved machine learning models for predictive analysis.
- Real-time API for job-to-candidate matching.

---

## Author
**Alexandre Amaral**  
For questions or collaboration, contact: [your-email@example.com](mailto:your-email@example.com)
