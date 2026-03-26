import re # for removing punctuation
import urllib.request # for downloading the file if not present
import math # for log in IDF computation
import collections # for counting word frequencies

import numpy as np
import nltk
nltk.download('stopwords')
import pandas as pd

from nltk.corpus import stopwords

class netflixRetrieval():

    def __init__(self):
        # csvs from the dataset folder
        title_url = 'https://raw.githubusercontent.com/azaan-f/netflix-nlp-recommender/main/datasets/netflix_titles.csv'
        title_url2 = 'https://raw.githubusercontent.com/azaan-f/netflix-nlp-recommender/main/datasets/titles.csv' # larger of the two i believe
        users = 'https://raw.githubusercontent.com/azaan-f/netflix-nlp-recommender/main/datasets/users.csv'

        titles_df = pd.read_csv(title_url) # swap if necessary

        # punctuation/stop word logic
        self.punctuations = '"\,<>./?@#$%^&*_~/!()-[]{};:'

        try:
            _ = stopwords.words('english')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))

        self.dataset = titles_df


    # normalizer
    def normalize(self, text):
        text = "" if text is None else str(text)
        text = text.strip().lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]

        return " ".join(tokens)
    

    def preprocess_descriptions(self, description_col="description"):
        self.dataset[description_col] = (
            self.dataset[description_col]
            .fillna("")
            .astype(str)
            .apply(self.normalize)
        )

        return self.dataset
    

    def build_vocabulary(self, text_col="description", max_words=200):
        if self.dataset is None:
            raise Exception("data set not loaded")
    
        # count the ocurrance of the words in the dataset 
        count = collections.Counter()

        for doc in self.dataset[text_col].fillna("").astype(str):
            count.update(doc.split())

        most_common = count.most_common(max_words)
        words_only = [w for (w, c) in most_common]

        self.vocab = np.array(words_only, dtype=object)

    
    def text2BitVector(self,text):
        bitVector = np.zeros(self.vocab.size, dtype=np.int32)
        tokens = set(self.normalize(text).split())

        for i, w in enumerate(self.vocab):
            if w in tokens:
                bitVector[i] = 1 # if present, set to 1, otherwise it remains 0

        return bitVector
    

    def bit_vector_score(self, query,doc):
        q = self.text2BitVector(query)
        d = self.text2BitVector(doc)

        # compute the relevance using q and d
        relevance = float(q.dot(d))
        return relevance
    

    # executes the computation of the relevance score for each document
    def execute_search_BitVec(self, query, text_col="description"):
        query = self.normalize(query)

        relevances = np.zeros(self.dataset.shape[0])

        for i, doc in enumerate(self.dataset[text_col].fillna("").astype(str)):
            relevances[i] = self.bit_vector_score(query, doc)

        return relevances
    

    def compute_IDF(self,M,collection):
        self.IDF = np.zeros(self.vocab.size) # initialize the IDFs to zero
        doc_sets = [set(self.normalize(doc).split()) for doc in collection.fillna("").astype(str)]

        for i, w in enumerate(self.vocab):
            df = sum(1 for s in doc_sets if w in s)
            self.IDF[i] = math.log((M + 1) / (df + 1)) + 1 # formula from slides

    
    # returns the bit vector representation of the text
    def text2TFIDF(self,text, applyBM25_and_IDF=False):
        tokens = self.normalize(text).split()
        tfidfVector = np.zeros(self.vocab.size, dtype=float)

        if not tokens:
            return tfidfVector

        counts = collections.Counter(tokens) 
    
        for i, w in enumerate(self.vocab):
            tf = counts.get(w, 0)
            if tf > 0:
                tfidfVector[i] = tf * self.IDF[i]

        return tfidfVector
    

    def tfidf_score(self,query,doc, applyBM25_and_IDF=False):
        q = self.text2TFIDF(query)
        d = self.text2TFIDF(doc)

        relevance = float(q.dot(d))
        return relevance
    

    def execute_search_TF_IDF(self, query, text_col="description"):
        query = self.normalize(query)
        self.compute_IDF(self.dataset.shape[0], self.dataset[text_col])

        # global IDF
        relevances = np.zeros(self.dataset.shape[0])

        for i, doc in enumerate(self.dataset[text_col].fillna("").astype(str)):
            relevances[i] = self.tfidf_score(query, doc)

        return relevances
    

# ------- testing ------- #

if __name__ == "__main__":
    netflix = netflixRetrieval()

    print("Loaded dataset:")
    print(netflix.dataset.head())

    netflix.preprocess_descriptions()
    netflix.build_vocabulary()

    print("\nVocabulary size:", len(netflix.vocab))

    print("\nTop 5 results (Bit Vector):")
    scores = netflix.execute_search_BitVec("crime family mafia")
    top_idx = np.argsort(scores)[::-1][:5]
    print(netflix.dataset.iloc[top_idx][["title", "description"]])

    print("\nTop 5 results (TF-IDF):")
    scores_tfidf = netflix.execute_search_TF_IDF("crime family mafia")
    top_idx = np.argsort(scores_tfidf)[::-1][:5]
    print(netflix.dataset.iloc[top_idx][["title", "description"]])