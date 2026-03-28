import math # for log in IDF computation
import collections # for counting word frequencies
import numpy as np

from gensim.models import Word2Vec


class features:

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.normalize = loader.normalize

        self.vocab: np.ndarray = None
        self.IDF: np.ndarray = None
        self.w2v_model: Word2Vec = None


    def build_vocabulary(self, text_col="description", max_words=5000):
        # count the ocurrance of the words in the dataset 
        count = collections.Counter()

        for doc in self.dataset[text_col].fillna("").astype(str):
            count.update(doc.split())

        self.vocab = np.array(
            [w for w, _ in count.most_common(max_words)], dtype=object
        )

        return self
    

    def compute_IDF(self, collection):
        M = len(collection)
        self.IDF = np.zeros(self.vocab.size) # initialize the IDFs to zero
        doc_sets = [set(self.normalize(doc).split()) for doc in collection.fillna("").astype(str)]

        for i, w in enumerate(self.vocab):
            df = sum(1 for s in doc_sets if w in s)
            self.IDF[i] = math.log((M + 1) / (df + 1)) + 1 # formula from slides


    def text2BitVector(self,text):
        bitVector = np.zeros(self.vocab.size, dtype=np.int32)
        tokens = set(self.normalize(text).split())

        for i, w in enumerate(self.vocab):
            if w in tokens:
                bitVector[i] = 1 # if present, set to 1, otherwise it remains 0

        return bitVector
    

    # returns the bit vector representation of the text
    def text2TFIDF(self, text):
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
    

    def tfidf_score(self, query, doc): # a little cleaner
        return float(self.text2TFIDF(query).dot(self.text2TFIDF(doc)))
    

    def train_word2vec(self, text_col="description", vector_size=100, window=5): # based only on descriptions so far
        sentences = [
            doc.split() 
            for doc in self.dataset[text_col].fillna("").astype(str)
        ]
        
        self.w2v_model = Word2Vec(
            sentences, vector_size=vector_size, window=window, min_count=1, workers=4
        )

        return self
    

    # returns the average word vector for the text, or a zero vector if no words are in the model
    def text2W2V(self, text):  # average word vectors = doc vector
        tokens = self.normalize(text).split()

        vecs = [
            self.w2v_model.wv[t] 
            for t in tokens 
            if t in self.w2v_model.wv
        ]

        if not vecs:
            return np.zeros(self.w2v_model.vector_size)
        
        return np.mean(vecs, axis=0)