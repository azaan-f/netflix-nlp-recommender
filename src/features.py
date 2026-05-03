import math # for log in IDF computation
import collections # for counting word frequencies
import numpy as np
import numpy.linalg as la

from gensim.models import Word2Vec


class features:

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.normalize = loader.normalize

        self.vocab: np.ndarray = None
        self.IDF: np.ndarray = None
        self.w2v_model: Word2Vec = None


    def build_vocabulary(self, max_words=5000):
        # count the ocurrance of the words in the dataset 
        count = collections.Counter()

        for col in ("description", "genres", "production_countries"):
            for doc in self.dataset[col].fillna("").astype(str):
                count.update(doc.split())
 
        self.vocab = np.array(
            [w for w, _ in count.most_common(max_words)], dtype=object
        )
 
        # index lookup for fast vectorization
        self._word2idx = {w: i for i, w in enumerate(self.vocab)}
 
        return self
    
    # switching to a per-column IDF computation
    def compute_IDF(self, collection, field="description"):
        M = len(collection)
        idf = np.zeros(self.vocab.size)
        doc_sets = [set(self.normalize(doc).split()) for doc in collection.fillna("").astype(str)]

        for i, w in enumerate(self.vocab):
            df = sum(1 for s in doc_sets if w in s)
            idf[i] = math.log((M + 1) / (df + 1)) + 1

        setattr(self, f"IDF_{field}", idf)
        self.IDF = idf  # keep default for backwards compat


    def text2BitVector(self,text):
        bitVector = np.zeros(self.vocab.size, dtype=np.int32)
        tokens = set(self.normalize(text).split())

        for i, w in enumerate(self.vocab):
            if w in tokens:
                bitVector[i] = 1 # if present, set to 1, otherwise it remains 0

        return bitVector
    

    # returns the bit vector representation of the text
    def text2TFIDF(self, text, normalize=True, idf=None):
        tokens = self.normalize(text).split()
        vec = np.zeros(self.vocab.size, dtype=float)

        if not tokens:
            return vec
        
        idf_weights = idf if idf is not None else self.IDF
        counts = collections.Counter(tokens) 
    
        for w, tf in counts.items():
            idx = self._word2idx.get(w)
            if idx is not None:
                vec[idx] = tf * idf_weights[idx]

        if normalize: # normalize the vector to unit length, which is common for cosine similarity
            norm = la.norm(vec)
            if norm > 0:
                vec /= norm
 
        return vec
    

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
        
        # average the word vectors to get a single representation for the text
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
 
        return avg / norm if norm > 0 else avg