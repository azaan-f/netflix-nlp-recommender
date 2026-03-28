import numpy as np
import pandas as pd


class retrieval:
    
    w_tfidf = 0.5
    w_w2v = 0.2
    w_genre = 0.2
    w_actors = 0.1


    def __init__(self, loader, features, user_profiles):
        self.dataset = loader.dataset
        self.normalize = loader.normalize
        self.get_watched = loader.get_watched_movies
        self.features = features
        self.user_profiles = user_profiles

    
    def hybrid_score(self, desc_profile, genre_profile, actor_profile, doc_desc, doc_genre, doc_cast):
        ft = self.features

        # TF-IDF on description (0.5)
        tfidf_sim = float(ft.text2TFIDF(desc_profile).dot(ft.text2TFIDF(doc_desc)))
 
        # Word2Vec cosine similarity on description (0.2)
        w2v_q  = ft.text2W2V(desc_profile)
        w2v_d  = ft.text2W2V(doc_desc)
        norm   = np.linalg.norm(w2v_q) * np.linalg.norm(w2v_d)
        w2v_sim = float(np.dot(w2v_q, w2v_d) / norm) if norm > 0 else 0.0
 
        # TF-IDF on genre (0.2)
        genre_sim = float(ft.text2TFIDF(genre_profile).dot(ft.text2TFIDF(doc_genre)))
 
        # TF-IDF on cast (0.1)
        actor_sim = float(ft.text2TFIDF(actor_profile).dot(ft.text2TFIDF(doc_cast)))
 
        return (
            self.w_tfidf  * tfidf_sim +
            self.w_w2v    * w2v_sim   +
            self.w_genre  * genre_sim +
            self.w_actors * actor_sim
        )
    
    
    # to help speed up the process
    def precompute_doc_matrices(self):
        ft = self.features
        self.desc_matrix  = np.vstack([ft.text2TFIDF(d) for d in self.dataset["description"].fillna("").astype(str)])
        self.genre_matrix = np.vstack([ft.text2TFIDF(d) for d in self.dataset["listed_in"].fillna("").astype(str)])
        self.actor_matrix = np.vstack([ft.text2TFIDF(d) for d in self.dataset["cast"].fillna("").astype(str)])
        self.w2v_matrix   = np.vstack([ft.text2W2V(d)   for d in self.dataset["description"].fillna("").astype(str)])
        print("doc matrices precomputed.") # just to confirm this step is done before we start recommending

    
    # actual recommendation function
    def recommend_for_user(self, user_id, k=5, title_col="title"):
        desc_profile, genre_profile, actor_profile = (
            self.user_profiles.build_all(user_id)
        )

        if not desc_profile:
            return pd.DataFrame(columns=[title_col, "description", "score"])

        watched_keys = set(self.normalize(t) for t in self.get_watched(user_id))
        ft = self.features

        # query vectors
        q_desc  = ft.text2TFIDF(desc_profile)
        q_genre = ft.text2TFIDF(genre_profile)
        q_actor = ft.text2TFIDF(actor_profile)
        q_w2v   = ft.text2W2V(desc_profile)

        # dot products against precomputed matrices
        tfidf_scores = self.desc_matrix.dot(q_desc)
        genre_scores = self.genre_matrix.dot(q_genre)
        actor_scores = self.actor_matrix.dot(q_actor)

        q_norm     = np.linalg.norm(q_w2v)
        doc_norms  = np.linalg.norm(self.w2v_matrix, axis=1)
        norms      = doc_norms * q_norm
        w2v_scores = np.where(norms > 0, self.w2v_matrix.dot(q_w2v) / norms, 0.0)

        scores = (
            self.w_tfidf  * tfidf_scores +
            self.w_w2v    * w2v_scores   +
            self.w_genre  * genre_scores +
            self.w_actors * actor_scores
        )

        results = self.dataset.copy()
        results["score"] = scores
        results = results[~results["title_key"].isin(watched_keys)]
        results = results.sort_values("score", ascending=False)

        return results[[title_col, "description", "score"]].head(k)