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
            self.W_TFIDF  * tfidf_sim +
            self.W_W2V    * w2v_sim   +
            self.W_GENRE  * genre_sim +
            self.W_ACTORS * actor_sim
        )
    
    
    # actual recommendation function
    def recommend_for_user(self, user_id, k=5, title_col="title"):
        desc_profile, genre_profile, actor_profile = (
            self.user_profile.build_all(user_id)
        )
 
        if not desc_profile:
            return pd.DataFrame(columns=[title_col, "description", "score"])
 
        self.features.compute_IDF(self.dataset["description"])
 
        watched_keys = set(
            self.normalize(t) for t in self.get_watched(user_id)
        )
 
        scores = []
        for _, row in self.dataset.iterrows():
            score = self.hybrid_score(
                desc_profile,
                genre_profile,
                actor_profile,
                str(row.get("description", "")),
                str(row.get("listed_in",   "")),
                str(row.get("cast",        "")),
            )
            scores.append(score)
 
        results = self.dataset.copy()
        results["score"] = scores
        results = results[~results["title_key"].isin(watched_keys)]
        results = results.sort_values("score", ascending=False)
 
        return results[[title_col, "description", "score"]].head(k)