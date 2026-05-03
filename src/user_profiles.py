import numpy as np
import numpy.linalg as la


class user_profiles:

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.normalize = loader.normalize
        self.get_watched_movies = loader.get_watched_movies

    
    # returns column for the movies watched by the user
    def _watched_rows(self, user_id):
        watched_keys = set(self.normalize(t) for t in self.get_watched_movies(user_id))
        return self.dataset[self.dataset["title_key"].isin(watched_keys)]
 

    # helper to join the column for the movies watched by the user and normalize it
    def _join_col(self, rows, col):
        return self.normalize(
        " ".join(rows[col].fillna("").astype(str).tolist())
        )


    # joins the column for the movies watched by the user and normalizes it
    def _tfidf_profile_vector(self, rows, col, feat, idf=None):
        vecs = [
            feat.text2TFIDF(str(d), idf=idf)
            for d in rows[col].fillna("").astype(str)
        ]

        if not vecs:
            return np.zeros(feat.vocab.size)

        avg = np.mean(vecs, axis=0)
        norm = la.norm(avg)

        return avg / norm if norm > 0 else avg
    

    def build_genre_profile(self, user_id):
        rows = self._watched_rows(user_id)
        if rows.empty:
            return ""
        return self._join_col(rows, "genres")
 
 
    def build_country_profile(self, user_id):
        rows = self._watched_rows(user_id)
        if rows.empty:
            return ""
        return self._join_col(rows, "production_countries")
 
    
    # helper to join the column for the movies watched by the user and normalize it
    def build_all(self, user_id, feat=None):
        rows = self._watched_rows(user_id)
 
        if rows.empty:
            if feat is not None:
                z = np.zeros(feat.vocab.size)
                return z, z, z
            return "", "", ""
 
        if feat is not None:
            return (
                self._tfidf_profile_vector(rows, "description",          feat, idf=getattr(feat, "IDF_description", feat.IDF)),
                self._tfidf_profile_vector(rows, "genres",               feat, idf=getattr(feat, "IDF_genres",       feat.IDF)),
                self._tfidf_profile_vector(rows, "production_countries", feat, idf=getattr(feat, "IDF_country",      feat.IDF)),
            )
 
        # string fallback for debug printing
        return (
            self._join_col(rows, "description"),
            self._join_col(rows, "genres"),
            self._join_col(rows, "production_countries"),
        )