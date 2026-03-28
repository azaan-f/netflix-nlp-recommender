
class user_profiles:

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.normalize = loader.normalize
        self.get_watched_movies = loader.get_watched_movies

    
    # returns column for the movies watched by the user
    def _watched_rows(self, user_id):
        watched_keys = set(self.normalize(t) for t in self.get_watched_movies(user_id))
        return self.dataset[self.dataset["title_key"].isin(watched_keys)]
 

    # joins the column for the movies watched by the user and normalizes it
    def _join_col(self, rows, col):
        return self.normalize(
            " ".join(rows[col].fillna("").astype(str).tolist())
        )
    

    # i think these will help with the bugs since they will return empty strings 
    # if the user has not watched any movies, rather than throwing an error when 
    # trying to join an empty dataframe
    def build_desc_profile(self, user_id):
        rows = self._watched_rows(user_id)

        if rows.empty:
            return ""
        
        return self._join_col(rows, "description")
 

    def build_genre_profile(self, user_id):
        rows = self._watched_rows(user_id)

        if rows.empty:
            return ""
        
        return self._join_col(rows, "listed_in")
 

    def build_actor_profile(self, user_id):
        rows = self._watched_rows(user_id)

        if rows.empty:
            return ""
        
        return self._join_col(rows, "cast")
 

    def build_all(self, user_id):
        rows = self._watched_rows(user_id)

        if rows.empty:
            return "", "", ""
        
        return (
            self._join_col(rows, "description"),
            self._join_col(rows, "listed_in"),
            self._join_col(rows, "cast"),
        )