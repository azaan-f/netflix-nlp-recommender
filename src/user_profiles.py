
class user_profiles:

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.normalize = loader.normalize
        self.get_watched_movies = loader.get_watched_movies
        self.user_dataset = loader.user_dataset
        self.get_user_row = loader.get_user_row

    
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
    # def build_desc_profile(self, user_id):
    #     rows = self._watched_rows(user_id)

    #     if rows.empty:
    #         return ""
        
    #     return self._join_col(rows, "description")
    def build_desc_profile(self, user_id): 
        row = self.get_user_row(user_id)
        desc = row["WatchedDescriptions"]
        if desc == "": 
            return ""
        return self.normalize(desc)

 

    def build_genre_profile(self, user_id):
        # rows = self._watched_rows(user_id)
        row = self.get_user_row(user_id)
        gnr = row["WatchedGENRE"]
        if gnr == "":
            return ""

        # if rows.empty:
        #     return ""
        
        # return self._join_col(rows, "listed_in")
        return self.normalize(gnr)
 

    def build_actor_profile(self, user_id):
        # rows = self._watched_rows(user_id)
        row = self.get_user_row(user_id)
        cast = row["WatchedCAST"]
        # if rows.empty:
        #     return ""
        if cast == "": 
            return ""
        
        # return self._join_col(rows, "cast")
        return self.normalize(cast)
 

    def build_all(self, user_id):
        # rows = self._watched_rows(user_id)

        # if rows.empty:
        #     return "", "", ""
        
        # return (
        #     self._join_col(rows, "description"),
        #     self._join_col(rows, "listed_in"),
        #     self._join_col(rows, "cast"),
        # )
        d = self.build_desc_profile(user_id)
        g = self.build_genre_profile(user_id)
        c = self.build_actor_profile(user_id)
        if d == "" and g == "" and c == "" : 
            return "","",""
        return d,g,c