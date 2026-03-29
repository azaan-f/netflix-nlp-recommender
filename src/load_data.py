import re # for removing punctuation
import nltk
import pandas as pd

from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)


class load_data:
    # csvs from the dataset folder
    title_url = 'https://raw.githubusercontent.com/azaan-f/netflix-nlp-recommender/main/datasets/netflix_titles.csv'
    # title_url2 = 'https://raw.githubusercontent.com/azaan-f/netflix-nlp-recommender/main/datasets/titles.csv' # larger of the two i believe
    users = 'https://raw.githubusercontent.com/azaan-f/netflix-nlp-recommender/main/datasets/users.csv'

    def __init__(self):
        self.dataset = pd.read_csv(self.title_url)
        self.user_dataset = pd.read_csv(self.users, encoding='latin-1')

        self.stop_words = set(stopwords.words('english'))

        self.dataset["title_key"] = (
            self.dataset["title"]
            .fillna("")
            .astype(str)
            .apply(self.normalize)
        )


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
    

    # normalize a list of titles and return a set of unique normalized titles
    def normalize_title_list(self, titles):
        return set(self.normalize(title) for title in titles if str(title).strip())
    

    # switching to a column rather than just doing descriptions, should help fix bugs
    def preprocess_columns(self, col):
        self.dataset[col] = (
            self.dataset[col]
            .fillna("")
            .astype(str)
            .apply(self.normalize)
        )

        return self

    
    # columns to preprocess
    def preprocess_all_columns(self):
        for col in ("description", "listed_in", "cast"):
            self.preprocess_columns(col)
        self.optimizaiton()
        return self
    




    # helper functions
    def get_user_row(self, user_id):
        row = self.user_dataset[self.user_dataset['UserID'] == user_id]

        if row.empty:
            raise ValueError(f"user_id {user_id!r} not found")
        
        return row.iloc[0]
    

    def get_watched_movies(self, user_id): # should be a little cleaner
        watched = str(self.get_user_row(user_id)["WatchedMovies"])

        return [t.strip() for t in watched.split(",") if t.strip()]
    

    def get_eval_titles(self, user_id): # should be a little cleaner
        eval_movies = str(self.get_user_row(user_id)["EvaluationMoviesTheyWillLike"])

        return [t.strip() for t in eval_movies.split(",") if t.strip()]
    

    #function that builds up user data pre-model in order to limit localization of watched movie info 
    def optimizaiton(self) : 
        movDESC = []
        movCAST = []
        movGNRA = []
        for i in range(len(self.user_dataset)): 
            movWTCH = self.user_dataset["WatchedMovies"][i]
            movWTCH = movWTCH.split(",")
            movLS = []
            for j in movWTCH: 
                j = j.strip()
                if j != "":
                    movLS.append(j)
            descTXT = ""
            castTXT = ""
            gnraTXT = ""
            for t in movLS: 
                tk = self.normalize(t)
                disc = self.dataset[self.dataset["title_key"] == tk] 
                if not disc.empty: 
                    r = disc.iloc[0]
                    descTXT = descTXT + " " + r["description"]
                    castTXT = castTXT + " " + r["cast"]
                    gnraTXT = gnraTXT + " " + r["listed_in"] 
            movDESC.append(descTXT.strip())
            movCAST.append(castTXT.strip())
            movGNRA.append(gnraTXT.strip())
        self.user_dataset["WatchedDescriptions"] = movDESC
        self.user_dataset["WatchedCAST"] = movCAST
        self.user_dataset["WatchedGENRE"] = movGNRA
        return self