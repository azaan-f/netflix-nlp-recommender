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
        self.title_dataset = pd.read_csv(self.title_url)
        self.users_dataset = pd.read_csv(self.users, encoding='latin-1')

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