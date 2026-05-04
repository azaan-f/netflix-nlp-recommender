# Enhanced Netflix Recommendation System

Netflix's current recommendation system estimates how likely a user is to enjoy a title based on interactions with the service. Netflix also considers factors such as what members with similar tastes have watched, the time of day a user watches, preferred languages, viewing devices, and how long a user engages with individual titles. This project explores an NLP-based approach to recommendation by using title metadata and user watch history to generate content-based recommendations. The goal is to apply concepts from **CS 410: Text Information Systems**, including TF-IDF, Word2Vec, and similarity-based retrieval.

## About
This project is a content-based Netflix recommendation system that recommends titles based on a user's watch history. The recommender builds user profiles from previously watched titles and compares those profiles against a filtered Netflix title dataset using TF-IDF, Word2Vec, genre similarity, and production-country information.

### DISCLAIMER
The graphical UI for this project was built with assistance from AI tools. It is intended as a visual display layer for the recommendation results and does not replace or alter the underlying recommendation pipeline, feature extraction, scoring, or evaluation logic. This was not part of the initial project plans, but was included for the sake of usability/visual appeal for users. Consent to incorporate this feature was gained beforehand.

---

### Datasets

#### License
As mentioned in the original repository, this dataset is under [CC0: Public Domain Dedication.](https://creativecommons.org/publicdomain/zero/1.0/) and credit goes to Victor Soeiro, creator of this [dataset in Kaggle](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies).

The principal dataset used for this project comes from the GitHub repository [kaggle-netflix-tv-shows-and-movies](https://github.com/amirtds/kaggle-netflix-tv-shows-and-movies/tree/main). The dataset lists Netflix titles and related metadata, acquired in May 2022.

#### **Note — The version of the title dataset used in this project only uses a portion of the original data. It was filtered to include titles with United States production countries.* 

The original dataset contains more than 5,000 unique Netflix titles and 15 metadata columns. For this project, a smaller subset of variables was used to build and evaluate the recommendation system. The title recommendation dataset and the synthetic dataset of 100 users were both built from the filtered title data. A description of the following used variables has been provided below, paired with a brief description of their primary function in relation to the project.

#### Title Dataset:

* **`title`**: Matches watched and evaluation movies to the title dataset.
* **`description`**: Builds text-based similarity through TF-IDF and Word2Vec.
* **`genres`**: Builds genre-based similarity and adds a genre-overlap boost.
* **`production_countries`**: Supports country-profile similarity and the United States production-country filter.

#### User Dataset:
* **`UserID`**: Unique identifier for each user.
* **`WatchedMovies`**: Titles used to build the user's recommendation profile.
* **`EvaluationMoviesTheyWillLike`**: Held-out titles used to evaluate recommendation quality.

--- 

### Model Approach

The recommendation system follows a content-based retrieval approach, meaning that for each user, the model builds a profile from the titles they have already watched, then compares that profile against all unwatched titles in the filtered Netflix dataset.

The final score combines:

* **`Description TF-IDF Similarity`**: Measures word-level overlap between the user's watched-title descriptions and candidate-title descriptions.
* **`Word2Vec Similarity`**: Captures broader semantic similarity between descriptions.
* **`Genre Similarity`**: Compares the user's watched genres with candidate-title genres.
* **`Country Similarity`**: Lightly accounts for production-country patterns.
* **`Genre-overlap Bonus`**: Gives an extra boost to titles whose genres directly overlap with the user's watched-title genres.

Which are used to create the scoring formula:

```python
score =
    0.30 * description_tfidf_similarity
  + 0.30 * word2vec_description_similarity
  + 0.30 * genre_tfidf_similarity
  + 0.10 * country_tfidf_similarity
  + 0.10 * genre_overlap_bonus
```

## Installation & Usage

### Installation

Clone the repository:

```bash
git clone https://github.com/username/repo-name.git
cd repo-name
```

And install the necessary packages:

```bash
pip install numpy pandas nltk gensim pillow
```

#### **Note — This project also uses `tkinter` for the UI. Tkinter is included with most standard Python installations, but if it is missing, it may need to be installed through your Python distribution or system package manager.*

---

### Usage

**To launch the Netflix-style recommendation UI:**

```bash
python netflix_ui.py
```

The UI displays the model's recommendations in a simple Netflix-inspired layout that uses the same recommendation pipeline from the `src/` folder and shows the top recommended titles for a selected user profile. Upon running `netflix_ui.py`, it should look something like this. From here, different users can be selected and their recommendations can be visualized: 

**To run the recommendation and evaluation pipeline:**

```bash
python main.py
```
<br clear="center"/>

<img align="right" width="50%" alt="terminal output" src="https://github.com/user-attachments/assets/a9bd62b0-13f3-47d4-b8fd-0e8499dcacec" />

This script loads the filtered Netflix title dataset and user dataset, builds the TF-IDF and Word2Vec features, generates recommendations, and evaluates the model using Precision@K and Recall@K. For example, here are the results for the first user (U001) in the dataset after running `main.py`:

<br clear="center"/>


## Evaluation & Results
The recommender was evaluated using held-out titles from the synthetic user dataset.

* **Precision@K** measures the fraction of the top K recommendations that appear in the user's held-out evaluation list.
* **Recall@K** measures the fraction of the user's held-out evaluation titles that appear in the top K recommendations.

<br clear="center"/>

<img align="right" width="30%" alt="terminal output" src="https://github.com/user-attachments/assets/9186a122-75f9-44dd-baf1-8cd28b2bec23" />

Using the current scoring setup, the model captures broad content similarity but often ranks the exact held-out titles outside the top recommendations. This suggests that the system is better at identifying related titles than sharply ranking the exact evaluation titles near the top.

<br clear="center"/>

## Limitations
Some of the main limitations encountered during the project are the following:
* The system is fully content-based and does not use collaborative filtering.
* The user dataset is synthetic and only contains positive watch-history examples.
* The model does not use ratings, watch time, clicks, or explicit user feedback.
* Users with diverse watch histories can produce broad profiles that are harder to rank from.
* Evaluation is strict because each user has only a small number of held-out liked titles.

## Conclusion
Overall, this project demonstrates how NLP methods can be used to build a content-based recommendation system from title metadata and user watch history. By combining features such as TF-IDF, Word2Vec, genre similarity, production-country information, and a genre-overlap bonus, our system is able to recommend titles that are related to a user’s previous viewing history.

While the final model does not match the complexity of a real production recommendation system like Netflix’s, it provides a useful example of how text-based retrieval methods can be applied to recommendation tasks. Our evaluation results show that the model captures broad content similarity, but also highlight the difficulty of ranking exact held-out titles near the top when only limited user-history data is available.

Future improvements could include adding collaborative filtering, using real user-rating or watch-time data, experimenting with stronger sentence embeddings, and creating more detailed user profiles. Considering these limitations as well as time limitations, the project successfully applies core CS 410 concepts to a realistic recommendation problem and provides both a command-line evaluation pipeline and a visual UI for displaying recommendations, even if the project did not achieve everything it originally set out to.
