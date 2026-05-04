# Enhanced Netflix Recommendation System

Netflix's current recommendation system estimates how likely a user is to enjoy a title based on interactions with the service. Netflix also considers factors such as what members with similar tastes have watched, the time of day a user watches, preferred languages, viewing devices, and how long a user engages with individual titles. This project explores an NLP-based approach to recommendation by using title metadata and user watch history to generate content-based recommendations. The goal is to apply concepts from **CS 410: Text Information Systems**, including TF-IDF, Word2Vec, and similarity-based retrieval.

## About
This project is a content-based Netflix recommendation system that recommends titles based on a user's watch history. The recommender builds user profiles from previously watched titles and compares those profiles against a filtered Netflix title dataset using TF-IDF, Word2Vec, genre similarity, and production-country information.

### Datasets

#### License
As mentioned in the original repository, this dataset is under [CC0: Public Domain Dedication.](https://creativecommons.org/publicdomain/zero/1.0/) and all the credits goes to Victor Soeiro creator of this [dataset in Kaggle](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies).

The principal dataset used for this project comes from the GitHub repository [kaggle-netflix-tv-shows-and-movies](https://github.com/amirtds/kaggle-netflix-tv-shows-and-movies/tree/main). The dataset lists Netflix titles and related metadata, acquired in May 2022.

#### **Note — The version of the title dataset used in this project only uses a portion of the original data. It was filtered to include titles with United States production countries.* 

The original dataset contains more than 5,000 unique Netflix titles and 15 metadata columns. For this project, a smaller subset of variables was used to build and evaluate the recommendation system. The title recommendation dataset and the synthetic dataset of 100 users were both built from the filtered title data. A description of the following used variables has been provided below, followed by a brief description of their primary function in relation to the project.

#### Title Dataset:

* **title**: Matches watched and evaluation movies to the title dataset.
* **description**: Builds text-based similarity through TF-IDF and Word2Vec.
* **genres**: Builds genre-based similarity and adds a genre-overlap boost.
* **production_countries**: Supports country-profile similarity and the United States production-country filter.

#### User Dataset:
* **UserID**: Unique identifier for each user.
* **WatchedMovies**: Titles used to build the user's recommendation profile.
* **EvaluationMoviesTheyWillLike**: Held-out titles used to evaluate recommendation quality.


## Installation


## Usage


## Evaluation & Results


## Limitations

