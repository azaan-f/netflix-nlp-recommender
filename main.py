from src.load_data import load_data
from src.features import features
from src.user_profiles import user_profiles
from src.retrieval import retrieval
from src.evaluation import evaluation

loader  = load_data()
loader.preprocess_all_columns() 

feat    = features(loader)
profile = user_profiles(loader)
ret     = retrieval(loader, feat, profile)
eva     = evaluation(loader, ret)

feat.build_vocabulary()
feat.train_word2vec()

for user_id in loader.user_dataset["UserID"].tolist():
    recs = ret.recommend_for_user(user_id, k=5)
    results = eva.evaluate(user_id, k=5)
    print(f"{user_id} — Precision: {results['precision']:.2f}, Recall: {results['recall']:.2f}")

watched = loader.get_watched_movies("U02")
watched_keys = set(loader.normalize(t) for t in watched)
found = loader.dataset[loader.dataset["title_key"].isin(watched_keys)]
print(f"Watched: {len(watched)}, Found in dataset: {len(found)}")
print(found["title"].tolist())