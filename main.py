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
feat.compute_IDF(loader.dataset["description"], field="description")
feat.compute_IDF(loader.dataset["genres"],      field="genres")
feat.compute_IDF(loader.dataset["production_countries"], field="country")
ret.precompute_doc_matrices() 


# EDIT THIS TO TEST A DIFFERENT USER
test_id = loader.user_dataset["UserID"].tolist()[0] 


desc, genre, country = profile.build_all(test_id)
print(f"=== profile check: {test_id} ===")
print(f"  description    ({len(desc.split())} tokens): {desc[:80]}...")
print(f"  genre   ({len(genre.split())} tokens): {genre[:80]}")
print(f"  country ({len(country.split())} tokens): {country[:80]}")

watched  = loader.get_watched_movies(test_id)
eval_titles = loader.get_eval_titles(test_id)
print(f"\n  watched : {len(watched)} titles")
print(f"  eval    : {eval_titles}")


# Check how many eval titles actually exist in the dataset
eval_keys = set(loader.normalize(t) for t in eval_titles)
matches   = loader.dataset[loader.dataset["title_key"].isin(eval_keys)]
print(f"  eval titles found in dataset: {matches['title'].tolist()}")
if len(matches) < len(eval_titles):
    missing = eval_keys - set(matches["title_key"])
    print(f"  !! missing from dataset: {missing}")


# Run the full recommender and see where eval titles rank
print(f"\n=== full recommendation check: {test_id} ===")
recs_large = ret.recommend_for_user(test_id, k=len(loader.dataset))
for title in eval_titles:
    key = loader.normalize(title)
    rank = recs_large[recs_large["title"].apply(loader.normalize) == key]
    if not rank.empty:
        r = recs_large.index.get_loc(rank.index[0]) + 1
        score = rank.iloc[0]["score"]
        print(f"  '{title}' rank {r}, score {score:.4f}")
    else:
        print(f"  '{title}' not found in recs (missing from dataset)")


top5 = ret.recommend_for_user(test_id, k=5)
print(f"\n  top-5 recommendations:")
for _, row in top5.iterrows():
    print(f"    [{row['score']:.4f}] {row['title']}")


# evaluation
def evaluate_all_users(loader, ret, eva, k=25):
    precisions, recalls = [], []

    for user_id in loader.user_dataset["UserID"].tolist()[:60]:
        try:
            recs = ret.recommend_for_user(user_id, k=k)
            p = eva.precision_at_k(user_id, k=k, recs_df=recs)
            r = eva.recall_at_k(user_id,    k=k, recs_df=recs)
            precisions.append(p)
            recalls.append(r)
        except Exception as e:
            print(f"skipping {user_id}: {e}")

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls)    / len(recalls)    if recalls    else 0.0

    print(f"\nEvaluated {len(precisions)} users @ k={k}")
    print(f"Avg Precision@{k}: {avg_p:.4f}")
    print(f"Avg Recall@{k}:    {avg_r:.4f}")

    return avg_p, avg_r


evaluate_all_users(loader, ret, eva, k=25)