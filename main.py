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
feat.compute_IDF(loader.dataset["description"])
ret.precompute_doc_matrices() 


test_id = loader.user_dataset["UserID"].tolist()[0]
desc, genre, actor = profile.build_all(test_id)
print("Desc profile:", desc[:100])
print("Genre profile:", genre[:100])
print("Actor profile:", actor[:100])

eval_titles = loader.get_eval_titles(test_id)
eval_keys = set(loader.normalize(t) for t in eval_titles)
print("Eval keys:", eval_keys)

matches = loader.dataset[loader.dataset["title_key"].isin(eval_keys)]
print("Eval titles found in dataset:", matches["title"].tolist())

if not matches.empty:
    for _, row in matches.iterrows():
        idx = loader.dataset.index.get_loc(row.name)
        s = (ret.w_tfidf  * ret.desc_matrix[idx].dot(feat.text2TFIDF(desc)) +
             ret.w_genre  * ret.genre_matrix[idx].dot(feat.text2TFIDF(genre)) +
             ret.w_actors * ret.actor_matrix[idx].dot(feat.text2TFIDF(actor)))
        print(f"  {row['title']} score: {s:.4f}")

def evaluate_all_users(loader, ret, eva, k=50):
    precisions = []
    recalls    = []

    for user_id in loader.user_dataset["UserID"].tolist():
        try:
            recs = ret.recommend_for_user(user_id, k=k)
            p = eva.precision_at_k(user_id, k=k, recs_df=recs)
            r = eva.recall_at_k(user_id,    k=k, recs_df=recs)
            precisions.append(p)
            recalls.append(r)
        except Exception as e:
            print(f"Skipping {user_id}: {e}")

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls)    / len(recalls)    if recalls    else 0.0

    print(f"\nEvaluated {len(precisions)} users @ k={k}")
    print(f"Avg Precision@{k}: {avg_p:.4f}")
    print(f"Avg Recall@{k}:    {avg_r:.4f}")

    return avg_p, avg_r


evaluate_all_users(loader, ret, eva, k=50)