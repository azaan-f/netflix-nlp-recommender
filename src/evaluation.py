
class evaluation:

    def __init__(self, loader, retrieval):
        self.normalize      = loader.normalize
        self.get_eval       = loader.get_eval_titles
        self.retrieval      = retrieval


    # helper to count hits and get eval keys (for recall denominator)
    def _hits(self, recommended_titles, eval_titles):
        rec_keys  = [self.normalize(t) for t in recommended_titles]
        eval_keys = set(self.normalize(t) for t in eval_titles)
        return sum(1 for t in rec_keys if t in eval_keys), eval_keys
    

    # from earlier
    def precision_at_k(self, user_id, k=5, recs_df=None):
        if k == 0:
            return 0.0
        
        if recs_df is None:
            recs_df = self.retrieval.recommend_for_user(user_id, k=k)

        hits, _ = self._hits(recs_df["title"].tolist(), self.get_eval(user_id))
        return hits / k


    # from earlier
    def recall_at_k(self, user_id, k=5, recs_df=None):
        if recs_df is None:
            recs_df = self.retrieval.recommend_for_user(user_id, k=k)

        hits, eval_keys = self._hits(recs_df["title"].tolist(), self.get_eval(user_id))
        
        if not eval_keys:
            return 0.0
        
        return hits / len(eval_keys)


    # will show precision and recall for a user at k, using the retrieval's recommend_for_user
    def evaluate(self, user_id, k=5):
        recs_df = self.retrieval.recommend_for_user(user_id, k=k)

        return {
            "precision": self.precision_at_k(user_id, k=k, recs_df=recs_df),
            "recall":    self.recall_at_k(user_id,    k=k, recs_df=recs_df),
        }