import math

import numpy as np
import tensorflow as tf


def ndcg(relevance_scores):
    if len(relevance_scores)==0:
        return 0
    dcg = sum(score/math.log2(2+idx) for idx, score in enumerate(relevance_scores))
    idcg = sum(
        score / math.log2(2 + idx)
        for idx, score in enumerate(sorted(relevance_scores, reverse=True))
    )

    return 0 if idcg==0 else dcg/idcg
    

def evaluate_hit_rate_and_ndcg(user_index_dict, true_scores, predict_scores, top_k=None):
    if top_k is None:
        top_k = len(true_scores)
    user_ndcg_values = []
    user_hit_rates = []
    for user, item_indices in user_index_dict.items():
        user_scores_true = list(np.concatenate(true_scores[item_indices]))
        user_scores_predict = list(np.concatenate(predict_scores[item_indices]))

        user_scores_true_sort_index =np.argsort(user_scores_true)[::-1][:top_k]
        
        user_scores_predict_sort_index =np.argsort(user_scores_predict)[::-1][:top_k]
        
        relevance_scores = [ 1 if item_idx in user_scores_predict_sort_index else 0 for item_idx in user_scores_true_sort_index]
        
        user_ndcg = ndcg(relevance_scores)
        
        user_ndcg_values.append(user_ndcg)
        user_hit_rates.append(np.mean(relevance_scores))
    
    return np.mean(user_hit_rates), np.mean(user_ndcg_values)
        
        
    
    
    
    