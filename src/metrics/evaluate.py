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


def get_dcg(idx):
    return 1 / math.log2(idx+2)

def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg += math.log(2)/math.log2(i+2)
    return idcg

def normalizedDiscountedCumulativeGain(result, sorted_result): 
    dcg = discountedCumulativeGain(result)
    idcg = discountedCumulativeGain(sorted_result)
    return 0 if idcg == 0 else dcg / idcg

def discountedCumulativeGain(result):
    dcg = []
    for idx, val in enumerate(result): 
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2) 
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg)

def evaluate_hit_rate_and_ndcg_2(user_index_dict, positive_ratings, negative_ratings_user_dict, top_k=5):
    
    user_ndcg_values = []
    user_hit_rates = []
    
    for user, item_indices in user_index_dict.items():
        user_positive_ratings = list(np.concatenate(positive_ratings[item_indices]))
        user_positive_ratings_length = len(user_positive_ratings)
        target_length = min(user_positive_ratings_length, top_k)
        
        user_all_predict_ratings= user_positive_ratings + list(negative_ratings_user_dict[user])
        
        sort_index = np.argsort(user_all_predict_ratings)[::-1]
        
        # tmp_user_hr_list = []
        # tmp_user_dcg_list = []
        # for idx in range(top_k):
        #     rank = sort_index[idx]
        #     if rank >= user_positive_ratings_length: continue
        #     tmp_user_hr_list.append(1.0)
        #     tmp_user_dcg_list.append(get_dcg(idx))
        
        # idcg_val = get_idcg(target_length)
        relevance_score = []
        for idx in range(top_k):
            rank = sort_index[idx]
            if rank < user_positive_ratings_length:
                relevance_score.append(1)
            else:
                relevance_score.append(0)
        
        user_hit_rates.append(np.sum(relevance_score)/ target_length)
        user_ndcg_values.append(normalizedDiscountedCumulativeGain(relevance_score, sorted(relevance_score, reverse=True)))
    
    return np.mean(user_hit_rates), np.mean(user_ndcg_values)
                
            
        
        
        
        
        
        
    
    
    
    