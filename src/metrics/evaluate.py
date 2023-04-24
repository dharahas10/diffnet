import math

import numpy as np
import tensorflow as tf


def compute_ndcg(result, sorted_result):
    dcg = compute_dcg(result)
    idcg = compute_dcg(sorted_result)
    return 0 if idcg == 0 else dcg / idcg


def compute_dcg(result):
    dcg = []
    for idx, val in enumerate(result):
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator = np.log2(idx + 2)
        score = numerator / denominator
        dcg.append(score)
    return sum(dcg)


def evaluate_hit_rate_and_ndcg_2(user_index_dict, positive_ratings, negative_ratings_user_dict, top_k=5):
    user_ndcg_values = []
    user_hit_rates = []

    for user, item_indices in user_index_dict.items():
        user_positive_ratings = list(np.concatenate(positive_ratings[item_indices]))
        user_positive_ratings_length = len(user_positive_ratings)
        target_length = min(user_positive_ratings_length, top_k)

        user_all_predict_ratings = user_positive_ratings + list(negative_ratings_user_dict[user])

        sort_index = np.argsort(user_all_predict_ratings)[::-1]
        relevance_score = []
        for idx in range(top_k):
            rank = sort_index[idx]
            if rank < user_positive_ratings_length:
                relevance_score.append(1)
            else:
                relevance_score.append(0)

        user_hit_rates.append(np.sum(relevance_score) / target_length)
        user_ndcg_values.append(compute_ndcg(relevance_score, sorted(relevance_score, reverse=True)))

    return np.mean(user_hit_rates), np.mean(user_ndcg_values)
