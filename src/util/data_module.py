import bisect
import json
from collections import defaultdict
from typing import Tuple

import numpy as np
import numpy.typing as npt
from sklearn import neighbors

from data.dataset_type import DatasetType
from data.key_type import KeyType


class DataModule:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def load(self):

        # load user and item embeddings
        self.user_embeddings = self.__load_numpy_file__(KeyType.USER)
        self.item_embeddings = self.__load_numpy_file__(KeyType.ITEM)

        # load user and item mappings
        self.user_map = self.__load_mapper_json__(KeyType.USER)
        self.item_map = self.__load_mapper_json__(KeyType.ITEM)

        self.user_links = self.__load_key_type_links__(KeyType.USER)
        self.item_links = self.__load_key_type_links__(KeyType.ITEM)
        
        # load data and their respective links
        self.train_data = self.__load_ratings__(DatasetType.Train)
        self.validataion_data = self.__load_ratings__(DatasetType.Validation)
        self.test_data = self.__load_ratings__(DatasetType.Test)
        
        

    def __load_numpy_file__(self, key_type: KeyType) -> np.ndarray:
        return np.load(f"{self.data_dir}/{key_type.value}.embeddings.npy")

    def __load_mapper_json__(self, key_type: KeyType) -> dict:
        with open(f"{self.data_dir}/{key_type.value}_map.json") as f:
            data = json.load(f)
        return data

    def __load_key_type_links__(self, key_type: KeyType) -> Tuple(npt.ArrayLike, npt.ArrayLike):
        links = defaultdict(list)
        with open(f"{self.data_dir}/{key_type.value}.links", "r") as f:
            for line in f:
                neighbor1, neighbor2, reverse_connection = [int(x.strip()) for x in line.strip().split(",")]
                bisect.insort_left(links[neighbor1], neighbor2)
                if reverse_connection:
                    bisect.insort_left(links[neighbor2], neighbor1)

        # covert to indices and values for above links
        indices = []
        values = []

        for n1, neighbors in links.items():
            for n2 in neighbors:
                indices.append([n1, n2])
                values.append(1.0 / len(neighbors))

        indices = np.array(indices).astype(np.int64)
        values = np.array(values).astype(np.float32)
        return indices, values

    def __load_ratings__(self, dataset_type: DatasetType):
        
        ratings_by_user = defaultdict(list)
        user_consumed_items = defaultdict(list)
        item_consumed_users = defaultdict(list)
        with open(f"{self.data_dir}/{dataset_type.value}.ratings", "r") as f:
            for line in f:
                user, item, rating = [int(x.strip()) for x in line.strip().split(",")]
                # add user, item , rating info to ratings
                ratings_by_user[user].append((item, rating))
                # if item1, item3 is rated by user1 then user_consumed_items is {user1: [item1, item3 ]}
                bisect.insort_left(user_consumed_items[user], item)
                # if item1 is rated by user1 and user2 then item_consumed_users is {item1: [user1,user2]}
                bisect.insort_left(item_consumed_users[item], user)

        user_consumed_items_indices = []
        user_consumed_items_values = []
        for user, items in user_consumed_items.items():
            for item in items:
                user_consumed_items_indices.append([user, item])
                user_consumed_items_values.append(1.0 / len(items))

        user_consumed_items_indices = np.array(user_consumed_items_indices).astype(np.int64)
        user_consumed_items_values = np.array(user_consumed_items_values).astype(np.float32)

        item_consumed_users_indices = []
        item_consumed_users_values = []
        for item, users in item_consumed_users.items():
            for user in users:
                item_consumed_users_indices.append([item, user])
                item_consumed_users_values.append(1.0 / len(users))


        item_consumed_users_indices = np.array(item_consumed_users_indices).astype(np.int64)
        item_consumed_users_values = np.array(item_consumed_users_values).astype(np.float32)

        # sort each user's items
        for value in ratings_by_user.values():
            value.sort(lambda v:v[0])
        
        return {
            'ratings_by_user': ratings_by_user,
            'user_consumed_items': {
                'indices': user_consumed_items_indices,
                'values': user_consumed_items_values
            },
            'item_consumed_items': {
                'indices': item_consumed_users_indices,
                'values': item_consumed_users_values
            }
        }
    
    def get_training_batch(self, batch_index: int=0):
        num_users = len(self.user_map)
        
        batch_users = list(range(batch_index*self.batch_size, (batch_index+1)*self.batch_size))
        input_users = []
        input_items = []
        label_ratings = []
        ratings_by_user = self.train_data['ratings_by_user']
        for user in batch_users:
            if user not in ratings_by_user:
                continue
            input_users.append(user)
            input_items.append(ratings_by_user[user][0])
            label_ratings.append(ratings_by_user[user][1])
        
        input_users = np.reshape(input_users, [-1,1])
        input_items = np.reshape(input_items, [-1,1])
        label_ratings = np.reshape(label_ratings, [-1,1])
        
        # next batch index
        next_batch_index = batch_index+1 if batch_users[-1] < num_users-1 else -1
        
        return [input_users, input_items], label_ratings, next_batch_index
            
            