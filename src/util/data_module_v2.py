import bisect
import json
import logging
import math
from collections import defaultdict
from operator import index

import numpy as np

from data.dataset_type import DatasetType
from data.key_type import KeyType

log = logging.getLogger(__name__)


class DataModule:
    def __init__(self, data_dir, num_negatives=8, num_evaluate=1000, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.num_evaluate = num_evaluate

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
        self.validation_data = self.__load_ratings__(DatasetType.Validation)
        self.test_data = self.__load_ratings__(DatasetType.Test)

    def __load_numpy_file__(self, key_type: KeyType):
        return np.load(f"{self.data_dir}/{key_type.value}.embeddings.npy")

    def __load_mapper_json__(self, key_type: KeyType) -> dict:
        with open(f"{self.data_dir}/{key_type.value}_map.json") as f:
            data = json.load(f)
        return data

    def __load_key_type_links__(self, key_type: KeyType):
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
        return {"indices": indices, "values": values}

    def __load_ratings__(self, dataset_type: DatasetType):
        ratings_by_user = defaultdict(list)
        user_consumed_items = defaultdict(list)
        item_consumed_users = defaultdict(list)
        user_items_dict = defaultdict(list)
        with open(f"{self.data_dir}/{dataset_type.value}.ratings", "r") as f:
            for line in f:
                user, item, rating = [int(x.strip()) for x in line.strip().split(",")]
                # user -> items dict
                user_items_dict[user].append(item)
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
                item_consumed_users_values.append(1.0)

        item_consumed_users_indices = np.array(item_consumed_users_indices).astype(np.int64)
        item_consumed_users_values = np.array(item_consumed_users_values).astype(np.float32)

        # sort each user's items
        for value in ratings_by_user.values():
            value.sort(key=lambda v: v[0])

        # add negatives
        num_negatives = self.num_negatives
        if dataset_type == DatasetType.Test:
            num_negatives = self.num_evaluate

        user_negative_items_dict = defaultdict(list)
        num_items = len(self.item_map)
        for user, items in user_items_dict.items():
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                if j in items:
                    j = np.random.randint(num_items)
                user_negative_items_dict[user].append(j)
        return {
            "ratings_by_user": ratings_by_user,
            "user_consumed_items": {
                "indices": user_consumed_items_indices,
                "values": user_consumed_items_values,
            },
            "item_consumed_items": {
                "indices": item_consumed_users_indices,
                "values": item_consumed_users_values,
            },
            "user_items_dict": user_items_dict,
            "user_negative_items_dict": user_negative_items_dict,
        }

    def train_data_batch_generator(self):
        num_users = len(self.user_map)
        num_items = len(self.item_map)
        users_list = list(range(num_users))
        user_batches = [users_list[i : i + self.batch_size] for i in range(0, num_users, self.batch_size)]

        ratings_by_user = self.train_data["ratings_by_user"]
        user_negative_items_dict = self.train_data["user_negative_items_dict"]
        for user_batch in user_batches:
            input_users = []
            input_items = []
            label_ratings = []
            for user in user_batch:
                if user not in ratings_by_user:
                    continue
                for item_rating in ratings_by_user[user]:
                    input_users.append(user)
                    input_items.append(item_rating[0])
                    label_ratings.append(item_rating[1])

                # add negative data points
                user_negative_items = user_negative_items_dict[user]
                for item in user_negative_items:
                    input_users.append(user)
                    input_items.append(item)
                    label_ratings.append(0)

            input_users = np.reshape(np.array(input_users).astype(np.int64), [-1, 1])
            input_items = np.reshape(np.array(input_items).astype(np.int64), [-1, 1])
            label_ratings = np.reshape(np.array(label_ratings).astype(np.float32), [-1, 1])

            yield input_users, input_items, label_ratings

    def get_validation_data(self):  # sourcery skip: class-extract-method
        ratings_by_user = self.validation_data["ratings_by_user"]
        input_users = []
        input_items = []
        label_ratings = []

        user_negative_items_dict = self.train_data["user_negative_items_dict"]
        for user, item_ratings in ratings_by_user.items():
            for item_rating in item_ratings:
                input_users.append(user)
                input_items.append(item_rating[0])
                label_ratings.append(item_rating[1])

            # add negative data points
            user_negative_items = user_negative_items_dict[user]
            for item in user_negative_items:
                input_users.append(user)
                input_items.append(item)
                label_ratings.append(0)

        input_users = np.reshape(np.array(input_users).astype(np.int64), [-1, 1])
        input_items = np.reshape(np.array(input_items).astype(np.int64), [-1, 1])
        label_ratings = np.reshape(np.array(label_ratings).astype(np.float32), [-1, 1])

        return input_users, input_items, label_ratings

    def get_test_data_positive(self):
        ratings_by_user = self.test_data["ratings_by_user"]
        input_users = []
        input_items = []
        label_ratings = []
        user_index_dict = defaultdict(list)
        index_counter = 0
        for user, item_ratings in ratings_by_user.items():
            for item_rating in item_ratings:
                input_users.append(user)
                input_items.append(item_rating[0])
                label_ratings.append(item_rating[1])
                user_index_dict[user].append(index_counter)
                index_counter += 1

        input_users = np.reshape(np.array(input_users).astype(np.int64), [-1, 1])
        input_items = np.reshape(np.array(input_items).astype(np.int64), [-1, 1])
        label_ratings = np.reshape(np.array(label_ratings).astype(np.float32), [-1, 1])

        return input_users, input_items, label_ratings, user_index_dict

    def get_test_data_negative(self):
        user_negative_items_dict = self.test_data["user_negative_items_dict"]
        users_list = list(user_negative_items_dict.keys())
        user_batches = [users_list[i : i + self.batch_size] for i in range(0, len(users_list), self.batch_size)]

        for user_batch in user_batches:
            input_users = []
            input_items = []
            for user in user_batch:
                if user not in user_negative_items_dict:
                    continue
                # add negative data points
                user_negative_items = user_negative_items_dict[user]
                for item in user_negative_items:
                    input_users.append(user)
                    input_items.append(item)

            input_users = np.reshape(np.array(input_users).astype(np.int64), [-1, 1])
            input_items = np.reshape(np.array(input_items).astype(np.int64), [-1, 1])

            yield input_users, input_items, user_batch
