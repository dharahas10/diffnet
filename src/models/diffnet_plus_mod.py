import logging

import tensorflow as tf

from layers.fusion_layer import FusionLayer
from util.tf_helper import normalize_with_moments

log = logging.getLogger(__name__)


class DiffnetPlusMod(tf.keras.Model):
    def __init__(
        self, gcn_layers, dims, num_users, num_items, user_review_embeddings, item_review_embeddings, user_consumed_items, user_links, item_consumed_users, item_links, *args, **kwargs
    ) -> None:
        super(DiffnetPlusMod, self).__init__(*args, **kwargs)
        ## init variables

        self.low_att_std = 1.0
        self.dims = dims
        self.gcn_layers = gcn_layers
        self.num_users = num_users
        self.num_items = num_items
        self.user_review_embeddings = user_review_embeddings
        self.item_review_embeddings = item_review_embeddings
        self.user_consumed_items = user_consumed_items
        self.item_consumed_users = item_consumed_users
        self.user_links = user_links
        self.item_links = item_links
        log.info("Selected Model: DiffnetPlusMod")
        #### user related variables and layers
        ### describe variables
        ## node attention
        # consumed items
        self.user_consumed_items_sparse_values = tf.Variable(
            tf.random_normal_initializer(stddev=self.low_att_std)(shape=[len(self.user_consumed_items["indices"])], dtype=tf.float32),
            trainable=True,
            name="user_consumed_items_trainable_values_or_weights",
        )
        # neighbor users
        self.user_neighbors_sparse_values = tf.Variable(
            tf.random_normal_initializer(stddev=self.low_att_std)(shape=[len(self.user_links["indices"])], dtype=tf.float32),
            trainable=True,
            name="user_neighbors_sparse_values",
        )

        ## describe layers
        # reduce dimensions layer
        self.user_embeddings_reduce_dims = tf.keras.layers.Dense(
            self.dims,
            activation=tf.keras.activations.sigmoid,
            name="user_embeddings_reduce_dims",
        )
        # Embedding layer
        self.user_fusion_layer = FusionLayer(name="user_fusion_layer")

        ## Node attention
        # consumed items
        self.user_consumed_items_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="user_consumed_items_attention_layer_1")
        # neighbor users
        self.user_neighbors_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="user_neighbors_attention_layer_1")

        ## Graph attention

        # Graph attention for consumed items
        self.user_consumed_items_graph_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name="user_consumed_items_graph_attention_layer_1")
        self.user_consumed_items_graph_attention_layer_2 = tf.keras.layers.Dense(
            1,
            activation=tf.nn.leaky_relu,
            name="user_consumed_items_graph_attention_layer_2",
        )

        # Graph attention for neighbor users
        self.user_neighbors_graph_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name="user_neighbors_graph_attention_layer_1")
        self.user_neighbors_graph_attention_layer_2 = tf.keras.layers.Dense(
            1,
            activation=tf.nn.leaky_relu,
            name="user_neighbors_graph_attention_layer_2",
        )

        ### item related variables and layers

        ## describe variables
        self.item_consumed_users_sparse_values = tf.Variable(
            tf.random_normal_initializer(stddev=self.low_att_std)(shape=[len(self.item_consumed_users["indices"])], dtype=tf.float32),
            trainable=True,
            name="item_consumed_users_trainable_values_or_weights",
        )

        # neighbor items
        self.item_neighbors_sparse_values = tf.Variable(
            tf.random_normal_initializer(stddev=self.low_att_std)(shape=[len(self.item_links["indices"])], dtype=tf.float32),
            trainable=True,
            name="item_neighbors_sparse_values",
        )

        ## describe layers

        # reduce dimensions layer
        self.item_embedding_reduce_dims = tf.keras.layers.Dense(
            self.dims,
            activation=tf.keras.activations.sigmoid,
            name="item_embedding_reduce_dims",
        )
        # Embedding layer
        self.item_fusion_layer = FusionLayer(name="item_fusion_layer")

        # Node attention
        self.item_consumed_users_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="item_consumed_users_attention_layer_1")

        # neighbor items
        self.item_neighbors_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="item_neighbors_attention_layer_1")

        # Graph attention
        self.item_consumed_users_graph_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name="item_consumed_users_graph_attention_layer_1")
        self.item_consumed_users_graph_attention_layer_2 = tf.keras.layers.Dense(
            1,
            activation=tf.nn.leaky_relu,
            name="item_consumed_users_graph_attention_layer_2",
        )

        self.item_neighbors_graph_attention_layer_1 = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name="item_neighbors_graph_attention_layer_1")
        self.item_neighbors_graph_attention_layer_2 = tf.keras.layers.Dense(
            1,
            activation=tf.nn.leaky_relu,
            name="item_neighbors_graph_attention_layer_2",
        )

    # @profile(stream=fp)
    # @tf.function
    def call(self, inputs, training=False):
        user_input, item_input = inputs
        ## user embeddings

        # normalize user review embeddings
        user_review_embeddings_norm = normalize_with_moments(self.user_review_embeddings, axes=[0, 1])
        user_review_embeddings_norm = self.user_embeddings_reduce_dims(user_review_embeddings_norm)
        user_review_embeddings_norm = normalize_with_moments(user_review_embeddings_norm, axes=[0, 1])
        # fusion layer
        user_fusion_embeddings = self.user_fusion_layer(user_review_embeddings_norm)

        ## item embeddings

        # normalize item review embeddings
        item_review_embeddings_norm = normalize_with_moments(self.item_review_embeddings, axes=[0, 1])
        item_review_embeddings_norm = self.item_embedding_reduce_dims(item_review_embeddings_norm)
        item_review_embeddings_norm = normalize_with_moments(item_review_embeddings_norm, axes=[0, 1])
        # fusion layer
        item_fusion_embeddings = self.item_fusion_layer(item_review_embeddings_norm)

        user_gcn_layer_embeddings_list = [user_fusion_embeddings]
        item_gcn_layer_embeddings_list = [item_fusion_embeddings]

        current_gcn_layer = 0
        current_user_gcn_embeddings = user_fusion_embeddings
        current_item_gcn_embeddings = item_fusion_embeddings
        while current_gcn_layer < self.gcn_layers:
            ## user consumed items node attention
            # generate trained weights for each user consumed item
            user_consumed_items_values = tf.reshape(self.user_consumed_items_sparse_values, [-1, 1])
            user_consumed_items_values = self.user_consumed_items_attention_layer_1(user_consumed_items_values)
            user_consumed_items_values = tf.reduce_sum(tf.math.exp(user_consumed_items_values), axis=1)

            # convert to sparse tensor
            user_consumed_items_sparse_matrix = tf.SparseTensor(
                indices=self.user_consumed_items["indices"],
                values=user_consumed_items_values,
                dense_shape=[self.num_users, self.num_items],
            )
            # softmax values
            user_consumed_items_attention_matrix = tf.sparse.softmax(user_consumed_items_sparse_matrix)

            # matrix multiply user_consumed_items_attention_matrix with item_fusion_embeddings to get the updated updated user_embeddings based on consumed items
            user_embeddings_from_consumed_items = tf.sparse.sparse_dense_matmul(user_consumed_items_attention_matrix, current_item_gcn_embeddings)

            ## user social neighbors node attention
            # generate trained weights for each user neighbors
            user_neighbors_values = tf.reshape(self.user_neighbors_sparse_values, [-1, 1])
            user_neighbors_values = self.user_neighbors_attention_layer_1(user_neighbors_values)
            user_neighbors_values = tf.reduce_sum(tf.math.exp(user_neighbors_values), axis=1)

            # convert to sparse tensor
            user_neighbors_sparse_matrix = tf.SparseTensor(
                indices=self.user_links["indices"],
                values=user_neighbors_values,
                dense_shape=[self.num_users, self.num_users],
            )
            # softmax values
            user_neighbors_sparse_attention_matrix = tf.sparse.softmax(user_neighbors_sparse_matrix)

            # matrix multiply user_neighbors_sparse_attention_matrix with user_fusion_embeddings to get the updated user_embeddings based on user links/connections/neighbors
            user_embeddings_from_user_links = tf.sparse.sparse_dense_matmul(user_neighbors_sparse_attention_matrix, current_user_gcn_embeddings)

            user_consumed_items_graph_attention_embeddings = tf.concat([current_user_gcn_embeddings + user_embeddings_from_consumed_items], 1)
            user_consumed_items_graph_attention_embeddings = self.user_consumed_items_graph_attention_layer_1(user_consumed_items_graph_attention_embeddings)
            user_consumed_items_graph_attention_embeddings = self.user_consumed_items_graph_attention_layer_2(user_consumed_items_graph_attention_embeddings)
            user_consumed_items_graph_attention_embeddings = tf.math.exp(user_consumed_items_graph_attention_embeddings) + 0.7  # TODO try removing bias factor

            user_neighbors_graph_attention_embeddings = tf.concat([current_user_gcn_embeddings, user_embeddings_from_user_links], 1)
            user_neighbors_graph_attention_embeddings = self.user_neighbors_graph_attention_layer_1(user_neighbors_graph_attention_embeddings)
            user_neighbors_graph_attention_embeddings = self.user_neighbors_graph_attention_layer_2(user_neighbors_graph_attention_embeddings)
            user_neighbors_graph_attention_embeddings = tf.math.exp(user_neighbors_graph_attention_embeddings) + 0.3  # TODO try removing bias factor

            # compute weight/factor for consumed_items and neighbors
            user_total_attention_embeddings = user_consumed_items_graph_attention_embeddings + user_neighbors_graph_attention_embeddings
            user_consumed_items_attention_weight = user_consumed_items_graph_attention_embeddings / user_total_attention_embeddings
            user_neighbors_attention_weight = user_neighbors_graph_attention_embeddings / user_total_attention_embeddings

            # final user gcn embeddings
            user_gcn_embedding = 0.5 * current_user_gcn_embeddings + 0.5 * (
                user_consumed_items_attention_weight * user_embeddings_from_consumed_items + user_neighbors_attention_weight * user_embeddings_from_user_links
            )

            ## item node attention

            # item item attention embeddings
            # item_item_graph_attention_embeddings = self.item_item_graph_attention_layer_1(current_item_gcn_embeddings)
            # item_item_graph_attention_embeddings = self.item_item_graph_attention_layer_2(item_item_graph_attention_embeddings) + 1.0 # TODO check on bias
            ## user social neighbors node attention
            # generate trained weights fro each user neighbors
            item_neighbors_values = tf.reshape(self.item_neighbors_sparse_values, [-1, 1])
            item_neighbors_values = self.item_neighbors_attention_layer_1(item_neighbors_values)
            item_neighbors_values = tf.reduce_sum(tf.math.exp(item_neighbors_values), axis=1)

            # convert to sparse tensor
            item_neighbors_sparse_matrix = tf.SparseTensor(
                indices=self.item_links["indices"],
                values=item_neighbors_values,
                dense_shape=[self.num_items, self.num_items],
            )
            # softmax values
            item_neighbors_sparse_attention_matrix = tf.sparse.softmax(item_neighbors_sparse_matrix)

            # matrix multiply user_neighbors_sparse_attention_matrix with user_fusion_embeddings to get the updated user_embeddings based on user links/connections/neighbors
            item_embeddings_from_item_links = tf.sparse.sparse_dense_matmul(item_neighbors_sparse_attention_matrix, current_item_gcn_embeddings)

            item_neighbors_graph_attention_embeddings = tf.concat([current_item_gcn_embeddings, item_embeddings_from_item_links], 1)
            item_neighbors_graph_attention_embeddings = self.item_neighbors_graph_attention_layer_1(item_neighbors_graph_attention_embeddings)
            item_neighbors_graph_attention_embeddings = self.item_neighbors_graph_attention_layer_2(item_neighbors_graph_attention_embeddings)
            item_neighbors_graph_attention_embeddings = tf.math.exp(item_neighbors_graph_attention_embeddings) + 0.5  # TODO try removing bias factor

            # item consumed users embeddings
            item_consumed_users_values = tf.reshape(self.item_consumed_users_sparse_values, [-1, 1])
            item_consumed_users_values = self.item_consumed_users_attention_layer_1(item_consumed_users_values)
            item_consumed_users_values = tf.reduce_sum(tf.math.exp(item_consumed_users_values), axis=1)

            # convert to sparse tensor
            item_consumed_users_sparse_matrix = tf.SparseTensor(
                indices=self.item_consumed_users["indices"],
                values=item_consumed_users_values,
                dense_shape=[self.num_items, self.num_users],
            )
            # softmax values
            item_consumed_users_sparse_attention_matrix = tf.sparse.softmax(item_consumed_users_sparse_matrix)

            # multiply item_consumed_users_sparse_attention_matrix with user_fusion_embeddings to get the updated item_embeddings based on users
            item_embeddings_from_consumed_users = tf.sparse.sparse_dense_matmul(item_consumed_users_sparse_attention_matrix, current_user_gcn_embeddings)

            # compute attention embeddings for items based on users
            item_consumed_users_graph_attention_embeddings = tf.concat([current_item_gcn_embeddings, item_embeddings_from_consumed_users], 1)
            item_consumed_users_graph_attention_embeddings = self.item_consumed_users_graph_attention_layer_1(item_consumed_users_graph_attention_embeddings)
            item_consumed_users_graph_attention_embeddings = self.item_consumed_users_graph_attention_layer_2(item_consumed_users_graph_attention_embeddings)
            item_consumed_users_graph_attention_embeddings = tf.math.exp(item_consumed_users_graph_attention_embeddings) + 0.5  # TODO check bias weight later

            # compute weight/factor for consumed_users and items
            item_total_attention_embeddings = item_neighbors_graph_attention_embeddings + item_consumed_users_graph_attention_embeddings
            item_consumed_users_attention_weight = item_consumed_users_graph_attention_embeddings / item_total_attention_embeddings
            item_neighbors_attention_weight = item_neighbors_graph_attention_embeddings / item_total_attention_embeddings

            item_gcn_embedding = 0.5 * current_item_gcn_embeddings + 0.5 * (
                item_consumed_users_attention_weight * item_embeddings_from_consumed_users + item_neighbors_attention_weight * item_embeddings_from_item_links
            )

            # update gcn embeddings
            user_gcn_layer_embeddings_list.append(user_gcn_embedding)
            item_gcn_layer_embeddings_list.append(item_gcn_embedding)

            current_user_gcn_embeddings = user_gcn_embedding
            current_item_gcn_embeddings = item_gcn_embedding
            current_gcn_layer += 1

        user_gcn_embeddings_final = tf.concat(user_gcn_layer_embeddings_list, 1)
        item_gcn_embeddings_final = tf.concat(item_gcn_layer_embeddings_list, 1)

        user_input_latent_embeddings = tf.gather_nd(user_gcn_embeddings_final, user_input)
        item_input_latent_embeddings = tf.gather_nd(item_gcn_embeddings_final, item_input)

        # prediction
        predict_vector = tf.multiply(
            user_input_latent_embeddings,
            item_input_latent_embeddings,
            name="prediction_vector",
        )
        return tf.math.sigmoid(tf.math.reduce_sum(predict_vector, 1, keepdims=True), name="prediction")
