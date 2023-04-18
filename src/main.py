import logging
import logging.config
import os
import sys
import time
from datetime import datetime

import tensorflow as tf

from metrics.evaluate import evaluate_hit_rate_and_ndcg
from models.diffnet_plus import DiffnetPlus
from models.difnet_plus_mod import DiffnetPlusMod
from util.data_module import DataModule

LOG_DIR = "./logs"

def setup_logging():
    """Load logging configuration"""
    config_path = "./config/logging.ini"

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    timestamp = datetime.now().strftime("%y%m%d-%H:%M:%S")
    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": f"{LOG_DIR}/{timestamp}.log"},
    )
    

fp = open("memory_reports/main.log", "w+")
from memory_profiler import profile


@profile(stream=fp)
def train_epoch_batch(model, optimizer, epoch_loss_avg, input_users, input_items, label_ratings):
    with tf.GradientTape() as tape:
        y_predict = model([input_users, input_items], training=True)
        #compute loss
        loss_value = tf.nn.l2_loss(label_ratings-y_predict, name="training_loss")

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg.update_state(loss_value)
    del y_predict, grads

@profile(stream=fp)
def train_epoch(epoch, model, optimizer, data_module):
    log = logging.getLogger(__name__)
    epoch_loss_avg = tf.keras.metrics.Mean()
    start_time = time.time()
    for step, (input_users, input_items, label_ratings) in enumerate(data_module.train_data_batch_generator(), start=1):
        log.info(f"Current epoch: {epoch} and step: {step}")
        train_epoch_batch(model, optimizer, epoch_loss_avg, input_users, input_items, label_ratings) # type: ignore
    epoch_time = time.time() - start_time



    # validation
    validation_loss_avg = tf.keras.metrics.Mean()
    validation_input_users, validation_input_items, validation_label_ratings, validation_user_index_dict = data_module.get_validation_data()
    validation_y_predict = model([validation_input_users, validation_input_items])
    validation_loss_value = tf.nn.l2_loss(validation_label_ratings-validation_y_predict, name="validation_loss")
    validation_loss_avg.update_state(validation_loss_value)

    # metrics hit_rate and ndcg
    hit_rate, ndcg = evaluate_hit_rate_and_ndcg(validation_user_index_dict, validation_label_ratings, validation_y_predict.numpy())

    log.info(f"Epoch: {epoch}: Loss: {epoch_loss_avg.result()} Validation Loss: {validation_loss_avg.result()}  Validation HR: {hit_rate} Validation NDCG: {ndcg} Time Elapsed:{epoch_time}")


@profile(stream=fp)
def main():
    ## hyperparameter
    dims=300
    gcn_layers = 2
    epochs=1
    batch_size=32
    top_k=20
    log = logging.getLogger(__name__)
    log.info("Diffnet model training started")

    data_dir = "./data/yelp_10"
    if not os.path.isdir(data_dir):
        log.error("Data directory not found")
        sys.exit()

    # load data
    log.info("Loading dataset")

    data_module = DataModule(data_dir, batch_size=batch_size)
    data_module.load()

    train_data= data_module.train_data
    log.info("Data loaded !!!!!!")
    model = DiffnetPlus(gcn_layers=gcn_layers,
                        dims=dims,
                        num_users=len(data_module.user_map),
                        num_items=len(data_module.item_map),
                        user_review_embeddings=data_module.user_embeddings,
                        item_review_embeddings=data_module.item_embeddings,
                        user_consumed_items=train_data['user_consumed_items'],
                        item_consumed_users=train_data['item_consumed_items'],
                        user_links=data_module.user_links,
                        item_links=data_module.item_links)


    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    ## train the model
    for epoch in range(epochs):
        train_epoch(epoch, model, optimizer, data_module) # type: ignore


    # test
    test_loss_avg = tf.keras.metrics.Mean()
    test_input_users, test_input_items, test_label_ratings, test_user_index_dict = data_module.get_test_data()
    test_y_predict = model([test_input_users, test_input_items])
    test_loss_value = tf.nn.l2_loss(test_label_ratings-test_y_predict, name="test_loss")
    test_loss_avg.update_state(test_loss_value)

    # metrics hit_rate and ndcg
    hit_rate, ndcg = evaluate_hit_rate_and_ndcg(test_user_index_dict, test_label_ratings, test_y_predict.numpy(), top_k=top_k)
    log.info(f"Test Loss: {test_loss_avg.result()}  Test HR: {hit_rate} Test NDCG: {ndcg}")

if __name__ == "__main__":
    # setup logging
    setup_logging()
    main() # type: ignore