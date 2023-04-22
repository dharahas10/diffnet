import argparse
import gc
import logging
import logging.config
import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from metrics.evaluate import evaluate_hit_rate_and_ndcg, evaluate_hit_rate_and_ndcg_2
from models.diffnet_plus import DiffnetPlus
from models.difnet_plus_mod import DiffnetPlusMod
from util.data_module_v2 import DataModule

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
    

# fp = open("memory_reports/main.log", "w+")
# from memory_profiler import profile


# @profile(stream=fp)
# @tf.function
def train_epoch_batch(model, optimizer, epoch_loss_avg, input_users, input_items, label_ratings):
    log = logging.getLogger(__name__)
    
    with tf.GradientTape() as tape:
        y_predict = model([input_users, input_items], training=True)
        #compute loss
        loss_value = tf.nn.l2_loss(label_ratings-y_predict, name="training_loss")

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg.update_state(loss_value)
    del y_predict, grads

# @profile(stream=fp)
# @tf.function
def train_epoch(epoch, model, optimizer, epoch_loss_avg, data_module):
    log = logging.getLogger(__name__)
    
    for step, (input_users, input_items, label_ratings) in enumerate(data_module.train_data_batch_generator(), start=1):
        # log.info(f"Current epoch: {epoch} and step: {step} and GC collected: {gc.collect()} count: {gc.get_count()}")
        # log.info(f"Current epoch: {epoch} and step: {step}")
        train_epoch_batch(model, optimizer, epoch_loss_avg, input_users, input_items, label_ratings) # type: ignore


# @profile(stream=fp)
def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--model_name', type=str, default="DiffnetPlusMod", metavar='N', help='Model')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training')
    parser.add_argument('--dims', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--gcn_layers', type=int, default=2, metavar='N', help='GCN layers')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--num_negatives', type=int, default=8, metavar='N', help='number negative ratings in train')
    parser.add_argument('--num_evaluate', type=int, default=1000, metavar='N', help='number of false positive ratings in test for evaluation')
    args = parser.parse_args()
    
    ## hyperparameter
    dims=args.dims
    gcn_layers = args.gcn_layers
    epochs=args.epochs
    batch_size=args.batch_size
    num_negatives=args.num_negatives
    num_evaluate=args.num_evaluate
    learning_rate=args.lr
    log = logging.getLogger(__name__)

    data_dir = "./data/yelp_10"
    if not os.path.isdir(data_dir):
        log.error("Data directory not found")
        sys.exit()

    log.info(f"Current config: dims: {dims} gcn_layers: {gcn_layers} epochs: {epochs} batch_size: {batch_size} num_negatives: {num_negatives} and num_evaluate={num_evaluate} and learning_rate={learning_rate}")
    # load data
    log.info(f"Loading dataset for dir: {data_dir}")

    data_module = DataModule(data_dir,num_negatives=num_negatives, num_evaluate=num_evaluate, batch_size=batch_size)
    data_module.load()
    log.info("Data loaded successful!!!!!!")
    train_data= data_module.train_data

    if args.model_name=="DiffnetPlusMod":
        model = DiffnetPlusMod(gcn_layers=gcn_layers,
                            dims=dims,
                            num_users=len(data_module.user_map),
                            num_items=len(data_module.item_map),
                            user_review_embeddings=data_module.user_embeddings,
                            item_review_embeddings=data_module.item_embeddings,
                            user_consumed_items=train_data['user_consumed_items'],
                            item_consumed_users=train_data['item_consumed_items'],
                            user_links=data_module.user_links,
                            item_links=data_module.item_links)
    else:
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ## train the model
    for epoch in range(1, epochs+1):
        epoch_loss_avg = tf.keras.metrics.Mean()
        start_time = time.time()
        train_epoch(epoch, model, optimizer, epoch_loss_avg, data_module) # type: ignore
        epoch_time = time.time() - start_time

        # validation 
        validation_loss_avg = tf.keras.metrics.Mean()
        validation_input_users, validation_input_items, validation_label_ratings = data_module.get_validation_data()
        validation_y_predict = model([validation_input_users, validation_input_items])
        validation_loss_value = tf.nn.l2_loss(validation_label_ratings-validation_y_predict, name="validation_loss")
        validation_loss_avg.update_state(validation_loss_value)

        # metrics hit_rate and ndcg
        # hit_rate, ndcg = evaluate_hit_rate_and_ndcg(validation_user_index_dict, validation_label_ratings, validation_y_predict.numpy())

        # test
        test_loss_avg = tf.keras.metrics.Mean()
        test_input_users, test_input_items, test_label_ratings, test_user_index_dict = data_module.get_test_data_positive()
        test_y_predict = model([test_input_users, test_input_items])
        test_loss_value = tf.nn.l2_loss(test_label_ratings-test_y_predict, name="test_loss")
        test_loss_avg.update_state(test_loss_value)

        test_negative_predictions_user_dict = {}
        for input_users, input_items, user_batch_list in data_module.get_test_data_negative():
            negative_predictions = model([input_users, input_items])
            negative_predictions_user_index = np.reshape(negative_predictions, (-1,num_evaluate))

            for index, user in enumerate(user_batch_list):
                test_negative_predictions_user_dict[user] = negative_predictions_user_index[index]

        hit_rate_5, ndcg_5 = evaluate_hit_rate_and_ndcg_2(test_user_index_dict, test_y_predict.numpy(), test_negative_predictions_user_dict, top_k=5)
        hit_rate_10, ndcg_10 = evaluate_hit_rate_and_ndcg_2(test_user_index_dict, test_y_predict.numpy(), test_negative_predictions_user_dict, top_k=10)
        hit_rate_15, ndcg_15 = evaluate_hit_rate_and_ndcg_2(test_user_index_dict, test_y_predict.numpy(), test_negative_predictions_user_dict, top_k=15)

        # rmse
        # test_predict_and_true = np.concatenate([test_label_ratings, test_y_predict])
        rmse = tf.keras.metrics.RootMeanSquaredError()
        rmse.update_state(test_label_ratings, test_y_predict)
        
        #mse
        mse = tf.keras.losses.MeanSquaredError()
        mse_val = mse(test_label_ratings, test_y_predict).numpy()


        # # metrics hit_rate and ndcg
        # hit_rate, ndcg = evaluate_hit_rate_and_ndcg(test_user_index_dict, test_label_ratings, test_y_predict, top_k=top_k)
        # log.info(f"Test Loss: {test_loss_avg.result()}  Test HR: {hit_rate} Test NDCG: {ndcg}")

        log.info(f"Epoch: {epoch}: Time Elapsed:{epoch_time} Loss: {epoch_loss_avg.result()} Validation Loss: {validation_loss_avg.result()}")
        log.info(f"Test loss: {test_loss_avg.result()} MSE: {mse_val} RMSE: {rmse.result()}")
        log.info(f"\t Test HR(5): {hit_rate_5}\tTest NDCG(5): {ndcg_5}")
        log.info(f"\t Test HR(10): {hit_rate_10}\tTest NDCG(10): {ndcg_10}")
        log.info(f"\t Test HR(15): {hit_rate_15}\tTest NDCG(15): {ndcg_15}")

if __name__ == "__main__":
    # setup logging
    setup_logging()
    main() # type: ignore