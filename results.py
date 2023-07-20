import json
import os
from pprint import pprint

import matplotlib.pyplot as plt


def print_results(results):
    print("========================\n\n")
    print("Last epoch results for each variation\n\n")
    for result in results:
        epochs = result["epoch"]
        last_epoch = epochs[-1]
        print(f'{result["model"]} : {result["hyperparameters"]} and results: {last_epoch}\n\n')
        # break
    print("========================\n\n")


def plot_training_loss(results: list, model):
    epochs_range = range(1, 101)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training loss")
    plt.title("Training Loss for training model variations")
    for result in results:
        if result["model"] != model:
            continue
        hyperparameters = result["hyperparameters"]
        if model == "DiffnetPlus":
            result_label = f'Diffnet++_gcn={hyperparameters["gcn_layers"]}_lr={hyperparameters["learning_rate"]}'
        else:
            result_label = f'RelationalNet_model_gcn={hyperparameters["gcn_layers"]}_lr={hyperparameters["learning_rate"]}'
        epochs = result["epoch"]
        training_loss_vals = [e["train_loss"] for e in epochs]
        pprint(training_loss_vals)

        plt.plot(epochs_range, training_loss_vals, label=result_label)
        # break
    plt.legend()
    plt.show()


if __name__ == "__main__":
    files = os.listdir("./out")
    files = [f for f in files if f.endswith(".json")]
    # pprint(files)

    results = []
    for filename in files:
        with open(f"./out/{filename}", "r") as f:
            results.append(json.load(f))

    # pprint(results[0]["hyperparameters"])
    # print_results(results)
    plot_training_loss(results, "DiffnetPlusMod")
