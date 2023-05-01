import itertools

model_name = ["DiffnetPlusMod", "DiffnetPlus"]
dims = [32, 64, 128]
batch_size = [100, 250, 500, 1000]
lr = [0.001, 0.0025, 0.005, 0.0001]
gcn = [2, 3, 4]
neg = [8, 10]
evals = [1000, 500]
epochs = [500]


with open("./run_script.sh", "w") as f:
    for model, dim, size, rate, layer, n_val, n_eval, e in itertools.product(model_name, dims, batch_size, lr, gcn, neg, evals, epochs):
        f.write(f"python src/main.py --model_name={model} --batch_size={size} --dims={dim} --lr={rate} --gcn_layers={layer} --epochs={e} --num_negatives={n_val} --num_evaluate={n_eval}\n")
        f.write("sleep 5\n")
