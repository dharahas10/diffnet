import itertools

model_name = ["DiffnetPlus", "DiffnetPlusMod"]
dims = [64]
batch_size = [500]
lr = [0.0025, 0.0005]
gcn = [2, 3]
neg = [10]
evals = [1000]
epochs = [100]


with open("./run_script.sh", "w") as f:
    for model, dim, size, rate, layer, n_val, n_eval, e in itertools.product(model_name, dims, batch_size, lr, gcn, neg, evals, epochs):
        f.write(f"python src/main.py --model_name={model} --batch_size={size} --dims={dim} --lr={rate} --gcn_layers={layer} --epochs={e} --num_negatives={n_val} --num_evaluate={n_eval}\n")
        f.write("sleep 5\n")
