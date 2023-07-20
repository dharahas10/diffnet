python src/main.py --model_name=DiffnetPlus --batch_size=500 --dims=64 --lr=0.0025 --gcn_layers=2 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
python src/main.py --model_name=DiffnetPlus --batch_size=500 --dims=64 --lr=0.0025 --gcn_layers=3 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
python src/main.py --model_name=DiffnetPlus --batch_size=500 --dims=64 --lr=0.0005 --gcn_layers=2 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
python src/main.py --model_name=DiffnetPlus --batch_size=500 --dims=64 --lr=0.0005 --gcn_layers=3 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
python src/main.py --model_name=DiffnetPlusMod --batch_size=500 --dims=64 --lr=0.0025 --gcn_layers=2 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
python src/main.py --model_name=DiffnetPlusMod --batch_size=500 --dims=64 --lr=0.0025 --gcn_layers=3 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
python src/main.py --model_name=DiffnetPlusMod --batch_size=500 --dims=64 --lr=0.0005 --gcn_layers=2 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
python src/main.py --model_name=DiffnetPlusMod --batch_size=500 --dims=64 --lr=0.0005 --gcn_layers=3 --epochs=100 --num_negatives=10 --num_evaluate=1000
sleep 5
