python model/train.py --config v4/v4_new_config.json --name mlp_v4_new_data --seeds 0 1 2 3 4 --set train.epochs=50 train.patience=10
python model/predict.py --config v4/v4_new_config.json --run model/runs/mlp_v4_new_data --out v4/submission_v4_new.csv --tta 2
