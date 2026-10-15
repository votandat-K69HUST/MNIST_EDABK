python model/train.py --config v6_config.json --name mlp_v6_hard_cases --seeds 0 1 2 --set train.epochs=40 train.patience=10
python model/predict.py --config v6_config.json --run model/runs/mlp_v6_hard_cases --out v6/submission_v6.csv --tta 2
