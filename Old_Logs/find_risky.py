import pandas as pd
import torch
from glob import glob
import sys
sys.path.append('model')
from predict import load_models, predict_proba, read_test

ids, X = read_test('test.csv')
models = load_models(glob('v4/models/model_seed*.pt'))
proba = predict_proba(models, X, tta_shifts=2)
conf, preds = torch.max(proba, dim=1)
preds = preds.numpy() + 10
conf = conf.numpy()

low_conf_idx = torch.where(torch.tensor(conf) < 0.65)[0]
results = []
for idx in low_conf_idx:
    p = proba[idx]
    top2_val, top2_idx = torch.topk(p, 2)
    top1_class, top2_class = top2_idx.numpy() + 10
    top1_conf, top2_conf = top2_val.numpy()
    results.append({
        'id': ids[idx],
        'pred': int(top1_class),
        'conf': round(float(top1_conf), 3),
        'runner_up': int(top2_class),
        'runner_up_conf': round(float(top2_conf), 3)
    })

df = pd.DataFrame(results).sort_values('conf')
print(f'Total low confidence (<0.65): {len(df)}')
if len(df) > 0:
    def make_pair(row):
        a, b = int(row['pred']), int(row['runner_up'])
        return f"{min(a,b)} vs {max(a,b)}"
    df['pair'] = df.apply(make_pair, axis=1)
    print('\nMost confusing pairs (count):')
    print(df['pair'].value_counts().head(10).to_string())
    print('\nTop 15 most unsure samples (lowest confidence):')
    print(df.drop(columns=['pair']).head(15).to_string(index=False))
