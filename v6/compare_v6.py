import pandas as pd
df1 = pd.read_csv('v4/submission_v4_new.csv')
df2 = pd.read_csv('v6/submission_v6.csv')
diff = (df1['label'] != df2['label']).sum()
total = len(df1)
print(f'So luong mau thay doi du doan (V4_new -> V6): {diff}/{total} ({(diff/total)*100:.2f}%)')
changes = df1[df1['label'] != df2['label']][['id', 'label']].copy()
changes.columns = ['id', 'old_label']
changes['new_label'] = df2[df2['label'] != df1['label']]['label']
if diff > 0:
    print('Cac mau cu the thay doi (V4_new -> V6):')
    for _, row in changes.iterrows():
        print(f"ID: {row['id']} | {row['old_label']} -> {row['new_label']}")
