import sys
import os
import pandas as pd
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GT = os.path.join(CUR_DIR, "test_labeled.csv")
GT_PATH = DEFAULT_GT if os.path.exists(DEFAULT_GT) else r"D:\2025\2025.2\LAB\AI_ML\Week_11\AI_HACKATHON\test_label_by_GEMINI\test_labeled.csv"

def evaluate_submission(sub_path, gt_path=GT_PATH):
    if not os.path.exists(sub_path):
        print(f"Error: File not found: {sub_path}")
        return
    
    gt = pd.read_csv(gt_path)
    sub = pd.read_csv(sub_path)
    
    if len(sub) != len(gt):
        print(f"Warning: Submission has {len(sub)} rows, but GT has {len(gt)} rows.")
    
    # Merge on ID
    merged = pd.merge(gt, sub, on='id', suffixes=('_gt', '_pred'))
    
    total = len(merged)
    correct_total = (merged['label_gt'] == merged['label_pred']).sum()
    acc_total = correct_total / total * 100
    
    pub_mask = merged['id'] < 220
    priv_mask = merged['id'] >= 220
    
    pub_total = pub_mask.sum()
    pub_correct = (merged.loc[pub_mask, 'label_gt'] == merged.loc[pub_mask, 'label_pred']).sum()
    pub_acc = pub_correct / pub_total * 100 if pub_total > 0 else 0
    
    priv_total = priv_mask.sum()
    priv_correct = (merged.loc[priv_mask, 'label_gt'] == merged.loc[priv_mask, 'label_pred']).sum()
    priv_acc = priv_correct / priv_total * 100 if priv_total > 0 else 0
    
    print("=" * 65)
    print(f"BẢNG CHẤM ĐIỂM TEST SET (ĐÁP ÁN: GEMINI TEST LABELED)")
    print(f"File: {sub_path}")
    print("=" * 65)
    print(f"🏆 TOÀN BỘ TEST (550 mẫu)   : {acc_total:6.2f}% ({correct_total}/{total})")
    print(f"🌐 PUBLIC SUBSET (220 mẫu)  : {pub_acc:6.2f}% ({pub_correct}/{pub_total})")
    print(f"🔒 PRIVATE SUBSET (330 mẫu) : {priv_acc:6.2f}% ({priv_correct}/{priv_total})")
    print("-" * 65)
    
    # Class-by-class accuracy
    print(f"{'Lớp':<6} | {'Số mẫu':<8} | {'Đúng':<6} | {'Độ chính xác':<12}")
    print("-" * 40)
    for c in sorted(merged['label_gt'].unique()):
        cls_mask = merged['label_gt'] == c
        cls_tot = cls_mask.sum()
        cls_corr = (merged.loc[cls_mask, 'label_gt'] == merged.loc[cls_mask, 'label_pred']).sum()
        cls_acc = cls_corr / cls_tot * 100
        print(f"{c:<6} | {cls_tot:<8} | {cls_corr:<6} | {cls_acc:6.2f}%")
        
    # Mẫu sai
    wrong = merged[merged['label_gt'] != merged['label_pred']]
    if len(wrong) > 0:
        print("\nCác mẫu bị sai:")
        print(wrong[['id', 'label_gt', 'label_pred']].to_string(index=False))
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "v7/submission_v7_blend.csv"
    evaluate_submission(path)
