import os
import sys
import time
import subprocess
import json
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "test_label_by_GEMINI"))

import train
import predict
import eval_against_gemini

def run_experiment(name, cfg_path, sub_path):
    print("=" * 70)
    print(f"BẮT ĐẦU HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH: {name}")
    print(f"Config: {cfg_path}")
    print("=" * 70)
    t0 = time.time()
    
    # 1. Huấn luyện
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_dir") or os.path.dirname(os.path.abspath(cfg_path))
    out_dir = os.path.join(ROOT, "model", "runs", cfg.get("name", name))
    seeds = cfg.get("ensemble", {}).get("seeds", [0])
    
    print(f"--- Bước 1: Huấn luyện {len(seeds)} seed(s) {seeds} ---")
    results = [train.train_one(cfg, s, out_dir, base) for s in seeds]
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)
    train_time = time.time() - t0
    print(f"Hoàn thành huấn luyện trong {train_time:.1f}s ({train_time/60:.2f} phút)")
    
    # 2. Suy luận (Inference + TTA=2)
    print(f"\n--- Bước 2: Tạo file Submission với TTA=2 ---")
    t_inf_0 = time.time()
    models_paths = [os.path.join(out_dir, f"model_seed{s}.pt") for s in seeds]
    models = predict.load_models(models_paths)
    ids, X = predict.read_test(os.path.join(ROOT, "test.csv"))
    proba = predict.predict_proba(models, X, tta_shifts=2)
    labels = proba.argmax(1).numpy() + predict.LABEL_OFFSET
    sub = predict.write_submission(ids, labels, sub_path)
    inf_time = time.time() - t_inf_0
    print(f"Đã xuất submission ({len(sub)} dòng) -> {sub_path} trong {inf_time:.2f}s")
    
    # 3. Đánh giá trên Ground Truth Gemini
    print(f"\n--- Bước 3: Chấm điểm trên Ground Truth (test_labeled.csv) ---")
    gt = pd.read_csv(os.path.join(ROOT, "test_label_by_GEMINI", "test_labeled.csv"))
    merged = pd.merge(gt, sub, on='id', suffixes=('_gt', '_pred'))
    total = len(merged)
    correct_total = (merged['label_gt'] == merged['label_pred']).sum()
    acc_total = correct_total / total * 100
    
    pub_mask = merged['id'] < 220
    priv_mask = merged['id'] >= 220
    pub_acc = (merged.loc[pub_mask, 'label_gt'] == merged.loc[pub_mask, 'label_pred']).sum() / pub_mask.sum() * 100
    priv_acc = (merged.loc[priv_mask, 'label_gt'] == merged.loc[priv_mask, 'label_pred']).sum() / priv_mask.sum() * 100
    
    print("-" * 60)
    print(f"KẾT QUẢ {name}:")
    print(f"  Toàn bộ (550 mẫu)   : {acc_total:.2f}% ({correct_total}/{total})")
    print(f"  Public LB (220 mẫu) : {pub_acc:.2f}%")
    print(f"  Private LB (330 mẫu): {priv_acc:.2f}%")
    print("-" * 60)
    
    return {
        "name": name,
        "train_time_sec": train_time,
        "acc_total": acc_total,
        "correct_total": int(correct_total),
        "pub_acc": pub_acc,
        "priv_acc": priv_acc,
        "best_epochs": [r["best_epoch"] for r in results],
        "val_score": float(np.mean([r["score"] for r in results])),
        "val_scores_all": [r["score"] for r in results]
    }

def main():
    start_all = time.time()
    res_81 = run_experiment("v8.1 (100k samples)", os.path.join(HERE, "v8_1_config.json"), os.path.join(HERE, "submission_v8_1.csv"))
    res_82 = run_experiment("v8.2 (31k handwritten only)", os.path.join(HERE, "v8_2_config.json"), os.path.join(HERE, "submission_v8_2.csv"))
    
    print("\n" + "=" * 75)
    print("BẢNG TỔNG KẾT SO SÁNH HIỆU NĂNG: V7 vs V8.1 vs V8.2")
    print("=" * 75)
    summary_df = pd.DataFrame([
        {
            "Mô hình": "v7 Baseline (Trial 4)",
            "Dữ liệu huấn luyện": "168,469 mẫu (m04, m03, m01, v5_new)",
            "Tỷ lệ phối trộn": "m04(1.0) : m03(1.0) : m01(0.2) : viết_tay(0.5)",
            "Val Acc (Real)": "96.43%",
            "Public LB": "96.82%",
            "Toàn bộ Test Ground Truth": "97.45% (536/550)"
        },
        {
            "Mô hình": res_81["name"],
            "Dữ liệu huấn luyện": "100,000 mẫu (bỏ m01)",
            "Tỷ lệ phối trộn": "m04(0.25) : m03(1.0) : viết_tay(0.5)",
            "Val Acc (Real)": f"{res_81['val_score']*100:.2f}%",
            "Public LB": f"{res_81['pub_acc']:.2f}%",
            "Toàn bộ Test Ground Truth": f"{res_81['acc_total']:.2f}% ({res_81['correct_total']}/550)"
        },
        {
            "Mô hình": res_82["name"],
            "Dữ liệu huấn luyện": "31,198 mẫu (100% viết tay)",
            "Tỷ lệ phối trộn": "Chỉ dùng v5_new_train.npz",
            "Val Acc (Real)": f"{res_82['val_score']*100:.2f}%",
            "Public LB": f"{res_82['pub_acc']:.2f}%",
            "Toàn bộ Test Ground Truth": f"{res_82['acc_total']:.2f}% ({res_82['correct_total']}/550)"
        }
    ])
    print(summary_df.to_string(index=False))
    print(f"\nTổng thời gian hoàn tất toàn bộ pipeline: {(time.time() - start_all)/60:.2f} phút.")

if __name__ == "__main__":
    main()
