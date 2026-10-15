# Báo Cáo Huấn Luyện & Pipeline Mô Hình Chuyên Gia Hard Case (`mlp_hard_case_specialist`)

---

## 1. Tổng Quan & Chiến Lược (Strategy)
Để khắc phục tình trạng suy giảm độ chính xác khi thêm các mẫu khó vào toàn bộ mô hình (như bản v6 cũ), chúng ta triển khai kiến trúc **Cascade Model (Reject Option Ensemble)**:
* **Mô hình chính (`v4_new`)**: Đạt 96.81% trên Leaderboard, chịu trách nhiệm nhận diện các mẫu thông thường với độ tự tin cao.
* **Mô hình chuyên gia (`mlp_hard_case_specialist`)**: Được huấn luyện chuyên biệt trên tập các ca khó (nét nhầm giữa 13/19, 10/20, 16/18). Chỉ kích hoạt khi độ tự tin của `v4_new` giảm xuống dưới ngưỡng **Confidence < 0.6**.

---

## 2. Dữ Liệu Huấn Luyện (Training Dataset)
Tập dữ liệu huấn luyện khó được tổng hợp từ:
1. **3,131 mẫu vẽ tay khó**: Sinh ra từ 31 mẫu nét khó do người dùng vẽ trên tool (`draw_tool.html`) kết hợp augmentation.
2. **708 mẫu khó đào từ dữ liệu thật (`v5_new_train.npz`)**: Gồm 236 mẫu chữ thật mà `v4_new` dự đoán sai hoặc phân vân (Confidence < 0.8), nhân bản 3 lần.
3. **Phối hợp dữ liệu trong khi train**:
   - `hard_cases_dataset.npz` (Hard cases): Trọng số **1.0**
   - `v5_new_train.npz` (Data thật tổng quát): Trọng số **0.3**
   - `m04_ligature_strokes` (Data tổng hợp nét nối): Trọng số **0.2**

---

## 3. Kiến Trúc & Siêu Tham Số (Hyperparameters)
* **Architecture**: MLP (512 -> 512 -> 256 -> 11)
* **Activation**: GELU + BatchNorm
* **Dropout**: 0.35
* **Optimizer**: AdamW (Learning Rate: 0.001, Weight Decay: 1e-4)
* **Scheduler**: Cosine Annealing (Warmup 2 epochs, Min LR fraction: 0.01)
* **Batch Size**: 128
* **Label Smoothing**: 0.05

---

## 4. Kết Quả Huấn Luyện Chi Tiết (Training Metrics)

| Seed | Epoch Tốt Nhất | Training Loss | Val Accuracy (`v5_new_val`) | Đường dẫn File Trọng Số |
|---|---|---|---|---|
| **Seed 0** | Epoch 28 | 0.6130 | **93.89%** | `model_seed0.pt` |
| **Seed 1** | Epoch 30 | 0.6135 | **94.12%** | `model_seed1.pt` |
| **Seed 2** | Epoch 19 | 0.6554 | **93.89%** | `model_seed2.pt` |
| **Trung bình Ensemble** | - | **0.6273** | **93.97%** | Ensemble 3 Seeds |

---

## 5. Kết Quả Chạy Pipeline Cascade Trên Tập Test (`test.csv`)
* **Cấu hình Ngưỡng (Threshold)**: `< 0.6`
* **Tổng số mẫu test**: 550 mẫu
* **Số mẫu có `v4_new` confidence < 0.6**: 22 mẫu (4.00%)
* **Số mẫu được Chuyên gia đồng ý (Giữ nguyên)**: 5 mẫu
* **Số mẫu được Chuyên gia điều chỉnh (Override)**: **17 mẫu**

### Chi tiết 17 ca được điều chỉnh nhãn:
* **19 -> 13**: 6 mẫu (IDs: 61, 93, 338, 476, 537, 325)
* **18 -> 10 / 13**: 2 mẫu (IDs: 162, 228)
* **15 -> 10 / 18**: 2 mẫu (IDs: 234, 332)
* **11 -> 12**: 2 mẫu (IDs: 147, 184)
* **20 -> 12**: 2 mẫu (IDs: 392, 414)
* **Các ca khác**: 3 mẫu (IDs: 8, 544, 548)

---

## 6. Danh Mục File Trong Thư Mục `v6/hard_case/`
* `hard_case_report.md`: Báo cáo tổng hợp chi tiết (File này).
* `hard_case_config.json`: File cấu hình thông số huấn luyện.
* `config_used.json`: Cấu hình thực tế đã được runtime sử dụng.
* `mine_hard_cases.py`: Script tự động đào mẫu khó từ train set.
* `predict_cascade.py`: Script chạy pipeline Cascade ghép 2 mô hình.
* `history_seed0.csv`, `history_seed1.csv`, `history_seed2.csv`: Log chi tiết quá trình train.
* `metrics.json`: Tổng hợp các chỉ số đánh giá JSON.
* `model_seed0.pt`, `model_seed1.pt`, `model_seed2.pt`: Trọng số mô hình chuyên gia.
