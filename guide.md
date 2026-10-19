# 📖 HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN (PROJECT GUIDE)
**Bài toán**: Nhận diện số viết tay hai chữ số từ **10 đến 20** (11 lớp)  
**Ràng buộc cốt lõi**: Mạng nơ-ron truyền thẳng **MLP thuần túy** (không dùng CNN/ViT theo thể lệ), tuân thủ 100% không rò rỉ dữ liệu test.  
**Thành tích cao nhất**: 
* **`v4_new`**: **0.96818** trên Kaggle Public Leaderboard (Top 1).
* **`v7_blend`**: **98.18%** (540/550 mẫu đúng) trên bộ Ground Truth đối chiếu.

---

## 🚀 1. CÀI ĐẶT MÔI TRƯỜNG (SETUP)

### Yêu cầu hệ thống:
* **Hệ điều hành**: Windows / Linux / macOS
* **Python**: Khuyến nghị Python **3.10** hoặc **3.11**
* **Tài nguyên**: Chạy hoàn toàn mượt mà trên **CPU** (không bắt buộc GPU).

### Các bước cài đặt:
1. **Clone repository về máy:**
   ```bash
   git clone <URL_REPOSITORY_CUA_BAN>
   cd AI_HACKATHON
   ```

2. **Khởi tạo môi trường ảo (khuyến nghị dùng Conda hoặc venv):**
   * *Dùng Conda:*
     ```bash
     conda create -n digit_env python=3.10 -y
     conda activate digit_env
     ```
   * *Hoặc dùng Python venv:*
     ```bash
     python -m venv venv
     # Kích hoạt trên Windows:
     venv\Scripts\activate
     # Kích hoạt trên Linux/macOS:
     source venv/bin/activate
     ```

3. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ 2. CHẠY DỰ ĐOÁN NGAY BẰNG CHECKPOINT CÓ SẴN (QUICK INFERENCE)

Toàn bộ các trọng số mô hình đã được huấn luyện tối ưu và lưu trữ sẵn trong thư mục `model/runs/`. Bạn có thể chạy tạo file nộp bài ngay mà **không cần train lại**:

### Cách 1: Xuất file nộp bài tối ưu nhất (`v7_blend` - Đạt 98.18%)
Đây là bản phối hợp Soft-Voting giữa 5 seeds $v4$ và 5 seeds $v7$ (tỷ lệ $0.6 \times v4 + 0.4 \times v7$):
```bash
python v7/create_v7_blend.py
```
👉 *Kết quả được tạo tại:* `v7/submission_v7_blend.csv`

---

### Cách 2: Chạy dự đoán mô hình `v7` đơn lẻ (5-Seed Ensemble Tuned)
Mô hình MLP 3 lớp sâu `[1024, 512, 256]` với hàm kích hoạt GELU và kỹ thuật TTA (Test-Time Augmentation):
```bash
python model/predict.py --config v7/v7_config.json --run model/runs/mlp_v7_best_tuned --out v7/submission_v7.csv --tta 2
```
👉 *Kết quả được tạo tại:* `v7/submission_v7.csv`

---

### Cách 3: Chạy dự đoán mô hình `v4_new` (Bản kỷ lục Kaggle 0.96818)
```bash
python model/predict.py --config v4/v4_new_config.json --run model/runs/mlp_v4_new_data --out v4/submission_v4_new.csv --tta 2
```
👉 *Kết quả được tạo tại:* `v4/submission_v4_new.csv`

---

## 🎯 3. TỰ ĐỘNG CHẤM ĐIỂM TEST OFFLINE (EVALUATION)

Để kiểm tra độ chính xác trên toàn bộ 550 mẫu test set so với đáp án Ground Truth (do Gemini thẩm định):
```bash
# Chấm điểm file v7_blend:
python eval_against_gemini.py v7/submission_v7_blend.csv

# Chấm điểm file v4_new:
python eval_against_gemini.py v4/submission_v4_new.csv
```
Script sẽ in ra bảng thống kê chi tiết:
* Tỷ lệ chính xác toàn bộ 550 mẫu.
* Độ chính xác trên tập Public (220 mẫu) và Private (330 mẫu).
* Độ chính xác chi tiết của từng lớp từ 10 đến 20.
* Danh sách ID các mẫu dự đoán sai (nếu có).

---

## 🛠️ 4. QUY TRÌNH HUẤN LUYỆN TỪ ĐẦU (TRAIN FROM SCRATCH)

Nếu bạn muốn tự huấn luyện lại từ số 0:

### Bước 1: Sinh dữ liệu ghép (Synthetic Data)
*(Nếu trong `data/output/` chưa có dữ liệu `m01`, `m03`, `m04`)*
```bash
# 1. Tải bộ dữ liệu gốc MNIST và QMNIST về data/raw/
python data/tools/download_data.py

# 2. Sinh tập m01 (ngẫu nhiên) và m03 (cùng writer)
python data/tools/run_all.py --workers 4

# 3. Sinh tập m04 (cùng writer kèm nét nối ligature)
python data/04_ligature_strokes/generate.py
```

### Bước 2: Huấn luyện mô hình đa hạt giống (5 Seeds Ensemble)
Huấn luyện 5 mô hình độc lập qua các seed `0, 1, 2, 3, 4` với Early Stopping:
```bash
python model/train.py --config v7/v7_config.json --name mlp_v7_best_tuned --seeds 0 1 2 3 4
```
Các checkpoint tốt nhất của từng seed sẽ tự động được lưu vào `model/runs/mlp_v7_best_tuned/model_seed<k>.pt`.

### Bước 3: Dự đoán và xuất kết quả
```bash
python model/predict.py --config v7/v7_config.json --run model/runs/mlp_v7_best_tuned --out v7/submission_v7.csv --tta 2
```

> **Mẹo (Chạy trọn gói):** Bạn có thể dùng `model/run_pipeline.py` để tự động hóa toàn bộ quy trình từ Train đến Predict:
> ```bash
> python model/run_pipeline.py --config v7/v7_config.json --out v7/submission_v7.csv
> ```

---

## 🔬 5. TÌM KIẾM SIÊU THAM SỐ (HYPERPARAMETER TUNING - TÙY CHỌN)

Dự án tích hợp sẵn công cụ Random Search để dò tìm cấu trúc mạng và Learning Rate tối ưu trên tập kiểm định thực tế:
```bash
python model/tune.py --config v7/v7_tune_base.json --trials 30 --space v7/v7_search_space.json
```
Kết quả của 30 lần thử nghiệm sẽ được ghi vào `model/runs/tune_v7_tuned/results.csv` và cấu hình tốt nhất được xuất ra tại `best_config.json`.

---

## 📁 6. CẤU TRÚC THƯ MỤC CHÍNH (REPOSITORY OVERVIEW)

```text
AI_HACKATHON/
├── guide.md                       # Tài liệu hướng dẫn này
├── requirements.txt               # Danh sách thư viện Python cần thiết
├── test.csv                       # Dữ liệu test của cuộc thi (550 mẫu)
│
├── data/                          # Toàn bộ mã nguồn và dữ liệu
│   ├── 01_mnist_random_concat/    # Ghép 2 số ngẫu nhiên từ MNIST
│   ├── 02_mnist_style_matched/    # Ghép theo cụm phong cách nét
│   ├── 03_qmnist_same_writer/     # Ghép cùng người viết (Same-Writer QMNIST)
│   ├── 04_ligature_strokes/       # Ghép cùng người viết + Nét nối mờ Bezier
│   ├── 05_self_handwritten/       # Dữ liệu viết tay thực tế
│   ├── 06_augmentation/           # Tăng cường hình học
│   ├── output/                    # Nơi chứa các file .npz đã sinh
│   ├── raw/                       # File gốc MNIST/QMNIST
│   └── tools/                     # Scripts download, stats đo lường, preview
│
├── model/                         # Kiến trúc mô hình & Huấn luyện
│   ├── mlp.py                     # Định nghĩa mạng MLP (Linear, BatchNorm, GELU, Dropout)
│   ├── train.py                   # Script huấn luyện từng seed (Early Stopping, Cosine LR)
│   ├── predict.py                 # Script Ensemble Soft-Voting + TTA xuất file nộp bài
│   ├── tune.py                    # Script Random Search tối ưu siêu tham số
│   ├── augment_torch.py           # Augmentation online (xoay, nghiêng, co/nở nét)
│   ├── data_utils.py              # Bộ nạp dữ liệu đa nguồn
│   └── runs/                      # Nơi lưu toàn bộ checkpoint .pt của các phiên bản
│
├── v1/ ... v7/                    # Lưu trữ các mốc thử nghiệm lịch sử, báo cáo và output
│   ├── v4/submission_v4_new.csv   # File nộp kỷ lục 0.96818 Kaggle
│   └── v7/submission_v7_blend.csv # File nộp kỷ lục 98.18% Ground Truth
└── test_label_by_GEMINI/          # Bộ Ground Truth & công cụ chấm đối chứng offline
    ├── test_labeled.csv           # Đáp án chuẩn 550 mẫu
    ├── eval_against_gemini.py     # Công cụ tự động chấm điểm với Ground Truth
    └── gemini_labeling_report.md  # Báo cáo kỹ thuật chi tiết quy trình gán nhãn
```

---

## ⚠️ 7. CÁC NGUYÊN TẮC QUAN TRỌNG KHI NỘP BÀI
1. **Tuân thủ thể lệ cuộc thi**: Tuyệt đối **KHÔNG dùng `test.csv` hay `test_labeled.csv` để huấn luyện hay fine-tune**.
2. **Định dạng file nộp**: File nộp phải có đúng 2 cột: `id` (0..549) và `label` (chữ số từ 10 đến 20).
3. **Mô hình khuyến nghị nộp**: File [`v7/submission_v7_blend.csv`](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/v7/submission_v7_blend.csv).
