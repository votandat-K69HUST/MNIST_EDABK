# Báo Cáo Phiên Bản V4 (Tối Ưu Hóa Siêu Tham Số - Hyperparameter Tuning)

Đây là phiên bản được tinh chỉnh kỹ lưỡng (Tuning) dựa trên nền tảng dữ liệu đã rất thành công của V3. Trọng tâm của V4 là tìm ra kiến trúc và các siêu tham số (Hyperparameters) giúp khai thác tối đa sức mạnh của mô hình MLP trên tập dữ liệu viết tay thật (Real Data).

---

## 1. Dữ Liệu Huấn Luyện & Đánh Giá
- **Huấn luyện (Train Mixture):** Kế thừa tỷ lệ pha trộn thành công rực rỡ từ V3 (tổng >166.000 mẫu/epoch):
  - `m04_ligature_strokes` (Trọng số 1.0)
  - `m03_qmnist_same_writer` (Trọng số 1.0)
  - `self_numbers_train` (Trọng số 0.5) - Oversample ảnh nét bút thật.
  - `m01_mnist_random_concat` (Trọng số 0.2)
- **Đánh giá (Validation):** Điểm khác biệt lớn nhất là bộ Tuning V4 được **đánh giá độc quyền trên tập `self_numbers_val.npz` (Tập test tay thật 100%)**. Điều này giúp mô hình không bị "ảo tưởng" điểm số trên dữ liệu sinh tự động.

---

## 2. Những Thay Đổi Về Mô Hình & Chi Tiết Cấu Hình (Hyperparameters)

Sau 20 vòng rà soát bằng Random Search (Tune), bộ tham số tối ưu nhất được chốt để train V4 bao gồm:

### 2.1. Cấu hình Mạng (Architecture)
*   **Kiến trúc:** MLP (Multilayer Perceptron).
*   **Lớp ẩn (Hidden Layers):** `[512, 512, 256]` (Mạng được tăng thêm độ sâu với một lớp `512` nơ-ron so với bản V3, giúp học các đặc trưng nét nối phức tạp hơn).
*   **Hàm kích hoạt (Activation):** `GELU`.
*   **Regularization:** Bật `BatchNorm` (`True`). Tăng mức `Dropout` lên **`0.367`** (Mạnh tay hơn mức 0.3 của V3 nhằm chống Overfit triệt để khi mạng sâu hơn).

### 2.2. Tối ưu hóa (Optimizer & Scheduler)
*   **Optimizer:** `AdamW`.
*   **Learning Rate (LR):** `0.00126` (Khởi tạo cao hơn một chút).
*   **Weight Decay:** Rất nhỏ `1.78e-05`.
*   **Scheduler:** `CosineAnnealingLR` (`warmup_epochs: 1`, giảm LR về `1%`).
*   **Label Smoothing:** Hạ về `0.0` (Mô hình được yêu cầu tự tin tối đa do mạng đã có đủ độ nhiễu phòng thủ từ Dropout và Augmentation).

### 2.3. Tham số Huấn Luyện & Augmentation
*   **Data Augmentation:** Xoay 8 độ, Kéo nghiêng 10 độ, Độ dày (`thickness_p`) 0.5, Nhiễu 0.05.
*   **Batch Size:** Hạ xuống **`128`** (Giúp gradient cập nhật chi tiết hơn so với 256 của V3).
*   **Cơ chế dừng sớm (Early Stopping):** Patience 10, Epoch tối đa 40.

---

## 3. Đánh Giá Kết Quả Cuối Cùng

### Điểm số Validation (Trên tập thực tế)
Chỉ với 1 Model đơn lẻ (Seed 0), Validation Accuracy đã chạm mốc kỷ lục **93.15%**. (Vượt xa mô hình đơn lẻ của V3).

### Cụm Ensemble & Chất Lượng Dự Đoán (`submission_v4.csv`)
Do điều kiện chạy máy, V4 đang sử dụng Ensemble **4 models** (Seeds 0, 1, 2, 3) kết hợp TTA = 2.
Kết quả đo đạc phân phối trên tập test chưa biết nhãn cực kỳ ấn tượng:
*   **Độ tự tin (Confidence) trung bình:** Đạt **0.950** (Tăng mạnh so với 0.870 của V3).
*   **Tỷ lệ lưỡng lự (Confidence < 0.6):** Giảm xuống vỏn vẹn **4.7%** (Chỉ bằng một nửa V3).
*   **Phân bố nhãn:** Cân bằng cực kỳ hoàn hảo xoay quanh mốc 50 ảnh/lớp. 
  - Thấp nhất: Lớp 10 và 14 (47 ảnh).
  - Cao nhất: Lớp 15 (55 ảnh).
  - Các lớp khó như 13, 17, 20 đều đạt mốc an toàn (52 ảnh).

**Kết luận:** V4 là phiên bản "Endgame" hoàn hảo của kiến trúc MLP. Mạng được tối ưu toán học sát với thực tế nhất và giải quyết gần như dứt điểm sự lưỡng lự ở các nét số nhiễu. 
**[CẬP NHẬT]** V4 đã chính thức đạt mốc **96.36% (0.96363)** trên Public Leaderboard! Một bước nhảy vọt khổng lồ và gần như chạm đến giới hạn sức mạnh tối đa của một mạng MLP thuần túy.
