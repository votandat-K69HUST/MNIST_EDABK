# Báo Cáo Cấu Hình & Kết Quả Huấn Luyện (MLP v2 Tuned Full)

## 1. Dữ Liệu Huấn Luyện & Đánh Giá
Các file dữ liệu đã được xử lý và lưu dưới định dạng `.npz` tại thư mục `data/output/`.

*   **Tập Train (Huấn luyện - Tổng cộng 111.642 mẫu/epoch):**
    *   Dữ liệu tổng hợp (MNIST Random): `data/output/m01_mnist_random_concat/train.npz` (**Lấy 44.657 mẫu** - Trọng số: 1.0)
    *   Dữ liệu tổng hợp (QMNIST Cùng người viết): `data/output/m03_qmnist_same_writer/train.npz` (**Lấy 44.657 mẫu** - Trọng số: 1.0)
    *   **Dữ liệu thật (Viết tay - Mới thêm):** `data/output/self_numbers_train.npz` (**Lấy 22.328 mẫu** - Trọng số: 0.5. Lưu ý: Bản gốc chỉ có 1.642 mẫu, được tự động Oversample để tránh bị mất cân bằng dữ liệu).
*   **Tập Validation (Đánh giá độc lập):**
    *   Dữ liệu tổng hợp: `m01_.../val.npz` và `m03_.../val.npz`
    *   **Dữ liệu thật:** `data/output/self_numbers_val.npz` (Gồm 411 mẫu hoàn toàn chưa từng thấy trong quá trình train). Đây là thước đo chính để chọn model tốt nhất.

---

## 2. Thông Số Mô Hình Tốt Nhất (Best Hyperparameters)
Cấu hình này được tìm ra thông qua quá trình Random Search tự động đánh giá trên tập Real Validation.

### Kiến trúc Mạng (Architecture)
*   **Các lớp ẩn (Hidden layers):** `[256]` (Chỉ dùng 1 lớp ẩn nhẹ gọn để chống Overfitting).
*   **Tổng số tham số (Params):** 204.299 tham số.
*   **Hàm kích hoạt (Activation):** `gelu`
*   **Kỹ thuật chuẩn hóa:** Bật `BatchNorm` (True).
*   **Kỹ thuật Regularization:** `Dropout: 0.4`

### Siêu tham số Huấn luyện (Training Params)
*   **Thuật toán tối ưu (Optimizer):** `AdamW`
*   **Learning Rate (lr):** `0.000697`
*   **Weight Decay:** `4.906e-05`
*   **Batch Size:** `256`
*   **Lập lịch LR (Scheduler):** `cosine` (Warmup: 1 epoch, Min LR frac: 0.01)
*   **Label Smoothing:** `0.0`
*   **Patience (Early Stopping):** 3 epochs (Dừng sớm nếu điểm trên tập thực tế không cải thiện).

### Data Augmentation (Làm giàu dữ liệu Online)
*   **Xoay (Rotate):** `4` độ
*   **Kéo nghiêng (Shear):** `5`
*   **Dịch chuyển (Shift):** `3` pixel
*   **Nhiễu (Noise std):** `0.03`
*   **Đổi độ dày nét (Thickness p):** `0.3` (30% xác suất biến đổi độ dày).

---

## 3. Kết Quả Đánh Giá (Evaluation Results)
Mô hình được huấn luyện bằng kỹ thuật **Ensemble** với 3 seed khác nhau (0, 1, 2) trên toàn bộ dữ liệu (111.642 mẫu).

*   **Seed 0:**
    *   Dừng sớm tại Epoch 9.
    *   Điểm trên tập thật (`real:output/self_numbers_val.npz`): **85.89%**
    *   Điểm trên tập ảo (Synthetic): ~94.41%
*   **Seed 1:**
    *   Dừng sớm tại Epoch 27.
    *   Điểm trên tập thật (`real:output/self_numbers_val.npz`): **87.35%**
    *   Điểm trên tập ảo (Synthetic): ~95.94%
*   **Seed 2:**
    *   Dừng sớm tại Epoch 13.
    *   Điểm trên tập thật (`real:output/self_numbers_val.npz`): **86.13%**
    *   Điểm trên tập ảo (Synthetic): ~95.12%

**-> Điểm chính xác trung bình (Average Validation Score trên tập Real): 86.45%**

*Lưu ý:* Việc dùng mạng kiến trúc siêu nhỏ gọn kết hợp Augmentation giúp mô hình duy trì điểm số cao trên dữ liệu thực tế (86.45%) đồng thời rất bền bỉ (robust), ít bị overfit trên tập dữ liệu Public Leaderboard sắp tới. Khi chạy dự đoán test, bạn nên giữ cờ `--tta 2` để kết quả Ensemble được tối đa hóa!
