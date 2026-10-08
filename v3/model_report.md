# Báo Cáo Phiên Bản V3 Super (Kết hợp Tinh Hoa V1 & V2)

Đây là phiên bản mô hình và dữ liệu hoàn thiện nhất, kết hợp các thế mạnh từ bản V1 (nét nối, phong cách) và V2 (dữ liệu thật, dàn giáo huấn luyện tối ưu).

---

## 1. Những Thay Đổi Về Dữ Liệu (Data Pipeline)

### 1.1. Lọc nhiễu dữ liệu viết tay (Real Data)
*   **Phân tích ban đầu:** Quét 2.053 mẫu viết tay từ file CSV. Phát hiện 10 mẫu bị nhiễu nặng (tỉ lệ khung hình `Aspect Ratio` quá bất thường: > 2.0 hoặc < 0.3, chữ bị bẹp dúm hoặc nghiêng ngả phi lý).
*   **Xử lý:** Loại bỏ hoàn toàn 10 mẫu nhiễu này. Tập dữ liệu sạch còn **2.043 mẫu**, được chia tỷ lệ 80/20 (Train: 1.634 mẫu, Val: 409 mẫu).
*   **Đặc điểm nét:** Trung bình độ dày nét là ~89 pixels. Khai thác đặc điểm này để đẩy độ biến thiên nét (Thickness Augmentation) lên 0.5.

### 1.2. Phối trộn nguồn dữ liệu (Data Mixing)
Tổng số lượng mẫu đưa vào huấn luyện mỗi epoch được đẩy lên mức kỷ lục **166.633 mẫu**, với cấu trúc trọng số được thiết kế lại:
*   **[Thêm lại từ V1] M04 Ligature Strokes (Trọng số 1.0):** Cung cấp 61.716 mẫu chứa các nét nối tự nhiên giữa 2 chữ số (đây là "chìa khóa" giúp V1 từng đạt điểm cao).
*   **[Giữ nguyên] M03 QMNIST Same Writer (Trọng số 1.0):** Cung cấp 61.716 mẫu đảm bảo phong cách viết 2 số đồng nhất.
*   **[Oversample] Self Numbers Train (Trọng số 0.5):** 1.634 mẫu viết tay thật được nhân bản lên thành 30.858 mẫu để mô hình quen với nét tay người thật.
*   **[Giảm mạnh] M01 MNIST Random (Trọng số 0.2):** Giảm số lượng mẫu ghép ngẫu nhiên (thiếu tự nhiên) xuống chỉ còn 12.343 mẫu, chỉ đóng vai trò làm đa dạng ngoại lệ.

---

## 2. Những Thay Đổi Về Mô Hình & Chi Tiết Cấu Hình (Hyperparameters)

Phiên bản V3 đã được tinh chỉnh lại toàn bộ thông số huấn luyện để mô hình hội tụ tốt nhất trên lượng dữ liệu lớn.

### 2.1. Cấu hình Mạng (Architecture)
*   **Kiến trúc:** MLP (Multilayer Perceptron).
*   **Lớp ẩn (Hidden Layers):** `[512, 256]` (mở rộng sức chứa so với V2 `[256]` do lượng dữ liệu tăng lên hơn 166k mẫu/epoch và độ phức tạp cao từ nét nối).
*   **Hàm kích hoạt (Activation):** `GELU` (giúp đạo hàm mượt hơn ReLU, đặc biệt hiệu quả với cấu trúc mạng sâu và rộng).
*   **Regularization:** `Dropout: 0.3`, có sử dụng `BatchNorm` (`batchnorm: true`). Input dropout được set bằng 0.

### 2.2. Tối ưu hóa (Optimizer & Scheduler)
*   **Optimizer:** `AdamW` (Adam với Weight Decay chuẩn xác hơn).
*   **Learning Rate (LR):** Khởi tạo ở mức `0.001`.
*   **Weight Decay:** `0.0001` (Tránh overfit).
*   **Scheduler:** `CosineAnnealingLR` (Hạ nhiệt dần dần) với `warmup_epochs: 1` và `min_lr_frac: 0.01` (LR giảm về thấp nhất là 1% so với ban đầu).
*   **Khác:** Cắt gradient (`grad_clip: 1.0`), làm mịn nhãn (`label_smoothing: 0.05`).

### 2.3. Data Augmentation (Tăng cường dữ liệu)
Được cấu hình dựa trên đo đạc thực tế của 2000 mẫu viết tay:
*   **Xoay (Rotate):** Lên tới 8 độ.
*   **Kéo nghiêng (Shear):** Lên tới 10 độ.
*   **Dịch chuyển (Shift):** 3 pixel.
*   **Thay đổi độ dày nét (Thickness_p):** Đặt ở mức `0.5` (mô phỏng được nét bút bi, bút lông, hay ảnh bị mờ).
*   **Nhiễu (Noise Std):** Đặt ở mức `0.05`.

### 2.4. Tham số Huấn Luyện (Training Params)
*   **Số Epoch tối đa:** `30` (nhưng có `patience: 5` để early stopping).
*   **Batch Size:** `256` (Cân bằng giữa tốc độ và độ ổn định của gradient).
*   **Ensemble:** Huấn luyện độc lập 3 mô hình từ 3 seed khác nhau (`0, 1, 2`).
*   **Test-Time Augmentation (TTA):** TTA Shifts bằng `2` khi dự đoán trên tập Test.

---

## 3. Đánh Giá Kết Quả Cuối Cùng

### Điểm số Validation (Trên tập thực tế 409 mẫu)
*   **Seed 0:** Đạt 92.42%
*   **Seed 1:** Đạt 92.18%
*   **Seed 2:** Đạt 92.67%
*   **-> Điểm trung bình:** **92.42%** (Tăng nhảy vọt so với mức ~86.4% của bản V2 và vượt đỉnh cũ 91.36% của V1).

### Chất lượng Dự đoán (`submission_v3_super.csv`)
Sinh file nộp bằng phương pháp Ensemble + TTA (Test-Time Augmentation = 2).
*   **Độ tự tin (Confidence) trung bình:** Đạt **0.870** (Kỷ lục cao nhất từ đầu dự án).
*   **Tỷ lệ lưỡng lự (Confidence < 0.6):** Giảm xuống vỏn vẹn **8.9%**.
*   **Phân bố nhãn:** Cực kỳ đồng đều, bám sát kỳ vọng 50 ảnh/lớp (Min: 46 ảnh ở lớp 14, Max: 56 ảnh ở lớp 15). Không có hiện tượng thiên vị cục bộ.

**Kết luận:** V3 là một mô hình cực kỳ Robust, có khả năng khái quát hóa (Generalization) mạnh nhất nhờ sự cân bằng tuyệt vời giữa Data tổng hợp thông minh (Ligatures) và Data tay thật sạch sẽ!
