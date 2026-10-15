# 02 – Ghép MNIST theo phong cách gần nhau

**Ý tưởng:** người thật viết hai chữ của cùng một số với độ nghiêng, độ dày nét, cỡ chữ tương đồng. MNIST không có writer id
nên **xấp xỉ** bằng đặc trưng ảnh: [độ nghiêng (mu11/mu02), độ dày nét (2·diện tích/chu vi), tỉ lệ khung w/h] (`common/features.py`).
Chọn chữ thứ nhất ngẫu nhiên, lấy `candidates` ứng viên cho chữ thứ hai và chọn cái **gần nhất** (softmax theo `temperature` để còn đa dạng).

**Chạy:** `python generate.py` → `../output/m02_mnist_style_matched/`. Lần đầu tính đặc trưng 70.000 ảnh (~30 giây) và cache ở `../raw/features_mnist.npy`.

**Tham số:** `pairing.candidates=24`, `pairing.temperature=0.3` (0 = luôn chọn gần nhất), `pairing.feature_weights=[nghiêng, dày, tỉ lệ]`;
`compose.shear_individual_std=2` và `size_ratio_std=0.04` (lệch nhỏ giữa hai chữ).

**Kiểm chứng:** khoảng chênh trung bình (z-score) giữa hai chữ: ghép ngẫu nhiên → nghiêng 1.18, dày 1.09; style_matched → nghiêng 0.54, dày 0.48.
