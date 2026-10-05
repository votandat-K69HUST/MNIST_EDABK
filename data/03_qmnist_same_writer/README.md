# 03 – Ghép chữ số cùng một writer (QMNIST)

**Ý tưởng (bài MDW, arXiv 2512.00676):** QMNIST là MNIST có thêm metadata, gồm **writer id** (cột 2 của file nhãn). Với mỗi mẫu:
chọn ngẫu nhiên một writer có đủ cả hai chữ số cần thiết, lấy ảnh chữ hàng chục và hàng đơn vị **của người đó**. Phong cách (nghiêng, dày, cỡ) tự nhiên đồng nhất.

**Nguồn:** `qmnist-train` (60k) + `qmnist-test` (60k) = 120.000 ảnh, 1.074 writer (539 + 535, rời nhau). Tải bằng `tools/download_data.py`.
`source.kind = "nist"` dùng toàn bộ xnist (~400k ảnh, nhiều chữ/writer hơn; file rất lớn).

**Chia train/val THEO WRITER** (`split.val_fraction=0.1`): writer của val không xuất hiện ở train → val phản ánh khả năng tổng quát sang người viết mới.

**Chạy:** `python generate.py` → `../output/m03_qmnist_same_writer/`.
**Tham số:** `pairing.mode="same_writer"`; lệch nhỏ giữa hai chữ (`shear_individual_std=2`, `size_ratio_std=0.04`).
**Hạn chế (bài báo tự nêu):** vẫn là ghép từng chữ, không có nét nối khi viết liền → xem 04.
