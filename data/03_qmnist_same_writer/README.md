# 03 – Ghép chữ số cùng một writer (QMNIST)

Ý tưởng bài MDW (arXiv 2512.00676): QMNIST là MNIST có thêm **writer id** (cột 2 file nhãn). Mỗi mẫu chọn ngẫu nhiên một writer có đủ cả hai chữ cần thiết và lấy ảnh hai chữ **của người đó** → phong cách đồng nhất. Lệch nhỏ giữa hai chữ: `size_ratio_std = 0.04`, `shear_individual_std = 2°`.

Nguồn: `qmnist-train` + `qmnist-test` = 120.000 ảnh, 1.074 writer (539 + 535, rời nhau). `source.kind = "nist"` dùng toàn bộ xnist (~400k ảnh, rất lớn).
**Chia train/val THEO WRITER** (`split.val_fraction = 0.1`): writer của val không xuất hiện ở train.

**Chạy:** `python generate.py` → `../output/m03_qmnist_same_writer/{train,val}.npz`. Tham số như 01; `pairing.mode = "same_writer"`.
Lưu ý: QMNIST chứa lại các ảnh của MNIST, nên val của 03 (writer bị giữ lại) và train của 01 có thể chia sẻ ảnh chữ gốc → điểm val có thể hơi lạc quan.
