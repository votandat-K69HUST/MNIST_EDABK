# 01 – Ghép ngẫu nhiên chữ số MNIST (baseline)

Nhãn 10–19 = "1" + chữ d; nhãn 20 = "2" + "0". Mỗi chữ bốc **ngẫu nhiên** từ MNIST (70.000 ảnh) rồi ghép (xem `../README.md`, mục Quy trình ghép).
Hai chữ độc lập nên phong cách khác nhau: `size_ratio_std = 0.12`, `shear_individual_std = 8°`.

**Chạy:** `python generate.py` (hoặc `--n-train 11000 --n-val 1100 --workers 4 --seed 1`) → `../output/m01_mnist_random_concat/{train,val}.npz`

**Tham số (`config.json`):** `compose.gap`, `compose.baseline_std`, `compose.size_ratio_std`, `compose.shear_individual_std`, `compose.hr`, `layout.box`; `split` tách train/val **theo chỉ số ảnh gốc** (`val_fraction` 0.1, `split_seed`).
