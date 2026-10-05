# 01 – Ghép ngẫu nhiên chữ số MNIST (baseline)

**Ý tưởng:** nhãn 10–19 = chữ "1" + chữ d; nhãn 20 = "2" + "0". Mỗi chữ bốc **ngẫu nhiên** từ MNIST (70.000 ảnh), ghép cạnh nhau.
Đây là cách phổ biến trong các repo/bài báo (kingyiusuen, 1912.03035), có ngẫu nhiên hoá chồng lấn/khoảng cách.
Nhược điểm: hai chữ khác phong cách (nghiêng/dày/cỡ) → không giống số do một người viết (xem bài MDW).

**Chạy:** `python generate.py` (hoặc `--n-train 11000 --n-val 1100 --workers 4 --seed 1`)
**Ra:** `../output/m01_mnist_random_concat/{train,val}.npz`

**Tham số chính (`config.json`):**
- `compose.shear_individual_std = 8`, `size_ratio_std = 0.12`: để hai chữ **lệch** nghiêng/cỡ nhiều (đúng tinh thần "ngẫu nhiên").
- `compose.gap [-0.05, 0.35]`: khoảng cách giữa hai chữ theo đơn vị chiều cao chữ; âm = chồng lấn/chạm.
- `compose.thickness_hr`: thay đổi độ dày nét (dilate/erode) trên canvas phân giải cao (hr = 4×).
- `layouts`: cách đặt số vào 28×28: giữ tỉ lệ, vừa khung ~20×20, căn trọng tâm dưới pixel (đã chỉnh khớp thống kê test) – xem `../README.md` mục "Điểm chưa chắc chắn".
- `split`: tách train/val **theo chỉ số ảnh gốc** (val_fraction 0.1, `split_seed`) để ảnh gốc không nằm ở cả hai tập.

**Quy trình ghép (common/compose.py):** cắt sát nét → phóng 4× → nghiêng → đặt cạnh nhau (baseline lệch nhẹ) → dày/mỏng nét →
bố cục → thu nhỏ **một lần** về 28×28 (vùng nối mờ tự nhiên như ảnh thật bị resize) → gamma/làm mờ nhẹ.
