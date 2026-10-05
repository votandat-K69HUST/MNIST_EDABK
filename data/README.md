# data/ – Xây dựng và tái tạo dữ liệu train cho MNIST 10–20

Mục tiêu: sinh ảnh 28×28 xám chứa số 10–20 (11 class) từ các cách khác nhau, **tái tạo được hoàn toàn** từ config + seed.
Ràng buộc của đội: chỉ dùng MLP (xem `../the_le.md`). Tổng hợp nghiên cứu: `../tong_hop_phuong_phap_data.md`.

## Cài đặt & chạy nhanh
```bash
pip install -r requirements.txt            # numpy, scipy, pillow (pandas cho stats)
python tools/download_data.py              # tải MNIST + QMNIST vào raw/ (~100MB)
python tools/run_all.py --n-train 11000 --n-val 1100 --workers 4   # thử nhanh tất cả phương pháp
python tools/run_all.py                    # kích thước mặc định (55.000 train + 5.500 val mỗi phương pháp)
python tools/preview.py output/m04_ligature_strokes/train.npz      # xem lưới ảnh TỰ SINH (không dùng cho test!)
```
Đã kiểm thử với Python 3.14, numpy 2.5, scipy 1.18, Pillow 12 trên Windows.

## Các phương pháp
| Thư mục | Ý tưởng | Nguồn | Chia train/val |
|---|---|---|---|
| `01_mnist_random_concat` | Ghép ngẫu nhiên 2 chữ số MNIST (baseline) | MNIST | theo ảnh |
| `02_mnist_style_matched` | Ghép 2 chữ **có phong cách gần nhau** (độ nghiêng, độ dày nét, tỉ lệ) | MNIST | theo ảnh |
| `03_qmnist_same_writer` | Ghép 2 chữ **cùng một người viết** | QMNIST (1074 writer) | **theo writer** |
| `04_ligature_strokes` | Cùng writer + **nét nối mờ ở giữa** hai chữ | QMNIST | theo writer |
| `05_self_handwritten` | Công cụ **tự viết tay** (số 2 chữ số / chữ số rời) → dữ liệu thật; ghép cùng writer từ chữ số tự viết | bạn tự viết | theo người viết |
| `06_augmentation` | Augmentation hình học/nét/nhiễu (preset light/medium/strong) offline hoặc online | npz bất kỳ | chỉ train |

Mỗi thư mục có `README.md` (ý tưởng, tham số), `config.json` (toàn bộ tham số), `generate.py` (chạy được ngay).
Mã dùng chung ở `common/` (một engine duy nhất đọc config; các thư mục phương pháp chỉ khác config).

## Cấu trúc
```
data/
  common/       io_utils (tải/parse IDX), features (nghiêng/dày/cỡ), imgops, ligature, layout, compose,
                pairing, engine (sinh dữ liệu theo config), augment
  raw/          dữ liệu gốc tải về + cache đặc trưng (features_*.npy) – có thể xoá, sẽ tự tạo lại
  output/       <tên_bộ>/{train,val}.npz + config_used.json   (có thể xoá, tái tạo bằng generate.py)
  tools/        download_data, run_all, preview, stats, merge_datasets
  0x_*/         các phương pháp
```

## Định dạng đầu ra (`output/<tên>/train.npz`, `val.npz`)
- `X` uint8 `(N,28,28)`, nền đen (0), nét trắng (255) — khớp kiểu MNIST; `y` int `(N,)` ∈ {10..20}.
- Metadata để phân tích lỗi: `src_left`, `src_right` (chỉ số ảnh nguồn), `writer`, `gap`, `thick`, `lig` (0 không/1 nối/2 đuôi bút), `layout` (0 aspect, 1 stretch).
- Đọc: `X, y = np.load(p)["X"], np.load(p)["y"]`; chuẩn hoá `X/255.0`, `reshape(N, 784)` cho MLP.
- Nhãn cân bằng giữa 11 lớp. Nhãn 10–19 = chữ "1" + chữ d; nhãn 20 = "2" + "0".

## Tái tạo
- `config.json` + `seed` + dữ liệu gốc (MNIST/QMNIST) ⇒ kết quả giống hệt, **không phụ thuộc số workers** (đã kiểm tra).
- Mỗi lần chạy ghi `output/<tên>/config_used.json` (đủ tham số + seed + kích thước thực dùng).
- Đổi tham số: sửa `config.json` hoặc dùng `--n-train/--n-val/--seed/--workers/--out`.

## Quy trình đề xuất
1. `run_all.py` → 4 bộ tổng hợp. Xem `preview.py` kiểm tra bằng mắt (chỉ dữ liệu tự sinh).
2. Tự viết tay (05): (a) vài trăm **số 2 chữ số** từ nhiều người làm tập **validation thật**; (b) chữ số rời của bạn/bạn bè → sinh thêm bộ `m05_self_same_writer`.
3. `tools/merge_datasets.py` trộn các nguồn theo trọng số → `final_train.npz`; augmentation (06) chỉ cho train.
4. Chọn mô hình theo val **thật** (tự viết); không chạy theo public LB.

## ⚠ Điểm chưa chắc chắn – cần xử lý
1. **Bố cục ảnh test — ĐÃ ĐỐI CHIẾU bằng thống kê tổng hợp (`tools/stats.py test.csv`, không xem ảnh):**
   - Chiều lớn nhất của vùng chữ gần như luôn ≤ 20 px (≥20 ở 55% ảnh, hầu như không có >21), rộng và cao đều bị chặn ở 20 → **giữ tỉ lệ, vừa khung 20×20** kiểu MNIST.
   - Trọng tâm ảnh nằm đúng (13.5, 13.5) với độ lệch chuẩn ≈ 0.03–0.09 → **căn giữa theo trọng tâm, độ chính xác dưới pixel**.
   - Tỉ lệ rộng/cao trung bình 1.05; 43% ảnh cao hơn rộng; tương quan rộng–cao = −0.38 (số hai chữ số khá "gọn", không phải dải ngang dài).
   - Ít pixel xám trung gian (11% so với ~20% khi mặc định) → ảnh **sắc nét** (đã thêm `post.contrast`).
   - Cấu hình hiện tại trong 5 `config.json` (`layouts.aspect` box 19.5–21.5, `squeeze` 0.8–1.0, `subpixel`, `post.contrast` 1.5–2, `compose.gap` −0.15…0.05, `thickness_hr`) cho thống kê khớp test: mean pixel 36.5 vs 38.7, tỉ lệ mực 0.143 vs 0.153, bbox cao/rộng 16.9/17.2 vs 17.1/17.2, aspect 1.075 vs 1.053, mid-gray 0.115 vs 0.114.
   - Còn lệch nhẹ: độ lệch chuẩn trọng tâm của dữ liệu tự sinh ~0.09 (test ~0.03); ảnh test có thể hơi đậm hơn (mean pixel).
   - Thống kê tổng hợp không cho biết **có nét nối hay không**, nên `ligature.prob` vẫn là ước đoán.
2. **Tần suất nét nối thật** không có số liệu trong tài liệu đã tra; `ligature.prob = 0.3` chỉ là giá trị khởi điểm. Hãy chỉnh theo cách bạn thật sự viết.
3. QMNIST là dữ liệu bên ngoài (NIST/MNIST gốc). Đề không cấm dữ liệu ngoài nhưng yêu cầu đội "tự xây dựng" dữ liệu – nên hỏi ban tổ chức nếu muốn chắc; phương án 05 (tự viết) không có rủi ro này.
4. Chưa có `test.csv` để đối chiếu nên chưa so được phân phối.

## Tuân thủ luật cuộc thi
- Không dùng `test.csv` để train / gán nhãn / xem ảnh. `tools/stats.py` chỉ in thống kê tổng hợp; `tools/preview.py` chỉ dùng cho `.npz` tự sinh.
