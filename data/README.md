# data/ – Sinh dữ liệu train cho MNIST 10–20 (bản tối giản v2)

Sinh ảnh 28×28 xám chứa số 10–20 (11 lớp) bằng cách **ghép hai chữ số MNIST/QMNIST**, tái tạo hoàn toàn từ config + seed.
Chỉ dùng MLP (xem `../the_le.md`). Bản v1 phức tạp hơn (nét nối, dày/mỏng nét, độ tương phản, ghép theo phong cách, nhiều bố cục, augmentation offline) được **cất trong `_archive_v1/`**, không xoá.

## Chạy
```bash
pip install -r requirements.txt
python tools/download_data.py          # tải MNIST + QMNIST vào raw/
python tools/run_all.py                # sinh m01 + m03 (55.000 train + 5.500 val mỗi bộ) -> output/
python tools/preview.py output/m03_qmnist_same_writer/train.npz   # xem lưới ảnh TỰ SINH
```
Từng bộ: `cd 01_mnist_random_concat && python generate.py [--n-train N --n-val N --seed S --workers W]`.

## Hai phương pháp
| Thư mục | Ý tưởng | Nguồn | Chia train/val |
|---|---|---|---|
| `01_mnist_random_concat` | Hai chữ bốc ngẫu nhiên (phong cách không đồng nhất) | MNIST (70k) | theo ảnh |
| `03_qmnist_same_writer` | Hai chữ **cùng một người viết** (ý tưởng bài MDW, arXiv 2512.00676) | QMNIST (1074 writer) | theo **writer** |
| `05_self_handwritten` | Công cụ tự viết tay → tập val thật / chữ số của riêng bạn | bạn tự viết | theo người viết |

Nhãn 10–19 = chữ "1" + chữ d; nhãn 20 = "2" + "0". Nhãn cân bằng 11 lớp.

## Quy trình ghép (common/compose.py, layout.py) – chỉ những phần được giữ lại
1. Cắt sát nét từng chữ, phóng 4× (canvas phân giải cao).
2. **Lệch kích thước** giữa hai chữ (`compose.size_ratio_std`) và **độ nghiêng** từng chữ độc lập (`compose.shear_individual_std`, độ).
3. Đặt cạnh nhau với **khoảng cách `gap`** (đơn vị 20 px; âm = chồng lấn/chạm) và **lệch đường cơ sở** (`baseline_std`); chữ phải đặt thẳng đáy chữ trái + lệch.
4. **Thu nhỏ MỘT LẦN** về 28×28 theo quy ước chuẩn MNIST (`layout`): cắt sát nét, giữ tỉ lệ để vừa khung 20×20, căn giữa theo trọng tâm (`box=20`).

Không còn: nét nối, đổi độ dày nét, độ tương phản/gamma/làm mờ/nhiễu, ép ngang, nhiều kiểu bố cục, ghép theo phong cách, augmentation trong khâu sinh dữ liệu.
Các tham số giữ lại (`gap` [-0.05, 0.35], `baseline_std` 0.04, `size_ratio_std`, `shear_individual_std`: 0.12/8° cho 01 và 0.04/2° cho 03) là **giá trị tự đặt**, không có nguồn bài báo, và **không khớp với thống kê test** (v2 không dùng thông tin gì từ `test.csv`).

## Cấu trúc
```
data/
  common/   io_utils (tải/parse IDX), features (cắt nét), imgops, compose, layout, pairing (random | same_writer), engine
  raw/      dữ liệu gốc tải về (có thể xoá, tự tải lại)
  output/   <tên>/{train,val}.npz + config_used.json
  tools/    download_data, run_all, preview, stats, merge_datasets
  01_*, 03_*, 05_*   cấu hình + generate.py từng phương pháp
  _archive_v1/       bản cũ (02 style_matched, 04 ligature, 06 augmentation, ligature.py, augment.py, configs/engine/compose cũ, output cũ)
```

## Định dạng đầu ra
`X` uint8 `(N,28,28)` nền đen, nét trắng; `y` ∈ {10..20}; metadata `src_left`, `src_right`, `writer`, `gap`. Chuẩn hoá `X/255`, reshape `(N,784)` cho MLP.

## Tái tạo
Cùng `config.json` + `seed` + dữ liệu gốc ⇒ kết quả giống hệt, không phụ thuộc số workers. Mỗi lần chạy ghi `output/<tên>/config_used.json`.

## Hướng phát triển (để bạn tự thêm từng bước và đo trên val thật)
Bản v1 trong `_archive_v1/` có sẵn code tham khảo cho: nét nối (`common/ligature.py`), ghép theo phong cách (`features_v1.py`, `pairing_v1.py`), nhiều bố cục (`layout_v1.py`), dày/mỏng nét + tương phản (`compose_v1.py`), augmentation (`augment.py`, `06_augmentation/`). Nên thêm từng thành phần một và đo trên **tập chữ viết tay thật** (`05_self_handwritten`).

## Tuân thủ luật
Không dùng `test.csv` để train/gán nhãn/xem ảnh. `tools/stats.py` (thống kê tổng hợp) vẫn còn nhưng v2 **không** dùng nó để chỉnh dữ liệu.
