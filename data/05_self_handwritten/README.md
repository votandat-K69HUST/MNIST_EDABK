# 05 – Tự viết tay (dữ liệu thật)

**Vì sao:** test là chữ viết tay thật. Vài trăm mẫu thật rất có giá trị để (1) làm **tập validation thật** và (2) lấy phong cách của người viết thật.

## Bước 1 – Thu thập (`collect.py`, tkinter; nền đen, nét trắng như MNIST)
```bash
python collect.py --writer minh --mode number --per-label 30   # viết số 10..20 (mỗi nhãn 30 lần)
python collect.py --writer minh --mode digit  --per-label 40   # viết chữ số rời 0..9
```
Enter = lưu, Esc/C = xoá, S = bỏ qua. Ảnh lưu nguyên canvas ở `samples/<writer>/<mode>/<nhãn>_<id>.png` (xử lý lại được với bố cục khác).
Mỗi **người viết dùng một `--writer` riêng** (chia train/val theo người). Viết đa dạng: nối nét/không nối, nghiêng khác nhau, to/nhỏ.

## Bước 2 – Chuyển thành npz (`build_dataset.py`)
```bash
python build_dataset.py --mode numbers --layout aspect --box 20 --val-writers ban_a   # -> output/self_numbers{,_train,_val}.npz
python build_dataset.py --mode digits                                                  # -> output/self_digits.npz
```
`layout aspect --box 20` = quy ước MNIST giống dữ liệu tổng hợp (giữ tỉ lệ, vừa khung 20×20, căn trọng tâm); `stretch` chỉ để thử. Chữ rời cũng được chuẩn hoá kiểu MNIST.

## Bước 3 – (tuỳ chọn) Sinh dữ liệu cùng-writer từ chữ số tự viết
```bash
python generate.py     # dùng config.json: source.kind="self", self_path=../output/self_digits.npz  -> ../output/m05_self_same_writer/
```
Cần mỗi writer có đủ các chữ 0,1,2,…,9. Với ít writer, `split.val_fraction=0` (toàn bộ làm train); dùng `self_numbers_val.npz` làm val.
