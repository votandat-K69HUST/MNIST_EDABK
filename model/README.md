# model/ – MLP phân loại số 10–20 và tạo `output.csv`

Chỉ dùng **MLP** (Linear + BatchNorm + activation + Dropout). Không CNN, không mạng phức tạp hơn (ràng buộc của đội, xem `../the_le.md`).

## Pipeline
```
data/output/*.npz  --(trộn theo trọng số)-->  train  --(augmentation online, torch)-->  MLP  --(chọn epoch theo val)-->  model_seed<k>.pt
test.csv  --(/255, reshape)-->  ensemble (+TTA tuỳ chọn)  --(softmax TB, argmax + 10)-->  output.csv (id,label)
```
| File | Vai trò |
|---|---|
| `config.json` | **Toàn bộ siêu tham số**: nguồn dữ liệu + trọng số, augmentation, kiến trúc, optimizer, ensemble, đường dẫn output |
| `mlp.py` | Mô hình MLP (chuẩn hoá đầu vào lưu trong mô hình) |
| `augment_torch.py` | Augmentation theo batch: affine (xoay/scale/shear/dịch), dày–mỏng nét, gamma, nhiễu |
| `data_utils.py` | Nạp và trộn các file npz từ `data/` |
| `train.py` | Huấn luyện (AdamW/SGD/Adam, cosine + warmup, label smoothing, early stopping theo val) |
| `tune.py` | Random search siêu tham số, chọn theo val |
| `predict.py` | Inference trên `test.csv` → `output.csv` |
| `run_pipeline.py` | Chạy cả chuỗi |

## Chạy
```bash
# 0) Dữ liệu (nếu chưa có): python ../data/tools/run_all.py --workers 4
python train.py                              # train ensemble theo config.json (seeds 0,1,2) -> runs/mlp_v1/
python predict.py                            # -> ../output.csv  (id,label)
python run_pipeline.py                       # (sinh dữ liệu nếu thiếu) + train + predict
python run_pipeline.py --tune 20             # random search 20 lần, train lại bằng cấu hình tốt nhất, predict
```
**Ghi đè tham số nhanh:** `python train.py --set train.lr=0.001 model.hidden=[1024,512] train.epochs=40 --seeds 0`
**Tune riêng:** `python tune.py --trials 30 --epochs 12 --subsample 60000` → `runs/tune_mlp_v1/results.csv`, `best_config.json`.
Huấn luyện lại bằng cấu hình tốt nhất: `python train.py --config runs/tune_mlp_v1/best_config.json --name mlp_v1_tuned --set train.epochs=40` rồi `python predict.py --config runs/tune_mlp_v1/best_config.json --run runs/mlp_v1_tuned`.

`predict.py` tự kiểm tra: `id` duy nhất, đủ số dòng, `label` ∈ [10,20]. In phân bố nhãn dự đoán (test cân bằng 50/lớp nên phân bố lệch nhiều = dấu hiệu lệch miền dữ liệu). Tên file mặc định `../output.csv`; đề yêu cầu nộp tên `submission.csv` → dùng `--out ../submission.csv` hoặc đổi tên khi nộp.

## Siêu tham số chính (config.json)
- `data.train[]`: `path` + `weight` (tỉ lệ trộn các bộ `m01…m05`, `self_numbers_train.npz`); `data.total_train` (0 = tổng các nguồn).
- `data.val`: `synthetic` (cùng phân phối với train → **lạc quan**) và `real` (tự viết thật → đáng tin hơn). **Chọn model theo `real` nếu có, nếu không theo `synthetic`.**
- `augment.*`: rotate, scale, shear, shift, thickness_p, gamma, noise_std (`enabled=false` để tắt).
- `model.*`: `hidden` (danh sách kích thước lớp ẩn), `dropout`, `activation` (relu/gelu/silu/tanh/leaky_relu), `batchnorm`, `input_dropout`.
- `train.*`: `epochs`, `batch_size`, `optimizer`, `lr`, `weight_decay`, `scheduler` (cosine/constant), `warmup_epochs`, `label_smoothing`, `grad_clip`, `patience`.
- `ensemble.seeds`: mỗi seed một mô hình; xác suất được trung bình khi dự đoán. `predict.tta_shifts`: dịch ±N px lúc dự đoán (0 = tắt).

## Lưu ý quan trọng
- Điểm val tổng hợp **không phản ánh điểm trên test thật**: dữ liệu tự sinh cùng phân phối với train. Cần tập val chữ viết tay thật (`data/05_self_handwritten`) để đánh giá đúng và tune.
- Public LB chỉ 220 ảnh – đừng tune theo public LB (xem `../the_le.md`).
- Tuân thủ luật: `test.csv` chỉ dùng để chạy inference; không train/gán nhãn/xem ảnh test.
- Tái lập: seed trong config; `runs/<name>/config_used.json` + `model_seed*.pt` (có kèm config). BTC có thể yêu cầu code + weights.
