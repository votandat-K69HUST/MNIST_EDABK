# 06 – Augmentation

Vì dùng MLP (không có tính bất biến dịch chuyển như CNN) nên augmentation hình học quan trọng: dịch, xoay, scale, shear, biến dạng đàn hồi,
đổi độ dày nét, gamma, làm mờ, nhiễu thưa, cutout. Mã ở `../common/augment.py`; preset ở `config.json` (`light` / `medium` / `strong`).

**Offline:**
```bash
python augment_dataset.py --in ../output/m03_qmnist_same_writer/train.npz --factor 3 --preset medium
# -> train_aug_medium_x3.npz (gồm bản gốc + 3 bản augment, đã xáo)
```
**Online (trong vòng huấn luyện):**
```python
import sys; sys.path.insert(0, "data")
from common.augment import augment_batch
import json, numpy as np
cfg = json.load(open("data/06_augmentation/config.json"))["medium"]
rng = np.random.default_rng(0)
Xb_aug = augment_batch(Xb, rng, cfg)     # Xb: (B,28,28) uint8 hoặc float 0..1
```
Chỉ augment **train**; không augment val. `strong` có thể làm hỏng một số mẫu (đã xem thử) – dùng `medium` làm mặc định.
