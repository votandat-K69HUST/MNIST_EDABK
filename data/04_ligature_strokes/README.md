# 04 – Cùng writer + nét nối (ligature) mờ ở giữa

**Ý tưởng:** khi viết bằng bút, người ta hay kéo nét nối từ chữ này sang chữ kia; nét nối thường **mỏng và mờ ở giữa** (nhấc bút nhanh).
Tài liệu về chữ số dính nhau mô tả "ligature = nét đuôi của chữ trước kéo sang chữ sau" (nghiên cứu tách chữ dính trên NIST SD19).
Không nguồn nào mô phỏng ligature khi tạo dữ liệu huấn luyện, nên phần này tự thiết kế (`common/ligature.py`).

**Cách làm:** (nguồn + chia như 03) → sau khi đặt hai chữ trên canvas phân giải cao, vẽ đường Bezier bậc 2 từ **điểm ra** của chữ đầu
(điểm ngoài cùng bên phải trong ~45% phía dưới, vd. chân số 1) tới **điểm vào** chữ sau (điểm trái nhất trong nửa trên). Độ đậm:
đậm ở hai đầu, giảm còn `mid_alpha` ở giữa. Một phần là "đuôi bút" (`tail_prob`): nét ngắn hướng sang phải không chạm chữ sau.

**Tham số (`compose.ligature`):** `prob` (0.3 – giá trị khởi điểm, **không có số liệu thật**, hãy chỉnh theo cách bạn viết), `tail_prob`,
`curvature`, `mid_alpha` [0.35, 0.9], `fade_power`, `width_scale` (độ dày nét nối so với nét chữ), `exit_lower_frac`, `entry_upper_frac`.
`compose.gap [0.05, 0.5]` lớn hơn 03 để có chỗ cho nét nối. Metadata `lig` trong npz: 0 không, 1 nối đầy đủ, 2 đuôi bút.

**Chạy:** `python generate.py` → `../output/m04_ligature_strokes/`.
**Lưu ý:** nét nối từ chân "1" lên đầu chữ sau đôi khi tạo hình chữ V; nếu thấy quá nhiều, giảm `prob`/`width_scale`.
