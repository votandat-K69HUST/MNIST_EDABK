# BÁO CÁO KỸ THUẬT: PHƯƠNG PHÁP M04 - GHÉP CÙNG WRITER KÈM NÉT NỐI (LIGATURE STROKES)
**Dự án**: AI Hackathon - Nhận Diện Chữ Số Viết Tay (Lớp 10 đến 20)  
**Mã phương pháp**: `m04_ligature_strokes`  
**Ngày lập**: 11/10/2026  
**Trạng thái**: Hoàn thành & Được tích hợp làm thành phần chủ lực trong mô hình V4, V4_new, V7  

---

## 1. TỔNG QUAN & BỐI CẢNH RA ĐỜI

Trong bài toán nhận diện chữ số ghép (chuỗi từ 10 đến 20), một hiện tượng phổ biến khi người thật viết tay là **động học viết liền một mạch**:
* Ngòi bút không nhấc lên hoàn toàn khi chuyển từ chữ số hàng chục sang chữ số hàng đơn vị.
* Tạo ra một **nét nối (ligature stroke)** kéo từ đuôi chữ số thứ nhất sang đỉnh chữ số thứ hai.
* Nét nối này thường **mỏng hơn nét chính và bị mờ ở đoạn giữa** do tốc độ nhấc/lướt bút nhanh của tay người.

Các phương pháp ghép truyền thống (ghép rời rạc 2 ảnh MNIST) hoàn toàn không có đặc tính này, dẫn đến việc mô hình bị bối rối khi gặp các mẫu thi có nét nối hoặc dính liền. Phương pháp **`m04_ligature_strokes`** được thiết kế độc quyền nhằm giải quyết trọn vẹn điểm mù này.

---

## 2. QUY TRÌNH THU THẬP & TÁCH NGUỒN DỮ LIỆU (DATA SOURCING)

### A. Nguồn dữ liệu gốc (Source Kind: QMNIST)
* **Dataset sử dụng**: **QMNIST** (bộ dữ liệu mở rộng của MNIST được tái tạo từ NIST Special Database 19).
* **Vị trí lưu trữ**: `data/raw/` (chứa các file IDX: `qmnist-train-images-idx3-ubyte`, `qmnist-train-labels-idx2-ubyte`...).
* **Lý do chọn QMNIST thay vì MNIST chuẩn**: 
  * QMNIST lưu giữ đầy đủ **metadata người viết (`writer_id`)** gồm 1.074 người viết độc lập.
  * Cho phép chọn cặp chữ số được viết bởi **cùng một người (Same-Writer)**, đảm bảo tính nhất quán về độ nghiêng, bề dày nét mực, lực ấn và phong cách chữ.

### B. Cơ chế phân tách tránh rò rỉ (Writer Split)
* Để ngăn chặn rò rỉ phong cách người viết giữa tập huấn luyện và kiểm định:
  * **Tỷ lệ phân chia**: 90% Writer cho Train / 10% Writer cho Val (`val_fraction: 0.1`).
  * **Hạt giống cố định**: `split_seed: 123` để đảm bảo khả năng tái tạo 100%.
  * Một `writer_id` xuất hiện ở tập Train sẽ **tuyệt đối không xuất hiện** ở tập Val.

---

## 3. QUY TRÌNH TIỀN XỬ LÝ (PREPROCESSING PIPELINE)

Trước khi tiến hành ghép, từng chữ số đơn lẻ được xử lý sơ bộ qua hàm `_prep_digit()`:

1. **Cắt sát nét chữ số (Tight Crop)**:
   * Loại bỏ toàn bộ viền đen thừa bằng hàm `crop(img, thr=0.1)`.
2. **Đưa lên Canvas phân giải cao ($4\times$)**:
   * Thiết lập hệ số phân giải cao `hr = 4` (tương đương kích thước canvas cơ sở $\approx 112 \times 112$).
   * **Mục đích then chốt**: Việc vẽ nét nối và xoay nghiêng trên canvas $4\times$ giúp các đường cong không bị răng cưa, khi thu nhỏ về $28 \times 28$ vùng giao thoa sẽ mờ mịn tự nhiên như ảnh chụp quang học.
3. **Tạo biến thiên kích thước và độ nghiêng ngẫu nhiên**:
   * Độ lệch kích thước chung: `size_common_std: 0.05`
   * Tỷ lệ kích thước giữa 2 số: `size_ratio_std: 0.04` (số hàng chục to/nhỏ hơn số hàng đơn vị).
   * Góc nghiêng ngòi bút chung: `shear_common_std: 6.0°`
   * Góc nghiêng dao động riêng từng số: `shear_individual_std: 2.0°`
4. **Biến thiên độ dày nét ban đầu (`thickness_hr`)**:
   * Lấy mẫu ngẫu nhiên mức độ dày/mảnh nét: `[-1, 0, 1, 2]` với xác suất tương ứng `[0.2, 0.4, 0.3, 0.1]` để tạo sự đa dạng về ngòi bút (bút bi kim nét thanh hoặc bút dạ nét đậm).

---

## 4. QUY TRÌNH GHÉP VÀ THUẬT TOÁN TẠO NÉT NỐI (LIGATURE GENERATION)

Toàn bộ logic tạo nét nối được thực hiện trong module `data/common/ligature.py` và tích hợp tại `compose_v1.py`:

```
   Chữ số 1 (Trái)                           Chữ số 2 (Phải)
      +----+                                     +----+
      |    |                               p1 -> |*   | (Điểm vào: 50% đỉnh trên)
      |    |                                     |    |
p0 -> |   *| (Điểm ra: 45% đáy dưới)             |    |
      +----+                                     +----+
         \                                         /
          \__--~~ Nét nối Bezier bậc 2 (Mờ giữa) ~~/
```

### A. Xác định tọa độ Điểm Ra (`exit_point`) và Điểm Vào (`entry_point`)
* **Điểm ra ($p_0$)**: 
  * Tìm các pixel có mực ($>0.3$) nằm ở **45% phía dưới** của chữ số bên trái (`exit_lower_frac: 0.45`, thường là chân số 1).
  * Lấy phân vị $90\%$ của tọa độ cột (nằm sát mép phải nhất) và chọn ngẫu nhiên 1 điểm neo.
* **Điểm vào ($p_1$)**:
  * Tìm các pixel có mực nằm ở **50% phía trên** của chữ số bên phải (`entry_upper_frac: 0.5`).
  * Lấy phân vị $10\%$ của tọa độ cột (nằm sát mép trái nhất) và chọn ngẫu nhiên 1 điểm neo.

### B. Thuật toán vẽ đường cong Bezier bậc 2 (`draw_ligature`)
* Đường nối giữa $p_0(r_0, c_0)$ và $p_1(r_1, c_1)$ được điều khiển bởi điểm uốn cong:
  $$\text{Mid} = \frac{p_0 + p_1}{2}, \quad \text{Ctrl} = \text{Mid} + \vec{n} \cdot (\text{curv} \times \text{length})$$
  * `curvature`: Dao động ngẫu nhiên trong khoảng $[-0.3, 0.3]$.
* **Mô phỏng hiệu ứng nhấc bút (Fading Effect)**:
  * Nét vẽ được đóng dấu liên tiếp bằng các đĩa tròn mềm (`_stamp_max`).
  * Độ đậm (Alpha) dọc theo đường nối từ $t=0$ đến $t=1$:
    $$\text{Edge}(t) = |2t - 1|^{\text{fade\_power}}, \quad \alpha(t) = \text{mid\_alpha} + (1 - \text{mid\_alpha}) \times \text{Edge}(t)$$
  * Với `mid_alpha` từ $0.35$ đến $0.9$ và `fade_power = 2.0`:
    * Tại hai đầu neo ($t=0, t=1$): $\alpha = 1.0$ (đậm tuyệt đối ăn khớp vào thân chữ).
    * Tại điểm chính giữa ($t=0.5$): $\alpha$ giảm xuống mức `mid_alpha` (mờ thanh mảnh, thể hiện ngòi bút lướt nhanh).

### C. Cơ chế Đuôi Bút Phẩy Ngắn (`tail_prob: 0.25`)
* Với xác suất 25%, nét nối không nối chạm vào chữ số thứ hai mà chỉ là một **nét phẩy đuôi hướng chếch lên sang phải** rồi dừng lại (mô phỏng người viết nhấc hẳn bút lên không chạm vào số tiếp theo).

### D. Khoảng cách ghép (`gap`)
* `gap`: Được chọn trong khoảng $[-0.1, 0.12]$ (theo đơn vị chiều cao chữ). Giá trị âm cho phép 2 thân chữ số đứng rất sát hoặc chạm nhẹ vào nhau trước khi nét nối được vẽ đè lên.

---

## 5. QUY TRÌNH SCALE ẢNH VÀ ĐƯA VỀ BỐ CỤC CHUẨN MNIST ($28 \times 28$)

Sau khi hoàn tất vẽ trên canvas lớn, ảnh được đưa về chuẩn kích thước cuối cùng thông qua `layout_v1.py`:

1. **Cắt sát nét toàn phần (Tight Bounding Crop)**: Bỏ sạch phần lề trống của canvas.
2. **Khống chế kích thước hộp bao ngẫu nhiên (`box: [19.5, 21.5]`)**:
   * Giữ nguyên tỷ lệ khung hình $W/H$, co cụm chữ số sao cho cạnh lớn nhất nằm trong khoảng $19.5 - 21.5$ pixel (trung bình 20.0 pixel, khớp hoàn hảo với phân bố đo được từ tập viết tay thực tế).
3. **Căn giữa theo Trọng tâm khối lượng (`center: "com"`, `subpixel: true`)**:
   * Tính tọa độ trọng tâm điểm ảnh $(\bar{r}, \bar{c})$.
   * Dịch chuyển nội suy subpixel (độ chính xác dưới 1 pixel) để đưa trọng tâm vào đúng tâm hình $(13.5, 13.5)$.
4. **Hậu xử lý tăng cường tương phản (`contrast: [1.5, 2.0]`)**:
   * Kéo giãn phân bố mức xám quanh ngưỡng $0.5$ để đường nét sắc sảo, chống hiện tượng ảnh bị mờ đục sau khi resize.
   * Chuẩn hóa đỉnh sáng: Chia cho giá trị cực đại để đỉnh sáng đạt đúng 255.

---

## 6. KẾT QUẢ ĐẦU RA & VAI TRÒ TRONG CÁC MODEL V4, V7

* **Dung lượng sinh ra**: 
  * `train.npz`: **55.000 mẫu** (kèm nhãn $10 - 20$ cân bằng tuyệt đối 5.000 mẫu/lớp).
  * `val.npz`: **5.500 mẫu**.
  * File lưu trữ: `data/output/m04_ligature_strokes/train.npz`.
* **Đóng góp chiến lược**:
  * Bộ dữ liệu `m04` chiếm trọng số **1.0** (cao nhất trong các nguồn dữ liệu nhân tạo) trong file cấu hình huấn luyện của **`v4`**, **`v4_new`**, và **`v7`**.
  * Chính nhờ 55.000 mẫu có nét nối và chạm nét này, mô hình MLP đã học được cách tách biệt thông minh giữa: nét nối ngẫu nhiên và nét chính của các số nhạy cảm (như số 12, 13, 14, 15, 17, 18), ngăn chặn việc mô hình bị nhầm lẫn khi người thi viết chữ liền nét.
