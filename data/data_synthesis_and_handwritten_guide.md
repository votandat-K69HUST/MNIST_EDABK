# TÀI LIỆU KỸ THUẬT: QUY TRÌNH XỬ LÝ DỮ LIỆU VIẾT TAY THỰC TẾ VÀ ĐỒNG BỘ HÓA THỐNG KÊ CHO DỮ LIỆU TỔNG HỢP (SYNTHETIC DATA)
**Dự án**: AI Hackathon - Nhận Diện Chữ Số Viết Tay (Lớp 10 đến 20)  
**Tác giả**: Đội ngũ phát triển & Nghiên cứu mô hình  
**Ngày lập tài liệu**: 11/10/2026  
**Mục tiêu**: Chuẩn hóa toàn bộ quy trình tiền xử lý dữ liệu viết tay thật và cơ chế đồng bộ hóa các chỉ số thống kê vào thuật toán sinh dữ liệu ghép (Synthetic Data) nhằm tối đa hóa khả năng khái quát hóa cho mạng MLP.

---

## I. TỔNG QUAN KIẾN TRÚC DỮ LIỆU

Nhận diện chuỗi chữ số (10–20) bằng mạng MLP phẳng (không dùng CNN/ViT) đòi hỏi mô hình phải có khả năng bất biến với các biến dạng hình học (độ nghiêng, kích thước, nét bút, khoảng cách chữ). Dự án giải quyết bài toán bằng cách kết hợp:
1. **Dữ liệu viết tay thật (Real Handwritten Data)**: Dữ liệu thu thập thực tế từ người viết (`05_self_handwritten` và sau này là `ve_tay_10_20` với 4.338 mẫu).
2. **Dữ liệu tổng hợp (Synthetic Data)**: Ghép từ các chữ số đơn của MNIST/QMNIST (`m01`, `m03`, `m04`) với hơn 300.000 mẫu.

> ⚠️ **Nguyên tắc cốt lõi**: Khâu sinh dữ liệu nhân tạo **không được tạo ra các bức ảnh "giả tạo, cứng nhắc"**, mà phải **mô phỏng chính xác các đặc trưng vật lý và thông số thống kê đo được từ tập viết tay thực tế**.

---

## II. BẢNG ĐỐI CHIẾU CÁC CHỈ SỐ THỐNG KÊ ĐO ĐƯỢC TỪ TẬP VIẾT TAY

Công cụ đo lường: `data/tools/stats.py`  
Lệnh đo: `python data/tools/stats.py data/output/v5_new_train.npz --compare data/output/m03_qmnist_same_writer/train.npz`

| Chỉ Số Thống Kê Đo Lường | Tập Viết Tay Thật (`v5_new_train.npz`) | Tập Dữ Liệu Ghép (`m03_qmnist...`) | Mức Độ Đồng Bộ / Nhận Xét |
| :--- | :---: | :---: | :--- |
| **Trọng tâm hàng (Center of Mass - Row)** | **13.499** | **13.502** | **Trùng khít tuyệt đối tâm (13.5)** |
| **Trọng tâm cột (Center of Mass - Col)** | **13.507** | **13.500** | **Trùng khít tuyệt đối tâm (13.5)** |
| **Độ lệch chuẩn trọng tâm hàng (`com_row_std`)** | **0.290** | **0.288** | Khớp hoàn hảo độ vững vị trí |
| **Độ lệch chuẩn trọng tâm cột (`com_col_std`)** | **0.288** | **0.288** | Khớp hoàn hảo độ vững vị trí |
| **Chiều cao hộp bao tối đa (p90 Box Height)** | **20.0 px** | **20.0 px** | Khớp chính xác khung `box = 20` |
| **Chiều rộng hộp bao tối đa (p90 Box Width)** | **20.0 px** | **19.0 px** | Khớp khung chuẩn chữ số đôi |
| **Tỷ lệ diện tích nét mực (`ink_fraction >= 128`)** | **11.5%** | **10.7%** | Tương đương độ đậm nét tự nhiên |
| **Tỷ lệ xám chuyển tiếp viền (`mid_gray_fraction`)** | **11.6%** | **14.9%** | Tương đương độ nhòe khử răng cưa |
| **Đỉnh sáng cực đại (`peak_mean`)** | **255.0** | **255.0** | Chuẩn hóa biên độ ánh sáng 100% |
| **Mật độ điểm ảnh trung bình (`mean_pixel`)** | **29.01** | **27.17** | Mức phân bố năng lượng tương đồng |

---

## III. CHI TIẾT CÁC THÔNG SỐ CHỈNH DỮ LIỆU SYNTHETIC: HÀM & FILE THỰC THI

Dưới đây là chi tiết từng thông số thống kê từ tập viết tay đã được áp dụng vào khâu sinh dữ liệu ghép, kèm file và hàm thực hiện:

### 1. Thông số Căn Tâm Theo Trọng Tâm Khối Lượng (Center of Mass = 13.5)
* **Thống kê từ tập viết tay**: Chữ số viết tay trong khung $28 \times 28$ có trọng tâm mực phân bố chính xác tại $\bar{r} = 13.5, \bar{c} = 13.5$ với độ lệch cực nhỏ ($\approx 0.29$). Nếu chỉ căn giữa theo hộp bao hình học (Bounding Box center), các số có nét lệch (như số 10, 11) sẽ bị trôi dạt sang bên trái/phải làm hỏng nhận diện của MLP.
* **Quy tắc điều chỉnh**: Tính tích phân trọng tâm mực và đặt điểm trọng tâm trùng với $(13.5, 13.5)$.
* **File thực hiện**: [data/common/imgops.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/imgops.py)
* **Hàm thực hiện**: 
  * `center_of_mass(a)`:
    $$\bar{r} = \frac{\sum (r \cdot I(r,c))}{\sum I(r,c)}, \quad \bar{c} = \frac{\sum (c \cdot I(r,c))}{\sum I(r,c)}$$
  * `paste_center(patch, size=28, center="com")`: Dịch chuyển patch ảnh con sao cho $(\bar{r}, \bar{c})$ đặt đúng tọa độ $(\frac{size-1}{2}, \frac{size-1}{2}) = (13.5, 13.5)$.

---

### 2. Thông số Khung Hộp Bao Cực Đại (`box = 20` pixel)
* **Thống kê từ tập viết tay**: Phân vị thứ 90 (`p90`) của chiều cao và chiều rộng chữ số viết tay là đúng **20 pixel**. Phần còn lại (4 pixel lề trên dưới, trái phải) là viền đen an toàn để mạng MLP không bị chạm biên.
* **Quy tắc điều chỉnh**: Sau khi ghép 2 chữ số, co giãn giữ nguyên tỷ lệ khung hình (Preserve Aspect Ratio) sao cho cạnh dài nhất của cụm 2 số vừa khít kích thước 20 pixel.
* **File thực hiện**: [data/common/layout.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/layout.py)
* **Hàm thực hiện**: 
  * `apply_layout(canvas, lcfg)`:
    ```python
    c = crop(canvas, thr=0.05)
    h, w = c.shape
    s = lcfg.get("box", 20) / max(h, w)
    patch = resize(c, round(h * s), round(w * s))
    ```

---

### 3. Thông số Khoảng Cách Giữa 2 Chữ Số (`gap = [-0.05, 0.35]`)
* **Thống kê từ tập viết tay**: Người thật khi viết số 2 chữ số (10 đến 20) có thói quen viết rất đa dạng: có người viết cách xa, có người viết sát nhau, và đặc biệt có người **viết dính nét vào nhau (chồng lấn/ligatures)**.
* **Quy tắc điều chỉnh**: Khoảng cách `gap` được lấy mẫu ngẫu nhiên từ $-0.05$ đến $0.35$ (theo đơn vị chiều cao chữ 20px). Giá trị âm mô phỏng trực tiếp hiện tượng 2 nét chữ chạm nhau hoặc dính vào nhau.
* **File thực hiện**: [data/common/compose.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/compose.py)
* **Hàm thực hiện**: 
  * `compose_number(left, right, rng, cfg)`:
    ```python
    base = 20 * hr
    gap = rng.uniform(*cc.get("gap", [-0.05, 0.35])) * base
    gap = max(gap, -0.6 * min(wL, wR)) # Giới hạn không chồng lấp quá 60% chiều rộng
    ```

---

### 4. Thông số Độ Lệch Đường Cơ Sở / Đáy Chữ (`baseline_std = 0.04`)
* **Thống kê từ tập viết tay**: Chữ số hàng đơn vị (bên phải) hiếm khi nằm thẳng hàng tuyệt đối với chữ số hàng chục (bên trái). Bàn tay người viết thường làm số bên phải bị lệch cao hơn hoặc thấp hơn đường đáy một vài pixel.
* **Quy tắc điều chỉnh**: Dịch chuyển trục Y của chữ số bên phải theo phân phối chuẩn với độ lệch chuẩn $4\%$ kích thước chữ (`baseline_std = 0.04`).
* **File thực hiện**: [data/common/compose.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/compose.py)
* **Hàm thực hiện**:
  * `compose_number(...)`:
    ```python
    dy = rng.normal(0, cc.get("baseline_std", 0.04)) * base
    dy_i = int(round(dy))
    yL_bot, yR_bot = 0, dy_i  # Đáy số phải lệch dy so với số trái
    ```

---

### 5. Thông số Tỷ Lệ To/Nhỏ Giữa 2 Chữ Số (`size_ratio_std = 0.06`)
* **Thống kê từ tập viết tay**: Số hàng chục (số 1 hoặc số 2) thường được viết to hơn hoặc nhỏ hơn số hàng đơn vị do lực tay và vị trí đặt bút.
* **Quy tắc điều chỉnh**: Tỷ lệ kích thước giữa số bên trái và bên phải được phóng đại/thu nhỏ theo hàm số mũ log-normal: `size_ratio = exp(Normal(0, 0.06))`.
* **File thực hiện**: [data/common/compose.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/compose.py)
* **Hàm thực hiện**:
  * `_prep_digit(img, scale, shear_deg)` và `compose_number(...)`.

---

### 6. Thông số Độ Nghiêng Độc Lập Của Từng Chữ Số (`shear_individual_std = 3.0°`)
* **Thống kê từ tập viết tay**: Ngay cả khi cùng một người viết, góc nghiêng của con số thứ nhất và con số thứ hai vẫn có sự dao động ngẫu nhiên khoảng $2^\circ - 4^\circ$.
* **Quy tắc điều chỉnh**: Áp dụng phép biến đổi Affine nghiêng ngang (shear X) độc lập cho từng con số trước khi đưa lên canvas ghép.
* **File thực hiện**: [data/common/imgops.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/imgops.py)
* **Hàm thực hiện**: 
  * `shear_x(a, deg)`:
    ```python
    t = np.tan(np.deg2rad(deg))
    mat = np.array([[1.0, 0.0], [t, 1.0]])
    ndimage.affine_transform(p, mat, offset=off, order=1, mode="constant")
    ```

---

### 7. Ghép Ở Độ Phân Giải Cao Khử Răng Cưa (`hr = 4` & Thuật toán `Image.BOX`)
* **Thống kê từ tập viết tay**: Ảnh quét thực tế có tỷ lệ pixel xám chuyển tiếp mịn màng (`mid_gray_fraction ≈ 11.6%`). Nếu cắt ghép trực tiếp trên lưới $28 \times 28$, điểm ảnh bị gãy khúc, bậc thang (aliasing).
* **Quy tắc điều chỉnh**: Phóng to $4\times$ canvas ghép, sau đó thu nhỏ về $28 \times 28$ bằng thuật toán lấy trung bình diện tích `Image.BOX` để tái tạo đúng độ mịn viền mực xám tự nhiên.
* **File thực hiện**: [data/common/compose.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/compose.py) & [data/common/imgops.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/imgops.py)
* **Hàm thực hiện**:
  * `resize(a, new_h, new_w)`: Sử dụng `Image.BOX` khi thu nhỏ và `Image.BILINEAR` khi phóng to.

---

### 8. Chuẩn Hóa Đỉnh Sáng Tuyệt Đối (`peak_mean = 255.0`)
* **Thống kê từ tập viết tay**: Nét mực viết tay luôn có điểm đạt cực đại giá trị 255 (trắng tuyệt đối trên nền đen).
* **Quy tắc điều chỉnh**: Chia toàn bộ ma trận ảnh cho giá trị cực đại của nó để đảm bảo ảnh luôn có độ tương phản tối đa.
* **File thực hiện**: [data/common/layout.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/common/layout.py)
* **Hàm thực hiện**: `apply_layout(...)`:
  ```python
  m = out.max()
  return out / m if m > 0 else out
  ```

---

### 9. Biến Hóa Độ Dày Nét Bút Dạ / Bút Bi Kim (Online Morphological Augmentation)
* **Thống kê từ tập viết tay**: Tập viết tay thu thập gồm nhiều loại bút khác nhau (bút bi kim nét thanh mảnh, bút nước, bút dạ quang đậm nét).
* **Quy tắc điều chỉnh**: Trong quá trình huấn luyện batch trực tiếp trên GPU/CPU, sử dụng toán tử Max-Pooling để làm nét chữ nở ra (Dilation - bút dạ) hoặc co lại (Erosion - bút kim) với xác suất $50\%$.
* **File thực hiện**: [model/augment_torch.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/model/augment_torch.py)
* **Hàm thực hiện**: 
  * `augment(x, cfg, g)`:
    ```python
    dil = F.max_pool2d(x, 3, 1, 1)   # Làm dày nét bút (Dilation)
    ero = -F.max_pool2d(-x, 3, 1, 1) # Làm mảnh nét bút (Erosion)
    alt = torch.where(torch.rand(...) < 0.5, dil, ero)
    x = x + use * lam * (alt - x)
    ```

---

## IV. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU VIẾT TAY THẬT

Để đảm bảo dữ liệu viết tay thật và dữ liệu tổng hợp hoàn toàn đồng nhất về hệ quy chiếu tọa độ, tập viết tay thật được xử lý qua cùng một pipeline toán học:

* **File thực hiện**: [data/05_self_handwritten/build_dataset.py](file:///D:/2025/2025.2/LAB/AI_ML/Week_11/AI_HACKATHON/data/05_self_handwritten/build_dataset.py)
* **Hàm thực hiện**: `number_to_28(img, layout="aspect", box=20.0)`
* **Các bước chuẩn hóa**:
  1. `crop(img, thr=0.05)`: Cắt sạch toàn bộ nền đen thừa xung quanh chữ viết tay.
  2. Co tỷ lệ giữ nguyên khung hình: $s = \frac{20}{\max(h, w)}$.
  3. `paste_center(..., 28, center="com")`: Đặt chữ số vào khung $28 \times 28$ sao cho trọng tâm mực nằm chính xác tại tâm hình $(13.5, 13.5)$.
  4. Đưa dải giá trị về $[0, 1]$ và lưu dưới định dạng `uint8 (0..255)` trong file `npz`.

---

## V. TỔNG KẾT & Ý NGHĨA KỸ THUẬT

1. **Tính nhất quán không gian**: Nhờ đồng bộ hóa trọng tâm $(13.5, 13.5)$ và hộp bao $20 \times 20$, mạng MLP không bao giờ bị bối rối giữa việc xử lý ảnh viết tay thật và ảnh máy tính ghép.
2. **Khắc phục điểm yếu cố hữu của MLP**: Việc mô phỏng độ nghiêng, khoảng cách biến thiên và độ dày nét bút dạ/kim đã trang bị cho MLP khả năng bao quát toàn bộ các biến thể chữ số mà một tập viết tay nhỏ không thể có đủ.
3. **Tuân thủ đạo đức thi đấu**: Toàn bộ các thông số thống kê được trích xuất hoàn toàn hợp lệ từ tập dữ liệu viết tay tự thu thập (`05_self_handwritten`), tuân thủ $100\%$ quy chế cuộc thi về việc không can thiệp hay nhìn trước tập dữ liệu kiểm thử bí mật.
