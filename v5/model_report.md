# BÁO CÁO THỰC NGHIỆM CHI TIẾT: MÔ HÌNH V5 (PURE REAL DATA)
**Dự án**: AI Hackathon - Nhận Diện Chữ Số Viết Tay (Lớp 10 đến 20)  
**Thời gian cập nhật**: 11/10/2026  
**Chủ đề thử nghiệm**: *Đánh giá hiệu năng khi loại bỏ hoàn toàn dữ liệu nhân tạo (Synthetic Data) và chỉ huấn luyện trên dữ liệu viết tay thật (Pure Real Data).*

---

## 1. BỐI CẢNH & GIẢ THUYẾT THỬ NGHIỆM

Sau khi thu thập và tiền xử lý thành công bộ dữ liệu viết tay mới (`ve_tay_10_20 - data (1).csv`) nâng tổng số mẫu thật lên **4.338 mẫu** (Train: 3.470 mẫu / Validation: 868 mẫu), một giả thuyết lớn được đặt ra:
> *"Khi lượng dữ liệu viết tay thực tế đã tăng gấp đôi (từ 1.6K lên hơn 3.4K mẫu train), liệu mô hình MLP có thể tự học tốt trên 100% dữ liệu thật mà không cần dựa vào ~300.000 mẫu dữ liệu máy tính tổng hợp (Synthetic Data) nữa hay không?"*

Phiên bản **V5** được thiết kế độc quyền nhằm kiểm chứng câu hỏi chiến lược này:
* **V5**: Cắt bỏ 100% dữ liệu nhân tạo, chỉ huấn luyện thuần túy trên 3.470 mẫu viết tay thực tế.
* **Đối chứng với V4 / V4_new**: Giữ nguyên toàn bộ ~300.000 mẫu Synthetic kết hợp với tập viết tay thật.

---

## 2. BẢNG SO SÁNH ĐỐI ĐẦU TOÀN DIỆN (V4 VS V5)

| Tiêu Chí Đánh Giá | V4 Cũ (Real cũ + Synthetic) | V5 Mới (Chỉ Real Mới, 0 Synthetic) | V4_new Mới (Real Mới + Synthetic) |
| :--- | :---: | :---: | :---: |
| **Dữ Liệu Thật (Train)** | ~1.600 mẫu | **~3.470 mẫu** | **~3.470 mẫu** |
| **Dữ Liệu Tổng Hợp (Synthetic)** | ~300.000 mẫu | ❌ **0 mẫu (Cắt bỏ)** | **~300.000 mẫu** |
| **Số Seeds Huấn Luyện** | 5 seeds ensemble | Chỉ chạy 1 seed (`seed 0`) | 5 seeds ensemble |
| **Validation Accuracy** | 93.15% | **94.59%** | **96.10%** *(Đỉnh 96.19%)* |
| **Độ Tự Tin (Mean Confidence)** | 0.950 | **0.888** *(Giảm sút)* | **0.956** *(Rất vững)* |
| **Tỷ Lệ Mẫu Lưỡng Lự (< 0.6)** | 4.7% | **10.7%** *(Gấp 2.5 lần)* | **4.0%** |
| **Điểm Chấm Ground Truth (550 mẫu)** | 98.00% *(539/550)* | **91.64%** *(504/550)* | **97.82%** *(538/550)* |
| **Điểm Kaggle Public Leaderboard** | 0.96818 | **~0.88 - 0.91** *(Tụt sâu)* | **0.96818 (Kỷ lục)** |

---

## 3. CÁC BIẾN THỂ CỦA V5 ĐÃ THỬ NGHIỆM

Trong quá trình khảo sát nội bộ V5, 3 cấu hình đã được chạy thử:

| Cấu hình | Dữ liệu Train | Augmentation | Val Accuracy | Ground Truth Score (550 mẫu) | Hiện tượng ghi nhận |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **V5 Cũ (Chỉ 2K)** | ~1.6K mẫu | BẬT | 87.78% | 87.09% *(479/550)* | Dữ liệu quá ít, mô hình chật vật học, tự tin cực thấp (0.666). |
| **V5 Không Augment** | ~3.4K mẫu | TẮT | 82.89% | 80.91% *(445/550)* | Overfitting trầm trọng (Train Loss 0.01), tự tin ảo tưởng nhưng test trật lắt. |
| **V5 Mới (submission_v5_new)** | **~3.4K mẫu** | **BẬT** | **94.59%** | **91.64%** *(504/550)* | Học ổn định hơn, nhưng vẫn thiếu hụt trầm trọng so với mạng có Synthetic (~97%). |

---

## 4. PHÂN TÍCH CHUYÊN SÂU: VÌ SAO V5 THẤT BẠI KHI BỎ SYNTHETIC DATA?

Kết quả thực nghiệm đã đem lại câu trả lời rõ ràng về bản chất của mạng MLP trong bài toán thị giác máy tính:

### A. Giới hạn dung lượng biểu diễn của ~3.500 mẫu viết tay
* Mạng MLP không có cơ chế tích chập trượt (Convolution) để chia sẻ trọng số không gian. Để phân loại 11 lớp chữ số bị nghiêng, lệch, to nhỏ khác nhau, MLP cần hàng chục nghìn biến thể hình học.
* Khi chỉ có 3.470 mẫu thật, mô hình bị rơi vào tình trạng **thiếu vốn từ vựng không gian** (Spatial Sparsity). Nó nhớ được các mẫu trong tập train nhưng khi gặp nét bút của người viết mới trong tập Test, nó bị lúng túng (thể hiện qua tỷ lệ lưỡng lự tăng vọt lên 10.7%).

### B. Vai trò không thể thay thế của 300.000 mẫu Synthetic Data
* Dữ liệu tổng hợp (MNIST nối ngẫu nhiên, QMNIST cùng người viết, Ligatures nét nối) tuy là do máy ghép, nhưng chúng cung cấp một **khung xương kiến trúc cực kỳ vững chắc**:
  1. Dạy cho MLP hiểu cấu trúc chuẩn của từng con số độc lập.
  2. Tạo ra vô số biến thể về khoảng cách giữa chữ số hàng chục và hàng đơn vị.
  3. Dạy cho mô hình cách phân biệt ranh giới giữa 2 chữ số dính liền nét.
* Khi kết hợp: **Synthetic Data đóng vai trò làm bệ đỡ kiến thức nền tảng**, còn **Real Data đóng vai trò tinh chỉnh thực tế (Domain Adaptation)**.

### C. Sai lầm khi chỉ dùng 1 Seed ở V5
* Ở V5, việc chỉ chạy 1 seed (`seed 0`) khiến mô hình hoàn toàn phụ thuộc vào một điểm cực tiểu cục bộ ngẫu nhiên.
* Việc nâng cấp lên 5 seeds ở `v4_new` và `v7` đã chứng minh khả năng triệt tiêu phương sai và tăng độ ổn định thêm 2–3% trên tập kiểm thử.

---

## 5. BÀI HỌC CHIẾN LƯỢC & TỔNG KẾT

1. **Khẳng định nguyên lý cốt lõi**: Tuyệt đối **KHÔNG ĐƯỢC BỎ DỮ LIỆU TỔNG HỢP (Synthetic Data)** trong các bài toán huấn luyện MLP. Dữ liệu thật và dữ liệu tổng hợp không triệt tiêu nhau mà bổ trợ tương hỗ cho nhau.
2. **Quyết định định hướng**: 
   * Thử nghiệm V5 tuy không đạt điểm cao trên leaderboard nhưng là **bước đi thử nghiệm khoa học có giá trị nhất dự án**, giúp toàn bộ đội ngũ dừng việc đi vào ngõ cụt "thuần real data".
   * Ngay sau thử nghiệm V5, chúng ta quay trở lại tích hợp 4.3K mẫu thật vào nền tảng V4 để tạo ra **`v4_new`** (đạt 0.96818 kỷ lục).
   * Tiếp đó, lấy chính `v4_new` làm nền tảng để chạy Optuna tuning sâu và sinh ra phiên bản tối ưu **`v7`** và **`v7_blend`** (98.18% Ground Truth) hiện tại.
