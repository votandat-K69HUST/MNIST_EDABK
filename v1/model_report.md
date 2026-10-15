# BÁO CÁO KỸ THUẬT: PHIÊN BẢN V1 (INITIAL BASELINE)
**Dự án**: AI Hackathon - Nhận Diện Chữ Số Viết Tay (Lớp 10 đến 20)  
**Thời gian đóng gói**: 11/10/2026  
**Trạng thái**: Đã lưu trữ & Đóng gói hoàn chỉnh thành thư mục `v1/`  

---

## 1. TỔNG QUAN PHIÊN BẢN V1

Phiên bản **V1** là mô hình nền tảng đầu tiên của dự án, được thiết kế ngay sau khi hoàn thiện các bộ tạo dữ liệu tổng hợp phức tạp (có nét nối `m04`, ghép theo phong cách `m02`, ghép cùng writer `m03` và augmentation).

* **Mục tiêu**: Xây dựng một baseline MLP hoàn chỉnh từ đầu, thiết lập pipeline dự đoán tự động cho tập `test.csv`.
* **Dữ liệu huấn luyện**: Kết hợp các bộ dữ liệu tổng hợp ban đầu (`m01`, `m03`, `m04`) với dữ liệu tự viết tay ban đầu (`m05`).

---

## 2. KẾT QUẢ ĐÁNH GIÁ TRÊN GROUND TRUTH (test_labeled.csv)

Khi chấm điểm file dự đoán `v1/submission_v1.csv` trên bộ nhãn Ground Truth 550 mẫu:

| Tập Đánh Giá | Số Mẫu Đúng / Tổng Số | Độ Chính Xác (Accuracy) |
| :--- | :---: | :---: |
| **Toàn bộ Test Set (550 mẫu)** | **510 / 550** | **92.73%** |
| **Tập Public (220 mẫu)** | 206 / 220 | **93.64%** |
| **Tập Private (330 mẫu)** | 304 / 330 | **92.12%** |

### Hiệu suất từng lớp ở phiên bản V1:
* **Các lớp dự đoán rất tốt (≥ 98%)**: 12 (98%), 13 (98%), 15 (100%), 17 (100%), 20 (98%).
* **Các lớp còn gặp nhiều khó khăn**:
  * **Lớp 19**: Độ chính xác chỉ đạt **76.47%** (bị nhầm nhiều sang 13).
  * **Lớp 18**: Độ chính xác đạt **85.71%** (bị nhầm sang 13 và 11).
  * **Lớp 11**: Độ chính xác đạt **86.00%** (bị nhầm sang 12 và 20).

---

## 3. CÁC THÀNH PHẦN ĐƯỢC ĐÓNG GÓI TRONG THƯ MỤC V1

1. **`v1/submission_v1.csv`**: File kết quả dự đoán ban đầu trên `test.csv` (đổi tên từ `output_v1_before_simplify.csv`).
2. **`v1/configs/`**: Thư mục chứa toàn bộ cấu hình gốc của V1:
   * `model_config_v1.json`: Cấu hình kiến trúc mạng MLP và tham số train của V1.
   * `m01_config.json`, `m03_config.json`, `m05_config.json`.
3. **`v1/pipeline_scripts/`**: Toàn bộ mã nguồn xử lý đặc trưng phức tạp thế hệ đầu:
   * `compose_v1.py`: Thuật toán ghép tổng hợp đa phong cách.
   * `engine_v1.py`: Engine điều phối sinh dữ liệu v1.
   * `features_v1.py` & `pairing_v1.py`: Thuật toán trích xuất đặc trưng và ghép cặp chữ số.
   * `layout_v1.py`: Bố cục đa dạng (aspect, stretch...).
   * `ligature.py`: Thuật toán vẽ nét nối Bezier ban đầu.
   * `augment.py`: Module augmentation offline.
4. **`v1/model_report.md`**: Báo cáo tổng kết kỹ thuật này.
