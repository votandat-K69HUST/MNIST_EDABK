# Tổng hợp phương pháp xây dựng data train cho nhận diện số viết tay nhiều chữ số

> Mục đích: tham khảo cách các tác giả dựng dữ liệu để áp dụng cho bài MNIST 10–20 (xem `the_le.md`). Ràng buộc của người làm: chỉ dùng MLP, không CNN.
> Mức độ tin cậy: mình đọc trực tiếp text của các nguồn 1, 2, 3 và 4 (một phần). Chi tiết nào không đọc được thì ghi rõ "chưa xác nhận". Không có số liệu nào dưới đây tự suy ra.

## 1. Các nguồn đã đọc

### 1.1. MDW – Realistic Handwritten Multi-Digit Writer Number Recognition Challenges (arXiv 2512.00676, 2026)
- Link: https://arxiv.org/abs/2512.00676
- **Vấn đề họ chỉ ra:** MNIST gốc không giữ thông tin ai viết chữ số nào, nên việc trộn ngẫu nhiên các chữ MNIST thành chuỗi tạo ra số "không thực tế" (mỗi chữ từ một người khác nhau). Số thật thì tất cả chữ số do **cùng một người** viết.
- **Cách làm:** dùng QMNIST (MNIST có thêm metadata, trong đó có `writer id`) để ghép các chữ số **cùng một người viết** thành chuỗi. Với mỗi số: chọn ngẫu nhiên một writer, rồi với từng chữ số d chọn ngẫu nhiên một ảnh chữ d của writer đó.
- **Chia tập theo writer:** chia writer thành tập train và test, không cho một writer xuất hiện ở cả hai (giống cách MNIST gốc tách writer). Bộ script của họ (`create_MDW_data.py`) nhận danh sách writer train để sinh dữ liệu train mà không rò rỉ sang test.
- **Kết quả đáng chú ý:** classifier tốt trên chữ số đơn vẫn kém hơn rõ trên số nhiều chữ số (ví dụ CNN VGG-like: lỗi chữ đơn 0.83% nhưng lỗi chuỗi 5 chữ số 2.99%).
- **Hạn chế họ tự nêu:** chuỗi tạo bằng ghép từng chữ nên **không mô phỏng được động học viết liền một mạch** (nét nối, chữ dính nhau). Dữ liệu NIST cũng chỉ từ nhân viên điều tra dân số và học sinh trung học Mỹ thập niên 1990, có thể không đại diện cho người viết hiện nay.
- **Chưa xác nhận:** chi tiết kích thước, khoảng cách khi ghép (bản đọc được chỉ nói chung chung) và việc họ có so sánh trực tiếp "cùng writer vs ngẫu nhiên" bằng số liệu hay không.

### 1.2. kingyiusuen/handwritten-multi-digit-number-recognition (GitHub)
- Link: https://github.com/kingyiusuen/handwritten-multi-digit-number-recognition
- Sinh dữ liệu tổng hợp bằng cách **ghép chữ số MNIST**, ngẫu nhiên hoá **tỉ lệ chồng lấn (overlap)** và **padding** giữa các chữ số.
- Augmentation lúc train: scale, rotation, shear ngẫu nhiên.
- Kích thước mặc định: 20.000 train / 2.000 val / 2.000 test. Mô hình CRNN (CNN + BiLSTM + CTC) đạt khoảng 98.65% trên tập test tổng hợp.
- Lưu ý: test của họ cũng là dữ liệu tổng hợp cùng phân phối, nên con số 98.65% không nói gì về chữ viết thật.

### 1.3. Performing Arithmetic Using a Neural Network Trained on Digit Permutation Pairs (arXiv 1912.03035)
- Link: https://arxiv.org/pdf/1912.03035
- Ghép **hai ảnh MNIST ngẫu nhiên** (đúng nhãn) cạnh nhau thành một ảnh. Mỗi cặp chữ số (0–9)×(0–9) sinh 1.000 ảnh ngẫu nhiên.
- Họ tách train/test **theo cặp chữ số** (90 cặp train, 10 cặp test) và **theo ảnh MNIST gốc** (ảnh train từ tập MNIST train, ảnh test từ tập MNIST test), để kiểm tra khả năng tổng quát.
- Bài học: có thể sinh hàng chục nghìn mẫu từ vài chục nghìn ảnh gốc, nhưng phải tách nguồn ảnh gốc giữa train và validation để tránh đánh giá cao ảo.

### 1.4. Luận văn METU – Handwritten digit string segmentation and recognition using deep learning (etd.lib.metu.edu.tr, 12619615)
- Link: https://etd.lib.metu.edu.tr/upload/12619615/index.pdf
- Dùng **dữ liệu chuỗi thật (CVL-Strings)** thay vì ghép. Họ phân loại kiểu dính nhau (chạm một điểm, chạm nhiều điểm), nêu các phương pháp tách chữ dính (water reservoir, contour analysis, ligature analysis) trên NIST SD19 (3.500–5.000 cặp dính trong các nghiên cứu họ tổng kết).
- Họ tự dựng bộ chữ số riêng bằng cách trượt cửa sổ 10 px trên chuỗi và gán nhãn tay (36.442 mẫu), rồi tách ra 7.344 chữ số được phân loại đúng.
- Kết quả đáng chú ý: **bộ train riêng từ CVL (ít mẫu) cho kết quả kém hơn khi train bằng MNIST nhiều mẫu**, tức số lượng mẫu vẫn quan trọng.
- Phần lớn nội dung là thuật toán tách chữ, không phải sinh dữ liệu dính nhau; mình chưa tìm thấy đoạn nào mô tả mô phỏng ligature.

### 1.5. Tìm thấy nhưng chưa đọc kỹ
- Kaggle "MNIST 2 Digit Classification Dataset": ghép hai chữ MNIST trên canvas 128×128, vị trí ngẫu nhiên, nhãn 00–99.
- Kaggle "MNIST Multi-Digit Regression Dataset": ghép tới 8 chữ số, ảnh 28×224.
- Một nguồn nhắc đến nét nối: ligature là nét đuôi của chữ trước kéo sang chữ sau; hai chữ có thể chạm tại một điểm hoặc nối bằng ligature.

## 2. Tổng hợp các phương pháp xây dựng dữ liệu của các tác giả

| # | Phương pháp | Nguồn | Ưu | Nhược |
|---|---|---|---|---|
| A | Ghép ngẫu nhiên chữ số MNIST cạnh nhau | 1.3, 1.5 | Rất nhanh, vô hạn mẫu | Hai chữ khác phong cách, không thực tế |
| B | Ghép ngẫu nhiên + ngẫu nhiên hoá overlap/padding | 1.2 | Mô phỏng chữ gần/sát/dính nhau | Vẫn khác phong cách, chưa có nét nối thật |
| C | Ghép **cùng writer** (QMNIST/NIST writer id) | 1.1 | Giống số thật hơn (nghiêng, size, nét đồng nhất) | Cần QMNIST/NIST metadata; vẫn không có nét nối |
| D | Dùng chuỗi thật (CVL-Strings, dữ liệu thu thập) | 1.4 | Gần thực tế nhất | Phải thu thập/gán nhãn, tốn công, ít mẫu |
| E | Augmentation (scale, rotate, shear) | 1.2 | Tăng đa dạng | Không thay thế được dữ liệu thật |
| F | Tách train/val theo writer hoặc theo nguồn ảnh gốc | 1.1, 1.3 | Tránh rò rỉ, đánh giá đúng | Cần có writer id hoặc kiểm soát nguồn |

## 3. Điều tổng hợp được áp dụng cho bài 10–20 (chỉ MLP)
1. **Nên ghép cùng writer (C)** thay vì ngẫu nhiên. Ý bạn đưa ra (đồng nhất độ nghiêng, size) khớp với hướng của bài MDW. Cần dùng **QMNIST** (có `writer id`) hoặc NIST SD19 thay cho MNIST thường. Nếu chỉ có MNIST thường, phải xấp xỉ bằng cách chọn cặp có độ nghiêng/độ dày/chiều cao gần nhau.
2. **Ngẫu nhiên hoá overlap/padding (B):** gồm khoảng cách xa, sát và chạm nhau.
3. **Nét nối và vùng mờ ở giữa:** không nguồn nào mô phỏng nét nối khi tạo dữ liệu. Chính bài MDW nêu đây là hạn chế. Vì vậy phần này phải tự làm: vẽ nét nối từ chân chữ đầu sang chữ sau, và tỉ lệ mẫu có nét nối chọn theo quan sát của bạn khi tự viết (không có số liệu tham khảo).
4. **Dữ liệu thật (D) nhỏ nhưng quan trọng:** bài METU cho thấy ít mẫu thật không thắng được nhiều mẫu tổng hợp, nên cách hợp lý là trộn: nhiều mẫu ghép + một ít mẫu tự viết thật (dùng để train một phần, và giữ một phần riêng để validate).
5. **Chia train/val theo writer hoặc nguồn ảnh gốc (F)** để validation đáng tin.
6. **Augmentation (E)** nhẹ: scale, rotate, shear, dịch. Quan trọng hơn với MLP vì MLP không có tính bất biến dịch chuyển như CNN.
7. **Ghép ở độ phân giải cao rồi resize về 28×28 một lần** (đề xuất của mình, chưa phải của nguồn nào) để vùng nối bị mờ giống ảnh thật bị resize.

## 4. Việc nên làm tiếp
- Tải QMNIST (có writer id) và kiểm tra có dùng được cho cuộc thi không (luật đề chỉ cấm dùng test để nhìn/gán nhãn; không cấm dữ liệu ngoài, nhưng nên xác nhận với ban tổ chức nếu nghi ngờ).
- Thử so sánh hai bộ train (ghép ngẫu nhiên và ghép cùng writer) trên một tập validation chữ viết tay thật do bạn tự viết.
- Đọc kỹ toàn văn bài MDW (đã lưu PDF tạm trong thư mục kết quả của phiên làm việc) nếu cần chi tiết cách resize/ghép.
