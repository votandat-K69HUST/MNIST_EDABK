# Thể lệ cuộc thi: EDABK LAB – MNIST 10–20 Challenge

> Nguồn: `EDABK_AI.pdf` (EDABK LAB, Ngo Pham Minh Duc). File này tóm tắt đầy đủ yêu cầu để các phiên làm việc sau có thể tiếp tục mà không cần đọc lại PDF.

## 1. Bài toán
- Phân loại ảnh chữ số viết tay theo phong cách MNIST, **11 class: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20**.
- Mỗi mẫu là **một ảnh duy nhất chứa đầy đủ một số** (số 2 chữ số, ví dụ "17", "20"), không phải một chữ số đơn.
- Ảnh thực tế là chữ viết tay: hình dạng, khoảng cách giữa hai chữ số, độ dày nét có thể khác nhau nhiều.
- Về bản chất: bài toán phân loại 11 lớp (hoặc có thể thiết kế thành 2 chữ số riêng: chục + đơn vị, miễn dùng NN và ra được nhãn 10–20).

## 2. Dữ liệu huấn luyện
- **BTC KHÔNG cung cấp tập train.** Mỗi đội tự xây dựng dữ liệu cho class 10–20.
- Đội tự chọn:
  - cách tạo/thu thập dữ liệu (vd: tự viết, ghép 2 chữ số từ MNIST gốc, sinh tổng hợp, v.v.);
  - preprocessing và augmentation;
  - kiến trúc mạng nơ-ron và phương pháp huấn luyện.
- Mục tiêu: mô hình **tổng quát hoá tốt** trên tập test độc lập (phân phối là chữ viết tay thật → cần domain gap nhỏ).

## 3. Tập test (BTC giữ, không công bố nhãn)
| | Ảnh/class | Tổng |
|---|---|---|
| Public Leaderboard (40%) | 20 | 220 |
| Private Leaderboard (60%) | 30 | 330 |
| **Tổng** | 50 | **550** |

- Public LB cập nhật trong thời gian thi; **Private LB quyết định xếp hạng cuối**.
- Hệ quả: tránh overfit vào public LB; chọn model dựa trên validation tự dựng.
- Test cân bằng: đúng 50 ảnh/class (prior đều).

## 4. Định dạng dữ liệu
- Ảnh chuẩn hoá kiểu MNIST: **28×28, grayscale, pixel 0–255**, mỗi ảnh chứa một số hoàn chỉnh 10–20.
- Ảnh được flatten thành 784 giá trị.
- `test.csv`: header `id,pixel0,pixel1,...,pixel783`; mỗi dòng là một mẫu.
- Code đọc mẫu của BTC:
```python
import pandas as pd
import numpy as np

df = pd.read_csv("test.csv")
ids = df["id"].values
X = df.drop(columns=["id"]).values
X = X.reshape(-1, 28, 28)
X = X.astype(np.float32) / 255.0
```
- Lưu ý: PDF không nói rõ nền/nét (MNIST chuẩn: nền đen 0, nét trắng ~255). Khi dựng dữ liệu train nên bám đúng kiểu này, và so sánh thống kê pixel (mean, tỉ lệ pixel khác 0) của test với train **bằng số liệu, không nhìn ảnh** (xem luật 3).

## 5. Bài nộp (`submission.csv`)
```
id,label
0,17
1,10
2,20
3,14
```
- Hai cột `id,label`.
- `label` ∈ {10,...,20} (số nguyên).
- **Mỗi `id` trong `test.csv` xuất hiện đúng một lần** (đủ 550 dòng, không thiếu/trùng).

## 6. Đánh giá
- Metric: **Accuracy = số dự đoán đúng / tổng số mẫu**. Ví dụ 500/550 đúng → 90.91%.

## 7. Luật chính (bắt buộc tuân thủ)
1. Đội tự xây dựng training data và mô hình; **mô hình phải là mạng nơ-ron** (không dùng SVM/RF/KNN... làm model chính).
2. Tập test **chỉ dùng để chạy inference** bằng pipeline/mô hình tự động.
3. **Cấm visualize hoặc phục hồi ảnh từ `test.csv` để nhìn nhãn bằng mắt.** (Không `imshow`/lưu PNG ảnh test, không in ảnh test ra xem.)
4. **Cấm manual-label test set** hoặc dùng người để xác định trực tiếp nhãn test.
5. BTC có quyền yêu cầu đội top cung cấp **source code, training code, inference code và model weights** để kiểm tra → cần lưu code tái lập được, có seed, có weights, có script inference chạy lại ra đúng `submission.csv`.

> Gợi ý an toàn: cũng nên tránh pseudo-label/tự huấn luyện trên test theo cách mơ hồ; nếu muốn dùng, cần cân nhắc vì luật nói test "chỉ dùng để chạy inference". Mặc định: **không dùng test để train**.

## 7b. Ràng buộc bổ sung của người làm (KHÔNG có trong đề)
- **Chỉ được dùng NN cơ bản: MLP (fully-connected, Dense) nhiều lớp.** Mới học NN cơ bản.
- **CẤM dùng CNN (Conv2D, pooling kiểu conv...) và mọi mạng phức tạp hơn** (ResNet, RNN/LSTM, Transformer, autoencoder phức tạp, pretrained model...).
- Cho phép các thành phần cơ bản của MLP: Linear/Dense, ReLU/tanh/sigmoid, softmax, dropout, (batch norm nếu cần, cân nhắc), weight decay, optimizer SGD/Adam, early stopping.
- Input: flatten 784 → MLP → 11 output (softmax). Nên chuẩn hoá /255.
- Vì MLP không bất biến dịch chuyển như CNN, **augmentation (dịch, xoay, co giãn, nghiêng, nhiễu) và dữ liệu đa dạng càng quan trọng** để bù.

## 8. Tóm tắt
| Hạng mục | Thông tin |
|---|---|
| Classes | 10 đến 20 (11 classes) |
| Input | 28×28 grayscale |
| Training data | Đội tự tạo |
| Mô hình | Mạng nơ-ron |
| Test set | 550 ảnh, 50 ảnh/class |
| Metric | Accuracy |
| Public LB | 40% – 220 ảnh |
| Private LB | 60% – 330 ảnh |
| Xếp hạng cuối | Private Leaderboard |
| Thời gian thi | 2 tuần |

## 9. Hiện trạng thư mục dự án
- Thư mục: `D:\2025\2025.2\LAB\AI_ML\Week_11\AI_HACKATHON`
- Hiện chỉ có `EDABK_AI.pdf` (đề bài) và file này. **Chưa có `test.csv`, chưa có dữ liệu train, chưa có code.**
- Lưu ý: PDF xuất hiện lỗi font khi trích text, nên các ví dụ ảnh (10, 17, 20) chỉ là minh hoạ class, không có thông tin dữ liệu thêm.

## 10. Checklist việc cần làm / hướng tiếp cận gợi ý
1. **Dựng dữ liệu train** (phần khó nhất vì không có sẵn):
   - Cách phổ biến: ghép 2 ảnh MNIST (chữ số chục = "1" hoặc "2", đơn vị = 0–9) thành ảnh 28×28 (mỗi chữ số thu về ~14 px rộng) → sinh hàng chục nghìn mẫu cho 11 class (10..19 = "1"+d, 20 = "2"+"0").
   - Bổ sung: tự viết tay/scan, dataset khác (EMNIST, v.v.) nếu được phép, font viết tay.
   - Augmentation: dịch, xoay nhẹ, co giãn, nghiêng (shear), thay đổi độ dày nét (dilate/erode), nhiễu, biến dạng đàn hồi, thay đổi khoảng cách 2 chữ số, chữ số dính nhau/chồng lấn.
2. **Mô hình**: chỉ **MLP** (vd 784 → 512 → 256 → 11, ReLU + dropout). **Không dùng CNN hay mạng phức tạp hơn.** Có thể ensemble nhiều MLP (trung bình xác suất) vì vẫn là NN cơ bản.
3. **Validation**: tách một tập val tự dựng càng giống test (chữ viết tay thật, nếu tự viết được) để chọn model; không tune theo test.
4. **Inference script**: đọc `test.csv` → chuẩn hoá /255 → reshape (N,28,28) → predict → ghi `submission.csv` (`id,label`, label 10–20, đủ 550 dòng).
5. **Kiểm tra bài nộp**: số dòng = số id, id duy nhất, label ∈ [10,20], kiểu int.
6. **Lưu lại để tái lập**: seed, training code, weights, requirements, hướng dẫn chạy (vì BTC có thể kiểm tra).
7. Chọn model cuối dựa trên validation, không chạy theo public LB (chỉ 220 ảnh, dễ nhiễu; xếp hạng cuối theo private).

## 11. Điều KHÔNG được làm (nhắc lại ngắn gọn)
- Không xem ảnh test bằng mắt dưới bất kỳ hình thức nào / không gán nhãn tay cho test.
- Không dùng model không phải NN làm mô hình chính.
- Không dùng CNN hoặc bất kỳ mạng phức tạp hơn MLP (ràng buộc của người làm).
- Không nộp submission thiếu/trùng id hoặc label ngoài 10–20.

## 12. Thư mục dữ liệu
- `data/` chứa code + config để **tái tạo** các bộ dữ liệu train (ghép MNIST ngẫu nhiên, ghép theo phong cách, ghép cùng writer QMNIST, thêm nét nối, tự viết tay, augmentation). Xem `data/README.md` và `tong_hop_phuong_phap_data.md`.
- Việc còn mở: bố cục 2 chữ số trong ảnh test chưa rõ (xem mục "Điểm chưa chắc chắn" trong `data/README.md`).

## 13. Phát hiện về định dạng ảnh test (từ thống kê tổng hợp, KHÔNG xem ảnh)
- Vùng chữ ≤ 20×20 px, giữ tỉ lệ; trọng tâm ở (13.5, 13.5) chính xác dưới pixel → test đã qua chuẩn hoá kiểu MNIST (fit 20×20 + căn trọng tâm).
- Ảnh sắc nét (ít pixel xám trung gian), tỉ lệ mực ≈ 15%, mean pixel ≈ 38.7.
- Dữ liệu tự sinh đã chỉnh khớp các chỉ số này (xem `data/README.md`). Chạy lại: `python data/tools/stats.py test.csv --compare data/output/<bộ>/train.npz`.
