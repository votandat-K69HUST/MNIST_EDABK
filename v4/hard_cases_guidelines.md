# Báo Cáo Phân Tích Lỗi & Hướng Dẫn Thu Thập Dữ Liệu Bổ Sung V5
*(Dựa trên phân tích 33 ca có độ tự tin thấp nhất của V4 trên tập Test)*

Để vượt qua mốc 96.36% (chinh phục nốt ~20 mẫu khó nhất), chúng ta **không cần** thêm các số viết nắn nót. Chúng ta cần thu thập một mẻ data (khoảng 300 - 500 ảnh) hoàn toàn là **chữ viết ẩu, viết dị dạng có chủ đích**.

Dưới đây là danh sách các cặp số gây "lú lẫn" nhiều nhất cho AI và **cách cố tình viết sai** để đánh lừa/huấn luyện nó:

---

## 1. Mục Tiêu Tối Thượng: Cặp `13` vs `19` (Kẻ thù số 1 - 7 lần sai)
Đây là điểm yếu chí mạng nhất của mô hình hiện tại.
*   **Cách viết để tạo data:** 
    *   **Viết số 13:** Hãy cố tình viết số `3` mượt đến mức vòng cung phía trên gập sát lại, gần như dính vào nhau (nhìn lướt qua tưởng là số 9).
    *   **Viết số 19:** Cố tình viết số `9` nhưng bút bị trượt, hở toang phần đầu của vòng tròn, hoặc hất đuôi lên khiến nó giống y hệt chữ `3`.

## 2. Các Cặp Nhầm Lẫn Khác Cần Thu Thập Thêm

### Cặp `17` vs `19` (3 lần sai)
*   **Cách viết:** Viết số `7` cong đầu, kéo nét xéo mượt và *đặc biệt có thêm nét gạch ngang ở giữa*. Sự kết hợp của nét cong đầu và nét gạch giữa rất dễ bị AI nhìn nhầm thành cái bụng của số `9`.

### Cặp `15` vs `16` (3 lần sai)
*   **Cách viết:** Chìa khóa nằm ở vòng bụng dưới của số `5`. Hãy cố tình khoanh tròn cái bụng đó lại sao cho nét bút chạm ngược lên thân chữ, tạo thành một vòng khép kín y chang số `6`.

### Cặp `16` vs `18` (2 lần sai)
*   **Cách viết:** Viết số `6` nhưng nét móc trên cùng cong gập xuống sát bụng. Hoặc viết số `8` nhưng hở toang vòng tròn phía trên.

### Cặp `10` vs `18` (2 lần sai)
*   **Cách viết:** Viết số `0` méo mó, ép sát vào số `1`. Lực tay không đều khiến mực chập vào nhau tạo thành hình thắt nơ (vòng số 0 dính vào cạnh của số 1, thoạt nhìn như hai bụng của số 8).

### Cặp `12` vs `15` (1 ca cực khó)
*   **Cách viết:** Viết số `2` nhưng nét vòng cung phía trên quặp gắt xuống dưới, và nét móc ngang ở đáy bị uốn lượn cong cong lên (trông như một con giun). Nó sẽ tạo ra form ảo giác của số `5`.

---

## 3. Khuyến Nghị Khi Lấy Mẫu
*   **Sử dụng bút bi lỗi:** Tìm những cây bút bi bị tắc mực, viết lúc đậm lúc nhạt, có khi bị đứt nét hoàn toàn ở giữa chữ.
*   **Dính nét (Ligature):** Chủ động viết số hàng chục và hàng đơn vị đè lên nhau (ví dụ móc của số 1 chọc xuyên vào số 9, hoặc nét ngang của 7 gạch đè qua số 1).

Hãy giao tài liệu này cho đội viết data, dặn họ cố tình "múa bút" theo đúng các miêu tả trên. Khi có được mẻ data này, V5 chắc chắn sẽ ủi phẳng mọi chướng ngại vật!
