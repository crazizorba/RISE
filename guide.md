# Kịch bản Trình bày Dự án RISE (Self-Improving Robot Policy with Compositional World Model)

**Mục tiêu**: Kịch bản thuyết trình dành cho dự án RISE, tập trung phân tích và trình bày qua 3 khía cạnh cốt lõi: Implementation (Triển khai), Evaluation (Đánh giá), và Ablation (Nghiên cứu cắt bỏ).

---

## Phần 1: Implementation (Triển khai Hệ thống)

**Slide/Nội dung chính:** Giới thiệu tổng quan kiến trúc 3 phần của RISE, quy trình chuẩn bị dữ liệu và cách hệ thống học tự động trong "trí tưởng tượng".

**Kịch bản thuyết trình:**
> "Chào mọi người. Hôm nay chúng ta sẽ đi sâu vào cấu trúc và phương pháp hoạt động của RISE - một framework học chính sách tự cải thiện (self-improving robot policy) dành cho robot thông qua một Mô hình Thế giới Kết hợp (Compositional World Model).
> 
> Về mặt triển khai (**Implementation**), kiến trúc mã nguồn của RISE được chia thành 3 phần rõ rệt:
> 
> 1. **Dynamics Model (Mô hình mô phỏng Động lực học):** Xây dựng dựa trên nền tảng của mô hình sinh video LTX-Video. Mô hình này có nhiệm vụ dự đoán các khung hình tương lai dựa trên hành động điều khiển của robot. Quá trình huấn luyện bao gồm pre-training (huấn luyện trước) trên các bộ dữ liệu robot quy mô lớn (như Galaxea Open World, AgiBot World Alpha), sau đó được fine-tune lại cho các tác vụ cụ thể. Điểm đặc biệt là hệ thống sử dụng cùng lúc 3 góc nhìn camera (1 camera trên đầu, 2 camera ở cổ tay) ở độ phân giải $256 \times 192$.
> 
> 2. **Offline Policy & Value Model:** Dựa trên OpenPI, mô hình học chính sách ban đầu (Offline Policy) từ dữ liệu biểu diễn (demonstrations). Đi kèm với nó là một Mô hình Giá trị (Progress Value Model) giúp đánh giá xem một trạng thái có đang dẫn đến thành công hay không.
> 
> 3. **Online RL (Học tăng cường trực tuyến):** Đây là trái tim của hệ thống. Thay vì phải tương tác thật trên phần cứng tốn kém và mất thời gian, RISE thực hiện **RL in imagination** - dùng thuật toán học tăng cường để lấy mẫu và thử nghiệm các hành động hoàn toàn trong thế giới ảo do Dynamics Model sinh ra, từ đó cập nhật và tối ưu chính sách tự động."

---

## Phần 2: Evaluation (Đánh giá Hiệu năng)

**Slide/Nội dung chính:** Kết quả đạt được khi áp dụng mô hình chính sách vào điều khiển robot vật lý trong môi trường thực.

**Kịch bản thuyết trình:**
> "Chuyển sang phần **Evaluation** (Đánh giá), câu hỏi quan trọng nhất là: Mô hình học trong môi trường ảo thì có hoạt động tốt trên thực tế hay không? Nhóm tác giả đã tiến hành chạy thử nghiệm chính sách học được thẳng lên các tác vụ thao tác vật lý (dexterous manipulation) có độ khó và tính động cao.
> 
> Những gì RISE thể hiện so với các phương pháp tiếp cận truyền thống (baselines) là cực kỳ ấn tượng:
> - **Tác vụ Dynamic Brick Sorting (Phân loại gạch động):** Hiệu suất thành công tăng **+35%**.
> - **Tác vụ Backpack Packing (Đóng gói balo):** Cải thiện vượt bậc lên đến **+45%**.
> - **Tác vụ Box Closing (Đóng nắp hộp):** Cải thiện **+35%**.
> 
> Những con số này minh chứng rõ ràng rằng: Mô hình thế giới (World model) của RISE không chỉ tạo ra các đoạn video có tính thực tế cao về mặt thị giác, mà động lực học đằng sau nó đủ chính xác để cung cấp tín hiệu đúng đắn cho sự tiến hóa của robot trong đời thực."

---

## Phần 3: Ablation Study (Nghiên cứu Cắt bỏ)

**Slide/Nội dung chính:** Phân tích những yếu tố then chốt tạo nên sức mạnh của mô hình bằng cách lần lượt loại bỏ/thay đổi từng thành phần (Dựa theo luận điểm thiết kế của RISE).

**Kịch bản thuyết trình:**
> "Cuối cùng, chúng ta hãy xem xét phần **Ablation Study** (Nghiên cứu cắt bỏ) để bóc tách xem điều gì thực sự làm nên thành công của framework này. Thiết kế 'Compositional World Model' của RISE hoạt động hiệu quả nhờ vào các thành phần nào?
> 
> 1. **Vai trò của Value Model (Mô hình Giá trị):** Nếu ta loại bỏ mô hình giá trị và chỉ dùng tín hiệu phần thưởng thô ráp (sparse reward) từ kết quả cuối, quá trình học RL sẽ rất khó hội tụ hoặc rơi vào trạng thái ảo tưởng. Value Model của RISE cung cấp một lợi thế thông tin (informative advantages) mạnh mẽ giúp hướng dẫn RL cải thiện policy một cách đáng tin cậy.
> 
> 2. **Tầm quan trọng của Multi-view Dynamics:** Việc mô hình sinh ra cảnh quan tương lai dựa trên hành động được hỗ trợ bởi 3 góc camera độc lập. Nếu giảm bớt góc nhìn, mô hình thế giới sẽ mất đi nhận thức không gian 3D, từ đó dự đoán sai lệch quỹ đạo vật thể, làm hỏng quá trình học 'trong trí tưởng tượng'.
> 
> 3. **Giá trị của RL in Imagination:** Nếu chúng ta chỉ dừng lại ở bước học Offline (bắt chước theo dữ liệu có sẵn) mà bỏ qua bước tự cải thiện trực tuyến (Online RL) qua mô phỏng, robot sẽ không thể vượt qua mức trần (bottleneck) của dữ liệu con người tạo ra. Bước RL in Imagination chính là chìa khóa tạo ra khoảng cách hiệu năng +35% đến +45% mà chúng ta vừa thấy trong phần đánh giá.
> 
> Tóm lại, sự thành công của RISE đến từ thiết kế kết hợp chặt chẽ giữa một bộ mô phỏng ảo đa góc nhìn, một cơ chế đánh giá giá trị sắc bén, và một quy trình tự tối ưu học tăng cường không giới hạn."

---

## Các tài nguyên bổ sung để tham khảo trong quá trình chuẩn bị:
- **Paper:** [arXiv:2602.11075](https://arxiv.org/abs/2602.11075)
- **Tài liệu Code:** Xem thêm trong `docs/offline_learning.md`, `docs/dynamics_model.md` và `docs/online_training.md`.
- **Cấu trúc Dữ liệu:** Sử dụng chuẩn dữ liệu LeRobot với độ phân giải khuyến nghị `[256, 192]`.
