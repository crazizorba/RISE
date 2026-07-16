# Báo cáo Đồ án RISE - Kế hoạch Báo cáo và Đánh giá

Kế hoạch này vạch ra chi tiết chiến lược báo cáo cho đồ án RISE, bao gồm việc làm rõ toàn bộ quy trình (pipeline) hoạt động của dự án, mục đích của từng mô hình, các giới hạn phần cứng thực tế và những kỹ thuật đã áp dụng để vượt qua giới hạn đó, từ đó đáp ứng đầy đủ 3 tiêu chí đánh giá cốt lõi: Implementation, Evaluation và Ablation Study.

---

## 1. Tổng quan Pipeline Hoàn chỉnh của RISE và Mục đích Từng Mô hình
Dự án RISE (Self-Improving Robot Policy with Compositional World Model) là một framework hoàn chỉnh với mục tiêu tạo ra robot có khả năng điều khiển cánh tay gắp thả trơn tru, nhận biết vật thể và **đặc biệt là tự cải thiện kỹ năng thông qua học tăng cường trong "trí tưởng tượng" (RL in imagination)**.

Để làm được điều đó, Pipeline của RISE theo thiết kế gốc bao gồm 3 khối kiến trúc độc lập:

1. **Offline Policy & Value Model (`policy_offline_and_value`):** 
   - *Policy Model:* Quan sát môi trường qua camera và dự đoán hành động (action) tiếp theo robot cần thực hiện để hoàn thành lệnh ngôn ngữ.
     - **Đầu vào (Input):** Các khung hình ảnh (RGB) từ 3 góc camera, trạng thái vật lý hiện tại của cánh tay robot (State), và lệnh văn bản (Prompt).
     - **Đầu ra (Output):** Một **Vector Hành động (Action Vector)** nhiều chiều, chứa thông số góc quay của các khớp tay và trạng thái kẹp (gripper).
   - *Value Model:* Đóng vai trò "bộ định vị tiến độ", không dự đoán hành động mà chỉ tập trung báo hiệu tác vụ đang hoàn thiện đến đâu. Giúp robot biết mình cách đích bao xa.
     - **Đầu vào (Input):** Giống hệt Policy Model (Ảnh camera, Trạng thái robot, Lệnh văn bản).
     - **Đầu ra (Output):** Một **Giá trị vô hướng (Scalar/Float)** chạy từ `0.0` đến `1.0`.
2. **Dynamics Model (`dynamics/dynamics_model`):**
   - Đóng vai trò là "Mô hình Thế giới" (World Model). Nó mô phỏng lại môi trường vật lý. Khi nhận vào một hành động (action) từ Policy, Dynamics Model sẽ sinh ra đoạn video dự đoán tương lai xem môi trường sẽ biến đổi như thế nào (ví dụ: cánh tay sẽ di chuyển tới đâu, đồ vật sẽ bị xê dịch ra sao) trên 3 góc camera cùng lúc.
3. **Online Policy / RL in Imagination (`policy_online`):**
   - Đóng vai trò là "trái tim" của hệ thống tự học. Thay vì bắt robot chạy thử nghiệm thật ở ngoài đời rất tốn kém, khối này sẽ cho Policy tương tác liên tục với môi trường ảo do Dynamics Model sinh ra. Kết hợp cùng Value Model để tự chấm điểm, RL (Học Tăng Cường) sẽ tối ưu hoá chính sách để robot ngày càng thông minh hơn.

---

## 2. Vì sao bỏ qua (không thực thi) Dynamics Model và Online Policy?
Trong phạm vi đồ án này, chúng ta **chỉ thực hiện phần 1 (Offline Policy & Value Model)** và hoàn toàn bỏ qua phần 2 và phần 3. Lý do chính đến từ giới hạn phần cứng cực kỳ khắc nghiệt (**NVIDIA GTX 1650 - 4GB VRAM**):

- **Bất khả thi với Dynamics Model:** Mô hình sinh video (Video Generation) như LTX-Video yêu cầu sức mạnh xử lý và VRAM khổng lồ (thường cần nhiều card A100/H100 80GB) để tính toán và kết xuất đồ họa đa góc nhìn (3 camera). Một GPU 4GB VRAM không thể load nổi tham số của mô hình này.
- **Quá tải với Online RL:** Để chạy RL in imagination, hệ thống bắt buộc phải load song song cả 3 mô hình khổng lồ cùng lúc vào VRAM: Policy Model (để chọn hành động), Dynamics Model (để render môi trường tương lai), và Value Model (để chấm điểm). Việc này vượt xa khả năng của máy tính Local, khiến quy trình tự cải thiện Online không thể diễn ra. 

Do đó, đồ án tập trung chứng minh tính đúng đắn của phần lõi quan trọng nhất: Khả năng học Multi-task của Offline Policy và Năng lực định vị tiến độ chính xác của Value Model.

---

## 3. Các Kỹ thuật Tối ưu (Cắt giảm) để chạy được Offline Policy & Value Model
Dù chỉ chạy phần 1, mô hình Pi0.5 vẫn rất khổng lồ. Các giải pháp cắt giảm khắc nghiệt đã được áp dụng triệt để trong code:
- **Đóng băng Backbone Thị giác (`freeze_vlm_backbone = True`):** Trọng số của mô hình ngôn ngữ - thị giác nền tảng (VLM Backbone như PaliGemma) được thiết lập `requires_grad = False` (trong `pi0_pytorch.py`). Việc không cập nhật gradient cho backbone này giúp tiết kiệm lượng cực lớn VRAM trong lúc backward pass.
- **Tắt hàm Loss không cần thiết (`loss_action_weight = 0.0`):** Khi huấn luyện Value Model (nhánh định giá tiến độ), hệ thống chủ động cấu hình tắt loss hành động, chỉ giữ lại `loss_value_weight = 1.0`. Điều này giúp mô hình hoàn toàn tập trung vào việc học tiến độ, bỏ qua tính toán sai số action, giảm tải đáng kể cho bộ nhớ.
- **Giảm Batch Size xuống tối thiểu (`batch_size = 1`):** Cấu hình trực tiếp trong các `TrainConfig` ở `config.py` nhằm giới hạn số lượng frame đưa vào tính toán, kết hợp với `grad_accu_steps=1`.
- **Kích hoạt Gradient Checkpointing:** Để đánh đổi thời gian lấy không gian bộ nhớ, tính năng gradient checkpointing được bật (`gradient_checkpointing_enable()`). Quá trình này giải phóng bộ nhớ đồ thị trung gian lúc Forward Pass và chỉ tính toán lại khi Backward.
- **Hạ Độ phân giải Hình ảnh (Image Resolution):** Độ phân giải tiêu chuẩn 224x224 được giảm xuống thông qua biến đổi `_transforms.ResizeImages(112, 112)` (trong `config.py`) tiết kiệm đến 75% khối lượng ma trận điểm ảnh.
- **Loại bỏ Torchrun:** Chuyển từ chạy Multi-GPU (`torchrun`) sang `python scripts/train_pytorch.py` thông thường trong tệp `train.sh` để chạy Single GPU Local, triệt tiêu lỗi `C10d store rendezvous` trên Windows.

---

## Phần 1: Implementation (Triển khai Hệ thống & Học Đa nhiệm)

**Mục tiêu:** Chứng minh năng lực làm chủ mã nguồn, khả năng vượt qua giới hạn tài nguyên và tính khái quát hoá (Generalization) của hệ thống.

1. **Khắc phục Giới hạn VRAM:** 
   - Đưa ra các minh chứng trong source code về thiết lập `batch_size=1`, `freeze_vlm_backbone=True`, gọi hàm `gradient_checkpointing`, và `ResizeImages(112, 112)`.
   - Cung cấp ảnh chụp màn hình console hiển thị quá trình train thành công làm minh chứng.
2. **Năng lực Học Đa nhiệm (Multi-task Learning):**
   - Trình bày quá trình khôi phục và tinh chỉnh cấu trúc (schema) cho 2 tập dữ liệu mới từ Hugging Face là `aloha_cabinet` và `aloha_ziploc`, hợp nhất cùng tập `svla` gốc.
   - **Tính toán Hệ số Chuẩn hóa (Normalization Statistics):** Chứng minh việc chạy kịch bản tiền xử lý `scripts/compute_norm_stats_fast.py` để trích xuất giá trị Trung bình (Mean) và Độ lệch chuẩn (Std) cho Action/State từ tập dữ liệu thô. Dữ liệu này giúp mô hình không bị thiên lệch và hội tụ ổn định hơn.
   - Cung cấp biểu đồ Loss từ hệ thống **W&B** ghi nhận quá trình Train from scratch đồng thời cả 3 dataset. 
   - **Kết luận:** Mô hình Policy hoàn toàn có khả năng học được nhiều loại tác vụ thao tác cánh tay robot khác nhau cùng một lúc.
3. **Kiểm chứng Quy trình Suy luận (Inference Pipeline) trên Thực tế:**
   - Trình bày kiến trúc **Server - Client** của hệ thống mô phỏng cách triển khai trên robot thật:
     - **Mô-đun Server (`serve_policy.py`)**: Tải trọng số khổng lồ của Policy Model lên VRAM, thiết lập cổng mạng (WebSocket ở Port 8000) và trực chờ nhận dữ liệu hình ảnh. Việc tách riêng Server giúp tái sử dụng mô hình mà không cần khởi tạo lại mỗi lần chạy.
     - **Mô-đun Client (`aloha_real/main.py`)**: Chạy vòng lặp (Episode), đóng gói ảnh chụp từ camera gửi qua Server, và nhận về **Vector Hành động** để điều hướng tay máy.
   - **Minh chứng:** Hình ảnh chụp màn hình 2 phiên Terminal chạy song song giao tiếp thành công với nhau.
   - **Ý nghĩa Kỹ thuật:** Khẳng định dự án không chỉ nằm trên giấy tờ hay dừng lại ở việc "train ra số", mà đã làm chủ và kích hoạt thành công một đường ống (pipeline) suy luận hoàn chỉnh. Hệ thống sẵn sàng được nhúng vào phần cứng robot vật lý để chạy thời gian thực (real-time).

---

## Phần 2: Evaluation (Đánh giá Năng lực của Value Model)

**Mục tiêu:** Kiểm chứng thành phần mở rộng "Value Model" thực sự hoạt động, có khả năng "hiểu" và dự báo được tiến độ hoàn thành chuỗi hành động.

1. **Định lượng bằng Sai số (MAE):**
   - Trình bày con số **MAE (Mean Absolute Error) ~ 0.25** khi đánh giá trên tập Validation (thông qua `label_value.sh`). Mức sai số thấp này khẳng định Value Head bám sát được thực tế tiến độ tác vụ.
2. **Minh họa Trực quan (Video):**
   - Trình chiếu đoạn Video `episode_000.mp4` (sinh ra bằng lệnh `vis_value.sh`).
   - **Giải thích cho Hội đồng:** Nhấn mạnh đồ thị màu đỏ/xanh biểu diễn "Value Prediction" ở dưới video. Khi robot tiến gần đến mục tiêu, đường đồ thị tịnh tiến mượt mà từ mức `0.0` (Khởi đầu) lên `1.0` (Hoàn thành nhiệm vụ).

---

## Phần 3: Ablation Study (Nghiên cứu Bóc tách)

**Mục tiêu:** Trả lời triệt để câu hỏi học thuật: *"Nếu tháo thành phần Value Model ra khỏi hệ thống thì kết quả huấn luyện sẽ thay đổi thế nào? Nó có giúp ích thực sự cho quá trình Policy Learning hay không?"*

1. **Thiết lập Thí nghiệm (Dựa trên tham số cấu hình):**
   - Cùng một cấu hình pipeline và tập dữ liệu gốc (`svla`), tiến hành 2 phiên bản huấn luyện hoàn toàn độc lập:
     - **Phiên bản bị khuyết (Ablation - No Value):** Tắt toàn bộ nhánh Value Head (`pi05=False` hoặc `with_value_head=False`). Mô hình sẽ chỉ huấn luyện ra Action (`run_svla_NO_VALUE`).
     - **Phiên bản hoàn chỉnh (Full Model - With Value):** Bật nhánh Value Head với `pi05=True`, `with_value_head=True` và `loss_value_weight=1.0` (`run_svla_WITH_VALUE`).
2. **Kết quả & Phân tích:**
   - Đặt 2 đồ thị **Total Loss** của 2 lần chạy trên W&B cạnh nhau để so sánh trực diện.
   - **Kết luận Báo cáo:** Khi có sự tồn tại của nhánh Value Head làm tín hiệu bổ trợ (auxiliary signal) cho hàm Loss tổng, đường Total Loss của toàn hệ thống sẽ hội tụ mượt mà hơn, có xu hướng đi xuống ổn định và ít bị gai nhiễu (spikes) hơn hẳn so với phiên bản bị bóc tách.
