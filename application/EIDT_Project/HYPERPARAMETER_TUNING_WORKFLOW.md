# Luồng tìm kiếm siêu tham số UCB–VAE cho Deep SARSA

## 1. Mục tiêu

Mục tiêu của quy trình là tìm một cấu hình UCB–VAE có hiệu quả tốt và ổn định cho bài toán giao dịch trong notebook `test12345.ipynb`.

Quy trình không thay đổi bài toán nghiên cứu cốt lõi:

- Tác nhân vẫn là Deep SARSA.
- Không gian trạng thái vẫn có 7 chiều.
- Không gian hành động vẫn gồm 11 hành động (`7 → 32 → 11`).
- VAE chỉ được xem là mô hình tạo **novelty score tương đối** cho cặp trạng thái–hành động, không được diễn giải là epistemic uncertainty đã hiệu chuẩn đầy đủ.
- Hiệu quả được đánh giá trên dữ liệu held-out thông qua Sharpe Ratio, ARR và Max Drawdown.

## 2. Trạng thái triển khai hiện tại

Phần điều phối thí nghiệm hiện đang được chia thành hai nơi:

- `test12345.ipynb`: chứa dữ liệu, môi trường MDP, mạng Q, VAE, chính sách UCB, hàm huấn luyện và hàm chẩn đoán VAE.
- `ucb_vae_tuning.py`: chứa lưới siêu tham số, vòng lặp screening/validation, cách xếp hạng, checkpoint và trực quan hóa.

Thiết kế này giúp kiểm tra logic dễ hơn, nhưng chưa đáp ứng yêu cầu cuối cùng là một notebook độc lập trên Kaggle. Khi chuyển sang bản Kaggle, toàn bộ nội dung cần thiết trong `ucb_vae_tuning.py` sẽ được đưa vào các cell của notebook, loại bỏ import:

```python
from application.EIDT_Project.ucb_vae_tuning import run_two_stage_ucb_vae_tuning
```

Tài liệu này mô tả đúng luồng đang được triển khai để làm cơ sở cho bước chuyển đổi đó.

## 3. Luồng tổng thể

```text
Nạp và chia dữ liệu GOOD/BAD
            │
            ▼
Khóa backbone Deep SARSA dùng chung
            │
            ▼
VAE screening: 9 cấu hình × 3 seed
            │
            ▼
Chọn và khóa cấu hình VAE tốt nhất
            │
            ▼
UCB screening: 48 cấu hình × 3 seed
            │
            ▼
Xếp hạng và lấy Top 5
            │
            ▼
Huấn luyện lại Top 5 × 20 seed mới
            │
            ▼
Chọn cấu hình vững chắc cuối cùng
            │
            ▼
Xuất bảng, biểu đồ và best_config.json
```

Tổng số lượt huấn luyện đầy đủ:

```text
VAE screening       =  9 × 3  =  27
UCB screening       = 48 × 3  = 144
Top-5 validation    =  5 × 20 = 100
Tổng cộng                      = 271 lượt train
```

## 4. Bước 1 — Khóa backbone Deep SARSA

Backbone được giữ cố định trong mọi cấu hình để phép so sánh giữa các chính sách là công bằng:

| Thành phần | Giá trị cố định |
|---|---:|
| Episodes | 45 |
| Gamma | 0.95 |
| Alpha | 0.60 |
| Kiến trúc Q-network | 7 → 32 → 11 |
| Q-network learning rate | 5e-5 |
| Optimizer | Adam |
| Loss | Huber Loss |

Trước khi chạy, hàm điều phối kiểm tra các giá trị này. Nếu cấu hình nền bị thay đổi, chương trình dừng thay vì vô tình so sánh các tác nhân có backbone khác nhau.

`DEFAULT_CONFIG_UCB` luôn được sao chép trước khi cập nhật một cấu hình thử nghiệm. DataFrame train/test cũng được fingerprint trước và sau quá trình tuning để phát hiện thay đổi ngoài ý muốn.

## 5. Bước 2 — Tối ưu VAE trước

UCB sử dụng novelty score của VAE, vì vậy VAE được tối ưu trước khi quét tham số UCB.

Trong giai đoạn này, UCB được khóa ở mức mặc định an toàn:

```python
beta = 0.15
beta_decay = 0.91
```

Lưới VAE:

| Tham số | Giá trị |
|---|---|
| `vae_beta_kl` | 0.001, 0.005, 0.01 |
| `bootstrap_trajectories` | 4, 5, 6 |
| `vae_latent_dim` | 16 |

Số cấu hình VAE là `3 × 3 × 1 = 9`. Mỗi cấu hình được huấn luyện lại từ đầu với ba seed screening độc lập, mặc định là `42, 43, 44`.

### Chẩn đoán VAE

Mỗi lượt chạy thu thập các tín hiệu sau:

- VAE loss ở phần cuối quá trình huấn luyện.
- Mức cải thiện loss giữa đầu và cuối quá trình.
- Hệ số biến thiên của loss ở đoạn cuối để kiểm tra độ ổn định.
- Reconstruction MSE trên dữ liệu quen thuộc.
- KL divergence trên dữ liệu quen thuộc.
- Cảnh báo KL collapse khi KLD tiến quá gần 0.
- Tỷ lệ novelty giữa held-out data và familiar data.
- Tương quan giữa novelty score và trị tuyệt đối TD error của Q-network.
- Bình phương phần tương quan dương, dùng như một chỉ báo hỗ trợ thay vì tuyên bố VAE đã đo uncertainty chính xác.

Tập test trong mỗi regime chỉ được xem là held-out/OOD proxy. Nó không nhất thiết là OOD thực sự theo nghĩa phân phối thị trường hoàn toàn mới.

### Cách chọn VAE

VAE score ưu tiên:

- Loss giảm và ổn định ở cuối quá trình.
- Reconstruction error trên dữ liệu quen thuộc thấp.
- Không xảy ra KL collapse.
- Novelty trên held-out data cao hơn familiar data một cách hợp lý.
- Novelty có tương quan dương với TD error.

Cấu hình có VAE score cao nhất được khóa để chuyển sang giai đoạn UCB.

## 6. Bước 3 — Tối ưu chính sách UCB

Sau khi chọn VAE tốt nhất, toàn bộ tham số VAE được giữ cố định. Chỉ hai tham số điều khiển mức khám phá của UCB được quét:

| Tham số | Giá trị |
|---|---|
| `beta` / `beta0` | 0.03, 0.05, 0.15, 0.20, 0.30, 0.80, 0.85, 0.90 |
| `beta_decay` | 0.85, 0.88, 0.91, 0.92, 0.94, 0.95 |

Số cấu hình UCB là `8 × 6 = 48`. Mỗi cấu hình tiếp tục được chạy với ba seed screening.

Mỗi seed tạo ra các chỉ số:

- Final Profit.
- Sharpe Ratio.
- ARR (%).
- ROI (%).
- Max Drawdown (%).

Screening score được xây dựng theo hướng:

```text
Sharpe trung bình
− hệ số phạt × độ lệch chuẩn Sharpe
+ phần thưởng nhỏ cho ARR
− phần phạt Max Drawdown
− phần phạt bổ sung nếu vượt ngưỡng drawdown an toàn
```

Final Profit vẫn được lưu để phân tích nhưng không phải tiêu chí chính, vì mục tiêu là hiệu quả điều chỉnh theo rủi ro và độ ổn định qua nhiều seed.

## 7. Bước 4 — Chọn Top 5 và kiểm chứng 20 seed

Sau screening, 48 cấu hình UCB được sắp xếp theo screening score. Năm cấu hình đứng đầu được huấn luyện lại hoàn toàn từ đầu với 20 seed mới, mặc định từ `100` đến `119`.

Các seed validation không trùng với seed screening. Điều này giúp giảm khả năng chọn phải một cấu hình chỉ tốt do seed may mắn.

Cấu hình cuối cùng được chọn dựa trên:

- Sharpe trung bình cao.
- Độ lệch chuẩn Sharpe thấp.
- ARR tốt.
- Max Drawdown được kiểm soát.
- Không vượt quá ngưỡng drawdown mà người chạy đã cấu hình.

Kết quả cuối cùng được báo cáo dưới dạng `mean ± std`, đặc biệt là Sharpe Ratio.

## 8. Checkpoint và khả năng tiếp tục phân tích

Sau mỗi cấu hình, kết quả hiện có được ghi ngay xuống CSV. Nếu Kaggle session dừng, các cấu hình đã hoàn tất vẫn còn trong output của session và có thể tải về.

Các file chính:

| File | Nội dung |
|---|---|
| `vae_screening.csv` | Tổng hợp 9 cấu hình VAE |
| `VAE-xx_seeds.csv` | Kết quả từng seed của một cấu hình VAE |
| `ucb_screening.csv` | Tổng hợp 48 cấu hình UCB |
| `UCB-xx_screening_seeds.csv` | Kết quả từng seed ở screening |
| `top5_validation.csv` | Tổng hợp Top 5 sau 20 seed |
| `top5_validation_seeds.csv` | Toàn bộ 100 lượt validation |
| `best_config.json` | Cấu hình cuối cùng và thông tin seed |

## 9. Biểu đồ được tạo

Quy trình tạo năm biểu đồ:

1. Xếp hạng toàn bộ cấu hình VAE.
2. Heatmap VAE score theo `vae_beta_kl` và `bootstrap_trajectories`.
3. Xếp hạng toàn bộ 48 cấu hình UCB.
4. Heatmap Sharpe trung bình theo `beta` và `beta_decay`.
5. Biểu đồ Top 5 gồm Sharpe `mean ± std`, ARR và Max Drawdown.

Mục tiêu là mọi cấu hình đã thử đều xuất hiện trong bảng hoặc biểu đồ, thay vì chỉ hiển thị cấu hình thắng cuộc.

## 10. Cách chạy hiện tại

Ở cell tuning cuối notebook:

```python
RUN_HYPERPARAMETER_TUNING = True
```

Sau đó chạy notebook tuần tự từ đầu. Mặc định notebook chạy tuning trên regime `GOOD`:

```python
tuning_results_good = tune_hpg_ucb_vae(period="GOOD")
```

Có thể chạy riêng regime `BAD` bằng:

```python
tuning_results_bad = tune_hpg_ucb_vae(period="BAD")
```

Không nên tuning và đồng thời dùng chính cùng một tập BAD để đưa ra kết luận cuối cùng. Để báo cáo khoa học chặt chẽ, nên dành một giai đoạn thị trường hoặc một tập dữ liệu cuối hoàn toàn chưa được dùng trong bất kỳ bước chọn siêu tham số nào.

## 11. Hướng chuyển thành notebook độc lập cho Kaggle

Để notebook có thể chạy trên Kaggle mà không phụ thuộc file `.py` của repository, bước tiếp theo sẽ thực hiện:

1. Tạo một cell cấu hình chứa backbone, VAE grid, UCB grid và seed.
2. Đưa các hàm đánh giá, scoring, checkpoint và plotting từ `ucb_vae_tuning.py` vào một hoặc nhiều cell sau phần định nghĩa model.
3. Giữ `vae_quality_diagnostics` trong notebook.
4. Bỏ import helper `.py`.
5. Đổi thư mục output sang đường dẫn tương thích Kaggle, ví dụ `/kaggle/working/tuning_results/...`.
6. Thêm chế độ chạy thử nhỏ để kiểm tra pipeline trước khi bật đủ 271 lượt train.
7. Giữ chế độ đầy đủ đúng 3 seed screening, Top 5 và 20 seed validation cho kết quả chính thức.

Sau bước chuyển đổi này, `test12345.ipynb` sẽ chứa toàn bộ logic cần thiết và có thể upload/chạy độc lập trên Kaggle.
