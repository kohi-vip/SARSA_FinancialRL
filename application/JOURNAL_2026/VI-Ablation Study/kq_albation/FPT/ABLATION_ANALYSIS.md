# Phân tích kết quả ablation của mô hình FPT

## 1. Mục tiêu và phạm vi

Tài liệu này đánh giá đóng góp của ba thành phần trong mô hình `UNCERTAINTY_AWARE_UCB_009`:

1. confidence weighting;
2. cost network;
3. VAE uncertainty.

Phép so sánh chính dùng giai đoạn `BAD`, vì cả ba thí nghiệm ablation đều chỉ được chạy trên giai đoạn này. Mỗi cấu hình có 20 lần chạy với cùng tập seed từ 41 đến 60, do đó có thể so sánh ghép cặp theo seed. Baseline ở giai đoạn `GOOD` chỉ được dùng để nhận xét về ảnh hưởng của chế độ thị trường, không được trộn vào phép so sánh ablation.

Nguồn dữ liệu:

- [Baseline BAD](FPT_baseline/bad/three_strategy_summary.csv) và [kết quả từng seed](FPT_baseline/bad/rl_metrics.csv)
- [Bỏ confidence weighting](without_confident_weighting/ablation_summary.csv) và [kết quả từng seed](without_confident_weighting/rl_metrics.csv)
- [Bỏ cost network](without_cost_network/ablation_summary.csv) và [kết quả từng seed](without_cost_network/rl_metrics.csv)
- [Bỏ VAE uncertainty](without_vae/ablation_summary.csv) và [kết quả từng seed](without_vae/rl_metrics.csv)

## 2. Kết quả tổng hợp

| Cấu hình | Profit, mean ± SD | Profit median | ROI mean (%) | ARR mean (%) | Volatility mean (%) | Sharpe, mean ± SD | Max drawdown mean (%) | Violations mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline đầy đủ | **224.55 ± 186.64** | **194.12** | **22.45** | **10.40** | 16.02 | 0.523 ± 0.535 | 19.86 | 1.70 |
| Không confidence weighting | 219.83 ± 188.24 | 167.84 | 21.98 | 10.19 | 15.80 | **0.530 ± 0.476** | 19.60 | 1.70 |
| Không cost network | 217.50 ± 187.14 | 191.10 | 21.75 | 10.08 | 15.67 | 0.463 ± 0.624 | 19.47 | 1.45 |
| Không VAE uncertainty | 216.96 ± 190.22 | 161.89 | 21.70 | 10.05 | **15.33** | 0.507 ± 0.553 | **19.15** | **1.40** |

Tất cả cấu hình đều có collapse rate, no-trade rate và invalid-action rate bằng 0. Điều này cho thấy các biến thể ablation vẫn vận hành hợp lệ; khác biệt kết quả không đến từ việc policy bị sập, không giao dịch hoặc sinh hành động không hợp lệ.

## 3. Mức thay đổi so với baseline

| Ablation | Δ Profit | Profit tương đối | Δ Sharpe | Δ Volatility | Δ Max drawdown | Δ Violations |
|---|---:|---:|---:|---:|---:|---:|
| Không confidence weighting | -4.72 | -2.10% | +0.007 | -0.22 | -0.26 | 0.00 |
| Không cost network | -7.05 | -3.14% | -0.060 | -0.35 | -0.39 | -0.25 |
| Không VAE uncertainty | -7.58 | -3.38% | -0.016 | -0.69 | -0.72 | -0.30 |

Giá trị âm ở các cột volatility, drawdown và violations là cải thiện vì rủi ro hoặc số vi phạm giảm. Nhìn theo trung bình, baseline tạo lợi nhuận cao nhất, nhưng đổi lại có volatility, drawdown và số vi phạm cao nhất.

## 4. Kiểm định ghép cặp theo seed

Đặt `Δ = baseline − ablation`. Bảng dưới dùng paired t-test hai phía trên 20 cặp seed; khoảng tin cậy là 95% cho Δ trung bình.

| Ablation | Δ Profit (95% CI), p | Δ Sharpe (95% CI), p | Δ Volatility (95% CI), p |
|---|---:|---:|---:|
| Không confidence weighting | 4.72 [-17.20; 26.64], p=0.658 | -0.007 [-0.074; 0.061], p=0.836 | 0.22 [-0.32; 0.75], p=0.405 |
| Không cost network | 7.05 [-13.73; 27.83], p=0.486 | 0.060 [-0.036; 0.157], p=0.206 | 0.35 [-0.08; 0.78], p=0.103 |
| Không VAE uncertainty | 7.58 [-16.46; 31.63], p=0.517 | 0.016 [-0.050; 0.083], p=0.613 | **0.69 [0.11; 1.27], p=0.023** |

Kiểm định Wilcoxon ghép cặp cho cùng các chênh lệch cũng không tìm thấy khác biệt profit hoặc Sharpe có ý nghĩa (`p ≥ 0.813` cho profit và `p ≥ 0.841` cho Sharpe). Riêng volatility khi bỏ VAE uncertainty có `p=0.002`; volatility thấp hơn baseline ở 17/20 seed. Đây là tín hiệu khá nhất quán rằng nhánh VAE uncertainty trong cấu hình hiện tại làm policy chấp nhận biến động cao hơn, nhưng chưa có bằng chứng rằng mức biến động tăng đó tạo ra cải thiện lợi nhuận hoặc Sharpe ổn định.

Do có nhiều chỉ tiêu và nhiều phép thử, các giá trị p nên được xem là bằng chứng thăm dò, không phải xác nhận nhân quả cuối cùng.

## 5. Diễn giải theo từng thành phần

### 5.1. Confidence weighting

Bỏ confidence weighting làm profit trung bình giảm 2.10% và profit trung vị giảm từ 194.12 xuống 167.84, nhưng Sharpe trung bình tăng rất nhẹ từ 0.523 lên 0.530. Không chỉ tiêu chính nào khác biệt có ý nghĩa thống kê. Vì vậy, dữ liệu hiện tại chỉ gợi ý confidence weighting có thể hỗ trợ lợi nhuận; chưa đủ bằng chứng để khẳng định thành phần này cải thiện hiệu quả điều chỉnh theo rủi ro.

### 5.2. Cost network

Bỏ cost network làm profit giảm 3.14% và Sharpe giảm mạnh nhất trong ba ablation, từ 0.523 xuống 0.463. Tuy nhiên, biến thể này lại có volatility, drawdown và số violations thấp hơn. Kết quả phù hợp với một đánh đổi giữa hiệu suất và mức độ thận trọng, nhưng các khoảng tin cậy vẫn cắt qua 0. Cost network có tín hiệu hữu ích đối với Sharpe và lợi nhuận, song cần thêm seed hoặc nhiều giai đoạn thị trường để xác nhận.

### 5.3. VAE uncertainty

Bỏ VAE uncertainty làm profit trung bình giảm nhiều nhất, 3.38%, và profit trung vị giảm 16.60%. Tuy vậy, đây cũng là cấu hình có volatility, drawdown và violations thấp nhất. Chênh lệch volatility là kết quả rõ nhất của toàn bộ ablation, trong khi khác biệt profit và Sharpe không có ý nghĩa. Nói cách khác, VAE uncertainty đang thay đổi khẩu vị rủi ro rõ hơn là tạo ra lợi thế lợi nhuận đã được chứng minh.

## 6. Chi phí tính toán

Thời gian chạy trung bình mỗi seed giảm từ 519.08 giây ở baseline xuống:

- 511.37 giây khi bỏ confidence weighting, giảm 1.48%;
- 452.07 giây khi bỏ cost network, giảm 12.91%;
- 421.89 giây khi bỏ VAE uncertainty, giảm 18.72%.

Vì vậy, cost network và đặc biệt là VAE uncertainty cần chứng minh lợi ích ổn định hơn nếu mục tiêu triển khai coi trọng chi phí huấn luyện.

## 7. Bối cảnh và giới hạn

- Độ lệch chuẩn profit khoảng 187–190, rất lớn so với chênh lệch trung bình 5–8 giữa các cấu hình. Nhiễu giữa seed đang lấn át hiệu ứng ablation.
- Mean và median không hoàn toàn đồng thuận. Ví dụ, bỏ VAE uncertainty chỉ giảm profit mean 3.38% nhưng giảm median 16.60%, cho thấy phân phối có thể bị ảnh hưởng bởi một số lần chạy lợi nhuận cao.
- Các ablation chỉ có dữ liệu ở giai đoạn `BAD`. Chưa thể kết luận đóng góp của từng thành phần có ổn định ở giai đoạn `GOOD` hay các chế độ thị trường khác hay không.
- Buy-and-Hold ở giai đoạn `BAD` đạt profit 466.25 và Sharpe 0.791, cao hơn mọi cấu hình RL. Đây không phải phép ablation, nhưng cho thấy mô hình đầy đủ chưa vượt được benchmark thụ động trên tập kiểm thử này.
- Việc dùng cùng seed hỗ trợ kiểm định ghép cặp, nhưng không thay thế cho đánh giá trên nhiều cửa sổ thời gian, nhiều mã cổ phiếu hoặc walk-forward validation.

## 8. Kết luận

Baseline đầy đủ đứng đầu về profit trung bình, ARR và profit trung vị, nhưng lợi thế chỉ khoảng 2.10–3.38% về profit mean và chưa có ý nghĩa thống kê. Không có bằng chứng đủ mạnh rằng bất kỳ thành phần riêng lẻ nào cải thiện profit hoặc Sharpe một cách ổn định trên 20 seed.

Kết quả đáng chú ý nhất là VAE uncertainty làm tăng volatility một cách nhất quán, đồng thời tăng nhẹ profit trung bình nhưng không cải thiện Sharpe có ý nghĩa. Nếu ưu tiên hiệu quả tính toán và rủi ro thấp, biến thể không VAE là ứng viên hợp lý để thử tiếp. Nếu ưu tiên tối đa hóa profit, nên giữ baseline ở thời điểm hiện tại nhưng cần xác nhận lại bằng nhiều giai đoạn thị trường hơn trước khi kết luận kiến trúc đầy đủ là tốt nhất.

Đề xuất thí nghiệm tiếp theo: chạy toàn bộ bốn cấu hình trên cùng nhiều cửa sổ `GOOD`, `BAD` và đi ngang; tăng số seed hoặc dùng nhiều mã; xác định trước một chỉ tiêu chính (ví dụ Sharpe); rồi báo cáo paired bootstrap confidence interval và hiệu chỉnh multiple testing.
