# Báo cáo kiểm toán bất thường huấn luyện, đánh giá và XAI

Ngày kiểm toán: 17/08/2026  
Phạm vi chính: `Test_Three_Strategy_6Stocks.ipynb`, dữ liệu GOOD/BAD của 6 cổ phiếu, `final_result`, checkpoint và notebook RDX/MSX.  
Trạng thái tài liệu: tổng hợp các phát hiện đã kiểm chứng đến thời điểm hiện tại; báo cáo này không tự sửa mô hình hoặc kết quả.

## 1. Kết luận điều hành

Các vấn đề cần xử lý theo thứ tự ưu tiên:

1. **P0 — Kết quả GAS cũ không hợp lệ.** Bốn split GAS trước đây là bản sao chính xác của HPG. Các split GAS đã được tái tạo đúng từ `GAS_data.csv`, nhưng mọi scaler, metric, portfolio, checkpoint và hình GAS sinh trước lần sửa vẫn phải chạy lại.
2. **P0 — Hành động yêu cầu và hành động thực thi không đồng nhất.** Khi không có cổ phiếu, agent vẫn có thể chọn SELL; môi trường thực thi thành 0 nhưng Q-update vẫn gán transition cho SELL đã yêu cầu. Đây là invalid-action trap và xảy ra trên nhiều cổ phiếu, không riêng ACB.
3. **P0 — Notebook RDX dùng đầu vào không chuẩn hóa.** Mô hình được huấn luyện với frozen scaler nhưng `RDX_BAD_UCB_VAE.ipynb` truyền raw state trực tiếp vào Q-network.
4. **P0 — RDX của ACB đang có nguy cơ dùng nhầm checkpoint VCB.** Sau khi chạy simulation của sáu mã, biến `qsa` đang chứa VCB; cell RDX ACB chạy trước khi ACB được nạp lại.
5. **P1 — Sharpe không ổn định khi volatility gần 0.** Collapse hoàn toàn được ghi Sharpe bằng 0, trong khi collapse gần hoàn toàn có thể cho Sharpe `-194`. Mean Sharpe vì thế gây hiểu nhầm nghiêm trọng.
6. **P1 — UCB không đảm bảo khám phá cho Q-network.** Bootstrap ngẫu nhiên chỉ cập nhật VAE/Cost; Q-network học từ policy hiện tại. Seed 50–51 có thiên lệch SELL ban đầu và đôi khi không thoát được.
7. **P1 — Tổ chức output/checkpoint thiếu provenance.** Notebook không ghi hash dữ liệu, scaler hoặc code version vào metric; `final_result` không phải output trực tiếp của notebook và có nhiều quy ước thư mục không đồng nhất.

## 2. Bản đồ dữ liệu và code hiện tại

Notebook huấn luyện:

```text
application/EIDT_Project/Test_Three_Strategy_6Stocks.ipynb
```

Quy tắc nạp dữ liệu:

```python
DATA_ROOT / "train" / f"{period.lower()}_train_{ticker}.csv"
DATA_ROOT / "test"  / f"{period.lower()}_test_{ticker}.csv"
```

Các cửa sổ thực tế:

| Giai đoạn | Train | Test |
|---|---|---|
| GOOD | 02/01/2013–28/12/2018 | 02/01/2019–31/12/2021 |
| BAD | 02/01/2013–31/12/2021 | 04/01/2022–29/12/2023 |

Notebook mang tên “6Stocks” nhưng mỗi lần chỉ chạy một mã qua biến:

```python
TICKER = "HPG"
```

Output trực tiếp của notebook:

```text
/kaggle/working/old_test_three_strategy/{TICKER}
```

hoặc local:

```text
kaggle_working/old_test_three_strategy/{TICKER}
```

`application/EIDT_Project/final_result` là lớp kết quả đã được sao chép/tổ chức lại, không phải `RUN_ROOT` mà notebook trực tiếp ghi.

## 3. P0 — GAS từng sử dụng toàn bộ dữ liệu HPG

### 3.1. Bằng chứng trước khi sửa

Bốn cặp file có SHA-256 giống tuyệt đối:

```text
good_train_GAS.csv = good_train_HPG.csv
good_test_GAS.csv  = good_test_HPG.csv
bad_train_GAS.csv  = bad_train_HPG.csv
bad_test_GAS.csv   = bad_test_HPG.csv
```

Đối chiếu `bad_test_GAS.csv` cũ:

- Khớp close của `HPG_data.csv`: 100%.
- Khớp close của `GAS_data.csv`: 0%.
- Mean absolute close difference so với GAS thật: khoảng `53.3439`.

### 3.2. Trạng thái dữ liệu hiện tại

Bốn split GAS đã được ghi đè từ `GAS_data.csv` bằng đúng pipeline technical indicators hiện tại:

| File | Số dòng | Close đầu | Close cuối |
|---|---:|---:|---:|
| `good_train_GAS.csv` | 1.496 | 16.67 | 51.43 |
| `good_test_GAS.csv` | 752 | 51.55 | 64.51 |
| `bad_train_GAS.csv` | 2.248 | 16.67 | 64.51 |
| `bad_test_GAS.csv` | 498 | 68.34 | 64.67 |

Hiện không còn hash GAS–HPG trùng; bad test GAS khớp raw GAS 100% và không có NaN.

### 3.3. Những artifact GAS vẫn mất hiệu lực

Các artifact trong `final_result/GAS` được tạo từ dữ liệu sai và phải tái tạo:

- `rl_metrics.csv`
- `rl_episode_curves.csv`
- `rl_losses.csv`
- `rl_test_portfolios.csv`
- `buy_and_hold_metrics.csv`
- `buy_and_hold_portfolio.csv`
- `frozen_scaler_train_only.npz`
- `three_strategy_summary.csv`
- `xai_model_checkpoint.pt`
- checkpoint GAS trong `models/EIDT`
- tất cả hình tổng hợp có GAS
- mọi kết quả SHAP/RDX/MSX cho GAS dùng checkpoint cũ

Trước khi sửa dữ liệu, portfolio GAS và HPG giống tuyệt đối sau khi bỏ cột ticker; metric giống tuyệt đối sau khi bỏ `ticker` và `elapsed_seconds`; tensor Q/VAE/Cost trong checkpoint XAI GAS–HPG cũng giống tuyệt đối.

## 4. P0 — Invalid-action trap trong môi trường giao dịch

### 4.1. Logic hiện tại

Khi agent yêu cầu bán nhưng `position == 0`:

```python
executed = -min(-requested, self.position)
```

Ví dụ:

```text
requested = -5
position  = 0
executed  = 0
```

Nhưng trajectory lưu action index do policy yêu cầu:

```python
action, action_index, _ = action_scores(...)
next_state, reward, done, info = env.step(action)
actions.append(action_index)
```

`info["executed_action"]` không được dùng cho Q-update.

### 4.2. Hệ quả

```text
Policy chọn SELL -5
        ↓
Môi trường thực thi 0
        ↓
Danh mục không thay đổi
        ↓
Q-update gán transition no-op cho SELL -5
```

Mạng có thể học SELL khi không có cổ phiếu như một no-op hợp lệ. Vấn đề tương tự tồn tại nếu lượng BUY yêu cầu bị giới hạn bởi tiền mặt nhưng Q-update vẫn dùng action yêu cầu.

### 4.3. Phạm vi ảnh hưởng đã quan sát

Loại GAS–HPG khỏi thống kê do dữ liệu cũ bị trùng, các trajectory có ít nhất 95% bước portfolio đứng yên gồm:

| Mã | Mô hình | Giai đoạn | Seed | Tỷ lệ phẳng |
|---|---|---|---:|---:|
| ACB | UCB-VAE | GOOD | 51 | 99,20% |
| ACB | UCB-VAE | BAD | 51 | 99,60% |
| ACB | Epsilon | GOOD | 51 | 98,13% |
| FPT | UCB-VAE | GOOD | 50 | 100,00% |
| FPT | UCB-VAE | GOOD | 51 | 99,47% |
| FPT | Epsilon | GOOD | 50 | 97,61% |
| FPT | Epsilon | GOOD | 51 | 95,48% |
| FPT | Epsilon | BAD | 51 | 98,80% |
| SSI | UCB-VAE | BAD | 50 | 96,39% |
| VCB | UCB-VAE | GOOD | 50 | 99,34% |
| VCB | Epsilon | GOOD | 50 | 98,80% |
| VCB | Epsilon | GOOD | 51 | 98,94% |

Đây là lỗi hệ thống, không phải hiện tượng riêng của ACB.

## 5. P1 — Seed 50 và 51 tạo thiên lệch SELL có hệ thống

Ở trạng thái biên test với `cash=1000`, `position=0`, mạng Q vừa khởi tạo có hành vi:

- Seed 50: ưu tiên SELL `-3` trên ACB, FPT, SSI và VCB ở hầu hết giai đoạn.
- Seed 51: ưu tiên SELL `-5` trên gần như toàn bộ mã/giai đoạn.

Riêng ACB seed 51:

| Giai đoạn | Top action | Q top | Q thứ hai | Gap | Beta UCB ban đầu |
|---|---:|---:|---:|---:|---:|
| GOOD | -5 | 0.4232 | 0.2388 | 0.1843 | 0.03 |
| BAD | -5 | 0.4875 | 0.3925 | 0.0950 | 0.03 |

UCB novelty bonus được scale tối đa khoảng beta, tức `0.03`, rồi giảm về `0.01`. Khoảng cách Q ban đầu có thể lớn hơn bonus nhiều lần.

### Nguyên nhân UCB khó thoát

`random_bootstrap()` chỉ đưa trải nghiệm ngẫu nhiên vào replay buffer để cập nhật VAE và Cost-network. Q-network không được cập nhật từ bootstrap này. Q chỉ học từ trajectory do policy UCB hiện tại chọn.

Vì vậy, nếu policy ban đầu nghiêng về action SELL không khả thi:

1. Q không có behavioral exploration độc lập.
2. Transition no-op lại được gán cho action SELL.
3. Beta giảm nhanh từ `0.03` về `0.01`.
4. Policy có thể mắc kẹt theo seed.

Epsilon-Greedy có exploration khi train nhưng policy greedy cuối vẫn có thể collapse vì invalid-action mismatch chưa được sửa.

## 6. P1 — Bất thường Sharpe ratio

### 6.1. Công thức hiện tại

```python
returns = np.diff(portfolio) / abs(portfolio[:-1])
annual_return = mean(returns) * 252 * 100
volatility = std(returns) * sqrt(252) * 100
sharpe = 0 if volatility < 1e-12 else (annual_return - 2.0) / volatility
```

Đơn vị phần trăm giữa return, volatility và risk-free rate là nhất quán, nhưng ngưỡng `1e-12` quá thấp để xử lý portfolio gần như phẳng.

### 6.2. Hai kết quả đối nghịch nhưng cùng là collapse

- FPT UCB GOOD seed 50: không giao dịch bước nào, volatility = 0, code trả Sharpe = `0`.
- ACB UCB GOOD seed 51: chỉ thay đổi 6/747 bước, volatility = `0.01024%`, Sharpe = `-194.6116`.

Do đó:

- Collapse hoàn toàn được biểu diễn như Sharpe trung tính 0.
- Collapse gần hoàn toàn được biểu diễn như Sharpe cực âm.
- So sánh mean Sharpe bị méo dù hai trường hợp đều là chính sách gần như không giao dịch.

### 6.3. ACB seed 51

| Giai đoạn | Test profit | Volatility | Sharpe | Số bước portfolio thay đổi |
|---|---:|---:|---:|---:|
| GOOD | 0.19244 | 0.01024% | -194.6116 | 6/747 |
| BAD | -0.24235 | 0.01612% | -124.8413 | 2/498 |

Ảnh hưởng đến tổng hợp UCB ACB:

| Giai đoạn | Mean Sharpe | Median Sharpe | Mean bỏ seed 51 |
|---|---:|---:|---:|
| GOOD | -9.5148 | 0.8535 | 0.2271 |
| BAD | -6.5810 | 0.0419 | -0.3567 |

Không được âm thầm xóa seed 51. Cần định nghĩa trạng thái `NO_TRADE`/`NEAR_ZERO_VOL`, báo cáo tỷ lệ collapse và dùng median/robust statistics song song với mean.

## 7. P1 — Reward shaping chưa phụ thuộc mức độ exposure

Reward hiện tại trừ:

```python
reward -= W_RISK * abs(drawdown) + W_STABILITY * abs(asset_return)
```

`asset_return` là biến động của cổ phiếu, kể cả khi agent giữ toàn bộ tiền mặt và `position == 0`. Vì vậy agent không sở hữu tài sản vẫn bị stability penalty do thị trường biến động.

Đây là bất thường về ý nghĩa tài chính. Penalty stability nên được kiểm tra lại theo exposure/portfolio return, thay vì luôn dùng biến động tài sản độc lập với vị thế.

## 8. P1 — Sai lệch một điểm thời gian trong đánh giá metric

Đánh giá test prepend hàng cuối train:

```python
frame = concat([train.iloc[-1], test])
```

Portfolio vì thế có `len(test) + 1` điểm và điểm đầu ứng với ngày cuối train. Tuy nhiên `period_metrics()` được gọi với `dates=test["time"]`, chỉ có `len(test)` ngày và ngày bắt đầu là ngày đầu test.

Hàm hiện chỉ dùng ngày đầu/cuối để annualize nên không gây lỗi shape, nhưng thời gian của portfolio và thời gian dùng tính ARR lệch tại biên một phiên. Cần truyền đúng chuỗi ngày `[train_last_date, *test_dates]` hoặc thống nhất định nghĩa điểm portfolio đầu tiên.

## 9. P0/P1 — Bất thường trong notebook RDX/MSX

Notebook liên quan:

```text
application/XAI_analysis/RDX_2013_2017.ipynb
application/FAIR_2026/RDX_BAD_UCB_VAE.ipynb
```

### 9.1. Không áp dụng frozen scaler

Hai notebook đều dùng dạng:

```python
qsa(torch.Tensor(s).float())
```

Không có `frozen_scaler_train_only.npz` hoặc `scaler.transform`. Trong khi mô hình của pipeline `Test_Three_Strategy_6Stocks` được train và evaluate trên state đã chuẩn hóa.

Hệ quả: action, Q-value, RDX và MSX có thể được tính trên phân phối input hoàn toàn khác lúc train.

### 9.2. ACB RDX có nguy cơ dùng checkpoint VCB

Trình tự notebook:

1. Load ACB → simulation ACB.
2. Load lần lượt FPT, GAS, HPG, SSI, VCB → simulation tương ứng.
3. Định nghĩa hàm RDX.
4. Chạy RDX ACB bằng biến dùng chung `qsa`.

Tại bước 4, `qsa` đang chứa checkpoint VCB vì không có cell reload ACB ngay trước RDX ACB. Các mã FPT–VCB được reload trước phần RDX tương ứng, riêng ACB không được reload.

### 9.3. Risk và Stability không phân biệt action trong RDX

Trong balanced decomposition, Risk và Stability chỉ phụ thuộc state/time, được gán giống nhau cho cả 11 action. Khi lấy contrast:

```text
RDX = component(selected action) - component(compared action)
```

thì:

```text
ΔRisk = 0
ΔStability = 0
```

MSX vì vậy chủ yếu chỉ còn Profit và Position; kết quả “MSX size = 1” không chứng minh Risk/Stability không quan trọng, mà là hệ quả trực tiếp của công thức phân rã action-invariant.

### 9.4. Plot và tên artifact không đồng nhất

- Cell gọi plot đầu tiên truyền HPG thay vì ACB.
- HPG được plot lặp lại.
- ACB chỉ xuất ở cell cuối.
- Tên `RDX_ACB_2013_2017.png` không phản ánh dữ liệu BAD 2022–2023 hiện đang dùng.
- Có đường dẫn xuất hình hard-code lồng `SARSA_FinancialRL/SARSA_FinancialRL/Graph+Pic`.
- Notebook có cell uninstall/install Kaleido giữa luồng thực thi, gây thay đổi môi trường khi Run All.

### 9.5. Cửa sổ BAD thực tế

Các file `bad_test_{ticker}.csv` chỉ bao phủ 2022–2023. Nếu báo cáo/hình ghi BAD 2021–2023 thì nhãn không khớp dữ liệu test thực tế. Năm 2021 nằm trong bad train.

## 10. P1 — Checkpoint và provenance

### 10.1. Hai luồng checkpoint

Notebook huấn luyện không tải checkpoint từ `models/EIDT`; nó train từ đầu và lưu:

```text
{RUN_ROOT}/xai_model_checkpoint.pt
```

Trong repository còn có checkpoint bare state dict tại:

```text
models/EIDT/UCB_BAD
models/EIDT/UCB_GOOD
models/EIDT/Greedy_bad
models/EIDT/GREEDY_GOOD
```

Tensor của checkpoint bare GAS–HPG hiện khác nhau, trong khi checkpoint XAI GAS–HPG cũ trong `final_result` giống nhau. Không có manifest chứng minh checkpoint bare được sinh từ run/seed/data hash nào.

### 10.2. Lựa chọn checkpoint

Checkpoint XAI được chọn theo `train_sharpe`, tránh leakage test. Tuy nhiên Sharpe gần-zero-vol có thể không ổn định; cần thêm điều kiện eligibility như minimum trades/exposure/volatility trước khi xếp hạng.

### 10.3. Thiếu metadata bắt buộc

Metric/checkpoint hiện chưa lưu đầy đủ:

- SHA-256 train/test CSV
- SHA-256 scaler
- git commit/code hash
- requested/executed action distribution
- số giao dịch và turnover
- số invalid actions
- dependency versions
- nguồn checkpoint và seed cha

## 11. P1/P2 — Tổ chức file kết quả

Các bất nhất đã xác nhận:

- `Bad` và `bad` cùng tồn tại.
- `plot` và `plots` cùng tồn tại.
- ACB/BAD thiếu `three_strategy_summary.csv`.
- `final_result` được tổ chức ngoài `RUN_ROOT` của notebook, nhưng không có script/manifest copy chính thức.
- `RESUME=True` chỉ xét `model_key` và seed trong CSV hiện có; không xác minh ticker, period, input hash hoặc code version.
- CSV curve/loss được append trước khi metric được commit. Nếu run lỗi và chạy lại, có thể tồn tại dòng curve/loss lặp; report hiện dùng `drop_duplicates`, nhưng file gốc vẫn có thể chứa lịch sử không sạch.
- Notebook nguồn hiện không chứa output đã chạy; kết quả thực tế nằm ở thư mục CSV, làm tăng rủi ro nhầm notebook/config với artifact.

## 12. Những phần đã kiểm tra và chưa phát hiện sai đường dẫn

- ACB, FPT, SSI và VCB bad-test close khớp raw tương ứng 100%.
- Train và test không chồng lấn theo kiểm tra hiện tại.
- Frozen scaler được fit từ train-only trong notebook huấn luyện.
- Hai mô hình RL dùng chung scaler của đúng ticker/period trong một run.
- Checkpoint XAI được chọn bằng train metric, không dùng test metric.
- Sau khi sửa, GAS split khớp raw GAS và không còn trùng HPG.

Các xác nhận trên không loại trừ lỗi thuật toán/action handling đã nêu.

## 13. Kế hoạch sửa đề xuất

### Giai đoạn A — Sửa tính đúng trước khi chạy lại

1. Tạo action mask theo state:
   - `position == 0` → mask toàn bộ SELL.
   - Không đủ cash → mask/giới hạn BUY phù hợp.
2. Quyết định rõ semantics:
   - policy chỉ được chọn feasible action; hoặc
   - lưu và update theo `executed_action`, đồng thời phạt invalid request rõ ràng.
3. Ghi log `requested_action`, `executed_action`, position, cash, Q, bonus, penalty và score.
4. Bổ sung exploration tối thiểu cho Q-network của UCB hoặc thiết kế bonus theo Q-gap.
5. Sửa reward stability theo exposure/portfolio risk.
6. Định nghĩa Sharpe cho no-trade/near-zero-vol là `NaN` hoặc trạng thái riêng, không dùng 0.
7. Thêm trade count, turnover, exposure ratio và invalid-action rate vào metric.
8. Đồng bộ ngày portfolio với ngày dùng tính metric.
9. Thêm input/scaler/code hash vào metric và checkpoint.

### Giai đoạn B — Sửa XAI

1. Load frozen scaler đúng ticker/period trước mọi forward pass.
2. Không dùng biến model toàn cục luân phiên; dùng dictionary `models[ticker]`.
3. Reload/verify checkpoint ACB trước RDX ACB.
4. Assert checkpoint hash và ticker ngay trong notebook.
5. Thiết kế lại Risk/Stability để có thành phần phụ thuộc action nếu muốn diễn giải contrastive.
6. Sửa tên hình, cửa sổ thời gian và mapping plot.

### Giai đoạn C — Chuẩn hóa artifact

1. Một cấu trúc duy nhất:

```text
final_result/{ticker}/{period}/{run_id}/
```

2. `run_manifest.json` chứa data hash, code hash, seed, config, dependency và checkpoint hash.
3. Chuẩn hóa lowercase `good/bad`, `plots`.
4. Không copy thủ công từ Kaggle; dùng một bước import có kiểm tra hash.

## 14. Phạm vi cần chạy lại

### Nếu chỉ khôi phục GAS với thuật toán cũ

Chạy lại GAS:

- GOOD và BAD
- UCB-VAE, Epsilon-Greedy, Buy-and-Hold
- 20 seed 41–60
- scaler, portfolio, metrics, summary, checkpoint XAI
- toàn bộ hình và XAI có GAS

Kết quả các mã khác vẫn giữ cùng định nghĩa thuật toán cũ, nhưng policy collapse phải được công bố như limitation.

### Nếu sửa invalid-action/action mask hoặc reward

Đây là thay đổi thuật toán. Để so sánh khoa học công bằng phải chạy lại:

```text
6 cổ phiếu × 2 giai đoạn × 2 mô hình RL × 20 seed
```

cùng Buy-and-Hold, scaler, checkpoint, metric và XAI tương ứng. Không nên chỉ sửa và chạy riêng ACB.

## 15. Acceptance tests bắt buộc trước lần chạy công bố

### Data

- [ ] Mỗi split hash khác các ticker còn lại.
- [ ] Close khớp raw đúng ticker 100% trên các ngày giao nhau.
- [ ] Không NaN/Inf ở 7 feature.
- [ ] Train max date < test min date.
- [ ] Manifest lưu data hash.

### Environment/action

- [ ] Không thể chọn SELL khi position = 0, hoặc invalid request được phạt/log rõ ràng.
- [ ] Requested và executed action nhất quán trong Q-update.
- [ ] Unit test BUY bị giới hạn bởi cash.
- [ ] Unit test SELL bị giới hạn bởi position.
- [ ] Invalid-action rate bằng 0 nếu dùng masking.

### Training

- [ ] Kiểm tra action distribution theo seed ngay episode 1.
- [ ] Cảnh báo nếu no-trade > 95% bước.
- [ ] Cảnh báo nếu một action chiếm > 95% lượt chọn.
- [ ] UCB exploration thực sự thay đổi action của Q-policy.
- [ ] Checkpoint đủ metadata và hash.

### Metrics

- [ ] Portfolio và dates cùng chiều dài/định nghĩa biên.
- [ ] Sharpe no-trade không được ghi 0 như hiệu suất trung tính.
- [ ] Báo cáo mean, SD, median và IQR.
- [ ] Báo cáo trade count, turnover, exposure và collapse rate.
- [ ] Không tổng hợp GAS cũ.

### XAI

- [ ] Forward pass dùng frozen scaler đúng ticker.
- [ ] Checkpoint ticker/hash được assert.
- [ ] ACB không dùng model VCB.
- [ ] Nhãn thời gian đúng với dữ liệu thực tế.
- [ ] Giải thích rõ thành phần RDX nào action-invariant.

## 16. Quyết định sử dụng kết quả hiện tại

| Thành phần | Trạng thái |
|---|---|
| Split GAS hiện tại | Đã sửa, có thể dùng để chạy lại |
| Kết quả GAS cũ | Không hợp lệ |
| HPG cũ | Dữ liệu HPG đúng, nhưng vẫn chịu lỗi action/metric chung |
| ACB/FPT/SSI/VCB data | Chưa phát hiện nhầm ticker |
| Metric ACB mean Sharpe | Không nên dùng đơn độc |
| Các seed 50–51 collapse | Phải đánh dấu/giải thích hoặc chạy lại sau sửa thuật toán |
| Checkpoint XAI GAS cũ | Không hợp lệ |
| RDX/MSX hiện tại | Chưa đủ tin cậy do scaler/model mapping/decomposition |
| Hình tổng hợp chứa GAS | Phải tạo lại |

## 17. Nguyên tắc chỉnh sửa hệ thống

Không loại seed bất lợi sau khi xem kết quả. Hoặc:

1. Giữ thuật toán cũ, báo cáo đầy đủ collapse rate và robust statistics; hoặc
2. Sửa thuật toán theo quy tắc định trước, rồi chạy lại toàn bộ phạm vi so sánh.

Không trộn kết quả chạy bằng action handling cũ và mới trong cùng bảng/hình công bố.
