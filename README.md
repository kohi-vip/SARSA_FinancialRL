# SARSA-FinancialRL
Đây là bài nghiên cứu khoa học nội bộ thử nghiệm khả năng lập trình AI cho trading cổ phiếu.

## Cấu trúc Project

Dự án được tổ chức theo mô hình **Environment - Agent - Training**:

```
SARSA_FinancialRL/
│
├── environments/           # Môi trường trading
│   ├── __init__.py
│   └── env_stocktrading.py  # FinRL Stock Trading Environment
│
├── agents/                 # Các agent RL
│   ├── __init__.py
│   ├── sarsa/             # SARSA Agent
│   │   ├── __init__.py
│   │   ├── deep_sarsa_agent.py      # SARSA Agent gốc
│   │   └── deep_sarsa_agent_paper.py # SARSA Agent theo paper Yang et al.
│   └── dqn/               # DQN Agent
│       ├── __init__.py
│       └── dqn_agent.py   # DQN Agent theo paper Yang et al.
│
├── training/              # Scripts training
│   ├── __init__.py
│   ├── new_train.ipynb    # Training với FinRL environment
│   └── agent_training_fpt.ipynb  # Training cho FPT stock
│
├── data/                  # Dữ liệu cổ phiếu
│   ├── FPT_detail_*.csv
│   ├── HPG_detail_*.csv
│   └── ...
│
├── Deep_RL_paper_application/  # Implementation theo paper Yang et al.
├── dqn-stable-baselines3/      # DQN với Stable Baselines3
├── FinRL/                      # Thư viện FinRL
├── Papers+Books/              # Tài liệu nghiên cứu
├── trained_models/            # Models đã train
├── results/                   # Kết quả training
│
├── test.ipynb                 # Notebook test tổng hợp
├── fix_date_format.py         # Script xử lý dữ liệu
├── Import_VN30.ipynb          # Import dữ liệu VN30
├── data_verify.ipynb          # Verify dữ liệu
├── 1_vietnam_stock_vnstock3.ipynb
├── fpt_data_comparison.ipynb
├── test copy.ipynb
├── README.md
└── requirements.txt
```

## Agents

### 1. SARSA Agent
- **File**: `agents/sarsa/deep_sarsa_agent.py`
- **Mô tả**: SARSA Agent tích hợp với FinRL StockTradingEnv
- **Tính năng**: Discretize continuous action space, experience replay, target network

### 2. DQN Agent
- **File**: `agents/dqn/dqn_agent.py`
- **Mô tả**: Deep Q-Network Agent theo Yang et al. (2020)
- **Tính năng**: Experience replay, target network, epsilon-greedy

### 3. Deep SARSA Agent (Paper)
- **File**: `agents/sarsa/deep_sarsa_agent_paper.py`
- **Mô tả**: Deep SARSA Agent theo Yang et al. (2020)
- **Tính năng**: On-policy learning, smooth Q-value updates

## Environment

### FinRL Stock Trading Environment
- **File**: `environments/env_stocktrading.py`
- **Mô tả**: Môi trường trading cổ phiếu từ thư viện FinRL
- **Tính năng**: Portfolio management, transaction costs, technical indicators

## Training

Các notebook training:
- `training/new_train.ipynb`: Training với FinRL environment
- `training/agent_training_fpt.ipynb`: Training cho cổ phiếu FPT

## Cách sử dụng

1. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Xem demo**:
   ```bash
   python demo.py
   ```

3. **Import agent**:
   ```python
   from agents.sarsa.deep_sarsa_agent import DeepSARSAAgent
   from agents.dqn.dqn_agent import DQNAgent
   ```

4. **Khởi tạo environment**:
   ```python
   from environments.env_stocktrading import StockTradingEnv
   ```

5. **Training**:
   Chạy các notebook trong folder `training/`

## Tài liệu tham khảo

- Yang, H., Liu, X. Y., Zhong, S., & Walid, A. (2020). Deep reinforcement learning for automated stock trading: An ensemble strategy. In Proceedings of the first ACM international conference on AI in finance (pp. 1-8).
- FinRL: https://github.com/AI4Finance-Foundation/FinRL

## Thiết lập môi trường bằng Conda (khuyến nghị)

Nếu bạn dùng Conda/Miniconda, repository này có `environment.yml` để tái tạo môi trường chính xác.

- Cài Miniconda (nếu chưa có): tải từ https://docs.conda.io/en/latest/miniconda.html và cài đặt.
- Tạo môi trường từ file `environment.yml` (đã có sẵn trong repo):
   ```powershell
   # Tạo môi trường từ file (chứa Python 3.12 và các package cần thiết)
   conda env create -f environment.yml

   # Kích hoạt môi trường
   conda activate myenv
   ```

- Hoặc tạo thủ công (nếu bạn muốn tự chọn tên/phiên bản Python):
   ```powershell
   conda create -n myenv python=3.12
   conda activate myenv
   conda install matplotlib pytorch torchvision torchaudio -c pytorch -c conda-forge
   ```

- Nếu conda yêu cầu chấp nhận Terms of Service cho các channel mặc định, chạy các lệnh sau trước khi tạo env:
   ```powershell
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
   ```

- Đồng đội có thể tái tạo môi trường với:
   ```bash
   conda env create -f environment.yml
   ```

Ghi chú: sau khi cài Miniconda, bạn có thể cần khởi động lại terminal để `conda` được nạp vào shell.
