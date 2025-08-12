# SARSA-FinancialRL
Đây là bài nghiên cứu khoa học nội bộ thử nghiệm khả năng lập trình AI.

Thiết kế cấu trúc project như sau:

```plaintext financial_rl/ ├── configs/ # Cấu hình mô hình, môi trường, siêu tham số │ ├── default.yaml │ ├── env_config.yaml │ └── training_config.yaml │ ├── data/ # Dữ liệu gốc & đã tiền xử lý │ ├── raw/ # Dữ liệu thô (csv từ API, vnstock,...) │ ├── processed/ # Dữ liệu đã xử lý feature │ └── README.md │ ├── envs/ # Định nghĩa môi trường RL (gym-style) │ ├── single_stock_env.py │ ├── multi_stock_env.py │ └── utils.py │ ├── agents/ # Thuật toán RL │ ├── sarsa_agent.py │ ├── qlearning_agent.py │ ├── dqn_agent.py │ └── common/ # Policy, replay buffer, network,... │ ├── policy.py │ ├── networks.py │ └── utils.py │ ├── training/ # Script huấn luyện & đánh giá │ ├── train_single.py │ ├── train_multi.py │ └── evaluate.py │ ├── experiments/ # Kết quả thí nghiệm (log, tensorboard) │ ├── run_2025_08_12_10-30/ │ │ ├── logs/ │ │ ├── checkpoints/ │ │ └── metrics.csv │ └── README.md │ ├── notebooks/ # Notebook cho phân tích & thử nghiệm nhanh │ ├── EDA.ipynb │ ├── Backtest.ipynb │ └── FeatureEngineering.ipynb │ ├── tests/ # Unit tests để đảm bảo code chạy đúng │ ├── test_envs.py │ ├── test_agents.py │ └── test_training.py │ ├── requirements.txt # Python package ├── setup.py # Cho phép cài `pip install -e .` ├── README.md └── .gitignore ```

