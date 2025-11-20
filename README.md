# SARSA-FinancialRL
Đây là bài nghiên cứu khoa học nội bộ thử nghiệm khả năng lập trình AI cho trading cổ phiếu.

## Cấu trúc Project

Dự án được tổ chức theo mô hình **Data - Environment - Agent - Training**:

```
SARSA_FinancialRL/
├── README.md                           # Tài liệu chính của dự án
│
├── application/                        # Thư mục chứa các ứng dụng, ví dụ, kết quả và các file không liên quan trực tiếp đến code chính
│   ├── Deep_RL_paper_application/      # Ứng dụng Deep RL từ paper (di chuyển từ data/)
│   ├── example/                        # Ví dụ sử dụng (di chuyển từ root)
│   ├── results/                        # Kết quả thí nghiệm (di chuyển từ data/)
│   └── testing/                        # Scripts và notebooks testing
│
├── agents/                             # Thư mục chứa các agent RL
│   ├── dqn/                            # Agent DQN (Deep Q-Network)
│   │   ├── __init__.py
│   │   ├── dqn_agent.py                # Code chính cho DQN agent
│   │   └── __pycache__/
│   ├── d_sarsa/                        # Agent D-SARSA (biến thể SARSA)
│   │   ├── __init__.py
│   │   ├── d_sarsa.py                  # Code chính cho D-SARSA
│   │   └── __pycache__/
│   └── policy_gradient/                # Agent Policy Gradient
│       ├── __init__.py
│       ├── policy_gradient_agent.py    # Code chính cho Policy Gradient
│       └── __pycache__/
│
├── data/                               # Thư mục chứa dữ liệu và các công cụ xử lý
│   ├── *.csv files                     # Dữ liệu CSV cổ phiếu VN cũ trước khi tổ chức folder
│   │   ├── FPT_detail_2013_01_01_2024_12_31.csv
│   │   ├── FPT_detail_2013_2020.csv
│   │   ├── HPG_detail_2013_2020.csv
│   │   ├── MWG_detail_2013_2020.csv
│   │   ├── PLX_detail_2013_2020.csv
│   │   ├── SSI_detail_2013_01_01_2024_12_31.csv
│   │   ├── stock_prices_2013_2020.csv
│   │   ├── VCB_detail_2013_2020.csv
│   │   └── VIC_detail_2013_2020.csv
│   ├── data_kaggle/                    # Dữ liệu từ Kaggle (VN30 stocks)
│   │   ├── *.csv files (ACB.csv, BCM.csv, BID.csv, ..., VN30.csv)
│   │   └── kaggle_VN30.ipynb           # Notebook xử lý dữ liệu Kaggle
│   ├── data_processor/                 # Công cụ xử lý dữ liệu (subfolder trong data/)
│   │   ├── library_extracted/          # Lấy dữ liệu từ thư viện bên ngoài
│   │   └── other_dataset/              # Bộ dataset khác
│   ├── data_provider/                  # Cung cấp dữ liệu
│   │   ├── column_reform/              # Sửa đổi cột dữ liệu
│   │   └── feature_engineer/           # Kỹ thuật đặc trưng
│   ├── data_spliter/                   # Chia tách dữ liệu (train/val/test)
│   └── data_storer/                    # Lưu trữ dữ liệu
│       ├── data_test/                  # Dữ liệu test
│       ├── data_train/                 # Dữ liệu train
│       └── data_val/                   # Dữ liệu validation
│
├── environments/                       # Thư mục chứa môi trường RL
│   ├── __init__.py                     # Khởi tạo package environments
│   ├── __pycache__/
│   ├── env_stocktrading.py             # Code chính cho môi trường trading cổ phiếu
│   ├── portfolio_management_env/       # Subfolder cho môi trường quản lý danh mục
│   └── stock_trading_env/              # Subfolder cho môi trường trading cổ phiếu
│
├── training/                           # Thư mục training chính (hiện rỗng)
│
├── data_processor/                     # Công cụ xử lý dữ liệu (ở root, riêng biệt với data/data_processor/)
│   └── vnstock/                        # Xử lý dữ liệu từ thư viện vnstock
│       ├── __init__.py
│       ├── __pycache__/
│       └── data_filter.py              # Class VNStockDataProcessor (đã tạo)
│
├── FinRL/                              # Thư viện FinRL submodule tham khảo (external library)
│
└── *.ipynb files (còn lại ở root)       # Các notebook Jupyter còn lại
```
