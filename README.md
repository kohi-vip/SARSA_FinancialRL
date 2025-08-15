# SARSA-FinancialRL
Đây là bài nghiên cứu khoa học nội bộ thử nghiệm khả năng lập trình AI.

Thiết kế cấu trúc project như sau:
```
financial_rl/
│   ├── applications
│   	└── stock_trading
│   ├── agents
│   	└── SARSA
│   ├── meta
│   	├── data_processors
│   	├── env_stock_trading
│   	├── preprocessor
│   	├── data_processor.py
│       ├── meta_config_tickers.py
│   	└── meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
│
├── examples
├── unit_tests (unit tests to verify codes on env & data)
│   ├── environments
│   	└── test_env_cashpenalty.py
│   └── downloaders
│   	├── test_yahoodownload.py
│   	└── test_alpaca_downloader.py
├── setup.py
├── requirements.txt
└── README.md
└── .gitignore
```
