import pytest
import pandas as pd
from meta.preprocessor.vnstockdownloader import VnstockDownloader

# Test cơ bản: tải dữ liệu 1 mã
def test_fetch_single_stock():
    downloader = VnstockDownloader(
        start_date="2024-01-01",
        end_date="2024-01-15",
        ticker_list=["VNM"]
    )
    df = downloader.fetch_data()

    # Kiểm tra kiểu dữ liệu
    assert isinstance(df, pd.DataFrame)
    # Phải có các cột chính
    required_cols = ["date", "open", "high", "low", "close", "volume", "tic", "day"]
    for col in required_cols:
        assert col in df.columns

    # Ít nhất có 1 dòng dữ liệu
    assert len(df) > 0
    # Mã cổ phiếu đúng
    assert set(df["tic"].unique()) == {"VNM"}

# Test nhiều mã
def test_fetch_multiple_stocks():
    downloader = VnstockDownloader(
        start_date="2024-01-01",
        end_date="2024-01-10",
        ticker_list=["VNM", "HPG"]
    )
    df = downloader.fetch_data()

    # Kiểm tra có đủ 2 mã
    assert set(df["tic"].unique()).issubset({"VNM", "HPG"})
    # Ngày hợp lệ
    assert pd.to_datetime(df["date"]).min() >= pd.to_datetime("2024-01-01")
    assert pd.to_datetime(df["date"]).max() <= pd.to_datetime("2024-01-10")

# Test trường hợp không có data
def test_no_data():
    downloader = VnstockDownloader(
        start_date="1900-01-01",
        end_date="1900-01-10",
        ticker_list=["VNM"]
    )
    with pytest.raises(ValueError):
        downloader.fetch_data()
