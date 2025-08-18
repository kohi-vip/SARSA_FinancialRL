"""
Contains methods and classes to collect data from
VNStock3 API (https://pypi.org/project/vnstock/)
"""

from __future__ import annotations

import pandas as pd
from vnstock import Vnstock


class VnstockDownloader:
    """Provides methods for retrieving daily stock data from
    VNStock3 API

    Attributes
    ----------
        start_date : str
            start date of the data (YYYY-MM-DD)
        end_date : str
            end date of the data (YYYY-MM-DD)
        ticker_list : list
            a list of stock tickers

    Methods
    -------
    fetch_data()
        Fetches data from VNStock3 API
    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.api = Vnstock()

    def fetch_data(self, market: str = "stock") -> pd.DataFrame:
        """Fetches data from VNStock3 API

        Parameters
        ----------
        market : str
            market type ("stock", "derivative", ...), default "stock"

        Returns
        -------
        `pd.DataFrame`
            columns: date, open, high, low, close, volume, tic, day
        """
        data_df = pd.DataFrame()
        num_failures = 0

        for tic in self.ticker_list:
            try:
                # Lấy dữ liệu qua API
                temp_df = self.api.stock(symbol=tic).quote.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval="1D",
                )
                temp_df["tic"] = tic
                if len(temp_df) > 0:
                    data_df = pd.concat([data_df, temp_df], axis=0)
                else:
                    num_failures += 1
            except Exception as e:
                print(f"Failed to fetch {tic}: {e}")
                num_failures += 1

        if num_failures == len(self.ticker_list):
            raise ValueError("No data is fetched from VNStock3.")

        # chuẩn hóa cột
        data_df.rename(
            columns={
                "time": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            inplace=True,
        )

        # convert date
        data_df["date"] = pd.to_datetime(data_df["date"])
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.dt.strftime("%Y-%m-%d")

        # drop missing
        data_df = data_df.dropna().reset_index(drop=True)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        print("Shape of DataFrame: ", data_df.shape)
        return data_df

    def select_equal_rows_stock(self, df):
        """Chọn những cổ phiếu có số lượng rows gần bằng nhau"""
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
