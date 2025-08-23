"""
Preprocessor for VNStock data.
Tương thích với FinRL-style pipeline: downloader -> preprocessor -> data_processor
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler
from stockstats import StockDataFrame as Sdf

from meta.preprocessor.vnstockdownloader import VnstockDownloader


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Load dataset từ file CSV"""
    return pd.read_csv(file_name)


def data_split(df, start, end, target_date_col="date"):
    """Cắt dữ liệu theo thời gian"""
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

class FeatureEngineer:
    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=None,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        """
        tech_indicator_list: danh sách các chỉ báo kỹ thuật
        """
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list or ["macd", "rsi_14"]
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa cột, thêm chỉ báo và các feature khác"""
        df = self.clean_data(df)

        if self.use_technical_indicator:
            df = self.add_technical_indicators(df)

        if self.use_turbulence:
            df = self.add_turbulence(df)

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Đổi tên cột timestamp -> date nếu cần, sắp xếp dữ liệu"""
        if "timestamp" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"timestamp": "date"})

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "tic"])
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính chỉ báo kỹ thuật cho từng cổ phiếu"""
        for indicator in self.tech_indicator_list:
            if indicator.lower() == "macd":
                df["macd"] = self._macd(df)
            elif indicator.lower() == "rsi_14":
                df["rsi_14"] = self._rsi(df, 14)
            # Có thể bổ sung thêm chỉ báo khác
        return df

    def _macd(self, df: pd.DataFrame) -> pd.Series:
        """Moving Average Convergence Divergence"""
        macd_list = []
        for tic in df["tic"].unique():
            temp = df[df["tic"] == tic].copy()
            exp1 = temp["close"].ewm(span=12, adjust=False).mean()
            exp2 = temp["close"].ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            macd_list.append(macd_line)
        return pd.concat(macd_list)

    def _rsi(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Relative Strength Index"""
        rsi_list = []
        for tic in df["tic"].unique():
            temp = df[df["tic"] == tic].copy()
            delta = temp["close"].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            roll_up = up.rolling(window).mean()
            roll_down = down.rolling(window).mean()
            rs = roll_up / (roll_down + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_list.append(rsi)
        return pd.concat(rsi_list)

    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        # Nếu chỉ có 1 mã cổ phiếu thì không thể tính covariance
        if df['tic'].nunique() == 1:
            df['turbulence'] = 0
            return df

        # Pivot dữ liệu để tính biến động
        df_pivot = df.pivot(index="date", columns="tic", values="close")
        df_return = df_pivot.pct_change().dropna()

        turbulence_index = []
        for i in range(len(df_return)):
            current = df_return.iloc[: i + 1]
            turbulence = current.cov().values.sum()
            turbulence_index.append(turbulence)

        # Đảm bảo khớp độ dài
        if len(turbulence_index) != len(df_return.index):
        # Điền thêm 0 vào đầu cho các ngày thiếu
            diff = len(df_return.index) - len(turbulence_index)
            turbulence_index = [0] * diff + turbulence_index

        turbulence_series = pd.Series(turbulence_index, index=df_return.index)


        # Merge với df gốc
        df = df.merge(
            turbulence_series.rename("turbulence"),
            how="left",
            left_on="date",
            right_index=True,
        )

        return df.fillna(0)




class GroupByScaler:
    """Chuẩn hóa dữ liệu theo từng nhóm cổ phiếu"""

    def __init__(self, by="tic"):
        self.by = by
        self.scalers = {}

    def fit(self, df: pd.DataFrame):
        for key, group in df.groupby(self.by):
            scaler = StandardScaler()
            num_cols = group.select_dtypes(include=[np.number]).columns
            scaler.fit(group[num_cols])
            self.scalers[key] = (scaler, num_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for key, group in df.groupby(self.by):
            scaler, num_cols = self.scalers[key]
            df_copy.loc[group.index, num_cols] = scaler.transform(group[num_cols])
        return df_copy