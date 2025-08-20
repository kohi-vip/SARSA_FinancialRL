"""Vnstock data processor (compatible with FinRL-style DataProcessor wrapper).

Docs: https://vnstocks.com/docs/vnstock/thong-ke-gia-lich-su
Usage core:
    from vnstock import Vnstock
    stock = Vnstock().stock(symbol='ACB', source='VCI')
    df = stock.quote.history(start='2024-01-01', end='2025-03-19', interval='1D')
"""

from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf

# vnstock v3 API
from vnstock import Vnstock


class VnstockProcessor:
    """Provides methods for retrieving and processing Vietnam stock market data via vnstock v3.

    Notes
    -----
    - History columns from vnstock: ['time','open','high','low','close','volume'].
    - Supported intervals (per docs): '1m','5m','15m','30m','1H','1D','1W','1M'.
    - This processor normalizes columns to:
        ['timestamp','open','high','low','close','volume','tic'].
    """

    def __init__(self, source: str = "VCI"):
        # 'VCI' is recommended, 'TCBS' is also supported.
        self.source = source
        self.start: str | None = None
        self.end: str | None = None
        self.time_interval: str | None = None

    # ---------- helpers ----------
    def _convert_interval(self, time_interval: str) -> str:
        """Map FinRL-style intervals to vnstock intervals."""
        if time_interval is None:
            return "1D"
        m = {
            "1Min": "1m",
            "2Min": "2m",  # vnstock không có 2m; sẽ fallback về 1m bên dưới
            "5Min": "5m",
            "15Min": "15m",
            "30Min": "30m",
            "60Min": "1H",
            "90Min": "1H",  # gần đúng
            "1H": "1H",
            "1h": "1H",
            "1D": "1D",
            "1d": "1D",
            "5D": "1D",  # gần đúng
            "1W": "1W",
            "1M": "1M",
            "3M": "1M",  # gần đúng
        }
        if time_interval in m:
            return m[time_interval]
        # if already valid
        if time_interval in {"1m", "5m", "15m", "30m", "1H", "1D", "1W", "1M"}:
            return time_interval
        raise ValueError(f"Unsupported interval for vnstock: {time_interval}")

    # ---------- core API expected by DataProcessor ----------
    def download_data(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Download OHLCV for tickers from vnstock.

        Returns a DataFrame with columns:
            ['timestamp','open','high','low','close','volume','tic'].
        """
        interval = self._convert_interval(time_interval)
        self.start = start_date
        self.end = end_date
        self.time_interval = interval

        out = []
        for tic in ticker_list:
            # vnstock v3 syntax
            s = Vnstock().stock(symbol=tic, source=self.source)
            df = s.quote.history(start=start_date, end=end_date, interval=interval)
            if df is None or df.empty:
                continue
            # normalize columns
            df = df.copy()
            # ensure datetime and rename
            df["timestamp"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"])
            df["tic"] = tic
            df = df[
                ["timestamp", "open", "high", "low", "close", "volume", "tic"]
            ].sort_values("timestamp")
            out.append(df)

        if not out:
            raise ValueError("No data fetched from vnstock for given tickers/date range.")

        data_df = pd.concat(out, ignore_index=True)
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce")
        data_df = data_df.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align dates across tickers and forward-fill missing OHLC with previous close; set missing volume=0.

        Works for EOD ('1D') data. For intraday intervals, we leave rows as-is (no calendar available).
        """
        if df.empty:
            return df

        if self.time_interval != "1D":
            # Keep simple for intraday: sort and return
            return df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

        # Build the union of all trading days from data itself (no holiday table needed)
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        all_days = (
            df["date"].drop_duplicates().sort_values().reset_index(drop=True).to_frame()
        )
        all_days.columns = ["date"]

        cleaned = []
        for tic, g in df.groupby("tic", sort=False):
            g = g.sort_values("date")
            g = (
                all_days.merge(g, on="date", how="left")
                .drop(columns=["timestamp"])
                .rename(columns={"date": "timestamp"})
            )
            # carry-forward fill using previous close when OHLC missing
            # if the very first row is NaN (no history at start), fill zeros
            if pd.isna(g.loc[0, "close"]):
                g.loc[0, ["open", "high", "low", "close", "volume"]] = [0, 0, 0, 0, 0]

            for i in range(1, len(g)):
                if pd.isna(g.loc[i, "close"]):
                    prev_close = g.loc[i - 1, "close"]
                    g.loc[i, ["open", "high", "low", "close"]] = [
                        prev_close,
                        prev_close,
                        prev_close,
                        prev_close,
                    ]
                    g.loc[i, "volume"] = 0
            g["tic"] = tic
            cleaned.append(g)

        new_df = pd.concat(cleaned, ignore_index=True)
        # restore dtype/ordering
        new_df = new_df[
            ["timestamp", "open", "high", "low", "close", "volume", "tic"]
        ].sort_values(["timestamp", "tic"])
        return new_df.reset_index(drop=True)

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: List[str]
    ) -> pd.DataFrame:
        """Add technical indicators using stockstats, per ticker."""
        if data.empty:
            return data

        df = data.copy().sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())  # stockstats expects OHLCV columns present
        unique_ticker = stock.tic.unique()

        for ind in tech_indicator_list:
            ind_df = pd.DataFrame()
            for tic in unique_ticker:
                try:
                    temp = stock[stock.tic == tic][ind]
                    temp = pd.DataFrame(temp)
                    temp["tic"] = tic
                    temp["timestamp"] = df[df.tic == tic]["timestamp"].to_list()
                    ind_df = pd.concat([ind_df, temp], ignore_index=True)
                except Exception as e:
                    print(f"[indicator {ind}] {tic}: {e}")
            df = df.merge(
                ind_df[["tic", "timestamp", ind]], on=["tic", "timestamp"], how="left"
            )
        df = df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)
        return df

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach VNINDEX close as market risk proxy (column name 'VNI').

        Equivalent role to VIXY in the Yahoo processor.
        """
        if df.empty:
            return df
        s = Vnstock().stock(symbol="VNINDEX", source=self.source)
        vni = s.quote.history(start=self.start, end=self.end, interval=self.time_interval or "1D")
        if vni is None or vni.empty:
            # fallback: no market series
            df_out = df.copy()
            df_out["VNI"] = 0.0
            return df_out

        vni = vni.copy()
        vni["timestamp"] = pd.to_datetime(vni["time"])
        vni = vni.rename(columns={"close": "VNI"})[["timestamp", "VNI"]]
        out = df.merge(vni, on="timestamp", how="left").sort_values(
            ["timestamp", "tic"]
        )
        # forward fill VNI if missing
        out["VNI"] = out["VNI"].ffill().bfill()
        return out.reset_index(drop=True)

    # Turbulence identical logic to Yahoo variant but using our timestamp/tic
    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        df = data.copy()
        px = df.pivot(index="timestamp", columns="tic", values="close").pct_change()

        unique_time = df["timestamp"].unique()
        start = min(time_period, len(unique_time))
        turbulence_index = [0] * start
        count = 0
        for i in range(start, len(unique_time)):
            current = px[px.index == unique_time[i]]
            hist = px[
                (px.index < unique_time[i])
                & (px.index >= unique_time[i - time_period] if i - time_period >= 0 else px.index.min())
            ]
            # align on least-missing span
            filtered = hist.iloc[hist.isna().sum().min() :].dropna(axis=1)
            if filtered.empty:
                turbulence_index.append(0)
                continue
            cov = filtered.cov()
            diff = current[[c for c in filtered]].fillna(0) - np.mean(filtered, axis=0)
            temp = diff.values.dot(np.linalg.pinv(cov)).dot(diff.values.T)
            if temp > 0:
                count += 1
                turbulence_index.append(temp[0][0] if count > 2 else 0)
            else:
                turbulence_index.append(0)

        return pd.DataFrame({"timestamp": px.index, "turbulence": turbulence_index})

    def add_turbulence(
        self, df: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        if df.empty:
            return df
        turb = self.calculate_turbulence(df, time_period=time_period)
        out = df.merge(turb, on="timestamp", how="left").sort_values(
            ["timestamp", "tic"]
        )
        return out.reset_index(drop=True)

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform dataframe into (price, tech, market/turbulence) arrays."""
        df = df.copy()
        tickers = df.tic.unique()
        first = True
        for tic in tickers:
            if first:
                price = df[df.tic == tic][["close"]].values
                tech = df[df.tic == tic][tech_indicator_list].values
                if if_vix and "VNI" in df.columns:
                    risk = df[df.tic == tic]["VNI"].values
                else:
                    risk = df[df.tic == tic]["turbulence"].values
                first = False
            else:
                price = np.hstack([price, df[df.tic == tic][["close"]].values])
                tech = np.hstack([tech, df[df.tic == tic][tech_indicator_list].values])
        return price, tech, risk
