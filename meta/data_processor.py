from __future__ import annotations
import numpy as np
import pandas as pd

from meta.data_processors.processor_vnstock import VnstockProcessor


class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        if data_source == "vnstock":
            self.processor = VnstockProcessor()
        else:
            raise ValueError("Only vnstock is supported in this version.")

        self.tech_indicator_list = tech_indicator
        self.vix = vix

    def download_data(self, ticker_list, start_date, end_date, time_interval="1d") -> pd.DataFrame:
        return self.processor.download_data(ticker_list, start_date, end_date, time_interval)

    def clean_data(self, df) -> pd.DataFrame:
        return self.processor.clean_data(df)

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        return self.processor.add_technical_indicator(df, tech_indicator_list)

    def add_turbulence(self, df) -> pd.DataFrame:
        return self.processor.add_turbulence(df)

    def add_vix(self, df) -> pd.DataFrame:
        return self.processor.add_vix(df)

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        # fill nan and inf values with 0
        if tech_array.size > 0:
            tech_array[np.isnan(tech_array)] = 0
            tech_array[np.isinf(tech_array)] = 0
        return price_array, tech_array, turbulence_array
