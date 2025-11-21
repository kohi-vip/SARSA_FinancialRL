import pandas as pd
import numpy as np
import talib as ta

# Hàm tính chỉ báo kĩ thuật MACD
def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Tính MACD (Moving Average Convergence Divergence) - chỉ trả về cột MACD
    """
    df = df.copy()
    df['MACD'] = ta.MACD(df['close'], fast, slow, signal)[0]  # macd line
    return df

# Hàm tính chỉ báo kĩ thuật RSI
def calculate_rsi(df, period=14):
    """
    Tính RSI (Relative Strength Index)
    """
    df = df.copy()
    df['RSI'] = ta.RSI(df['close'], period)
    return df

# Hàm tính chỉ báo kĩ thuật CCI
def calculate_cci(df, period=20):
    """
    Tính CCI (Commodity Channel Index)
    """
    df = df.copy()
    df['CCI'] = ta.CCI(df['high'], df['low'], df['close'], period)
    return df

# Hàm tính chỉ báo kĩ thuật ADX
def calculate_adx(df, period=14):
    """
    Tính ADX (Average Directional Index) - chỉ trả về cột ADX
    """
    df = df.copy()
    df['ADX'] = ta.ADX(df['high'], df['low'], df['close'], period)
    return df

def add_technical_indicators(df, start_date=None, auto_adjust_start_date=False, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14, cci_period=20, adx_period=14):
    """
    Thêm các chỉ số kỹ thuật vào DataFrame: MACD, RSI, CCI, ADX
    Đầu vào: DataFrame với cột 'time' (string dd/mm/yyyy), 'open', 'high', 'low', 'close', 'volume'
    start_date: Ngày bắt đầu tính toán (datetime hoặc string dd/mm/yyyy). Nếu None, sử dụng ngày có đủ lịch sử (từ max_period ngày trở đi).
    auto_adjust_start_date: Nếu True, tự động điều chỉnh start_date về ngày gần nhất có đủ lịch sử. Nếu False, raise error nếu không đủ.
    macd_fast, macd_slow, macd_signal: Tham số cho MACD
    rsi_period: Tham số cho RSI
    cci_period: Tham số cho CCI
    adx_period: Tham số cho ADX
    Đầu ra: DataFrame với cột 'time' (string dd/mm/yyyy), OHLCV và 4 chỉ báo kỹ thuật, bắt đầu từ start_date (hoặc đã điều chỉnh)
    """
    if df.empty:
        raise ValueError("DataFrame không được rỗng.")

    df = df.copy()
    # Chuyển 'time' thành datetime nếu là string
    if isinstance(df['time'].iloc[0], str):
        df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y')

    # Tính max_period sớm
    max_period = max(macd_slow, rsi_period, cci_period, adx_period)

    # Xác định ngày bắt đầu
    if start_date is None:
        # Mặc định chọn ngày có đủ lịch sử (từ max_period ngày trở đi)
        if len(df) > max_period:
            start_date = df.iloc[max_period]['time']
        else:
            start_date = df['time'].min()
    else:
        # Chuyển start_date thành datetime nếu là string
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, format='%d/%m/%Y')
        if start_date not in df['time'].values:
            # Tìm ngày gần nhất >= start_date
            future_dates = df[df['time'] >= start_date]
            if future_dates.empty:
                raise ValueError(f"Không tìm thấy ngày >= {start_date} trong DataFrame.")
            start_date = future_dates['time'].min()

    # Tìm index của start_date
    start_idx = df[df['time'] == start_date].index[0]

    # Kiểm tra đủ lịch sử: cần ít nhất max_period ngày trước start_date
    if start_idx < max_period:
        if auto_adjust_start_date:
            # Tự động điều chỉnh start_date về ngày có đủ lịch sử
            new_start_idx = max_period
            if new_start_idx >= len(df):
                raise ValueError(f"DataFrame không đủ dữ liệu lịch sử. Cần ít nhất {max_period + 1} ngày, nhưng chỉ có {len(df)} ngày.")
            start_date = df.iloc[new_start_idx]['time']
            start_idx = new_start_idx
            print(f"Đã tự động điều chỉnh start_date về {start_date} để có đủ lịch sử.")
        else:
            raise ValueError(f"Không đủ dữ liệu lịch sử để tính chỉ số kỹ thuật. Cần ít nhất {max_period} ngày trước ngày {start_date}. Hiện tại chỉ có {start_idx} ngày. Sử dụng auto_adjust_start_date=True để tự động điều chỉnh.")

    df = calculate_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = calculate_rsi(df, period=rsi_period)
    df = calculate_cci(df, period=cci_period)
    df = calculate_adx(df, period=adx_period)

    # Loại bỏ NaN (các dòng đầu do tính toán chỉ số kỹ thuật)
    df = df.dropna().reset_index(drop=True)

    # Chỉ giữ lại cột time, OHLCV và 4 chỉ báo kỹ thuật
    required_columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'MACD', 'RSI', 'CCI', 'ADX']
    df = df[required_columns]

    # Lọc từ start_date trở đi
    df = df[df['time'] >= start_date].reset_index(drop=True)

    # Định dạng cột 'time' theo quy ước dd/mm/yyyy
    df['time'] = df['time'].dt.strftime('%d/%m/%Y')

    return df

# Ví dụ sử dụng:
# import pandas as pd
# from data.data_processor.feature_engineer.engineer_stat import add_technical_indicators
#
# df = pd.read_csv('data/data_storer/FPT_detail_2013_01_01_2024_12_31.csv')
# df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y')
# df = df.sort_values('time').reset_index(drop=True)
# start_date = pd.to_datetime('2020-01-01')
# df_with_indicators = add_technical_indicators(df, start_date=start_date, auto_adjust_start_date=True, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14, cci_period=20, adx_period=14)
# print(df_with_indicators.head())
