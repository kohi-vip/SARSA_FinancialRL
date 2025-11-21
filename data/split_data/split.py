"""
File này cho việc chia dữ liệu thành tập huấn luyện, kiểm tra và đánh giá.
Các hàm chính:
- split_data_two_parts: Chia dữ liệu thành hai phần (huấn luyện và kiểm tra).
- split_data_three_parts: Chia dữ liệu thành ba phần (huấn luyện, kiểm tra và đánh giá).
Input: DataFrame với cột 'time' (string dd/mm/yyyy), OHLCV và các chỉ báo kỹ thuật.
Output: Các DataFrame tương ứng với các phần đã chia.
"""

import pendulum
import pandas as pd

def split_data_two_parts(df, train_portion=0.75):
    """
    Chia dữ liệu thành hai phần: train và test.
    
    Parameters:
    - df: DataFrame với cột 'time' (string dd/mm/yyyy), OHLCV và các chỉ báo kỹ thuật.
    - train_portion: Tỷ lệ phần train (float, default 0.75).
    
    Returns:
    - train_df: DataFrame cho phần train.
    - test_df: DataFrame cho phần test.
    """
    # Sử dụng cột 'time' để parse timestamp
    if 'time' in df.columns:
        first_time = df['time'].iloc[0]
        if isinstance(first_time, str):
            try:
                # Strip possible brackets from string
                df = df.copy()
                df['time'] = df['time'].str.strip('[]')
                # Chuyển string dd/mm/yyyy thành datetime, rồi lấy timestamp
                dt_series = pd.to_datetime(df['time'], format='%d/%m/%Y', errors='coerce')
                if dt_series.isna().any():
                    invalid = df['time'][dt_series.isna()].head()
                    raise ValueError(f"Không thể parse một số giá trị 'time': {invalid.tolist()}")
                dt_list = dt_series.astype(int) / 10**9  # timestamp float
            except Exception as e:
                print(f"Lỗi parse string time: {e}. Giá trị mẫu sau strip: {df['time'].head()}")
                raise
        elif hasattr(first_time, 'timestamp'):  # datetime object
            dt_list = [dt.timestamp() for dt in df['time']]
        else:
            raise ValueError("Cột 'time' phải là string dd/mm/yyyy hoặc datetime object.")
    else:
        raise ValueError("DataFrame phải có cột 'time' (string dd/mm/yyyy hoặc datetime).")
    
    # Tính chỉ số chia
    train_ind = int(len(df) * train_portion)
    
    # Chia DataFrame
    train_df = df.iloc[:train_ind].copy()
    test_df = df.iloc[train_ind:].copy()
    
    return train_df, test_df
