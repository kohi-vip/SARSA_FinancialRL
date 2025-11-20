import pandas as pd

# Load the CSV file
file_path = 'data/FPT_detail_2013_01_01_2024_12_31.csv'
df = pd.read_csv(file_path)

# Convert 'time' column to datetime and then to DD-MM-YYYY format
df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%d-%m-%Y')

# Save back to CSV
df.to_csv(file_path, index=False)

print("Đã sửa cột 'time' thành format ngày-tháng-năm (DD-MM-YYYY).")