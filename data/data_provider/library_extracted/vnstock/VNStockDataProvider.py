import pandas as pd
from vnstock import Vnstock
from vnstock import Listing, Quote, Company, Finance, Trading, Screener

class VNStockDataProvider:
    """
    Lớp toàn diện để xử lý dữ liệu VNStock, bao gồm lấy dữ liệu OHLCV
    và lọc cho các mã cổ phiếu đã chọn.

    Lớp này tích hợp thư viện vnstock để lấy dữ liệu và cung cấp
    các tùy chọn lọc linh hoạt cho các thí nghiệm tài chính.
    """

    def __init__(self, source='VCI'):
        """
        Khởi tạo VNStockDataProvider.

        Tham số:
        - source (str): Nguồn dữ liệu ('VCI' hoặc 'TCBS').
        """
        self.source = source
        self.vnstock = Vnstock()

    def get_ohlcv_data(self, symbols, start_date, end_date, interval='1D', verbose=True):
        """
        Lấy dữ liệu OHLCV cho các mã đã chỉ định từ vnstock bằng lớp Quote.

        Tham số:
        - symbols (list hoặc str): Danh sách mã cổ phiếu hoặc chuỗi mã đơn.
        - start_date (str): Ngày bắt đầu theo định dạng 'YYYY-MM-DD'.
        - end_date (str): Ngày kết thúc theo định dạng 'YYYY-MM-DD'.
        - interval (str): Khoảng thời gian dữ liệu ('1D' cho hàng ngày, '1W' cho hàng tuần, v.v.).
        - verbose (bool): Có in thông tin tiến trình hay không.

        Trả về:
        - pd.DataFrame: DataFrame OHLCV kết hợp cho tất cả mã với cột [date, open, high, low, close, volume, symbol].
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = []
        failed_symbols = []

        if verbose:
            print("="*80)
            print("LẤY DỮ LIỆU OHLCV TỪ VNSTOCK")
            print("="*80)
            print(f"📋 Danh sách mã: {symbols}")
            print(f"📅 Khoảng thời gian: {start_date} đến {end_date}")
            print(f"⏱️  Phiên: {interval}")
            print(f"📡 Nguồn: {self.source}")
            print("-"*80)

        for symbol in symbols:
            try:
                if verbose:
                    print(f"📡 Đang lấy dữ liệu cho {symbol}...")
                # Sử dụng instance Vnstock stock để lấy dữ liệu lịch sử
                stock = self.vnstock.stock(symbol=symbol, source=self.source)
                data = stock.quote.history(start=start_date, end=end_date, interval=interval)
                data['symbol'] = symbol  # Thêm cột symbol
                all_data.append(data)
                if verbose:
                    print(f"✅ {symbol}: {len(data)} dòng")
            except Exception as e:
                failed_symbols.append(symbol)
                if verbose:
                    print(f"❌ {symbol}: Lỗi - {str(e)}")

        if not all_data:
            if verbose:
                print("⚠️  Không có dữ liệu nào được lấy thành công!")
            return pd.DataFrame()

        combined_data = pd.concat(all_data, ignore_index=True)

        # Đảm bảo cột date là datetime và định dạng thành 'date' theo yyyy/mm/dd
        if 'time' in combined_data.columns:
            combined_data['date'] = pd.to_datetime(combined_data['time']).dt.strftime('%Y/%m/%d')
            combined_data.drop('time', axis=1, inplace=True)
        elif 'Date' in combined_data.columns:
            combined_data['date'] = pd.to_datetime(combined_data['Date']).dt.strftime('%Y/%m/%d')
            combined_data.drop('Date', axis=1, inplace=True)
        else:
            # Giả sử cột đầu tiên là date
            date_col = combined_data.columns[0]
            combined_data['date'] = pd.to_datetime(combined_data[date_col]).dt.strftime('%Y/%m/%d')
            combined_data.drop(date_col, axis=1, inplace=True)

        # Sắp xếp lại cột thành [date, open, high, low, close, volume, symbol]
        combined_data = combined_data[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

        # Sắp xếp theo symbol và date
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)

        if verbose:
            print("\n" + "="*80)
            print("KẾT QUẢ")
            print("="*80)
            print(f"✅ Tổng số dòng dữ liệu: {len(combined_data):,}")
            print(f"✅ Số mã thành công: {len(symbols) - len(failed_symbols)}")
            if failed_symbols:
                print(f"❌ Mã thất bại ({len(failed_symbols)}): {', '.join(failed_symbols)}")

            print(f"\n📋 Các cột dữ liệu:")
            for col in combined_data.columns:
                print(f"   • {col}")

            print(f"\n📊 Thống kê:")
            print(f"   • Số mã: {combined_data['symbol'].nunique()}")
            print(f"   • Khoảng thời gian: {pd.to_datetime(combined_data['time']).min().date()} đến {pd.to_datetime(combined_data['time']).max().date()}")
            print(f"   • Khoảng thời gian: {pd.to_datetime(combined_data['date']).min().date()} đến {pd.to_datetime(combined_data['date']).max().date()}")

            print(f"\n📄 Mẫu dữ liệu (10 dòng đầu):")
            print(combined_data.head(10))

            print(f"\n💾 Dữ liệu đã sẵn sàng")
            print(f"   Shape: {combined_data.shape} (rows, columns)")
            print(f"   Memory usage: {combined_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            print("\n✅ HOÀN TẤT!")

        return combined_data

    def filter_selected_ohlcv(self, ohlcv_data, selected_symbols=None, verbose=True):
        """
        Filter OHLCV data for selected stock symbols.

        Parameters:
        - ohlcv_data (pd.DataFrame): The OHLCV DataFrame (from get_ohlcv_data or external).
        - selected_symbols (list): List of stock symbols to filter. If None, uses default 10 symbols.
        - verbose (bool): Whether to print detailed information.

        Returns:
        - pd.DataFrame: Filtered DataFrame with selected symbols.
        """
        if selected_symbols is None:
            selected_symbols = ['ACB', 'BCM', 'BVH', 'FPT', 'GAS', 'GVR', 'HPG', 'MSN', 'MWG', 'SSI']

        if verbose:
            print("="*80)
            print("LỌC DỮ LIỆU OHLCV CHO CÁC MÃ CỔ PHIẾU ĐÃ CHỌN")
            print("="*80)
            print(f"📋 Danh sách: {selected_symbols}")
            print(f"📊 Số mã cổ phiếu: {len(selected_symbols)}")
            print("-"*80)

        if ohlcv_data.empty:
            if verbose:
                print("⚠️  Không có dữ liệu để lọc!")
            return pd.DataFrame()

        # Lọc dữ liệu cho các mã đã chọn
        selected_ohlcv = ohlcv_data[ohlcv_data['symbol'].isin(selected_symbols)].copy()

        # Sắp xếp theo symbol và date
        selected_ohlcv = selected_ohlcv.sort_values(['symbol', 'date']).reset_index(drop=True)

        if verbose:
            print("\n" + "="*80)
            print("KẾT QUẢ")
            print("="*80)
            print(f"✅ Tổng số dòng dữ liệu: {len(selected_ohlcv):,}")
            print(f"✅ Số mã trong DataFrame: {selected_ohlcv['symbol'].nunique()}")

            # Kiểm tra mã nào có/không có
            found_symbols = selected_ohlcv['symbol'].unique().tolist()
            missing_symbols = [s for s in selected_symbols if s not in found_symbols]

            if missing_symbols:
                print(f"⚠️  Các mã không có dữ liệu ({len(missing_symbols)}): {', '.join(missing_symbols)}")

            # Thống kê chi tiết
            print(f"\n📋 Các cột dữ liệu:")
            for col in selected_ohlcv.columns:
                print(f"   • {col}")

            print(f"\n📊 Thống kê:")
            print(f"   • Số mã: {selected_ohlcv['symbol'].nunique()}")
            print(f"   • Khoảng thời gian: {selected_ohlcv['date'].min().date()} đến {selected_ohlcv['date'].max().date()}")
            print(f"   • Trung bình ngày/mã: {len(selected_ohlcv) / selected_ohlcv['symbol'].nunique():.0f}")

            # Thống kê theo từng mã
            print(f"\n📈 Số ngày giao dịch theo từng mã:")
            symbol_counts_selected = selected_ohlcv['symbol'].value_counts().sort_index()
            for symbol, count in symbol_counts_selected.items():
                print(f"   {symbol}: {count:,} ngày")

            # Hiển thị mẫu dữ liệu
            print(f"\n📄 Mẫu dữ liệu OHLCV (10 dòng đầu):")
            print(selected_ohlcv.head(10))

            print(f"\n📄 Mẫu dữ liệu OHLCV (10 dòng cuối):")
            print(selected_ohlcv.tail(10))

            print(f"\n💾 Dữ liệu đã lọc")
            print(f"   Shape: {selected_ohlcv.shape} (rows, columns)")
            print(f"   Memory usage: {selected_ohlcv.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            print("\n✅ HOÀN TẤT!")

        return selected_ohlcv

    def get_and_filter_ohlcv(self, symbols, start_date, end_date, interval='1D', selected_symbols=None, verbose=True):
        """
        Combined method: Fetch OHLCV data and filter for selected symbols.

        Parameters:
        - symbols (list or str): Symbols to fetch data for.
        - start_date (str): Start date.
        - end_date (str): End date.
        - interval (str): Data interval.
        - selected_symbols (list): Symbols to filter after fetching. If None, uses default.
        - verbose (bool): Verbosity.

        Returns:
        - pd.DataFrame: Filtered OHLCV data.
        """
        ohlcv_data = self.get_ohlcv_data(symbols, start_date, end_date, interval, verbose)
        if ohlcv_data.empty:
            return pd.DataFrame()
        return self.filter_selected_ohlcv(ohlcv_data, selected_symbols, verbose)