import pandas as pd
from vnstock import Vnstock
from vnstock import Listing, Quote, Company, Finance, Trading, Screener

class VNStockDataProcessor:
    """
    A comprehensive class for processing VNStock data, including fetching OHLCV data
    and filtering for selected stock symbols.

    This class integrates vnstock library for data retrieval and provides
    flexible filtering options for financial experiments.
    """

    def __init__(self, source='VCI'):
        """
        Initialize the VNStockDataProcessor.

        Parameters:
        - source (str): Data source ('VCI' or 'TCBS').
        """
        self.source = source
        self.vnstock = Vnstock()

    def get_ohlcv_data(self, symbols, start_date, end_date, interval='1D', verbose=True):
        """
        Fetch OHLCV data for specified symbols from vnstock using Quote class.

        Parameters:
        - symbols (list or str): List of stock symbols or single symbol string.
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        - interval (str): Data interval ('1D' for daily, '1W' for weekly, etc.).
        - verbose (bool): Whether to print progress information.

        Returns:
        - pd.DataFrame: Combined OHLCV DataFrame for all symbols.
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
                # Use Vnstock stock instance for historical data
                stock = self.vnstock.stock(symbol=symbol, source=self.source)
                data = stock.quote.history(start=start_date, end=end_date, interval=interval)
                data['symbol'] = symbol  # Add symbol column
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

        # Ensure date column is datetime
        if 'time' in combined_data.columns:
            combined_data['date'] = pd.to_datetime(combined_data['time'])
        elif 'Date' in combined_data.columns:
            combined_data['date'] = pd.to_datetime(combined_data['Date'])
        else:
            # Assume first column is date
            date_col = combined_data.columns[0]
            combined_data['date'] = pd.to_datetime(combined_data[date_col])

        # Sort by symbol and date
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
            print(f"   • Khoảng thời gian: {combined_data['date'].min().date()} đến {combined_data['date'].max().date()}")
            print(f"   • Trung bình ngày/mã: {len(combined_data) / combined_data['symbol'].nunique():.0f}")

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

    def get_ohlcv_data(self, symbols, start_date, end_date, interval='1D', verbose=True):
        """
        Fetch OHLCV data for specified symbols from vnstock.

        Parameters:
        - symbols (list or str): List of stock symbols or single symbol string.
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        - interval (str): Data interval ('1D' for daily, '1W' for weekly, etc.).
        - verbose (bool): Whether to print progress information.

        Returns:
        - pd.DataFrame: Combined OHLCV DataFrame for all symbols.
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
            print("-"*80)

        for symbol in symbols:
            try:
                if verbose:
                    print(f"📡 Đang lấy dữ liệu cho {symbol}...")
                data = self.trading.stock_historical_data(symbol, start_date, end_date, interval=interval)
                data['symbol'] = symbol  # Add symbol column
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

        # Ensure date column is datetime
        if 'time' in combined_data.columns:
            combined_data['date'] = pd.to_datetime(combined_data['time'])
        elif 'Date' in combined_data.columns:
            combined_data['date'] = pd.to_datetime(combined_data['Date'])
        else:
            # Assume first column is date
            date_col = combined_data.columns[0]
            combined_data['date'] = pd.to_datetime(combined_data[date_col])

        # Sort by symbol and date
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
            print(f"   • Khoảng thời gian: {combined_data['date'].min().date()} đến {combined_data['date'].max().date()}")
            print(f"   • Trung bình ngày/mã: {len(combined_data) / combined_data['symbol'].nunique():.0f}")

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