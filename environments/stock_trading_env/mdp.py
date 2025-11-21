"""
file này định nghĩa MDP cho môi trường giao dịch cổ phiếu sử dụng các chỉ số kỹ thuật MACD, RSI, CCI và ADX.
Các hàm chính:
- get_features: Lấy các đặc trưng chỉ báo kỹ thuật từ một bản ghi.
- update_state: Cập nhật trạng thái dựa trên hành động và bản ghi mới.
- reward: Tính phần thưởng dựa trên trạng thái hiện tại và trạng thái tiếp theo.
- simulate: Mô phỏng quá trình tương tác với môi trường dựa trên chính sách.
- interact_test: Thử nghiệm chính sách trên dữ liệu huấn luyện và kiểm tra.


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StockTradingMDP:
    def __init__(self, balance_init=1000, k=5, min_balance=-100):
        self.balance_init = balance_init  # số dư ban đầu (đơn vị tiền)
        self.k = k  # số lượng cổ phiếu tối đa có thể nắm giữ
        self.min_balance = min_balance  # ngưỡng số dư tối thiểu (tolerance)
        self.A = [a for a in range(-self.k, self.k + 1, 1)]  # hành động

    # chuyển tiếp trạng thái
    def get_features(self, s, new_record):
        """
        Lấy các đặc trưng chỉ báo kỹ thuật từ một bản ghi
        Trả về: [MACD, RSI, CCI, ADX]
        """
        # Các cột: date(0), open(1), high(2), low(3), close(4), volume(5), symbol(6), MACD(7), RSI(8), CCI(9), ADX(10)
        return [
            float(new_record['MACD']),
            float(new_record['RSI']),
            float(new_record['CCI']),
            float(new_record['ADX'])
        ]

    def update_state(self, s, a, new_record):
        # s: (giá, số dư, số cổ phiếu)
        price, balance, shares = s[0], s[1], s[2]
        # a: (0 là giữ, -k là bán, +k là mua)

        # Ràng buộc
        # nếu là bán, kiểm tra có đủ số cổ phiếu để bán
        if a < 0:
            if shares <= abs(a):
                a = -shares
        elif a > 0:  # if buying, check if there are enough balance
            # nếu mua, kiểm tra số dư có đủ theo ngưỡng `min_balance`
            if balance - (a * price) < self.min_balance:
                possible_balance = np.array([balance - (a_ * price) for a_ in range(a)]) >= self.min_balance
                a = np.argmax(possible_balance)
        new_shares = shares + a
        new_balance = balance - (a * price)
        
        # apply fee (approx 0.1%)
        # áp phí giao dịch (xấp xỉ 0.1%)
        new_balance -= (a * price) * 1e-3
        
        # update state: [price, balance, shares, MACD, RSI, CCI, ADX]
        # cập nhật trạng thái: [giá, số dư, số cổ phiếu, MACD, RSI, CCI, ADX]
        features = self.get_features(s, new_record)
        return [float(new_record['close']), float(new_balance), float(new_shares)] + features

    # reward
    def reward(self, s, s_next):
        return (s[1] + s[0] * s[2]) - (s_next[1] + s_next[0] * s_next[2])

    # tương tác
    def simulate(self, series, state_init, pi, greedy, eps=0.2):
        Rs = list()
        actions = list()
        states = [state_init]
        for index, row in series.iterrows():
            a = pi(states[-1], greedy=greedy, eps=eps)
            actions.append(a)
            states.append(self.update_state(states[-1], a, row))
            Rs.append(self.reward(states[-2], states[-1]))
        return states, Rs, actions

    def interact_test(self, pi, train_series, test_series, series_name='test', verbose=True):
        if series_name == 'test':
            series = test_series
            prev_series = train_series
            prev_ind = -1
        elif series_name == 'train':
            series = train_series[1:]
            prev_series = train_series
            prev_ind = 0

        # Khởi tạo trạng thái: [giá, số dư, số cổ phiếu, MACD, RSI, CCI, ADX]
        prev_row = prev_series.iloc[prev_ind]
        state_init = [
            float(prev_row['close']), 
            self.balance_init, 
            0,
            float(prev_row['MACD']),
            float(prev_row['RSI']),
            float(prev_row['CCI']),
            float(prev_row['ADX'])
        ]

        # bắt đầu một chuỗi trạng thái (trajectory)
        states, rewards, actions = self.simulate(series, state_init, pi, True)

        # in chi tiết / trực quan hóa
        portforlio = np.array([s[1] + s[0] * s[2] for s in states])
        if verbose:
            print("Profit at The End of Trajactory:", portforlio[-1] - self.balance_init)

            plt.style.use('dark_background')
            plt.plot(series['close'])
            plt.title("Price")
            plt.xlabel("Time (1 day inter val)")
            plt.ylabel("Price ($)")
            plt.show()

            plt.style.use('dark_background')
            plt.plot([s[2] for s in states])
            plt.title("Number of Shares")
            plt.xlabel("Time (1 day interval)")
            plt.ylabel("Num shares")
            plt.show()

            plt.style.use('dark_background')
            plt.plot(portforlio)
            plt.title("Portfolio ($)")
            plt.xlabel("Time (1 day inter val)")
            plt.ylabel("Portfolio ($)")
            plt.show()

            plt.style.use('dark_background')
            plt.plot(portforlio - self.balance_init)
            plt.title("Trading Profit")
            plt.xlabel("Time (1 day interval)")
            plt.ylabel("Profit ($)")
            plt.show()
        
        return portforlio[-1] - self.balance_init