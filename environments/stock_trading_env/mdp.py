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
        return (s_next[1] + s_next[0] * s_next[2]) - (s[1] + s[0] * s[2])

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

    def plot_trading_history(self, states, actions, prices, title="Trading History"):
        """
        Vẽ biểu đồ lịch sử giao dịch với 4 subplots:
        1. Giá cổ phiếu và điểm mua/bán
        2. Số lượng cổ phiếu nắm giữ
        3. Số dư tài khoản
        4. Tổng giá trị portfolio
        
        Args:
            states: list of states [price, balance, shares, MACD, RSI, CCI, ADX]
            actions: list of actions
            prices: array of prices
            title: title for the plot
        """
        # Robust plotting: derive price sequence from `states` to ensure alignment
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Basic validation
        if not isinstance(states, (list, tuple)) or len(states) == 0:
            raise ValueError("`states` must be a non-empty list of state vectors")

        # Derive series from states to guarantee consistent lengths and alignment
        state_prices = [float(s[0]) for s in states]
        balances = [float(s[1]) for s in states]
        shares = [float(s[2]) for s in states]
        portfolio = [balances[i] + state_prices[i] * shares[i] for i in range(len(states))]

        x = np.arange(len(state_prices))

        # 1. Price chart with buy/sell markers. Actions correspond to transitions
        # from state[i] -> state[i+1], so plot action markers at index i+1
        axes[0].plot(x, state_prices, label='Price', color='blue', linewidth=1.5)

        # compute action x positions (shift by +1 to align with next state/price)
        n_actions = len(actions) if actions is not None else 0
        action_positions = (np.arange(n_actions) + 1).tolist()

        buy_positions = [pos for idx, pos in enumerate(action_positions) if actions[idx] > 0]
        sell_positions = [pos for idx, pos in enumerate(action_positions) if actions[idx] < 0]

        if buy_positions:
            axes[0].scatter(buy_positions, [state_prices[pos] for pos in buy_positions],
                            color='green', marker='^', s=80, label='Buy', alpha=0.8, zorder=5)
        if sell_positions:
            axes[0].scatter(sell_positions, [state_prices[pos] for pos in sell_positions],
                            color='red', marker='v', s=80, label='Sell', alpha=0.8, zorder=5)

        axes[0].set_title(f'{title} - Price & Trading Signals', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (days)')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # 2. Shares held
        axes[1].plot(x, shares, label='Shares Held', color='orange', linewidth=2)
        axes[1].fill_between(x, shares, alpha=0.3, color='orange')
        axes[1].set_title('Shares Held Over Time', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time (days)')
        axes[1].set_ylabel('Number of Shares')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        # 3. Balance
        axes[2].plot(x, balances, label='Cash Balance', color='purple', linewidth=2)
        axes[2].axhline(y=self.balance_init, color='gray', linestyle='--',
                       label=f'Initial Balance (${self.balance_init})', alpha=0.7)
        axes[2].set_title('Cash Balance Over Time', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (days)')
        axes[2].set_ylabel('Balance ($)')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        # 4. Portfolio value
        axes[3].plot(x, portfolio, label='Portfolio Value', color='green', linewidth=2)
        axes[3].axhline(y=self.balance_init, color='gray', linestyle='--',
                       label=f'Initial Value (${self.balance_init})', alpha=0.7)

        profit_mask = np.array([p >= self.balance_init for p in portfolio])
        loss_mask = ~profit_mask

        axes[3].fill_between(x, portfolio, self.balance_init,
                            where=profit_mask, alpha=0.3, color='green', label='Profit')
        axes[3].fill_between(x, portfolio, self.balance_init,
                            where=loss_mask, alpha=0.3, color='red', label='Loss')

        axes[3].set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Time (days)')
        axes[3].set_ylabel('Portfolio Value ($)')
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

        return fig, axes