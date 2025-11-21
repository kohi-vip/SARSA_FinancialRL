# Quản lý rủi ro / Đầu tư
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

balance_init = 1000  # số dư ban đầu (đơn vị tiền)
k = 5  # số lượng cổ phiếu tối đa có thể nắm giữ
min_balance = -100  # ngưỡng số dư tối thiểu (tolerance)

# hành động
A = [a for a in range(-k, k+1, 1)]

# chuyển tiếp trạng thái
def get_features(s, new_record):
    """
    Lấy các đặc trưng chỉ báo kỹ thuật từ một bản ghi
    Trả về: [macd, rsi, cci, adx]
    """
    # Các cột: date(0), open(1), high(2), low(3), close(4), volume(5), symbol(6), macd(7), rsi(8), cci(9), adx(10)
    return [
        float(new_record['macd']),
        float(new_record['rsi']),
        float(new_record['cci']),
        float(new_record['adx'])
    ]

def update_state(s, a, new_record):
    # s: (giá, số dư, số cổ phiếu)
    price, balance, shares = s[0], s[1], s[2]
    # a: (0 là giữ, -k là bán, +k là mua)

    # Ràng buộc
    # nếu là bán, kiểm tra có đủ số cổ phiếu để bán
    if a < 0:
        if shares <= abs(a):
            a = -shares
    elif a > 0: # if buying, check if there are enough balance
        # nếu mua, kiểm tra số dư có đủ theo ngưỡng `min_balance`
        if balance - (a * price) < min_balance:
            possible_balance = np.array([balance - (a_ * price) for a_ in range(a)]) >= min_balance
            a = np.argmax(possible_balance)
    new_shares = shares + a
    new_balance = balance - (a * price)
    
    # apply fee (approx 0.1%)
    # áp phí giao dịch (xấp xỉ 0.1%)
    new_balance -= (a * price) * 1e-3
    
    # update state: [price, balance, shares, macd, rsi, cci, adx]
    # cập nhật trạng thái: [giá, số dư, số cổ phiếu, macd, rsi, cci, adx]
    features = get_features(s, new_record)
    return [float(new_record['close']), float(new_balance), float(new_shares)] + features

# reward
def reward(s, s_next):
    return (s[1] + s[0]*s[2]) - (s_next[1] + s_next[0]*s_next[2])

# tương tác
def simulate(series, state_init, pi, greedy, eps=0.2):
    Rs = list()
    actions = list()
    states = [state_init]
    for index, row in series.iterrows():
        a = pi(states[-1], greedy=greedy, eps=eps)
        actions.append(a)
        states.append(update_state(states[-1], a, row))
        Rs.append(reward(states[-2], states[-1]))
    return states, Rs, actions

def interact_test(pi, series_name='test', verbose=True):
    if series_name == 'test':
        series = test_series
        prev_series = train_series
        prev_ind = -1
    elif series_name == 'train':
        series = train_series[1:]
        prev_series = train_series
        prev_ind = 0

    # Khởi tạo trạng thái: [giá, số dư, số cổ phiếu, macd, rsi, cci, adx]
    prev_row = prev_series.iloc[prev_ind]
    state_init = [
        float(prev_row['close']), 
        balance_init, 
        0,
        float(prev_row['macd']),
        float(prev_row['rsi']),
        float(prev_row['cci']),
        float(prev_row['adx'])
    ]

    # bắt đầu một chuỗi trạng thái (trajectory)
    states, rewards, actions = simulate(series, state_init, pi, True)

    # in chi tiết / trực quan hóa
    portforlio = np.array([s[1] + s[0]*s[2] for s in states])
    if verbose:
        print("Profit at The End of Trajactory:", portforlio[-1] - balance_init)

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
        plt.plot(portforlio - balance_init)
        plt.title("Trading Profit")
        plt.xlabel("Time (1 day interval)")
        plt.ylabel("Profit ($)")
        plt.show()
    
    return portforlio[-1] - balance_init