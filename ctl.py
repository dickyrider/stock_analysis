import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

ticker = 'NVDA'
start_date = '2020-01-01'
end_date = '2025-03-17'

nvda = yf.download("NVDA", start=start_date, end=end_date)
nvda.columns = [col[0] for col in nvda.columns]    
nvda.reset_index(inplace=True)   

def label_trends(prices, threshold=0.05):

    n = len(prices)
    labels = np.zeros(n, dtype=int)
    
    # 初始價格與初始設置
    initial_price = prices[0]
    current_trend = None   # 尚未確定趨勢，初始化為 None
    current_extreme = initial_price
    trend_start = 0        # 確認趨勢的開始下標

    # 判斷第一個趨勢轉折點
    for i in range(1, n):
        if prices[i] >= initial_price * (1 + threshold):
            current_trend = 1
            current_extreme = prices[i]
            trend_start = i
            break
        elif prices[i] <= initial_price * (1 - threshold):
            current_trend = -1
            current_extreme = prices[i]
            trend_start = i
            break

    # 如果後續一直未觸及閾值，全部標記為 0
    if current_trend is None:
        return labels

    # 將 trend_start 之前的標記設為 0（或依需求處理）
    labels[:trend_start] = 0
    # 將第一個確定趨勢點標記
    labels[trend_start] = current_trend

    # 從趨勢確立點後逐日檢查，更新標籤
    for i in range(trend_start + 1, n):
        price = prices[i]
        if current_trend == 1:  # 當前上漲趨勢
            if price > current_extreme * (1 + threshold):
                current_extreme = price
                labels[i] = current_trend
            # 當價格從極值回落達到15%時，轉為下跌趨勢
            if price < current_extreme * (1 - threshold):
                current_trend = -1
                current_extreme = price
                labels[i] = current_trend
        elif current_trend == -1:  # 當前下跌趨勢
            if price < current_extreme * (1 - threshold) :
                current_extreme = price
                labels[i] = current_trend
            # 當價格從極值上升達到15%時，轉為上漲趨勢
            if price > current_extreme * (1 + threshold):
                current_trend = 1
                current_extreme = price
                labels[i] = current_trend
        
    return labels



# 取出收盤價並生成標籤
prices = nvda['Close'].values
labels = label_trends(prices)

# 將標籤加入 DataFrame
nvda['Trend Label'] = labels

# 將標籤向前移一格（用下一日的標籤對應當前日特徵）
nvda['Shifted Trend Label'] = nvda['Trend Label'].shift(-1)

# 輸出最後20筆資料，注意最後一筆因為前移後缺失標籤會是 NaN
print(nvda[['Date', 'Close', 'Trend Label', 'Shifted Trend Label']].tail(20))

# 可視化：原始趨勢標籤與前移後的趨勢標籤
plt.figure(figsize=(12,6))
plt.plot(nvda['Date'], nvda['Close'], label='Close Price', color='blue')

# 標記前移後的標籤（僅顯示非空標籤）
shifted = nvda.dropna(subset=['Shifted Trend Label'])
up_shifted = shifted[shifted['Shifted Trend Label'] == 1]
down_shifted = shifted[shifted['Shifted Trend Label'] == -1]
plt.scatter(up_shifted['Date'], up_shifted['Close'], label='Up Trend (Shifted)', facecolors='none', edgecolors='green', marker='^', s=100)
plt.scatter(down_shifted['Date'], down_shifted['Close'], label='Down Trend (Shifted)', facecolors='none', edgecolors='red', marker='v', s=100)

plt.title(f'{ticker} Close Price with Original and Forward Shifted Trend Labels')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()