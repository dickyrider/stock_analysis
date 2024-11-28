import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
import os


sp500_df = yf.download("^GSPC", start='2019-11-11', end='2024-11-08')['Close']
sp500_returns_df = sp500_df.pct_change().dropna()

stock_tickers = [
    'PG',   # Procter & Gamble Co.
    'KO',   # The Coca-Cola Company
    'PEP',  # PepsiCo, Inc.
    'WMT',  # Walmart Inc.
    'COST', # Costco Wholesale Corporation
    'DUK',  # Duke Energy Corporation
    'SO',   # The Southern Company
    'NEE',  # NextEra Energy, Inc.
    'D',    # Dominion Energy, Inc.
    'AEP',  # American Electric Power Company, Inc.
    'T',    # AT&T Inc.
    'VZ',   # Verizon Communications Inc.
    'XOM',  # Exxon Mobil Corporation
    'CVX',  # Chevron Corporation
    'JNJ',  # Johnson & Johnson
    'GOLD', # Barrick Gold Corporation
    'NEM',  # Newmont Corporation 
    'WPM',  # Wheaton Precious Metals Corp.
    'PFE',  # Pfizer Inc.
    'MRK',  # Merck & Co., Inc.
    'ABT',  # Abbott Laboratories
    'UNH'   # UnitedHealth Group Incorporated
]


url = 'https://zh.wikipedia.org/zh-hk/S%26P_500%E6%88%90%E4%BB%BD%E8%82%A1%E5%88%97%E8%A1%A8'

response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')
links = soup.find_all('a', class_='external text')
sp500_lst = []
for link in links[1:]:
    if len(link.text) < 6:
        sp500_lst.append(link.text)
    else:
        break
sp500_lst


data_dict = {}
stock_lst = []

for stock_ticker in sp500_lst:
    stock_df = yf.download(stock_ticker, start='2019-11-11', end='2024-11-08')['Close']
    if len(stock_df) > 1200:
        stock_returns_df = stock_df.pct_change().dropna()  # 计算股票收益率
        data_dict[stock_ticker] = stock_df
        stock_lst.append(stock_ticker)

if data_dict:
    # 将每个 DataFrame 转换为普通列并合并
    stock_data_list = [df for df in data_dict.values()]
    
    # 合并所有 DataFrame
    stock_data = pd.concat(stock_data_list, axis=1)  # 横向合并
    stock_data.columns = [f"{ticker}_Close" for ticker in data_dict.keys()]  # 直接设置列名




# 合并S&P 500和股票数据
stock_data = pd.DataFrame(data_dict)
stock_data


# 合并S&P 500和股票收益率数据
combined_data = pd.concat([sp500_df, stock_data], axis=1).dropna()
combined_data.columns = ['SP500'] + stock_lst[:len(combined_data.columns) - 1]  # 确保
combined_data


# 进行线性回归分析
results = {}
if not combined_data.empty:
    for stock in data_dict.keys():
        if f"{stock}" in combined_data.columns:
            corr, p_value = pearsonr(combined_data['SP500'], combined_data[f"{stock}"])
            r_squared = corr ** 2  # 计算 R 平方值
            results[stock] = {
                'Pearson r': corr,
                'R-squared': r_squared,
                'P-value': p_value
            }
    results_df = pd.DataFrame.from_dict(results, orient='index')


results_df[results_df['R-squared']>0.5]

results_df