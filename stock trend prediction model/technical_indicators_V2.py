import pandas as pd
import numpy as np
from scipy.signal import hilbert

def compute_all_indicators(df):

    def EMA(series, period):
        return series.ewm(span=period, adjust=False).mean()

    def T3(series, period=20, v=0.7):
        e1 = series.ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        e3 = e2.ewm(span=period, adjust=False).mean()
        e4 = e3.ewm(span=period, adjust=False).mean()
        e5 = e4.ewm(span=period, adjust=False).mean()
        e6 = e5.ewm(span=period, adjust=False).mean()
        c1 = -v**3
        c2 = 3 * v**2 + 3 * v**3
        c3 = -6 * v**2 - 3 * v - 3 * v**3
        c4 = 1 + 3 * v + 3 * v**2 + v**3
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    def WMA(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
        )

    def ADX_func(df_local, period=14):
        df_local = df_local.copy()
        df_local['UpMove'] = df_local['High'] - df_local['High'].shift(1)
        df_local['DownMove'] = df_local['Low'].shift(1) - df_local['Low']
        df_local['+DM'] = np.where(
            (df_local['UpMove'] > df_local['DownMove']) & (df_local['UpMove'] > 0),
            df_local['UpMove'],
            0
        )
        df_local['-DM'] = np.where(
            (df_local['DownMove'] > df_local['UpMove']) & (df_local['DownMove'] > 0),
            df_local['DownMove'],
            0
        )
        tr1 = df_local['High'] - df_local['Low']
        tr2 = np.abs(df_local['High'] - df_local['Close'].shift(1))
        tr3 = np.abs(df_local['Low'] - df_local['Close'].shift(1))
        df_local['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_local['ATR'] = df_local['TR'].rolling(window=period).mean()
        df_local['+DI'] = 100 * (df_local['+DM'].rolling(window=period).sum() / df_local['ATR'].rolling(window=period).sum())
        df_local['-DI'] = 100 * (df_local['-DM'].rolling(window=period).sum() / df_local['ATR'].rolling(window=period).sum())
        df_local['DX'] = 100 * np.abs(df_local['+DI'] - df_local['-DI']) / (df_local['+DI'] + df_local['-DI'])
        adx = df_local['DX'].rolling(window=period).mean()
        return adx, df_local['+DI'], df_local['-DI'], df_local['+DM'], df_local['-DM']

    def CCI_func(df_local, period=20):
        TP = (df_local['High'] + df_local['Low'] + df_local['Close']) / 3
        sma = TP.rolling(window=period).mean()
        mad = TP.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (TP - sma) / (0.015 * mad)

    def CMO_func(series, period=14):
        diff = series.diff()
        up = diff.copy()
        up[up < 0] = 0
        down = -diff.copy()
        down[down < 0] = 0
        sum_up = up.rolling(window=period).sum()
        sum_down = down.rolling(window=period).sum()
        return 100 * (sum_up - sum_down) / (sum_up + sum_down)

    def RSI_func(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        RS = avg_gain / avg_loss
        return 100 - (100 / (1 + RS))

    def ultimate_oscillator_func(df_local, short=7, medium=14, long=28):
        ult = (4 * df_local['Close'].diff(short) +
               2 * df_local['Close'].diff(medium) +
               df_local['Close'].diff(long)) / 7
        return ult

    def williams_R_func(df_local, period=14):
        high_max = df_local['High'].rolling(window=period).max()
        low_min = df_local['Low'].rolling(window=period).min()
        return -100 * (high_max - df_local['Close']) / (high_max - low_min)

    def ATR_func(df_local, period=14):
        tr1 = df_local['High'] - df_local['Low']
        tr2 = np.abs(df_local['High'] - df_local['Close'].shift(1))
        tr3 = np.abs(df_local['Low'] - df_local['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def chaikin_ad_func(df_local):
        mf_multiplier = ((df_local['Close'] - df_local['Low']) - (df_local['High'] - df_local['Close'])) / \
                        (df_local['High'] - df_local['Low']).replace(0, np.nan)
        mf_volume = mf_multiplier * df_local['Volume']
        return mf_volume.cumsum()

    def hilbert_transform_func(series):
        analytic_signal = hilbert(series.fillna(0))
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        return amplitude_envelope, instantaneous_phase

    def MFI_func(df_local, period=14):
        TP = (df_local['High'] + df_local['Low'] + df_local['Close']) / 3
        MF = TP * df_local['Volume']
        delta = TP.diff()
        pos_mf = np.where(delta > 0, MF, 0)
        neg_mf = np.where(delta < 0, MF, 0)
        pos_mf = pd.Series(pos_mf, index=df_local.index)
        neg_mf = pd.Series(neg_mf, index=df_local.index)
        pos_mf_sum = pos_mf.rolling(window=period).sum()
        neg_mf_sum = neg_mf.rolling(window=period).sum().replace(0, np.nan)
        mfr = pos_mf_sum / neg_mf_sum
        return 100 - (100 / (1 + mfr))

    df = df.copy()

    # 1. Trend and Range Indicators ---------------------------
    # Bollinger Bands (calculated using the 'Close' price, window=20, multiplier=2)
    window_bb = 20
    mult = 2
    df['BB_Middle'] = df['Close'].rolling(window=window_bb).mean()
    df['BB_Std'] = df['Close'].rolling(window=window_bb).std()
    df['BB_Upper'] = df['BB_Middle'] + mult * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - mult * df['BB_Std']

    period_dema = 20
    ema_series = EMA(df['Close'], period_dema)
    ema_of_ema = EMA(ema_series, period_dema)
    df['DEMA'] = 2 * ema_series - ema_of_ema

    df['SMA'] = df['Close'].rolling(window=period_dema).mean()
    df['T3'] = T3(df['Close'], period=20, v=0.7)
    df['TMA'] = df['Close'].rolling(window=period_dema).mean().rolling(window=period_dema).mean()
    df['WMA'] = WMA(df['Close'], period=20)

    # Midpoint Indicators
    df['Midpoint_over_period'] = (df['Close'].rolling(window=period_dema).max() +
                                  df['Close'].rolling(window=period_dema).min()) / 2
    df['Midpoint_Price'] = (df['High'] + df['Low']) / 2
    df['Average_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['Median_Price'] = (df['High'] + df['Low']) / 2
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Weighted_Close_Price'] = (df['High'] + df['Low'] + 2 * df['Close']) / 4

    # 2. Momentum, Oscillation, and Trend Reversal Indicators ---------
    adx, plus_DI, minus_DI, plus_DM, minus_DM = ADX_func(df, period=14)
    df['ADX'] = adx
    df['Plus_DI'] = plus_DI
    df['Minus_DI'] = minus_DI
    df['Plus_DM'] = plus_DM
    df['Minus_DM'] = minus_DM
    df['ADXR'] = (df['ADX'] + df['ADX'].shift(14)) / 2

    fast_period = 12
    slow_period = 26
    ema_fast = EMA(df['Close'], fast_period)
    ema_slow = EMA(df['Close'], slow_period)
    df['APO'] = ema_fast - ema_slow

    # Aroon Indicators (period=25)
    def aroon_calculation(series_high, series_low, period=25):
        aroon_up = series_high.rolling(window=period + 1).apply(
            lambda x: 100 * (len(x) - 1 - np.argmax(x)) / (len(x) - 1), raw=True)
        aroon_down = series_low.rolling(window=period + 1).apply(
            lambda x: 100 * (len(x) - 1 - np.argmin(x)) / (len(x) - 1), raw=True)
        return aroon_up, aroon_down

    df['Aroon_Up'], df['Aroon_Down'] = aroon_calculation(df['High'], df['Low'], period=25)
    df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']

    df['BOP'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan)
    df['CCI'] = CCI_func(df, period=20)
    df['CMO'] = CMO_func(df['Close'], period=14)
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = EMA(df['MACD'], 9)
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['MFI'] = MFI_func(df, period=14)
    df['PPO'] = 100 * (ema_fast - ema_slow) / ema_slow

    df['ROC'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
    df['ROC_Percentage'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['ROC_Ratio'] = df['Close'] / df['Close'].shift(1)
    df['ROC_Ratio_100'] = (df['Close'] / df['Close'].shift(1)) * 100

    df['RSI'] = RSI_func(df['Close'], period=14)

    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()
    df['Stoch_%K'] = stoch_k
    df['Stoch_%D'] = stoch_d
    df['Stoch_%K_High'] = stoch_k.rolling(window=14).max()
    df['Stoch_%K_Low'] = stoch_k.rolling(window=14).min()

    # Stochastic RSI (first compute RSI)
    rsi_series = RSI_func(df['Close'], period=14)
    min_rsi = rsi_series.rolling(window=14).min()
    max_rsi = rsi_series.rolling(window=14).max()
    stoch_rsi = (rsi_series - min_rsi) / (max_rsi - min_rsi)
    df['StochRSI_K'] = stoch_rsi.rolling(window=3).mean()
    df['StochRSI_D'] = df['StochRSI_K'].rolling(window=3).mean()

    df['Ultimate_Oscillator'] = ultimate_oscillator_func(df, short=7, medium=14, long=28)
    df['Ultimate_Oscillator_ROC'] = df['Ultimate_Oscillator'].diff(1)
    df["Williams_%R"] = williams_R_func(df, period=14)

    # 3. Volatility and Range Width Indicators --------------------
    df['ATR'] = ATR_func(df, period=14)
    tr1 = df['High'] - df['Low']
    tr2 = np.abs(df['High'] - df['Close'].shift(1))
    tr3 = np.abs(df['Low'] - df['Close'].shift(1))
    df["True_Range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["Normalized_ATR"] = df["ATR"] / df["Close"]

    # 4. Money Flow and Accumulation Indicators --------------------
    df['Chaikin_AD'] = chaikin_ad_func(df)
    df['Chaikin_AD_Fast'] = EMA(df['Chaikin_AD'], 3)
    df['Chaikin_AD_Slow'] = EMA(df['Chaikin_AD'], 10)
    df['Chaikin_AD_Osc'] = df['Chaikin_AD_Fast'] - df['Chaikin_AD_Slow']

    # 5. Hilbert Transform Series ----------------
    hilb_amp, hilb_phase = hilbert_transform_func(df['Close'])
    df['Hilbert_Amplitude'] = hilb_amp
    df['Hilbert_Phase'] = hilb_phase
    df['Hilbert_Dominant_Cycle_Period'] = df['Hilbert_Phase'].diff().abs()
    df['Hilbert_SineWave'] = np.sin(df['Hilbert_Phase'])
    df['Hilbert_LeadSineWave'] = np.sin(df['Hilbert_Phase'] + np.pi / 4)  # shifted ahead by 45°
    df['Hilbert_Trend_vs_Cycle'] = np.where(df['Hilbert_SineWave'] > df['Hilbert_LeadSineWave'], 1, 0)

    return df