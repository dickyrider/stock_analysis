import yfinance as yf
import pandas as pd
from functools import reduce
from technical_indicators_V2 import compute_all_indicators
from continuou_trend_labelling import ctl_label
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

#accessing data
data_path = #path

for filename in os.listdir(data_path):
    if filename[:len('T10Y2Y.csv')] == 'T10Y2Y.csv':
        #10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        T10Y2Y_data = os.path.join(data_path,filename)
    if filename[:len('DFF')] == 'DFF':
        #Effective Federal Funds Rate
        dff_data = os.path.join(data_path,filename)


T10Y2Y_df = pd.read_csv(T10Y2Y_data, parse_dates=["observation_date"])
T10Y2Y_df.rename(columns={"observation_date": "Date"}, inplace=True)
dff_df = pd.read_csv(dff_data, parse_dates=["observation_date"])
dff_df.rename(columns={"observation_date": "Date"}, inplace=True)


#access data from yfinance
start_day = "2012-01-01"
end_day = "2025-03-15"
ticker = 'NVDA'
stock_df = yf.download(ticker, start=start_day, end=end_day)
stock_df.columns = [col[0] for col in stock_df.columns]
stock_df.reset_index(inplace=True)  

etf_tickers = ["DBC", "XLE", "GLD", "TIP",'QCOM']
etf_data = yf.download(etf_tickers, start=start_day, end=end_day)
etf_close_df = etf_data.iloc[:, :len(etf_tickers)]
etf_close_df.columns = etf_close_df.columns.get_level_values(1)
etf_close_df.reset_index(inplace=True) 

spy_df = yf.download('SPY', start=start_day, end=end_day)
spy_df.columns = spy_df.columns.get_level_values(0)
spy_df.reset_index(inplace=True)
spy_df = spy_df.rename(columns={'Close': 'spy_Close'})
spy_dataset_df =  spy_df.iloc[:, :2]

# Calculate moving average and the difference
spy_dataset_df['spy_MA20'] = spy_dataset_df['spy_Close'].rolling(window=20).mean()
spy_dataset_df['spy_MA50'] = spy_dataset_df['spy_Close'].rolling(window=50).mean()
spy_dataset_df['spy_MA200'] = spy_dataset_df['spy_Close'].rolling(window=200).mean()
spy_dataset_df['MA20_50_diff'] = spy_dataset_df['spy_MA20'] - spy_dataset_df['spy_MA50']    
spy_dataset_df['MA50_200_diff'] = spy_dataset_df['spy_MA50'] - spy_dataset_df['spy_MA200'] 


# Merge all dataframe
data_frames = [stock_df, dff_df, T10Y2Y_df, etf_close_df, spy_dataset_df]
merged_data_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="left"), data_frames)

all_data = compute_all_indicators(merged_data_df)

# Label data
labeled_data_df = ctl_label(all_data)
labeled_data_df.set_index('Date', inplace=True)
labeled_data_df.to_csv(#path, index=False)

clean_data_df = labeled_data_df.dropna().copy()

clean_data_df.columns

clean_data_df

features = ['Close', 'High', 'Low', 'Open', 'Volume', 'DFF', 'T10Y2Y',
            'DBC', 'GLD', 'QCOM', 'TIP', 'XLE', 'spy_Close', 'spy_MA20', 
            'spy_MA50', 'spy_MA200', 'MA20_50_diff', 'MA50_200_diff',
            'BB_Middle', 'BB_Std', 'BB_Upper','BB_Lower', 'DEMA', 'SMA',
            'T3', 'WMA', 'Midpoint_over_period','Midpoint_Price', 'ADX',
            'Plus_DI', 'Minus_DI', 'Plus_DM','Minus_DM', 'APO', 'Aroon_Up',
            'Aroon_Down', 'Aroon_Oscillator','BOP', 'CCI', 'CMO', 'Momentum',
            'MACD','MFI', 'PPO', 'ROC','RSI', 'Stoch_%K', 'Stoch_%D','StochRSI_K',
            'StochRSI_D', 'Ultimate_Oscillator','Ultimate_Oscillator_ROC', 'Williams_%R',
            'ATR', 'Chaikin_AD','Hilbert_Amplitude', 'Hilbert_Phase','Hilbert_SineWave',
            'Hilbert_LeadSineWave', 
            ]

features_off = ['Minus_DM','Williams_%R','Plus_DM','Stoch_%K','StochRSI_K','Aroon_Down','ATR','Ultimate_Oscillator_ROC',
            'RSI','ROC','MA20_50_diff','Hilbert_SineWave','StochRSI_D','BB_Std','CMO','DFF'
            ]

target = ['Shifted Trend Label']

def preprocess_data(df):
    # Data cleaning and preprocessing
    df = df.copy()
    
    # Data formatting
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df.dropna()

clean_data_df = preprocess_data(clean_data_df)

#Split data for training
train_size = int(len(clean_data_df)*0.7)
train_data = clean_data_df.iloc[:train_size]
test_data = clean_data_df.iloc[train_size:]

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]


smote = SMOTE(random_state=42, sampling_strategy=0.5, k_neighbors=3)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

X_train, y_train = X_train_resampled, y_train_resampled

#Initialize the XGBClassifier for binary classification.
model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=1,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=0.8,
    eval_metric='logloss',
    tree_method='hist',
    n_estimators=200,
    early_stopping_rounds=50
)



# Set up Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Perform cross-validation 
cv_results = xgb.cv(
    params=model.get_xgb_params(),
    dtrain=xgb.DMatrix(X_train, label=y_train),
    num_boost_round=1000,
    folds=tscv,
    early_stopping_rounds=50,
    verbose_eval=50
)

# Determine the optimal number of boosting rounds (trees) based on the cross-validation results.
best_n_estimators = cv_results.shape[0]
model.set_params(n_estimators=best_n_estimators)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=10
)


# Evaluate Model 

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['downward', 'upward']))

importances = model.feature_importances_

# 將重要性與特徵名稱組成一個 Series，並依照重要性排序
feat_importances = pd.Series(importances, index=X_train.columns)
feat_importances = feat_importances.sort_values(ascending=False)

print("\nFeature Importances:")
print(feat_importances)

# 繪製特徵重要性長條圖
plt.figure(figsize=(10, 6))
feat_importances.plot(kind='bar')
plt.title("Feature Importances")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.tight_layout()
plt.show()
