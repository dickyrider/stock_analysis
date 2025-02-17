import yfinance as yf
import numpy as np
import pandas as pd
from technical_indicators_V2 import compute_all_indicators
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib


#Use SP500 data for training
data = yf.download("^GSPC", start="2015-01-01", end="2025-01-01")

#Columns formatting
data.columns = [col[0] for col in data.columns]

#Add technical indicators into data
ta_data = compute_all_indicators(data)

#Adding target feature
ta_data['Typical_Price'] = (ta_data['High'] + ta_data['Low'] + ta_data['Close']) / 3
ta_data['Pct_Change'] = ta_data['Typical_Price'].pct_change()
ta_data['Downward_Trend'] = (ta_data['Pct_Change'].shift(-1) < -0.001).astype(int)



data_clean = ta_data.dropna().copy()


features = ['Adj Close', 'High', 'Low', 'Open', 'Volume', 'BB_Middle', 'BB_Std', 'BB_Upper',
       'BB_Lower', 'DEMA', 'SMA', 'T3', 'WMA', 'Midpoint_over_period',
       'Midpoint_Price', 'ADX', 'Plus_DI', 'Minus_DI', 'Plus_DM',
       'Minus_DM', 'APO', 'Aroon_Up', 'Aroon_Down', 'Aroon_Oscillator',
       'BOP', 'CCI', 'CMO', 'Momentum', 'MACD','MFI', 'PPO', 'ROC','RSI', 
       'Stoch_%K', 'Stoch_%D','StochRSI_K', 'StochRSI_D', 'Ultimate_Oscillator',
       'Ultimate_Oscillator_ROC', 'Williams_%R', 'ATR', 'Chaikin_AD',
       'Hilbert_Amplitude', 'Hilbert_Phase',
       'Hilbert_SineWave','Hilbert_LeadSineWave', 'Hilbert_Trend_vs_Cycle'
       ]


target = ['Downward_Trend']

def preprocess_data(df):
    # Data cleaning and preprocessing
    df = df.copy()
    
    # Data formatting
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df.dropna()

data_clean = preprocess_data(data_clean)

data_clean

#Split data for training
train_size = int(len(data_clean)*0.7)
train_data = data_clean.iloc[:train_size]
test_data = data_clean.iloc[train_size:]

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]


#Initialize the XGBClassifier for binary classification.
model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=1.5,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=1,
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

print("\nClassification report：")
print(classification_report(y_test, y_pred, target_names=['normal', 'downward']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
