# stacking_prototype.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from utils.data_utils import load_m5_data
from models.base.sarimax_model import SARIMAXModel
from models.base.lstm_model import LSTMModel
from models.base.lightgbm_model import LightGBMModel

# Load full data and define CV
item, store = "HOBBIES_1_001", "CA_1"
df = load_m5_data(item, store)
y = df["demand"].values
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

oof_preds = np.zeros((len(y), 3))  # SARIMAX, LSTM, LGBM

for idxs_train, idxs_val in kf.split(df):
    train, val = df.iloc[idxs_train], df.iloc[idxs_val]
    y_train, y_val = y[idxs_train], y[idxs_val]
    
    # SARIMAX
    sar = SARIMAXModel().train(train["demand"])
    oof_preds[idxs_val, 0] = sar.predict(steps=len(val))
    
    # LSTM
    lstm = LSTMModel().train(train["demand"].values)
    oof_preds[idxs_val, 1] = lstm.predict(train["demand"].values, steps=len(val))
    
    # LightGBM
    lgbm = LightGBMModel().train(train, train["demand"])
    feats = lgbm.create_features(pd.concat([train, val]), "demand")
    X_val = feats.iloc[len(train):].drop(columns=["date","demand"])
    oof_preds[idxs_val, 2] = lgbm.predict(X_val)

# Train meta-model
meta = Ridge(alpha=1.0)
meta.fit(oof_preds, y)
print("Meta-model coefficients:", meta.coef_)

# Evaluate stacking on full series (train predictions)
stacked_pred = meta.predict(oof_preds)
print("Stacking MAE:", mean_absolute_error(y, stacked_pred))