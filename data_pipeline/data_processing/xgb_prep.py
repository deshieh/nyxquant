import numpy as np
import pandas as pd

class XGBPrep:
    """XGBoost 数据预处理"""    
    def __init__(self, future_horizon=2, lag_days=10, training = True):
        self.future_horizon = future_horizon
        self.lag_days = lag_days
        self.training = training

    #特征工程
    def process(self, df):
        df = df.copy().dropna()
        df["weekday"] = df.index.weekday
        df["month"] = df.index.month
        df["is_month_end"] = df.index.is_month_end.astype(int)
        base_features = [
            "open", "high", "low", "close", "volume", 
            "macd_dif_bfq", "macd_dea_bfq", "bias1_bfq", "dpo_bfq",
            "mtm_bfq", "mtmma_bfq", "roc_bfq", "maroc_bfq",
            "obv_bfq", "vr_bfq","brar_ar_bfq", "brar_br_bfq", 
            "psy_bfq", "rsi_bfq_6", "rsi_bfq_12",
            "atr_bfq", "cr_bfq", "emv_bfq", "mass_bfq",
        ]
        df = df[["weekday", "month", "is_month_end", *base_features]].copy()
        #增加滞后特征
        lagged_features = {}
        for col in base_features:
            for lag in range(self.lag_days):
                lagged_features[f"{col}_lag{lag}"] = df[col].shift(lag)
        df = pd.concat([df, pd.DataFrame(lagged_features, index=df.index)], axis=1)

        #增加目标变量
        if self.training:
            df["future_return"] = np.log(df["close"].shift(-self.future_horizon) / df["close"]) 

        df.dropna(inplace=True)     

        return df
        

    
       
