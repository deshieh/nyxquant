import numpy as np

class TransformerPrep:
    """Transformer 数据预处理"""
    def __init__(self, future_horizon=10, training=True):

        self.future_horizon = future_horizon
        self.training = training

    def process(self, df):
        df = df.copy().dropna()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(window=self.future_horizon).std()
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
        df = df[["weekday", "month", "is_month_end", "log_return", "volatility", *base_features]].copy()

        # 增加目标变量
        if self.training:
            df["future_return"] = np.log(df["close"].shift(-self.future_horizon) / df["close"])  # `T+10` 收益率
            df["future_volatility"] = df["future_return"].rolling(self.future_horizon).std()  # `T+10` 波动率
            df["future_return_direction"] = np.sign(df["future_return"])
            for i in range(1, self.future_horizon + 1):
                df[f"future_log_return_{i}"] = np.sign(df["log_return"].shift(-i))
            df.dropna(inplace=True)
            future_cols = [f"future_log_return_{i}" for i in range(1, self.future_horizon + 1)]
            df["return_consistency"] = df[future_cols].eq(df["future_return_direction"], axis=0).mean(axis=1)

        df.dropna(inplace=True)     

        return df