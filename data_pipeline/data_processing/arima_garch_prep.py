import numpy as np

class ArimaGarchPrep:
    """ ARIMA-GARCH 预处理 """

    def process(self, df):
        df.dropna(inplace=True)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 1000
        df.dropna(inplace=True)
        return df
