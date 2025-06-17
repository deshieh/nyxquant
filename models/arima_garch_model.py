from datetime import timedelta
import logging
import numpy as np
from pmdarima import ARIMA, auto_arima
from arch import arch_model
from data_pipeline.data_container import data_container
import pandas as pd


class ArimaGarchModel:
    """ ARIMA-GARCH 训练 & 预测 """
    def __init__(self, arima_order):
        """ 载入 (p, d, q) 参数 """
        self.arima_order = arima_order
        self.arima_model = None
        self.garch_model = None    

    @classmethod
    def load(cls, model_data):
        """ ✅ 加载 `ARIMA-GARCH`，实例化对象 """
        arima_order = model_data["arima_order"]
        return cls(arima_order)    

    @staticmethod
    def train(asset_type, asset, current_date):
        """ 训练 ARIMA-GARCH """

        # 计算训练数据时间窗口
        start_date = current_date - timedelta(days=1000)
        df = data_container.data_manager().get_data(asset_type, asset, start_date, current_date)
        df = data_container.arima_garch_prep().process(df)
        returns = df["log_return"].dropna()

        # 拟合ARIMA获取pq
        arima_model = auto_arima(returns, stepwise=False, suppress_warnings=True)
        order = arima_model.order

        model_data = {"arima_order": order}
        model_path = f"models/ARIMA-GARCH_{asset}.pkl"
        from models.model_container import model_container
        model_container.model_registry().register_model(asset_type, "ARIMA-GARCH", model_path, model_data)
        print(f"✅ ARIMA-GARCH 训练完成 {asset}，存储参数: {order}")

    def predict(self,  asset_type, asset, current_date):
        """ 预测均值 & 波动率 """  

        # 计算预测数据时间窗口
        start_date = current_date - timedelta(days=200)
        df = data_container.data_manager().get_data(asset_type, asset, start_date, current_date)
        df = data_container.arima_garch_prep().process(df)    
        returns = df["log_return"].dropna()
        last_close_price = df["close"].iloc[-1]

        # ARIMA 拟合
        self.arima_model = ARIMA(order = self.arima_order,  suppress_warnings=True).fit(returns)
        # 获取一步预测的预测值
        arima_pred = self.arima_model.predict(n_periods=10).iloc[-1] / 1000

        # GARCH(1,1) 拟合 & 预测波动率
        residuals = self.arima_model.resid()

        self.garch_model = arch_model(residuals, vol="Garch", p=1, q=1).fit(disp="off")

        garch_forecast = self.garch_model.forecast(horizon=1)

        garch_pred = (np.sqrt(garch_forecast.variance.values[-1][0])) / 1000

        # 转换到价格空间
        predicted_mean = last_close_price * np.exp(arima_pred + 0.5 * garch_pred**2)
        predicted_vol = predicted_mean * garch_pred

        print(f"📊 ARIMA-GARCH 价格预测 {asset} 日期 {current_date} 均值 {predicted_mean:.4f}, 波动率 {predicted_vol:.4f}")
        return predicted_mean, predicted_vol