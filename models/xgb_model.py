from datetime import timedelta
import xgboost as xgb
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler



class XGBModel:
    def __init__(self, model=None):
        self.model = model
    
    @classmethod
    def load(cls, model_path):
        """加载模型,返回封装实例"""
        model = joblib.load(model_path)
        return cls(model)



    @staticmethod
    def train(asset_type, asset, current_date):
        from data_pipeline.data_container import data_container
        prep = data_container.xgb_prep(future_horizon=2, training=True)
        start_date = current_date - timedelta(days=5000)   # 训练所需数据量
        df = data_container.data_manager().get_data(asset_type, asset, start_date, current_date)
        df = prep.process(df)                      
        X = df.drop(columns=["future_return"])
        y = df["future_return"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = xgb.XGBRegressor(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_scaled, y)
        # ✅ 训练完成后，存储模型 & 注册到 ModelRegistry
        model_path = f"models/XGB_{asset}.pkl"
        scaler_path = f"models/XGB_{asset}_scaler.pkl"
        from models.model_container import model_container
        joblib.dump(scaler, scaler_path)
        wrapper = XGBModel()
        wrapper.model = model
        model_container.model_registry().register_model(asset_type, "XGB", model_path, wrapper)

    def predict(self, asset_type, asset, current_date):
        from data_pipeline.data_container import data_container
        prep = data_container.xgb_prep(future_horizon=3, training=False)        
        start_date = current_date - timedelta(days=30)
        df = data_container.data_manager().get_data(asset_type, asset, start_date, current_date)
        df = prep.process(df)
        scaler_path = f"models/XGB_{asset}_scaler.pkl"
        scaler = joblib.load(scaler_path)
        X = df.iloc[[-1]]
        X_scaled = scaler.transform(X)
        return self.model.predict(X_scaled)[0]