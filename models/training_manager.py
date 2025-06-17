import logging
import traceback
from infrastructure.global_config import config_container

class TrainingManager:
    """ 训练管理器，负责触发定期重训练 """
    def __init__(self):
        from models.model_container import model_container
        self.model_registry = model_container.model_registry()
        self.training_records = {}  # 记录每个模型的最新训练时间
        # **手动映射 model_type 到 train() 方法**
        self.model_train_methods = {
            "Transformer": model_container.transformer_train,
            "ARIMA-GARCH": model_container.arima_garch_train,
            "XGB": model_container.xgb_train,
        }        

    def should_train(self, asset_type, asset, model_type, current_date):
        train_interval = config_container.config().TRAINING_INTERVAL.get(asset_type, {}).get(model_type, 30)
        last_train_date = self.training_records.get((asset_type, model_type))

        if last_train_date is None or (current_date - last_train_date).days >= train_interval:
            print(f"last_train_date:{last_train_date}，开始训练")
            return self.train_model(asset_type, asset, model_type, current_date)
        logging.info(f"⏭️ {asset_type} ({model_type}) 近期已训练，跳过训练")
        return False

    
    def train_model(self, asset_type, asset, model_type, current_date):
        """ 触发模型训练，并确保训练失败不会影响记录 """
        print("触发训练函数")
        if model_type not in self.model_train_methods:
            logging.error(f"❌ 未知模型类型: {model_type}")
            return False

        try:
            print(f"🚀 开始训练 {model_type} ({asset}) on {current_date}")
            train_function = self.model_train_methods[model_type]
            train_function(asset_type, asset, current_date)

            # ✅ **训练成功后，才更新训练记录**
            self.training_records[(asset_type, model_type)] = current_date
            print(f"✅ 训练完成: {model_type} ({asset}) on {current_date}")
            return True

        except Exception as e:
            print(f"❌ 训练失败: {model_type} ({asset}) on {current_date}, 错误信息: {e}")
            print(traceback.format_exc())  # ✅ 打印完整异常信息
            return False  # 训练失败