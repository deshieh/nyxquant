import logging
import pickle
import joblib
import torch

class ModelRegistry:
    """ 统一管理模型注册、存储 & 加载 """

    def __init__(self):
        self.model_store = {}  # 存储已加载的模型
        self.model_metadata = {}  # 存储模型文件路径

    def register_model(self, asset_type, model_type, model_path, model_obj=None):
        """ 注册模型：存储路径 & 加载模型对象 """

        key = (asset_type, model_type)

        print(f"开始注册模型{key}")

        if model_type == "Transformer":
            # ✅ Transformer 存路径 + torch 模型
            torch.save(model_obj.state_dict(), model_path)
            self.model_metadata[key] = {"path": model_path}
            print(f"模型路径储存成功：{self.model_metadata[key]}")
            self.model_store[key] = model_obj  # 直接存入模型
            print("Transformer实例对象储存成功")
        
        elif model_type == "XGB":
            joblib.dump(model_obj, model_path)
            self.model_metadata[key] = {"path": model_path}
            print(f"[XGBoost] 模型路径储存成功：{self.model_metadata[key]}")
            self.model_store[key] = model_obj  # 直接存入模型
            print("[XGBoost] 实例对象储存成功")


        elif model_type == "ARIMA-GARCH":
            # ✅ ARIMA-GARCH 仅存 (p, d, q) 参数
            self.model_metadata[key] = {"path": model_path}  # 存路径
            with open(model_path, "wb") as f:
                pickle.dump(model_obj, f)
            self.model_store[key] = None  # 需要 `load_model()` 动态加载

        print(f"✅ {model_type} 模型注册完成: {model_path}")

    def get_model(self, asset_type, model_type):
        """ ✅ 获取模型：如果已加载，则直接返回，否则从路径加载 """
        key = (asset_type, model_type)
        print(f"开始获取模型{key}")
        if key in self.model_store and self.model_store[key] is not None:
            print("从对象储存中获取模型")
            return self.model_store[key]  # ✅ 直接返回已加载的实例，避免重复加载
        
        if key not in self.model_metadata:
            print(f"❌ {asset_type} - {model_type} 模型未注册")
            return None
        
        model_path = self.model_metadata[key]["path"]
        from models.model_container import model_container
        try:
            if model_type == "Transformer":
                model = model_container.transformer()  
                model.load_state_dict(torch.load(model_path))
                model.eval()

            elif model_type == "ARIMA-GARCH":
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                print(model_data)
                model = model_container.arima_garch_load(model_data)

            elif model_type == "XGB":
                model = model_container.xgb_load(model_path)


            else:
                logging.error(f"❌ 未知模型类型: {model_type}")
                return None

            self.model_store[key] = model  # ✅ 存入实例
            print("新加载的模型实例对象已储存")
            return model

        except Exception as e:
            print(f"❌ 加载模型失败: {model_type} ({asset_type}), 错误信息: {e}")
            return None