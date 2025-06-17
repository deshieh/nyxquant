from dependency_injector import containers, providers
from models.arima_garch_model import ArimaGarchModel
from models.training_manager import TrainingManager
from models.transformer_model import TransformerModel
from models.xgb_model import XGBModel
from models.model_registry import ModelRegistry


class ModelContainer(containers.DeclarativeContainer):

    #模型容器
    transformer = providers.Factory(TransformerModel)
    arima_garch = providers.Factory(ArimaGarchModel)
    xgb = providers.Factory(XGBModel)

    #模型训练容器
    transformer_train = providers.Factory(TransformerModel.train)
    arima_garch_train = providers.Factory(ArimaGarchModel.train)
    xgb_train = providers.Factory(XGBModel.train)

    #模型加载容器
    arima_garch_load = providers.Factory(ArimaGarchModel.load)
    xgb_load = providers.Factory(XGBModel.load)

    #模型注册表容器
    model_registry = providers.Singleton(ModelRegistry)

    #模型训练管理容器
    training_manager = providers.Singleton(TrainingManager)

model_container = ModelContainer()