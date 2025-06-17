import os
import json
from pathlib import Path
from dotenv import load_dotenv
from dependency_injector import containers, providers

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1️⃣  加载 .env 文件

env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

# 2️⃣  加载 config.json
config_path = os.path.join(current_dir, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config_json = json.load(f)

class Config:
    """全局配置类"""

    # 3️⃣  读取环境变量
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")


    # 4️⃣  读取 JSON 固定配置
    DEFAULT_DATA_SOURCE = config_json.get("default_data_source", "tushare")
    LOG_LEVEL = config_json.get("log_level", "INFO")
    CACHE_ENABLED = config_json.get("cache_enabled", False)
    TRAINING_INTERVAL = config_json.get("training_interval",{})



    
    #获取指定配置
    def get_value(self, *keys, default=None):
        for key in keys:
            if isinstance(config_json,dict):
                value = value.get(key,default)
            else:
                return value



# 5️⃣  依赖注入（DI）
class ConfigContainer(containers.DeclarativeContainer):
    """全局配置容器"""
    config = providers.Singleton(Config)

# 6️⃣  获取全局配置对象
config_container = ConfigContainer()