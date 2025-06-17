from datetime import timedelta
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TransformerModel(nn.Module):
    """ Transformer 预测未来收益率、波动率、return_consistency """

    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, nhead=4, sequence_length=500):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, sequence_length, hidden_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc_return = nn.Linear(hidden_dim, 1)   # 预测未来收益率
        self.fc_volatility = nn.Linear(hidden_dim, 1)  # 预测 T+10 波动率
        self.fc_consistency = nn.Linear(hidden_dim, 1)   # 预测 return_consistency

    def forward(self, x):
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        pred_return = self.fc_return(x[:, -1, :])  
        pred_volatility = self.fc_volatility(x[:, -1, :])  
        pred_consistency = torch.sigmoid(self.fc_consistency(x[:, -1, :]))  # 归一化到 0~1 之间
        return pred_return, pred_volatility, pred_consistency 

    @staticmethod
    def train(asset_type, asset, current_date):
        """ Transformer 训练 """
        from data_pipeline.data_container import data_container
        prep = data_container.transformer_prep(future_horizon=10, training=True)
        start_date = current_date - timedelta(days=5000)   # 训练所需数据量
        df = data_container.data_manager().get_data(asset_type, asset, start_date, current_date)
        df = prep.process(df)
        feature_cols = ["log_return", "volatility", "momentum", "volume_change"]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        scaler_path = f"models/Transformer_{asset}_scaler.pkl"
        joblib.dump(scaler, scaler_path)


        # 滑动窗口
        X, y_return, y_volatility, y_consistency = [], [], [], []
        for i in range(len(df) - 500):
            X.append(df.iloc[i:i+500][["log_return", "volatility", "momentum", "volume_change"]].values)
            y_return.append(df.iloc[i+500]["future_return"])
            y_volatility.append(df.iloc[i+500]["future_volatility"])
            y_consistency.append(df.iloc[i+500]["return_consistency"])

        X = torch.tensor(X).float()
        y_return = torch.tensor(y_return).float().unsqueeze(1)
        y_volatility = torch.tensor(y_volatility).float().unsqueeze(1)
        y_consistency = torch.tensor(y_consistency).float().unsqueeze(1)

        dataset = TensorDataset(X, y_return, y_volatility, y_consistency)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = TransformerModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion_return = nn.MSELoss()
        criterion_volatility = nn.MSELoss()
        criterion_consistency = nn.BCELoss()  

        for epoch in range(10):
            for batch_X, batch_y_return, batch_y_volatility, batch_y_consistency in loader:
                optimizer.zero_grad()
                pred_return, pred_volatility, pred_consistency = model(batch_X)
                loss = (criterion_return(pred_return, batch_y_return) +
                        criterion_volatility(pred_volatility, batch_y_volatility) +
                        criterion_consistency(pred_consistency, batch_y_consistency))
                loss.backward()
                optimizer.step()

        # ✅ 训练完成后，存储模型 & 注册到 ModelRegistry
        model_path = f"models/Transformer_{asset}.pth"
        from models.model_container import model_container
        model_container.model_registry().register_model(asset_type, "Transformer", model_path, model)

    def predict(self, asset_type, asset, current_date):
        """ 预测 """
        from data_pipeline.data_container import data_container
        prep = data_container.transformer_prep(future_horizon=10, training=False)
        start_date = current_date - timedelta(days=500)
        df = data_container.data_manager().get_data(asset_type, asset, start_date, current_date)
        df = prep.process(df)
        scaler_path = f"models/Transformer_{asset}_scaler.pkl"
        scaler = joblib.load(scaler_path)
        feature_cols = ["log_return", "volatility", "momentum", "volume_change"]
        df[feature_cols] = scaler.transform(df[feature_cols])
        X = torch.tensor(df[["log_return", "volatility", "momentum", "volume_change"]].values[-500:], dtype=torch.float32, requires_grad=False).unsqueeze(0)
        with torch.no_grad():
            pred_return, pred_volatility, pred_consistency = self(X)

        return pred_return.item(), pred_volatility.item(), pred_consistency.item()

