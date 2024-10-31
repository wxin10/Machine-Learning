import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

if __name__ == "__main__":
    # 加载数据
    file_path = '../data/processed/combined_preprocessed_data.csv'
    data = load_data(file_path)

    # 加载选择的特征
    selected_features = pd.read_csv('../data/selected_features.csv').values.flatten()

    # 分离特征和标签
    X = data.iloc[:, selected_features]
    y = data['Label']
n
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型，使用所有CPU核心
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, '../models/random_forest_model.pkl')
