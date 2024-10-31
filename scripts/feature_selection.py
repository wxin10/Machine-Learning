import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# 特征选择
def feature_selection(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    selector = SelectFromModel(rf_model, prefit=True)
    selected_features = selector.get_support(indices=True)

    return selected_features

if __name__ == "__main__":
    # 加载数据
    file_path = '../data/processed/combined_preprocessed_data.csv'
    data = load_data(file_path)

    # 分离特征和标签
    X = data.drop(columns=['Label'])
    y = data['Label']

    # 特征选择
    selected_features = feature_selection(X, y)

    # 保存选中的特征
    pd.DataFrame(selected_features).to_csv('../data/selected_features.csv', index=False)
