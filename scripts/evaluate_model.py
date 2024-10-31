import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import joblib


# 加载模型
def load_model(model_path):
    model = joblib.load(model_path)
    return model


# 加载预处理后的数据
def load_preprocessed_data(file_path):
    data = pd.read_csv(file_path)
    return data


if __name__ == "__main__":
    # 加载预处理后的数据
    file_path = '../data/processed/combined_preprocessed_data.csv'
    data = load_preprocessed_data(file_path)

    # 加载选择的特征
    selected_features = pd.read_csv('../data/selected_features.csv').values.flatten()

    # 分离特征和标签
    X = data.iloc[:, selected_features]  # 使用相同的特征
    y = data['Label']  # 标签列

    # 加载已经训练好的模型
    model_path = '../models/random_forest_model.pkl'
    model = load_model(model_path)

    # 设置模型为多核运行，加快速度
    model.set_params(n_jobs=-1)  # 使CPU全速运行

    # 分割数据集
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 生成混淆矩阵和分类报告
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 保存混淆矩阵和分类报告为CSV文件
    pd.DataFrame(cm).to_csv('../results/confusion_matrix.csv', index=False)

    # 将分类报告保存为 CSV 文件
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('../results/classification_report.csv', index=True)
