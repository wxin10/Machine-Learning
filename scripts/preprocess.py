import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# 读取和合并CSV文件
def load_and_merge_csv(folder_path):
    # 获取文件夹中所有CSV文件的路径
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 初始化一个空的DataFrame
    combined_data = pd.DataFrame()

    # 依次读取每个CSV文件并进行合并
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"正在读取文件: {file}")
        data = pd.read_csv(file_path)

        # 去除列名中的空格
        data.columns = data.columns.str.strip()

        print(f"文件 {file} 的列名: {data.columns.tolist()}")
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    return combined_data


# 处理缺失值
def handle_missing_values(data):
    # 填充缺失值为0或使用中位数等方法
    data = data.fillna(0)
    return data


# 处理无效值（无穷大或超出范围的值）
def handle_invalid_values(data):
    # 将无穷大值替换为NaN，然后使用0填充NaN
    data = data.replace([float('inf'), float('-inf')], float('nan'))
    data = data.fillna(0)  # 你也可以使用其他方式填充，如data.fillna(data.mean())
    return data


# 标签编码
def encode_labels(data, label_column='Label'):
    if label_column in data.columns:
        label_encoder = LabelEncoder()
        data[label_column] = label_encoder.fit_transform(data[label_column])
        print(f"已对标签 '{label_column}' 进行编码。")
        return data, label_encoder
    else:
        print(f"警告：数据中没有找到 '{label_column}' 列，跳过标签编码。")
        return data, None


# 特征标准化
def scale_features(data, feature_columns):
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data, scaler


# 数据分割
def split_data(data, label_column='Label', test_size=0.2):
    X = data.drop(columns=[label_column])
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 加载并合并数据集
    folder_path = '../data/CICIDS2017/'  # 使用相对路径，确保该路径正确指向CSV文件夹
    combined_data = load_and_merge_csv(folder_path)

    # 处理缺失值
    combined_data = handle_missing_values(combined_data)

    # 处理无效值（如无穷大）
    combined_data = handle_invalid_values(combined_data)

    # 提取特征列（假设最后一列是标签列，需检查数据是否一致）
    feature_columns = combined_data.columns[:-1]  # 忽略标签列

    # 标签编码
    combined_data, label_encoder = encode_labels(combined_data)

    # 如果存在 'Label' 列则继续处理
    if label_encoder is not None:
        # 特征标准化
        combined_data, scaler = scale_features(combined_data, feature_columns)

        # 分割数据集为训练集和测试集
        X_train, X_test, y_train, y_test = split_data(combined_data)

        # 保存处理后的数据（可选）
        combined_data.to_csv('../data/processed/combined_preprocessed_data.csv', index=False)
        print("数据预处理完成并保存！")
    else:
        print("由于缺少 'Label' 列，跳过预处理。")
