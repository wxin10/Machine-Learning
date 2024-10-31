import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 绘制混淆矩阵
def plot_confusion_matrix(file_path):
    cm = pd.read_csv(file_path)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig('../results/confusion_matrix.png')
    plt.show()

# 绘制分类报告的条形图
def plot_classification_report(file_path):
    # 加载分类报告
    report_df = pd.read_csv(file_path)

    # 打印数据以检查文件内容
    print(report_df)

    # 使用适当的列名，根据实际数据结构设置
    report_df.columns = ['class', 'precision', 'recall', 'f1-score', 'support']

    # 过滤掉非数字类行（如 accuracy, macro avg 等）
    filtered_df = report_df[~report_df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])]

    # 可视化 precision 列
    plt.figure(figsize=(10, 6))
    sns.barplot(x=filtered_df['class'], y=filtered_df['precision'].astype(float))
    plt.title('Precision by Class')
    plt.xticks(rotation=90)
    plt.savefig('../results/precision_by_class.png')  # 保存图像
    plt.show()

if __name__ == "__main__":
    # 加载并可视化混淆矩阵
    plot_confusion_matrix('../results/confusion_matrix.csv')

    # 加载并可视化分类报告
    plot_classification_report('../results/classification_report.csv')
