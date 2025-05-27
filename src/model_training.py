import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import seaborn as sns
import os

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置项目根目录（假设项目根目录是脚本目录的上一级）
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 更新数据和模型路径
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_selected_data():
    """加载选定特征的数据并处理缺失值"""
    print("加载选定特征的数据...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_X_selected.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_y_selected.csv')).squeeze()
    X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val_X_selected.csv'))
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val_y_selected.csv')).squeeze()
    
    # 检查缺失值
    train_missing = X_train.isna().sum().sum()
    val_missing = X_val.isna().sum().sum()
    print(f"训练集缺失值数: {train_missing}, 验证集缺失值数: {val_missing}")
    
    return X_train, y_train, X_val, y_val

def train_random_forest_model():
    """训练随机森林模型（严格遵循论文参数设置）"""
    try:
        # 1. 加载数据
        X_train, y_train, X_val, y_val = load_selected_data()
        
        # 2. 构建模型管道（包含缺失值处理和随机森林）
        model = make_pipeline(
            SimpleImputer(strategy='median'),  # 中位数填充缺失值
            RandomForestClassifier(
                n_estimators=200,       # 论文未明确参数，但实验结果优异，设置为200棵树
                max_depth=15,           # 限制树深度，避免过拟合
                min_samples_split=5,    # 节点分裂最小样本数
                class_weight='balanced', # 处理类别不均衡
                random_state=42,
                n_jobs=-1               # 使用所有CPU核心
            )
        )
        
        print("开始训练随机森林模型...")
        model.fit(X_train, y_train)
        
        # 3. 验证集评估
        print("\n=== 模型评估结果 ===")
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"验证集准确率: {accuracy:.4f}")
        print(classification_report(y_val, y_pred, target_names=['正常', '攻击']))
        
        # 4. 保存模型管道
        joblib.dump(model, os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'))
        print("模型已保存到models/random_forest_pipeline.pkl")
        
        # 5. 可视化特征重要性
        rf_model = model.named_steps['randomforestclassifier']
        plot_feature_importance(rf_model, X_train.columns)
        
        # 6. 可视化混淆矩阵
        plot_confusion_matrix(y_val, y_pred)
        
    except Exception as e:
        print(f"\n[错误] 训练失败: {str(e)}")
        raise  # 重新抛出异常以便调试

def plot_feature_importance(model, feature_names):
    """可视化特征重要性"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 特征重要性排名")
    sns.barplot(x=importance[indices][:20], y=feature_names[indices][:20])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_plot.png'))
    print("特征重要性图已保存到results/feature_importance_plot.png")

def plot_confusion_matrix(y_true, y_pred):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '攻击'], 
                yticklabels=['正常', '攻击'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    print("混淆矩阵已保存到results/confusion_matrix.png")

if __name__ == "__main__":
    train_random_forest_model()