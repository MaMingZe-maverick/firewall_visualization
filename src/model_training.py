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
    
    # 确保标签是字符串类型（多分类需要）
    y_train = y_train.astype(str)
    y_val = y_val.astype(str)
    
    # 检查缺失值
    train_missing = X_train.isna().sum().sum()
    val_missing = X_val.isna().sum().sum()
    print(f"训练集缺失值数: {train_missing}, 验证集缺失值数: {val_missing}")
    
    return X_train, y_train, X_val, y_val

def train_random_forest_model():
    """训练随机森林模型（多分类）"""
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
                n_jobs=-1,              # 使用所有CPU核心
                oob_score=True          # 计算袋外分数
            )
        )
        
        print("开始训练随机森林模型...")
        model.fit(X_train, y_train)
        
        # 3. 验证集评估
        print("\n=== 模型评估结果 ===")
        y_pred = model.predict(X_val)
        
        # 获取所有类别（按字母顺序排序）
        classes = sorted(set(y_val) | set(y_pred))
        
        # 多分类评估报告
        print("分类报告:")
        print(classification_report(y_val, y_pred, target_names=classes))
        
        # 计算整体准确率
        accuracy = accuracy_score(y_val, y_pred)
        print(f"整体准确率: {accuracy:.4f}")
        
        # 计算袋外分数（OOB Score）
        rf_model = model.named_steps['randomforestclassifier']
        print(f"袋外分数 (OOB Score): {rf_model.oob_score_:.4f}")
        
        # 4. 保存模型管道
        joblib.dump(model, os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'))
        print("模型已保存到models/random_forest_pipeline.pkl")
        
        # 5. 可视化特征重要性
        plot_feature_importance(rf_model, X_train.columns)
        
        # 6. 可视化混淆矩阵
        plot_confusion_matrix(y_val, y_pred, classes)
        
    except Exception as e:
        print(f"\n[错误] 训练失败: {str(e)}")
        raise  # 重新抛出异常以便调试

def plot_feature_importance(model, feature_names):
    """可视化特征重要性（多分类）"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Feature importance ranking")
    sns.barplot(x=importance[indices][:20], y=feature_names[indices][:20])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_plot.png'))
    print("特征重要性图已保存到results/feature_importance_plot.png")

def plot_confusion_matrix(y_true, y_pred, classes):
    """可视化混淆矩阵（多分类）"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Prediction label')
    plt.ylabel('Real label')
    plt.title('confusion matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    print("混淆矩阵已保存到results/confusion_matrix.png")

if __name__ == "__main__":
    train_random_forest_model()