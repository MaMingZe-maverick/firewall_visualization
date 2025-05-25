import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import seaborn as sns
import os

def load_selected_data():
    """加载选定特征的数据并处理缺失值"""
    X_train = pd.read_csv('../data/processed/train_X_selected.csv')
    y_train = pd.read_csv('../data/processed/train_y_selected.csv').squeeze()
    X_val = pd.read_csv('../data/processed/val_X_selected.csv')
    y_val = pd.read_csv('../data/processed/val_y_selected.csv').squeeze()
    
    # 检查并打印缺失值情况
    print("\n=== 缺失值统计 ===")
    print("训练集缺失值数:", X_train.isna().sum().sum())
    print("验证集缺失值数:", X_val.isna().sum().sum())
    
    return X_train, y_train, X_val, y_val

def train_model():
    try:
        # 加载数据
        X_train, y_train, X_val, y_val = load_selected_data()
        
        # 创建包含缺失值处理的模型管道
        model = make_pipeline(
            SimpleImputer(strategy='median'),  # 用中位数填充缺失值
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )
        
        print("\n训练随机森林模型...")
        model.fit(X_train, y_train)
        
        # 验证集评估
        print("\n=== 验证集性能 ===")
        y_pred = model.predict(X_val)
        print(classification_report(y_val, y_pred, target_names=['正常', '攻击']))
        
        # 保存整个管道（包含预处理和模型）
        joblib.dump(model, '../models/random_forest_pipeline.pkl')
        print("\n模型管道已保存到 ../models/random_forest_pipeline.pkl")
        
        # 可视化特征重要性（需要从管道中提取随机森林模型）
        rf_model = model.named_steps['randomforestclassifier']
        plot_feature_importance(rf_model, X_train.columns)
        
        # 可视化混淆矩阵
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
    plt.savefig('../results/feature_importance_plot.png')
    print("\n特征重要性图已保存到 results/feature_importance_plot.png")

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
    plt.savefig('../results/confusion_matrix.png')
    print("混淆矩阵已保存到 results/confusion_matrix.png")

if __name__ == "__main__":
    os.makedirs('../results', exist_ok=True)
    train_model()