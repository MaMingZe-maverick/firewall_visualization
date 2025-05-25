import pandas as pd
import numpy as np
import joblib
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import os

# 配置参数
TOP_K = 25
RANDOM_STATE = 42
SAMPLE_SIZE = 50_000

class FeatureSelectorWrapper:
    """包装器类替代lambda函数"""
    def __init__(self, scores):
        self.scores = scores
    
    def __call__(self, X, y):
        return self.scores

def load_data_with_prefixes():
    """加载带有z_和mm_前缀的特征数据"""
    print("正在加载预处理后的数据...")
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
    X_val = pd.read_csv('../data/processed/X_val.csv')
    y_val = pd.read_csv('../data/processed/y_val.csv').squeeze()
    
    # 确保只选择数值特征（z_和mm_前缀）
    num_cols = [col for col in X_train.columns if col.startswith(('z_', 'mm_'))]
    return X_train[num_cols], y_train, X_val[num_cols], y_val

def create_feature_selector(X_train, y_train, k=TOP_K):
    """处理预处理数据的特征选择器"""
    # 1. 数据采样加速
    sample_size = min(SAMPLE_SIZE, len(X_train))
    if len(X_train) > sample_size:
        X_sampled, y_sampled = resample(
            X_train, y_train,
            n_samples=sample_size,
            stratify=y_train,
            random_state=RANDOM_STATE
        )
    else:
        X_sampled, y_sampled = X_train, y_train
    
    # 2. 处理可能的缺失值
    print("正在检查缺失值...")
    if X_sampled.isna().any().any():
        imputer = SimpleImputer(strategy='median')
        X_sampled = pd.DataFrame(imputer.fit_transform(X_sampled), columns=X_sampled.columns)
    
    # 3. 计算特征重要性
    print("正在计算特征重要性...")
    mi_scores = mutual_info_classif(
        X_sampled, 
        y_sampled,
        random_state=RANDOM_STATE
    )
    
    # 4. 选择特征
    top_indices = np.argsort(mi_scores)[-k:][::-1]
    selected_features = X_train.columns[top_indices]
    
    # 5. 构建最终选择器（使用包装类替代lambda）
    selector = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('selector', SelectKBest(FeatureSelectorWrapper(mi_scores), k=k))
    ])
    selector.fit(X_train, y_train)
    
    return selector, selected_features, mi_scores

def save_selected_data(X, y, selected_features, output_prefix):
    """保存选定特征的数据"""
    X_selected = X[selected_features]
    X_selected.to_csv(
        f'../data/processed/{output_prefix}_X_selected.csv',
        index=False
    )
    y.to_csv(
        f'../data/processed/{output_prefix}_y_selected.csv',
        index=False,
        header=['attack']
    )

if __name__ == "__main__":
    # 1. 加载预处理后的数据
    X_train, y_train, X_val, y_val = load_data_with_prefixes()
    
    # 2. 特征选择
    print("开始特征选择...")
    selector, selected_features, mi_scores = create_feature_selector(X_train, y_train, k=TOP_K)
    
    # 3. 保存结果
    joblib.dump(selector, '../models/feature_selector.pkl')
    
    pd.Series(selected_features).to_csv(
        '../data/processed/selected_features.txt',
        index=False,
        header=False
    )
    
    # 4. 生成特征重要性报告
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(
        '../results/feature_importance.csv',
        index=False
    )
    
    # 5. 保存选定数据
    print("保存结果...")
    save_selected_data(X_train, y_train, selected_features, 'train')
    save_selected_data(X_val, y_val, selected_features, 'val')
    
    # 6. 打印报告
    print("\n=== 特征选择报告 ===")
    print(f"原始特征数: {X_train.shape[1]}")
    print(f"选定特征数: {len(selected_features)}")
    print(f"Top {TOP_K} 特征:\n{selected_features[:TOP_K]}")
    print("\n处理完成！")