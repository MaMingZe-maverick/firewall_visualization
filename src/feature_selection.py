import pandas as pd
import numpy as np
import joblib
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
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

# 配置参数（严格遵循论文实验设置）
TOP_K = 25  # 选择前25个特征
RANDOM_STATE = 42
SAMPLE_SIZE = 50000  # 采样大小，加速计算

def load_data_with_prefixes():
    """加载带有z_和mm_前缀的数值特征数据"""
    print("正在加载预处理后的数据...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze()
    X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv')).squeeze()
    
    # 仅选择数值特征（z_和mm_前缀），论文中使用信息增益处理数值特征
    num_cols = [col for col in X_train.columns if col.startswith(('z_', 'mm_'))]
    print(f"加载的数值特征数: {len(num_cols)}")
    return X_train[num_cols], y_train, X_val[num_cols], y_val

def calculate_information_gain(X, y):
    """计算特征的信息增益（使用sklearn的mutual_info_classif）"""
    # 论文中提到使用信息增益算法（公式1-5）
    # mutual_info_classif在分类问题中等同于信息增益
    scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return scores

def information_gain_wrapper(X, y):
    """可序列化的信息增益评分函数包装器"""
    return calculate_information_gain(X, y)

def create_feature_selector(X_train, y_train, k=TOP_K):
    """创建基于信息增益的特征选择器"""
    print(f"开始特征选择（选择前{k}个特征）...")
    
    # 1. 数据采样（加速计算）
    sample_size = min(SAMPLE_SIZE, len(X_train))
    if len(X_train) > sample_size:
        print(f"数据采样：从{len(X_train):,}条样本中采样{sample_size:,}条...")
        X_sampled, y_sampled = resample(
            X_train, y_train,
            n_samples=sample_size,
            stratify=y_train,
            random_state=RANDOM_STATE
        )
    else:
        X_sampled, y_sampled = X_train, y_train
    
    # 2. 处理缺失值
    if X_sampled.isna().any().any():
        print("检测到缺失值，使用中位数填充...")
        imputer = SimpleImputer(strategy='median')
        X_sampled = pd.DataFrame(imputer.fit_transform(X_sampled), columns=X_sampled.columns)
    
    # 3. 计算信息增益分数
    print("计算特征信息增益...")
    mi_scores = calculate_information_gain(X_sampled, y_sampled)
    
    # 4. 选择Top-K特征
    top_indices = np.argsort(mi_scores)[-k:][::-1]
    selected_features = X_train.columns[top_indices]
    print(f"Top-{k}特征: {selected_features}")
    
    # 5. 构建特征选择管道（使用可序列化的函数）
    selector = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('selector', SelectKBest(score_func=information_gain_wrapper, k=k))
    ])
    selector.fit(X_train, y_train)
    
    return selector, selected_features, mi_scores

def save_selected_data(X, y, selected_features, output_prefix):
    """保存选定特征的数据"""
    X_selected = X[selected_features]
    X_selected.to_csv(
        os.path.join(PROCESSED_DATA_DIR, f'{output_prefix}_X_selected.csv'),
        index=False
    )
    y.to_csv(
        os.path.join(PROCESSED_DATA_DIR, f'{output_prefix}_y_selected.csv'),
        index=False,
        header=['attack']
    )
    print(f"已保存{output_prefix}集选定特征数据")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. 加载预处理后的数据
    X_train, y_train, X_val, y_val = load_data_with_prefixes()
    
    # 2. 特征选择
    selector, selected_features, mi_scores = create_feature_selector(X_train, y_train, k=TOP_K)
    
    # 3. 保存特征选择器
    joblib.dump(selector, os.path.join(MODELS_DIR, 'feature_selector.pkl'))
    print("特征选择器已保存到models/feature_selector.pkl")
    
    # 4. 保存选定特征列表
    pd.Series(selected_features).to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'selected_features.txt'),
        index=False,
        header=False
    )
    
    # 5. 生成特征重要性报告
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'information_gain': mi_scores
    }).sort_values('information_gain', ascending=False)
    
    feature_importance.to_csv(
        os.path.join(RESULTS_DIR, 'feature_importance.csv'),
        index=False
    )
    print("特征重要性报告已保存到results/feature_importance.csv")
    
    # 6. 保存选定数据
    save_selected_data(X_train, y_train, selected_features, 'train')
    save_selected_data(X_val, y_val, selected_features, 'val')
    
    # 7. 打印总结报告
    print("\n=== 特征选择完成 ===")
    print(f"原始特征数: {X_train.shape[1]}")
    print(f"选定特征数: {len(selected_features)}")
    print(f"Top 5 特征: {selected_features[:5]}")