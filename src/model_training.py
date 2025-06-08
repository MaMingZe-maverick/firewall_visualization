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
from sklearn.utils.class_weight import compute_class_weight

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

def load_data(chunksize=100000):
    """加载预处理后的数据，使用分块加载以优化内存使用"""
    print("加载预处理后的数据...")
    
    # 尝试导入tqdm
    try:
        from tqdm import tqdm
        progress = tqdm
    except ImportError:
        progress = lambda x: x
    
    # 分块加载训练数据
    print("加载训练数据...")
    train_chunks_X = []
    train_chunks_y = []
    
    # 获取总行数
    total_rows = sum(1 for _ in open(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))) - 1
    total_chunks = (total_rows + chunksize - 1) // chunksize
    
    for chunk_X, chunk_y in progress(zip(
        pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), chunksize=chunksize),
        pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), chunksize=chunksize)
    ), total=total_chunks, desc="Loading training data"):
        train_chunks_X.append(chunk_X)
        train_chunks_y.append(chunk_y.squeeze())
        
        # 定期清理内存
        if len(train_chunks_X) >= 5:
            train_chunks_X = [pd.concat(train_chunks_X, ignore_index=True)]
            train_chunks_y = [pd.concat(train_chunks_y, ignore_index=True)]
    
    X_train = pd.concat(train_chunks_X, ignore_index=True)
    y_train = pd.concat(train_chunks_y, ignore_index=True)
    
    # 清理内存
    del train_chunks_X, train_chunks_y
    import gc
    gc.collect()
    
    # 分块加载验证数据
    print("加载验证数据...")
    val_chunks_X = []
    val_chunks_y = []
    
    total_rows = sum(1 for _ in open(os.path.join(PROCESSED_DATA_DIR, 'X_val.csv'))) - 1
    total_chunks = (total_rows + chunksize - 1) // chunksize
    
    for chunk_X, chunk_y in progress(zip(
        pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val.csv'), chunksize=chunksize),
        pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv'), chunksize=chunksize)
    ), total=total_chunks, desc="Loading validation data"):
        val_chunks_X.append(chunk_X)
        val_chunks_y.append(chunk_y.squeeze())
    
    X_val = pd.concat(val_chunks_X, ignore_index=True)
    y_val = pd.concat(val_chunks_y, ignore_index=True)
    
    # 清理内存
    del val_chunks_X, val_chunks_y
    gc.collect()
    
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
        X_train, y_train, X_val, y_val = load_data()
        
        # 2. 计算类别权重 - 重点优化DoS和Probe的识别
        print("\n计算类别权重以优化DoS和Probe识别...")
        
        # 获取所有类别
        classes = np.unique(y_train)
        
        # 计算类别的原始频率
        class_counts = y_train.value_counts()
        print("类别分布:")
        for cls, count in class_counts.items():
            print(f"- {cls}: {count:,}")
        
        # 自定义权重 - 重点提升DoS和Probe的权重
        custom_weights = {
            'DoS': 25.0,   # 显著提高DoS权重
            'Probe': 20.0, # 显著提高Probe权重
            'R2L': 3.0,
            'U2R': 5.0,
            'normal': 1.0
        }
        
        # 创建样本权重向量
        sample_weights = y_train.map(custom_weights).values
        
        # 3. 构建模型管道（包含缺失值处理和随机森林）
        model = make_pipeline(
            SimpleImputer(strategy='median'),  # 中位数填充缺失值
            RandomForestClassifier(
                n_estimators=500,       # 增加树的数量
                max_depth=None,          # 不限制深度
                min_samples_split=10,    # 增加分裂限制防止过拟合
                min_samples_leaf=5,     # 添加叶节点最小样本限制
                max_features='sqrt',    # 限制每棵树使用的特征数
                class_weight='balanced', 
                random_state=42,
                n_jobs=-1,
                oob_score=True
            )
        )
        
        print("开始训练随机森林模型...")
        
        # 4. 使用样本权重训练模型
        model.fit(X_train, y_train, randomforestclassifier__sample_weight=sample_weights)
        
        # 5. 验证集评估
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
        
        # 6. 保存模型管道
        joblib.dump(model, os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'))
        print("模型已保存到models/random_forest_pipeline.pkl")
        
        # 7. 可视化特征重要性
        plot_feature_importance(rf_model, X_train.columns)
        
        # 8. 可视化混淆矩阵
        plot_confusion_matrix(y_val, y_pred, classes)
        
        # 9. 保存类别权重信息
        weight_info = pd.DataFrame({
            'class': list(custom_weights.keys()),
            'weight': list(custom_weights.values())
        })
        weight_info.to_csv(os.path.join(RESULTS_DIR, 'class_weights.csv'), index=False)
        print("类别权重信息已保存到results/class_weights.csv")
        
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