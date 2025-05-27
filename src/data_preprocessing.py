import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import json
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

# 定义KDD Cup'99数据集的41个特征和标签
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

def load_and_filter_data(data_path):
    """加载原始数据并进行初步过滤"""
    print("正在加载原始数据...")
    
    # 检查文件是否有.gz后缀
    if data_path.endswith('.gz'):
        data = pd.read_csv(data_path, names=columns, compression='gzip')
    else:
        data = pd.read_csv(data_path, names=columns)
    
    # 过滤协议类型为tcp或udp的数据（论文提到处理IP和TCP流量）
    data = data[data['protocol_type'].isin(['tcp', 'udp'])]
    
    # 过滤无效字节数（确保src_bytes和dst_bytes为正）
    data = data[(data['src_bytes'] > 0) & (data['dst_bytes'] > 0)]
    
    print(f"原始数据行数: {len(data):,} -> 过滤后行数: {len(data):,}")
    return data

def preprocess_data(data):
    """数据预处理：特征编码、标准化、归一化"""
    print("开始数据预处理...")
    
    # 分离分类特征和数值特征
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['label']]
    
    # 1. 分类特征独热编码
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    
    # 2. 数值特征标准化（Z-Score）
    zscaler = StandardScaler()
    zscaled_data = zscaler.fit_transform(data[numerical_cols])
    zscaled_df = pd.DataFrame(
        zscaled_data, 
        columns=[f'z_{col}' for col in numerical_cols]
    )
    
    # 3. 数值特征归一化（Min-Max）
    def minmax_scale(series):
        arr = series.values
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    minmax_data = data[numerical_cols].apply(minmax_scale)
    minmax_df = minmax_data.add_prefix('mm_')
    
    # 合并所有特征
    processed_data = pd.concat([
        zscaled_df, 
        minmax_df, 
        encoded_df, 
        data['label']
    ], axis=1)
    
    print(f"预处理后特征数: {processed_data.shape[1]}")
    return processed_data, zscaler, encoder

def process_labels(data):
    """处理标签：将攻击类型映射为0（正常）和1（攻击）"""
    # 论文中提到将攻击分为DoS、R2L、U2R、Probe等类型
    # 简化处理：正常样本为0，所有攻击样本为1
    normal_label = 'normal.'
    attack_types = {
        normal_label: 0,
        **{k: 1 for k in data['label'].unique() if k != normal_label}
    }
    
    y = data['label'].map(attack_types).fillna(1)
    print(f"标签分布 - 正常样本: {sum(y==0):,}, 攻击样本: {sum(y==1):,}")
    return y, attack_types

if __name__ == "__main__":
    # 1. 加载数据
    data = load_and_filter_data(os.path.join(RAW_DATA_DIR, 'kddcup.data_10_percent'))
    
    # 2. 数据预处理
    processed_data, zscaler, encoder = preprocess_data(data)
    
    # 3. 处理标签
    y, attack_types = process_labels(processed_data)
    X = processed_data.drop('label', axis=1)
    
    # 4. 划分数据集（训练集/验证集/测试集，分层抽样）
    print("划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"数据集大小 - 训练集: {len(X_train):,}, 验证集: {len(X_val):,}, 测试集: {len(X_test):,}")
    
    # 5. 保存预处理后的数据和模型
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False, header=['attack'])
    X_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_val.csv'), index=False, header=['attack'])
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False, header=['attack'])
    
    pd.to_pickle(zscaler, os.path.join(MODELS_DIR, 'zscaler.pkl'))
    pd.to_pickle(encoder, os.path.join(MODELS_DIR, 'encoder.pkl'))
    
    # 保存攻击类型映射
    with open(os.path.join(PROCESSED_DATA_DIR, 'attack_mapping.json'), 'w') as f:
        json.dump(attack_types, f)
    
    print("预处理完成！数据已保存到data/processed目录")