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

def load_and_filter_data(data_path, chunksize=100000):
    """加载原始数据并进行初步过滤，使用分块处理以处理大型数据集"""
    print("正在加载并过滤原始数据...")
    
    filtered_chunks = []
    total_rows = 0
    filtered_rows = 0
    
    # 使用tqdm显示进度
    try:
        from tqdm import tqdm
        progress = tqdm
    except ImportError:
        progress = lambda x: x
    
    # 分块读取数据
    chunks = pd.read_csv(data_path, names=columns, 
                        compression='gzip' if data_path.endswith('.gz') else None,
                        chunksize=chunksize)
    
    for chunk in progress(chunks):
        total_rows += len(chunk)
        
        # 过滤协议类型为tcp或udp的数据
        chunk = chunk[chunk['protocol_type'].isin(['tcp', 'udp'])]
        
        # 过滤无效字节数
        chunk = chunk[(chunk['src_bytes'] > 0) & (chunk['dst_bytes'] > 0)]
        
        filtered_rows += len(chunk)
        if len(chunk) > 0:
            filtered_chunks.append(chunk)
        
        # 定期清理内存
        if len(filtered_chunks) > 10:
            filtered_chunks = [pd.concat(filtered_chunks, ignore_index=True)]
    
    # 合并所有过滤后的数据
    data = pd.concat(filtered_chunks, ignore_index=True)
    print(f"原始数据行数: {total_rows:,} -> 过滤后行数: {filtered_rows:,}")
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
    """处理标签：将攻击类型映射为多分类标签"""
    # 按照论文中的分类：正常、DoS、Probe、R2L、U2R
    attack_mapping = {
        'normal.': 'normal',
        # DoS攻击（11种）
        'back.': 'DoS', 'land.': 'DoS', 'neptune.': 'DoS', 'pod.': 'DoS', 
        'smurf.': 'DoS', 'teardrop.': 'DoS', 'apache2.': 'DoS', 'udpstorm.': 'DoS',
        'processtable.': 'DoS', 'worm.': 'DoS', 'mailbomb.': 'DoS',
        # Probe攻击（6种）
        'satan.': 'Probe', 'ipsweep.': 'Probe', 'nmap.': 'Probe', 'portsweep.': 'Probe',
        'mscan.': 'Probe', 'saint.': 'Probe',
        # R2L攻击（14种）
        'ftp_write.': 'R2L', 'guess_passwd.': 'R2L', 'imap.': 'R2L', 'multihop.': 'R2L',
        'phf.': 'R2L', 'spy.': 'R2L', 'warezclient.': 'R2L', 'warezmaster.': 'R2L',
        'xlock.': 'R2L', 'xsnoop.': 'R2L', 'snmpguess.': 'R2L', 'snmpgetattack.': 'R2L',
        'httptunnel.': 'R2L', 'sendmail.': 'R2L',
        # U2R攻击（7种）
        'buffer_overflow.': 'U2R', 'loadmodule.': 'U2R', 'rootkit.': 'U2R', 'perl.': 'U2R',
        'sqlattack.': 'U2R', 'xterm.': 'U2R', 'ps.': 'U2R'
    }
    
    # 处理未知攻击类型
    y = data['label'].map(attack_mapping)
    unknown_attacks = y.isna()
    if unknown_attacks.any():
        print(f"警告: 发现 {unknown_attacks.sum()} 条未知攻击类型的样本")
        # 将未知攻击类型归类为"其他"
        y.fillna('other', inplace=True)
    
    # 打印类别分布
    class_counts = y.value_counts()
    print("标签分布:")
    for cls, count in class_counts.items():
        print(f"- {cls}: {count:,}")
    
    return y, attack_mapping

def main():
    """执行数据预处理的主函数"""
    # 1. 加载10%数据作为训练数据
    print("\n=== 处理训练数据（10%数据集）===")
    train_data = load_and_filter_data(os.path.join(RAW_DATA_DIR, 'kddcup.data_10_percent'))
    
    # 2. 数据预处理（使用训练数据拟合预处理器）
    processed_train_data, zscaler, encoder = preprocess_data(train_data)
    
    # 3. 处理训练数据标签
    y_train_full, attack_types = process_labels(processed_train_data)
    X_train_full = processed_train_data.drop('label', axis=1)
    
    # 4. 划分训练集和验证集
    print("\n划分训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2,  # 使用20%作为验证集
        stratify=y_train_full,
        random_state=42
    )
    
    print(f"训练集大小: {len(X_train):,}, 验证集大小: {len(X_val):,}")
    
    # 5. 加载完整数据集作为测试集
    print("\n=== 处理测试数据（完整数据集）===")
    test_data = load_and_filter_data(os.path.join(RAW_DATA_DIR, 'kddcup.data.gz'))
    
    # 6. 使用已拟合的预处理器处理测试数据
    # 分离分类特征和数值特征
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in test_data.columns if col not in categorical_cols + ['label']]
    
    # 使用已拟合的编码器和缩放器转换测试数据
    encoded_test = encoder.transform(test_data[categorical_cols])
    encoded_test_df = pd.DataFrame(
        encoded_test,
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    
    zscaled_test = zscaler.transform(test_data[numerical_cols])
    zscaled_test_df = pd.DataFrame(
        zscaled_test,
        columns=[f'z_{col}' for col in numerical_cols]
    )
    
    # 对测试集进行Min-Max缩放
    minmax_test = test_data[numerical_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)
    minmax_test_df = minmax_test.add_prefix('mm_')
    
    # 合并测试集特征
    processed_test_data = pd.concat([
        zscaled_test_df,
        minmax_test_df,
        encoded_test_df,
        test_data['label']
    ], axis=1)
    
    # 处理测试集标签
    y_test, _ = process_labels(processed_test_data)
    X_test = processed_test_data.drop('label', axis=1)
    
    print(f"测试集大小: {len(X_test):,}")
    
    # 7. 保存所有处理后的数据和模型
    print("\n保存处理后的数据...")
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
    
    print("\n预处理完成！所有数据已保存到data/processed目录")

if __name__ == "__main__":
    main()