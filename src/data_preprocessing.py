import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import json
import os

# 确保输出目录存在
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# 完整41个字段定义
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

def custom_minmax_scale(series):
    arr = series.values
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val == min_val:
        return pd.Series(np.zeros_like(arr))
    return pd.Series((arr - min_val) / (max_val - min_val))

def load_and_filter_data(data_path):
    data = pd.read_csv(data_path, names=columns, compression='gzip')
    data = data[data['protocol_type'].isin(['tcp', 'udp'])]
    data = data[(data['src_bytes'] > 0) & (data['dst_bytes'] > 0)]
    return data

def preprocess_data(data):
    # 处理分类特征
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # 处理数值特征
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['label']]
    
    # 标准化和归一化
    zscaler = StandardScaler()
    zscaled_data = zscaler.fit_transform(data[numerical_cols])
    zscaled_df = pd.DataFrame(zscaled_data, columns=[f'z_{col}' for col in numerical_cols])
    
    minmax_data = data[numerical_cols].apply(custom_minmax_scale)
    minmax_df = minmax_data.add_prefix('mm_')
    
    # 合并所有特征
    processed_data = pd.concat([zscaled_df, minmax_df, encoded_df, data['label']], axis=1)
    
    return processed_data, zscaler, encoder

if __name__ == "__main__":
    # 加载和预处理数据
    data = load_and_filter_data('../data/raw/kddcup.data.gz')
    processed_data, zscaler, encoder = preprocess_data(data)
    
    # 定义攻击类型映射
    attack_types = {
        'normal.': 0,
        **{k: 1 for k in [
            'back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 
            'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 
            'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 
            'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'
        ]}
    }
    
    # 处理标签
    y = processed_data['label'].map(attack_types).fillna(1)
    X = processed_data.drop('label', axis=1)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # 保存数据
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False, header=['attack'])
    X_val.to_csv('../data/processed/X_val.csv', index=False)
    y_val.to_csv('../data/processed/y_val.csv', index=False, header=['attack'])
    X_test.to_csv('../data/processed/X_test.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False, header=['attack'])
    
    # 保存预处理对象
    pd.to_pickle(zscaler, '../models/zscaler.pkl')
    pd.to_pickle(encoder, '../models/encoder.pkl')
    
    # 保存攻击类型映射
    with open('../data/processed/attack_mapping.json', 'w') as f:
        json.dump(attack_types, f)
    
    print("预处理完成！数据已保存到data/processed目录")