import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置项目根目录（假设项目根目录是脚本目录的上一级）
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 更新数据和模型路径
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_test_data(chunksize=100000):
    """分块加载测试数据以处理大型数据集"""
    print("加载测试数据...")
    
    # 获取总行数
    total_rows = sum(1 for _ in open(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))) - 1
    total_chunks = (total_rows + chunksize - 1) // chunksize
    
    X_chunks = []
    y_chunks = []
    
    # 使用tqdm显示进度
    for i, (chunk_X, chunk_y) in enumerate(tqdm(zip(
        pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), chunksize=chunksize),
        pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), chunksize=chunksize)
    ), total=total_chunks, desc="Loading test data")):
        X_chunks.append(chunk_X)
        y_chunks.append(chunk_y.squeeze())
        
        # 定期清理内存
        if len(X_chunks) >= 5:
            X_chunks = [pd.concat(X_chunks, ignore_index=True)]
            y_chunks = [pd.concat(y_chunks, ignore_index=True)]
    
    X_test = pd.concat(X_chunks, ignore_index=True)
    y_test = pd.concat(y_chunks, ignore_index=True)
    
    # 清理内存
    del X_chunks, y_chunks
    import gc
    gc.collect()
    
    # 确保标签是字符串类型
    y_test = y_test.astype(str)
    
    print(f"测试数据加载完成，共 {len(X_test):,} 条记录")
    return X_test, y_test

def evaluate_model_on_test_data(batch_size=10000):
    """在完整测试集上评估模型性能（添加阈值调整）"""
    try:
        start_time = time.time()
        
        # 1. 加载模型
        print("加载模型...")
        model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'))
        
        # 2. 加载测试数据
        X_test, y_test = load_test_data()
        
        # 3. 设置DoS和Probe的阈值
        dos_threshold = 0.3  # 降低DoS的判定阈值
        probe_threshold = 0.3  # 降低Probe的判定阈值
        
        # 获取所有类别
        classes = model.classes_
        print(f"模型类别: {classes}")
        
        # 4. 分批预测以处理大型数据集
        print("开始预测...")
        predictions = []
        
        for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
            batch_X = X_test.iloc[i:i+batch_size]
            
            # 使用概率预测代替直接分类
            batch_proba = model.predict_proba(batch_X)
            
            # 应用自定义阈值
            batch_pred = []
            for probs in batch_proba:
                # 获取DoS和Probe的索引
                dos_idx = np.where(classes == 'DoS')[0][0]
                probe_idx = np.where(classes == 'Probe')[0][0]
                
                # 应用自定义阈值
                if probs[dos_idx] >= dos_threshold:
                    batch_pred.append('DoS')
                elif probs[probe_idx] >= probe_threshold:
                    batch_pred.append('Probe')
                else:
                    # 对于其他类别，选择概率最高的
                    batch_pred.append(classes[np.argmax(probs)])
            
            predictions.extend(batch_pred)
        
        y_pred = np.array(predictions)
        
        # 5. 评估性能
        print("\n=== 模型在完整测试集上的评估结果 ===")
        
        # 获取所有类别（按字母顺序排序）
        classes = sorted(set(y_test) | set(y_pred))
        
        # 计算整体准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"整体准确率: {accuracy:.4f}")
        
        # 生成分类报告
        print("\n分类报告:")
        class_report = classification_report(y_test, y_pred, target_names=classes)
        print(class_report)
        
        # 保存分类报告到文件
        with open(os.path.join(RESULTS_DIR, 'test_classification_report.txt'), 'w') as f:
            f.write(f"测试集大小: {len(X_test):,}\n")
            f.write(f"整体准确率: {accuracy:.4f}\n\n")
            f.write(class_report)
        
        # 6. 可视化混淆矩阵
        print("\n生成混淆矩阵...")
        plot_confusion_matrix(y_test, y_pred, classes)
        
        # 7. 计算每类攻击的检测率和误报率
        print("\n计算每类攻击的检测率和误报率...")
        calculate_detection_rates(y_test, y_pred, classes)
        
        elapsed_time = time.time() - start_time
        print(f"\n评估完成！总耗时: {elapsed_time:.2f} 秒")
        
        # 8. 保存阈值信息
        threshold_info = {
            'dos_threshold': dos_threshold,
            'probe_threshold': probe_threshold,
            'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'elapsed_time': elapsed_time
        }
        with open(os.path.join(RESULTS_DIR, 'threshold_info.json'), 'w') as f:
            json.dump(threshold_info, f)
        
    except Exception as e:
        print(f"\n[错误] 评估失败: {str(e)}")
        raise

def plot_confusion_matrix(y_true, y_pred, classes):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('完整测试集上的混淆矩阵')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'test_confusion_matrix.png'))
    print(f"混淆矩阵已保存到 {os.path.join(RESULTS_DIR, 'test_confusion_matrix.png')}")

def calculate_detection_rates(y_true, y_pred, classes):
    """计算每类攻击的检测率和误报率"""
    # 创建混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # 计算每类的检测率和误报率
    detection_rates = {}
    false_alarm_rates = {}
    
    for i, cls in enumerate(classes):
        # 检测率 = TP / (TP + FN)
        true_positives = cm[i, i]
        false_negatives = np.sum(cm[i, :]) - true_positives
        detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # 误报率 = FP / (FP + TN)
        false_positives = np.sum(cm[:, i]) - true_positives
        true_negatives = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + true_positives
        false_alarm_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        detection_rates[cls] = detection_rate
        false_alarm_rates[cls] = false_alarm_rate
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Class': classes,
        'Detection Rate': [detection_rates[cls] for cls in classes],
        'False Alarm Rate': [false_alarm_rates[cls] for cls in classes]
    })
    
    # 保存结果
    results.to_csv(os.path.join(RESULTS_DIR, 'detection_rates.csv'), index=False)
    print(f"检测率和误报率已保存到 {os.path.join(RESULTS_DIR, 'detection_rates.csv')}")
    
    # 打印结果
    print("\n每类攻击的检测率和误报率:")
    for cls in classes:
        print(f"{cls}: 检测率 = {detection_rates[cls]:.4f}, 误报率 = {false_alarm_rates[cls]:.4f}")

if __name__ == "__main__":
    evaluate_model_on_test_data()