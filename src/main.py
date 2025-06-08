import os
import sys
import time
from datetime import datetime

def print_section(title):
    """打印带格式的节标题"""
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50 + "\n")

def run_pipeline():
    """运行完整的训练和评估流程"""
    start_time = time.time()
    
    try:
        # 记录开始时间
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 数据预处理
        print_section("第1步：数据预处理")
        from data_preprocessing import main as preprocess_data
        processed_data_path = os.path.join('data', 'processed', 'X_train.csv')
        
        if not os.path.exists(processed_data_path):
            print("开始数据预处理...")
            preprocess_data()
        else:
            print("检测到已处理的数据文件，跳过预处理步骤")
            print("如需重新预处理，请删除:")
            print(f"- {processed_data_path}")
        
        # 2. 模型训练
        print_section("第2步：模型训练")
        import model_training
        if not os.path.exists(os.path.join('models', 'random_forest_pipeline.pkl')):
            print("开始模型训练...")
            model_training.train_random_forest_model()
        else:
            print("检测到已训练的模型文件，跳过训练步骤")
            print("如需重新训练，请删除 models/random_forest_pipeline.pkl 文件")
        
        # 3. 模型评估
        print_section("第3步：模型评估")
        import model_evaluation
        print("在完整测试集上评估模型性能...")
        model_evaluation.evaluate_model_on_test_data()
        
        # 计算总运行时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print_section("完成")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总运行时间: {hours}小时 {minutes}分钟 {seconds}秒")
        print("\n评估结果已保存到 results 目录")
        
    except KeyboardInterrupt:
        print("\n\n[中断] 用户终止了程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[错误] 程序执行失败: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # 确保在正确的目录中运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # 运行流程
    run_pipeline()