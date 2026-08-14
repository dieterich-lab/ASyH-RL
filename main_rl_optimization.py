# -*- coding: utf-8 -*-
"""
Main entry point for Cardio-Privacy-GMM optimization.
Synchronized with 80/10/10 data splitting.
"""
import os
import sys
import pandas as pd
import numpy as np
from rl_optimizer import RLOptimizer
from sklearn.compose import ColumnTransformer


def main():
    print("=" * 80)
    print("GMM Hyperparameter Optimization (RL-driven with Val Set)")
    print("=" * 80)
    
    # 1. 路径配置
    train_data_path = 'puf_narrow/train.csv' # 80%
    val_data_path = 'puf_narrow/val.csv'     # 10%
    
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        print("错误: 找不到 train.csv 或 val.csv。请先运行 split_data.py！")
        sys.exit(1)
    
    # 2. 初始化优化器 (解决 TypeError: 统一参数名为 val_path)
    df = pd.read_csv(train_data_path, sep=',', on_bad_lines='skip')
    # discrete_columns = ['类型', 'agegroup', 'sex', 'location', 'status', 'SAHfh', 'drink', 'smoke',
                        # 'HBP', 'DM', 'IS', 'CAD', 'number', 'shape1', 'maxgroup', 'shape2']
    # discrete_columns = ['cohort']
    cont_columns = []
    for col in df.columns:
        if df[col].nunique() > 20:
            cont_columns.append(col)

    discrete_columns = list([col for col in df.columns if col not in cont_columns])
    # import ipdb; ipdb.set_trace()
    
    optimizer = RLOptimizer(
        train_path=train_data_path, 
        val_path=val_data_path, # 这里的参数名必须叫 val_path
        discrete_columns=discrete_columns
    )
    
    # 3. 执行优化
    print("\n[CLEO-RL] 优化中...")
    optimizer.optimize(episodes=30)
    
    # 4. 生成最终数据
    best_params = optimizer.get_best_params()
    print(f"\n最佳参数已选定: {best_params}")
    
    output_csv = 'synthetic_data_gmm_optimized_fixed.csv'
    optimizer.generate_with_best_params(output_csv, n_samples=6900)
    print(f"✓ 合成数据已保存至: {output_csv}")

if __name__ == "__main__":
    main()
