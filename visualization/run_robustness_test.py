# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from evaluator.independent_evaluator import IndependentEvaluator

class Config:
    def __init__(self):
        self.target_column = "status"  
        self.discrete_columns = [
            'agegroup', 'sex', 'location', 'status', 'SAHfh', 'drink', 'smoke',
            'HBP', 'DM', 'IS', 'CAD', 'number', 'shape1', 'maxgroup', 'shape2'
        ]

def run_experiment_10_times():
    print("="*60)
    print(" 10-times Robustness Test with Different Seeds ")
    print("="*60)

    cfg = Config()
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    models = {
        "CLEO (Ours)": "synthetic_data_gmm_optimized.csv",
        "TVAE": "synthetic_data_tvae.csv",
        "CTGAN": "synthetic_data_gan.csv"
    }

    # 预设10个不同的随机种子以产生波动
    seeds = [42, 123, 2026, 777, 888, 10, 99, 1024, 5, 2025]
    metrics_to_collect = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    final_results = {}

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: file {path} not found.")
            continue
            
        print(f"\n Evaluating {name} ...")
        run_scores = {m: [] for m in metrics_to_collect}
        
        for i in range(10):
            current_seed = seeds[i]
            # 传入随机种子确保每次训练的随机性
            evaluator = IndependentEvaluator(train_path, test_path, path, cfg, seed=current_seed)
            res = evaluator.evaluate()
            
            # 记录随机森林的指标
            rf_metrics = res['utility']['utility_scores']['RF']
            for m in metrics_to_collect:
                run_scores[m].append(rf_metrics[m])
            
            print(f"  Run {i+1}/10 (Seed {current_seed}) - F1: {rf_metrics['f1']:.4f}")

        final_results[name] = run_scores

    # --- 打印统计结果 (使用 +/- 替代特殊符号以防报错) ---
    print("\n" + "="*20 + " Final Results (Mean +/- SD) " + "="*20)
    report_rows = []
    for name, scores in final_results.items():
        row = {"Model": name}
        for m in metrics_to_collect:
            mean_val = np.mean(scores[m])
            std_val = np.std(scores[m])
            row[m] = f"{mean_val:.4f} +/- {std_val:.4f}"
        report_rows.append(row)

    df_report = pd.DataFrame(report_rows)
    print(df_report.to_string(index=False))

    # --- 显著性检验 ---
    if "CLEO (Ours)" in final_results and "TVAE" in final_results:
        f1_cleo = final_results["CLEO (Ours)"]["f1"]
        f1_tvae = final_results["TVAE"]["f1"]
        try:
            stat, p = wilcoxon(f1_cleo, f1_tvae)
            print(f"\nWilcoxon Test (F1): CLEO vs TVAE, p-value = {p:.5f}")
            if p < 0.05: print(" Significant Difference (*)")
        except Exception as e:
            print(f"\nWilcoxon Test skipped: {e}")

    df_report.to_csv("robustness_results.csv", index=False)

if __name__ == "__main__":
    run_experiment_10_times()