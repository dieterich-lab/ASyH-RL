# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from evaluator.independent_evaluator import IndependentEvaluator, CombinedConfig


def run_experiment_10_times():
    print("="*60)
    print(" 10-times Robustness Test with Different Seeds ")
    print("="*60)

    train_path = 'puf_wide/train.csv'
    test_path = 'puf_wide/test.csv'
    
    models = {
        "GMM": "tables4comparison/synthetic_data_gmm_wide_full.csv",
        "Gaussian Copula": "tables4comparison/synthetic_data_GaussianCopulaModel_wide_rl_40ep.csv",
        "TVAE": "tables4comparison/synthetic_data_TVAEModel_wide_rl_40ep.csv",
        "CTGAN": "tables4comparison/synthetic_data_CTGANModel_wide_rl_40ep.csv"
    }

    # Ԥ��10����ͬ����������Բ�������
    seeds = [42, 123, 2026, 777, 888, 10, 99, 1024, 5, 2025]
    metrics_to_collect = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    final_results = {}

    CombinedConfig.seeds = seeds
    CombinedConfig.discrete_columns = ['date_infection_exact', 'gender', 'ethnicity', 'gecco_537', 
                                       'gecco_552', 'gecco_550', 'gecco_551', 
                                       'gecco_538', 'gecco_553', 'gecco_665', 'gecco_658', 
                                       'gecco_655', 'gecco_522', 'gecco_524', 'gecco_656', 
                                       'gecco_533', 'gecco_534', 'gecco_535', 'gecco_536', 
                                       'gecco_528', 'gecco_529_a', 'gecco_529_b', 'gecco_530', 
                                       'gecco_664', 'gecco_539', 'gecco_540', 
                                       'gecco_541', 'gecco_542', 'gecco_543', 'gecco_544', 
                                       'gecco_545', 'gecco_546', 'gecco_547', 
                                       'gecco_548', 'score-eq5d-5l', 'Cohort.numeric']
    CombinedConfig.target_column = "score-eq5d-5l"
    cfg = CombinedConfig

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: file {path} not found.")
            continue
            
        print(f"\n Evaluating {name} ...")
        run_scores = {m: [] for m in metrics_to_collect}
        
        evaluator = IndependentEvaluator(train_path, test_path, path, cfg)
        res = evaluator.evaluate()
            
        final_results[name] = run_scores

    # --- ��ӡͳ�ƽ�� (ʹ�� +/- �����������Է�����) ---
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

    # --- �����Լ��� ---
    if "GMM" in final_results and "Gaussian Copula" in final_results:
        f1_cleo = final_results["GMM"]["f1"]
        f1_tvae = final_results["Gaussian Copula"]["f1"]
        try:
            stat, p = wilcoxon(f1_cleo, f1_tvae)
            print(f"\nWilcoxon Test (F1): GMM vs TVAE, p-value = {p:.5f}")
            if p < 0.05: print(" Significant Difference (*)")
        except Exception as e:
            print(f"\nWilcoxon Test skipped: {e}")

    df_report.to_csv("robustness_results.csv", index=False)

if __name__ == "__main__":
    run_experiment_10_times()
