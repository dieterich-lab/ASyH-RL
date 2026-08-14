"""
独立评估脚本 (Train/Test Split 版)

功能：
- 读取 train.csv, test.csv 和 synthetic_data_gmm_optimized.csv
- Fidelity: 对比 Synthetic vs Real Train (衡量拟合程度)
- Utility: 执行 TSTR (Train on Synthetic, Test on Real Test) 并与 Baseline (Train on Real Train, Test on Real Test) 对比
- 输出详细报告至 results/evaluation_report.txt
"""

from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# -----------------------------------------------------------
# 关键修改：引用 utility_evaluator_2
# -----------------------------------------------------------
try:
    # 尝试直接导入（如果你把它移到了当前目录）
    from utility_evaluator_2 import UtilityEvaluator
    from evaluator.fidelity_evaluator import FidelityEvaluator
except ImportError:
    try:
        # 如果还在 evaluator 文件夹里
        from evaluator.utility_evaluator_2 import UtilityEvaluator
        from evaluator.fidelity_evaluator import FidelityEvaluator
    except ImportError:
        print("严重警告: 无法导入 UtilityEvaluator，请检查文件名和路径！")

# -----------------------------
# 默认路径配置
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
DEFAULT_TEST_PATH = os.path.join(BASE_DIR, "test.csv")
# 这里指向你主程序生成的那个文件
REPORT_NAME = 'copula_gan'
DEFAULT_SYN_PATH = os.path.join(BASE_DIR,"synthetic_data_gmm_optimized.csv") # "synthetic_data_wo_clean.csv") 
DEFAULT_REPORT_PATH = os.path.join(BASE_DIR, "results", f"evaluation_{REPORT_NAME}.txt")

print(f"BASE DIR: {BASE_DIR}")

@dataclass
class CombinedConfig:
    target_column: str
    discrete_columns: Optional[List[str]] = None
    # 数据与报告编码
    csv_encoding: str = "gbk" # 你的数据如果是gbk，这里要是gbk
    report_encoding: str = "utf-8"
    # 相似度权重
    ks_weight: float = 0.30
    js_weight: float = 0.30
    corr_weight: float = 0.20
    tvd_weight: float = 0.20
    # 总分融合权重
    alpha_fidelity: float = 0.5
    # 随机种子 (用于 Utility 的多次评估)
    seeds: Tuple[int, ...] = (13, 27, 42, 77, 2025, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)


class IndependentEvaluator:
    def __init__(
        self,
        train_csv_path: str = DEFAULT_TRAIN_PATH,
        test_csv_path: str = DEFAULT_TEST_PATH,
        synthetic_csv_path: str = DEFAULT_SYN_PATH,
        config: Optional[CombinedConfig] = None
    ) -> None:
        if config is None:
            raise ValueError("必须提供 CombinedConfig（至少包含 target_column）。")
        self.config = config

        # 检查文件
        for path, name in [(train_csv_path, "训练集"), (test_csv_path, "测试集"), (synthetic_csv_path, "合成数据")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name}不存在: {path}")

        # 读取数据
        def load_csv(p):
            try: 
                df =  pd.read_csv(p, encoding=self.config.csv_encoding, encoding_errors="ignore", sep=',')
                if len(df.columns) <= 1:
                    df = pd.read_csv(p, encoding='utf-8', encoding_errors="ignore", sep=';')
                return df
            except: 
                df = pd.read_csv(p, encoding='utf-8', encoding_errors="ignore", sep=',')
                if len(df.columns) <= 1:
                    df = pd.read_csv(p, encoding='utf-8', encoding_errors="ignore", sep=';')
                return df

        self.train_df = load_csv(train_csv_path)
        self.test_df = load_csv(test_csv_path)
        self.syn_df = load_csv(synthetic_csv_path)

        # apply the class method `convert_numeric_strings_df` to 3 dataframes defined above
        self.train_df = self.convert_numeric_strings_df(self.train_df)
        self.test_df = self.convert_numeric_strings_df(self.test_df)
        self.syn_df = self.convert_numeric_strings_df(self.syn_df)

        # apply fillna to each of the dataframes, fill nan with median values
        self.train_df.fillna(self.train_df.median(numeric_only=True), inplace=True)
        self.test_df.fillna(self.test_df.median(numeric_only=True), inplace=True)
        self.syn_df.fillna(self.syn_df.median(numeric_only=True), inplace=True)

        print(f"数据加载完毕: Train={len(self.train_df)}, Test={len(self.test_df)}, Syn={len(self.syn_df)}")

        # 最小必要清洗：对齐列
        # 以 Train 的列为准
        common_cols = [c for c in self.train_df.columns if c in self.syn_df.columns]
        
        if self.config.target_column not in common_cols:
            # 尝试修复列名空格等问题
            common_cols = [c.strip() for c in common_cols]
            if self.config.target_column not in common_cols:
                 raise ValueError(f"目标列 `{self.config.target_column}` 缺失。")
            
        # 过滤列
        self.train_df = self.train_df[common_cols]
        self.syn_df = self.syn_df[common_cols]
        # Test 集也要对齐
        self.test_df = self.test_df[[c for c in common_cols if c in self.test_df.columns]]

        # 自动推断离散列
        if self.config.discrete_columns is None:
            self.config.discrete_columns = self._infer_discrete_columns(self.train_df)


    def convert_numeric_strings(self, series):
        """                                                                                                                                           
        Detects numeric values stored as strings (with either '.' or ',' as decimal 
        separator) in a pandas Series and converts them to float or int.     
        Non-numeric strings, NaNs, and already-numeric values are left unchanged.
                                                                                                                                                    
        Parameters                                                           
        ----------                                                           
        series : pd.Series                                                   
            Input column, potentially containing a mix of str, int, float, and NaN.
                            
        Returns                                                              
        -------                                                              
        pd.Series                                                            
            Series with numeric-looking strings converted to float/int.      
        """                                                                                                                                           
                                                                            
        # Regex pattern: optional sign, digits, optional decimal part (. or ,)
        pattern = re.compile(r'^-?\d+([.,]\d+)?$')                                                                                                    
                                        
        def _convert_value(val):                                             
            # 1. Leave NaN untouched                                         
            if pd.isna(val):                                                                                                                          
                return val                                                   
                                                                            
            # 2. If already int or float, leave as is
            if isinstance(val, (int, float, np.integer, np.floating)):
                return val                                                   
                                                                            
            # 3. If string, check if it matches numeric pattern              
            if isinstance(val, str):                                         
                val_stripped = val.strip()
                if pattern.match(val_stripped):
                    # Replace comma decimal separator with point
                    val_normalized = val_stripped.replace(',', '.')
                    float_val = float(val_normalized)
                    
                    # Convert to int if there's no meaningful decimal part
                    if float_val.is_integer():
                        return int(float_val)
                    else:
                        return float_val 
                else:
                    # Not a numeric-looking string -> leave unchanged
                    return val
            
            # 4. Any other type -> leave unchanged
            return val
        return series.apply(_convert_value)
    

    def convert_numeric_strings_df(self, df, columns=None):
        """                                                                                                                                           
        Apply convert_numeric_strings to multiple columns in a dataframe.
        If columns is None, applies to all object-dtype columns.
        """                                                                                                                                           
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include='object').columns.tolist()
            
        for col in columns:                                                  
            df[col] = self.convert_numeric_strings(df[col])          
            
        return df


    @staticmethod
    def _infer_discrete_columns(df: pd.DataFrame) -> List[str]:
        """简单的离散列推断"""
        discrete_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() <= 10:
                discrete_cols.append(col)
        return sorted(list(set(discrete_cols)))

    def compute_fidelity(self) -> Dict[str, float]:
        """
        计算 Fidelity: 比较 Synthetic 与 Real Train
        """
        print("正在计算 Fidelity (Syn vs Train)...")
        
        eval_cols = [c for c in self.config.discrete_columns if c in self.train_df.columns]
        
        # import ipdb; ipdb.set_trace()

        fidelity_eval = FidelityEvaluator(
            real_data=self.train_df, # 基准是训练集
            synthetic_data=self.syn_df,
            discrete_columns=eval_cols
        )
        
        uni = fidelity_eval.evaluate_univariate_distributions()
        corr = fidelity_eval.evaluate_correlation_similarity()
        tvd = fidelity_eval.evaluate_tvd()

        # 转换得分
        ks_score = float(max(0.0, min(1.0, 1.0 - uni.get("overall_ks_score", 1.0))))
        js_score = float(max(0.0, min(1.0, 1.0 - uni.get("overall_js_score", 1.0))))
        
        # 相关性映射 [-1, 1] -> [0, 1]
        raw_corr = corr.get("correlation_correlation", 0.0)
        corr_score = (float(raw_corr) + 1.0) / 2.0
        
        tvd_score = float(max(0.0, min(1.0, tvd.get("tvd_score", 0.0))))

        # 加权计算
        weights = np.array([self.config.ks_weight, self.config.js_weight, self.config.corr_weight, self.config.tvd_weight])
        scores = np.array([ks_score, js_score, corr_score, tvd_score])
        fidelity = float(np.dot(scores, weights / weights.sum()))

        return {
            "ks_score": ks_score,
            "js_score": js_score,
            "correlation_score": corr_score,
            "tvd_score": tvd_score,
            "fidelity_score": fidelity,
        }

    def compute_utility(self) -> Dict[str, float]:
        """
        计算 Utility: 多次运行 TSTR 并取中位数
        传入: Real Train (Baseline), Real Test (Validation), Synthetic (TSTR Train)
        """
        print("正在计算 Utility (TSTR)...")
        utility_scores = []
        
        for seed in self.config.seeds:
            # 实例化 UtilityEvaluator (传入三份数据)
            try:
                util_eval = UtilityEvaluator(
                    real_train=self.train_df,
                    real_test=self.test_df,
                    synthetic_data=self.syn_df,
                    target_column=self.config.target_column,
                    discrete_columns=self.config.discrete_columns
                )
                # 这里的 random_state 仅控制模型初始化的随机性
                report = util_eval.get_detailed_report(random_state=seed)
                
                # 提取 Utility Score (通常是 Ratio: TSTR Acc / Baseline Acc)
                score = report.get('utility_score', 0.0)
                utility_scores.append(score)
                
            except TypeError as e:
                print(f"错误: UtilityEvaluator 参数不匹配: {e}")
                print("请确保您已更新 utility_evaluator_2.py 以接受 real_train 和 real_test。")
                return {"utility_score": 0.0}
            except Exception as e:
                 print(f"Utility计算发生未知错误: {e}")
                 return {"utility_score": 0.0}

        # 聚合结果
        utility_final = float(np.median(utility_scores)) if utility_scores else 0.0
        
        return {
            "utility_score": utility_final,
            "utility_mean": float(np.mean(utility_scores)) if utility_scores else 0.0,
            "utility_std": float(np.std(utility_scores)) if utility_scores else 0.0,
            "raw_scores": utility_scores
        }

    def evaluate(self) -> Dict:
        # in `compute_fidelity()` resolved issues 
        # (one - caused by too short arrays arg-s of Pearson-R method)
        # and another one was related to dropna() pandas methods
        fidelity = self.compute_fidelity()
        # yet, it fails to work on the line below, the program freeezes for NA reason
        utility = self.compute_utility()
        
        alpha = self.config.alpha_fidelity
        combined = alpha * fidelity["fidelity_score"] + (1.0 - alpha) * utility["utility_score"]
        
        return {
            "fidelity": fidelity,
            "utility": utility,
            "combined_score": combined,
            "config": {
                "alpha": alpha,
                "target": self.config.target_column
            }
        }

def save_report(result: Dict, path: str, encoding: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding=encoding) as f:
        f.write("="*50 + "\n")
        f.write("  INDEPENDENT EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Combined Score: {result['combined_score']:.6f}\n")
        f.write(f"(Alpha: {result['config']['alpha']}, 1-Alpha: {1-result['config']['alpha']})\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("1. Fidelity Metrics (Syn vs Train)\n")
        f.write("-" * 30 + "\n")
        for k, v in result['fidelity'].items():
            f.write(f"{k:<20}: {v:.6f}\n")
            
        f.write("\n" + "-" * 30 + "\n")
        f.write("2. Utility Metrics (Test on Real Test)\n")
        f.write("-" * 30 + "\n")
        u = result['utility']
        f.write(f"Utility Score (Median): {u['utility_score']:.6f}\n")
        f.write(f"Mean: {u['utility_mean']:.6f}, Std: {u['utility_std']:.6f}\n")
        f.write(f"Raw Scores: {u['raw_scores']}\n")

if __name__ == "__main__":

    # 手动指定路径（为了保险起见，这里写死）
    train_path = "puf_narrow/train.csv"
    test_path = "puf_narrow/test.csv"
    # syn_path = "tables4comparison/synthetic_data_gmm_optimized_cut.csv" # 确保这个名字和你生成的一致
    # syn_path = "tables4comparison/out_uf_ctgan_util_v3.csv"
    # syn_path = "tables4comparison/out_uf_gaussian_copula_util_v3.csv"
    syn_path = "tables4comparison/out_uf_copula_gan_util_v3.csv"

    df_train = pd.read_csv(train_path, sep=';')
    if len(df_train.columns) <= 1:
        df_train = pd.read_csv(train_path, sep=',')

    cont_columns = []
    for col in df_train.columns:
        if df_train[col].nunique() > 20:
            cont_columns.append(col)

    discrete_columns = list([col for col in df_train.columns if col not in cont_columns])
    print(f"Continuous columns: {cont_columns}")
    print(f"Discrete columns: {discrete_columns}")

    # 配置 (根据你的数据列名修改)
    cfg = CombinedConfig(
        target_column="score-eq5d-5l_start", # "status",
        discrete_columns=discrete_columns, # [
            # 'agegroup', 'sex', 'location', 'status', 'SAHfh', 'drink', 'smoke',
            # 'HBP', 'DM', 'IS', 'CAD', 'number', 'shape1', 'maxgroup', 'shape2'
        # ],
        alpha_fidelity=0.5
    )

    evaluator = IndependentEvaluator(train_path, test_path, syn_path, cfg)
    results = evaluator.evaluate()
    
    save_report(results, DEFAULT_REPORT_PATH, "utf-8")
    print(f"\n评估完成！\n综合得分: {results['combined_score']:.4f}")
    print(f"完整报告已保存至: {DEFAULT_REPORT_PATH}")
