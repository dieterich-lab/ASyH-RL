"""
统计相似度评估器 (Fidelity Evaluator)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FidelityEvaluator:
    """统计相似度评估器"""
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                 discrete_columns: List[str] = None):
        """
        初始化评估器
        """
        # 确保真实数据和合成数据有相同的列
        common_cols = [col for col in real_data.columns if col in synthetic_data.columns]
        if len(common_cols) == 0:
            raise ValueError("真实数据和合成数据没有共同的列！")
        
        # import ipdb; ipdb.set_trace()
        
        # 只使用共同的列
        self.real_data = real_data[common_cols].copy()
        self.synthetic_data = synthetic_data[common_cols].copy()
        self.discrete_columns = [col for col in (discrete_columns if discrete_columns else []) 
                                if col in common_cols]
        self.continuous_columns = [col for col in common_cols 
                                   if col not in self.discrete_columns]
        
        # 处理连续列中的NaN值（用均值填充）
        for col in self.continuous_columns:
            if col in self.real_data.columns and self.real_data[col].isnull().any():
                fill_val = self.real_data[col].median()
                self.real_data[col] = self.real_data[col].fillna(fill_val)

            if col in self.synthetic_data.columns and self.synthetic_data[col].isnull().any():
                fill_val = self.synthetic_data[col].median()
                self.synthetic_data[col] = self.synthetic_data[col].fillna(fill_val)
        
    def evaluate_univariate_distributions(self) -> Dict[str, float]:
        """
        评估单变量分布相似度
        """
        results = {
            'ks_statistics': {},
            'js_divergences': {},
            'overall_ks_score': 0.0,
            'overall_js_score': 0.0
        }
        
        ks_scores = []
        js_scores = []
        
        # 评估连续变量
        for col in self.continuous_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                # KS检验
                ks_stat, _ = stats.ks_2samp(
                    self.real_data[col], # .fillna(self.real_data[col].median()), 
                    self.synthetic_data[col] # .fillna(self.synthetic_data[col].median())
                )
                results['ks_statistics'][col] = ks_stat
                ks_scores.append(ks_stat)
                
                # JS散度（基于直方图）
                try:
                    real_hist, bins = np.histogram(self.real_data[col], # .fillna(self.real_data[col].median()), 
                                                   bins=50, density=True)
                    syn_hist, _ = np.histogram(self.synthetic_data[col], # .fillna(self.synthetic_data[col].median()), 
                                              bins=bins, density=True)
                    
                    # 归一化
                    real_hist = real_hist / (real_hist.sum() + 1e-10)
                    syn_hist = syn_hist / (syn_hist.sum() + 1e-10)
                    
                    js_div = jensenshannon(real_hist + 1e-10, syn_hist + 1e-10)
                    results['js_divergences'][col] = js_div
                    js_scores.append(js_div)
                except:
                    pass
        
        # 评估离散变量
        for col in self.discrete_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                try:
                    # 获取类别分布
                    real_counts = self.real_data[col].value_counts(normalize=True)
                    syn_counts = self.synthetic_data[col].value_counts(normalize=True)
                    
                    # 对齐类别
                    all_categories = set(real_counts.index) | set(syn_counts.index)
                    real_probs = [real_counts.get(cat, 0) for cat in all_categories]
                    syn_probs = [syn_counts.get(cat, 0) for cat in all_categories]
                    
                    # 归一化
                    real_probs = np.array(real_probs)
                    syn_probs = np.array(syn_probs)
                    real_probs = real_probs / (real_probs.sum() + 1e-10)
                    syn_probs = syn_probs / (syn_probs.sum() + 1e-10)
                    
                    # JS散度
                    js_div = jensenshannon(real_probs + 1e-10, syn_probs + 1e-10)
                    results['js_divergences'][col] = js_div
                    js_scores.append(js_div)
                except:
                    pass
        
        # 计算总体得分（越接近0越好）
        results['overall_ks_score'] = np.mean(ks_scores) if ks_scores else 1.0
        results['overall_js_score'] = np.mean(js_scores) if js_scores else 1.0
        
        return results
    
    def evaluate_tvd(self) -> Dict[str, float]:
        """
        评估总变分距离 (Total Variation Distance, TVD)
        
        TVD(P, Q) = 0.5 * sum(|P(x) - Q(x)|)
        范围: [0, 1]，0表示完全相同，1表示完全不同
        
        Returns:
            包含各列的TVD和总体TVD得分的字典
        """
        results = {
            'tvd_by_column': {},
            'overall_tvd': 0.0,
            'tvd_score': 0.0  # 转换为得分 (1 - TVD)
        }
        
        tvd_values = []
        
        # 评估连续变量
        for col in self.continuous_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                try:
                    # 使用直方图将连续变量离散化
                    real_values = self.real_data[col].fillna(self.real_data[col].median()).values
                    syn_values = self.synthetic_data[col].fillna(self.synthetic_data[col].median()).values
                    
                    if len(real_values) > 0 and len(syn_values) > 0:
                        # 确定统一的bins
                        min_val = min(real_values.min(), syn_values.min())
                        max_val = max(real_values.max(), syn_values.max())
                        
                        # 使用50个bins
                        bins = np.linspace(min_val, max_val, 51)
                        
                        # 计算直方图（概率分布）
                        real_hist, _ = np.histogram(real_values, bins=bins, density=False)
                        syn_hist, _ = np.histogram(syn_values, bins=bins, density=False)
                        
                        # 归一化为概率分布
                        real_probs = real_hist / (real_hist.sum() + 1e-10)
                        syn_probs = syn_hist / (syn_hist.sum() + 1e-10)
                        
                        # 计算TVD: 0.5 * sum(|P(x) - Q(x)|)
                        tvd = 0.5 * np.sum(np.abs(real_probs - syn_probs))
                        
                        results['tvd_by_column'][col] = tvd
                        tvd_values.append(tvd)
                except Exception as e:
                    # 如果计算失败，跳过该列
                    pass
        
        # 评估离散变量
        for col in self.discrete_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                try:
                    # 获取类别分布
                    real_counts = self.real_data[col].value_counts(normalize=True)
                    syn_counts = self.synthetic_data[col].value_counts(normalize=True)
                    
                    # 对齐类别
                    all_categories = set(real_counts.index) | set(syn_counts.index)
                    real_probs = np.array([real_counts.get(cat, 0) for cat in all_categories])
                    syn_probs = np.array([syn_counts.get(cat, 0) for cat in all_categories])
                    
                    # 归一化
                    real_probs = real_probs / (real_probs.sum() + 1e-10)
                    syn_probs = syn_probs / (syn_probs.sum() + 1e-10)
                    
                    # 计算TVD
                    tvd = 0.5 * np.sum(np.abs(real_probs - syn_probs))
                    
                    results['tvd_by_column'][col] = tvd
                    tvd_values.append(tvd)
                except Exception as e:
                    # 如果计算失败，跳过该列
                    pass
        
        # 计算总体TVD
        results['overall_tvd'] = np.mean(tvd_values) if tvd_values else 1.0
        
        # 转换为得分 (1 - TVD)，得分越高越好
        results['tvd_score'] = 1.0 - results['overall_tvd']
        
        return results
    
    def evaluate_correlation_similarity(self) -> Dict[str, float]:
        """
        评估相关性矩阵相似度
        """
        results = {}
        
        # 只使用连续变量计算相关性
        if len(self.continuous_columns) > 0:
            # try:
            real_corr = self.real_data[self.continuous_columns].corr()
            syn_corr = self.synthetic_data[self.continuous_columns].corr()
            
            # Frobenius范数距离
            frobenius_dist = np.linalg.norm(real_corr - syn_corr, 'fro')
            results['frobenius_distance'] = frobenius_dist
            
            # 相关系数矩阵的相关性
            real_corr_flat = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
            syn_corr_flat = syn_corr.values[np.triu_indices_from(syn_corr.values, k=1)]
            
            if len(real_corr_flat) >= 2: # 0:
                corr_corr, _ = stats.pearsonr(real_corr_flat, syn_corr_flat)
                results['correlation_correlation'] = corr_corr
            else:
                results['correlation_correlation'] = 1.0
            
            # 平均绝对误差
            mae = np.mean(np.abs(real_corr - syn_corr))
            results['correlation_mae'] = mae
                
            # except Exception as e:
            #     # import ipdb; ipdb.set_trace()
            #     print(f"相关性评估出错: {e}")
            #     results['frobenius_distance'] = 0.0
            #     results['correlation_correlation'] = 1.0
            #     results['correlation_mae'] = 0.0
        else:
            results['frobenius_distance'] = 0.0
            results['correlation_correlation'] = 1.0
            results['correlation_mae'] = 0.0
        
        return results
    
    def compute_fidelity_score(self) -> float:
        """
        计算综合统计相似度得分 
        """
        # 获取各项指标
        univariate_results = self.evaluate_univariate_distributions()
        correlation_results = self.evaluate_correlation_similarity()
        tvd_results = self.evaluate_tvd()
        
        # KS统计量转换为得分（1 - KS）
        ks_score = 1.0 - univariate_results['overall_ks_score']
        
        # JS散度转换为得分（1 - JS）
        js_score = 1.0 - univariate_results['overall_js_score']
        
        # 相关性得分
        corr_score = correlation_results['correlation_correlation']
        
        # TVD得分（已经转换为得分形式）
        tvd_score = tvd_results['tvd_score']
        
        # 加权平均（包含TVD）
        weights = [0.3, 0.3, 0.2, 0.2]  # KS, JS, Correlation, TVD
        fidelity_score = (weights[0] * ks_score + 
                         weights[1] * js_score + 
                         weights[2] * corr_score +
                         weights[3] * tvd_score)
        
        return max(0.0, min(1.0, fidelity_score))
    
    def get_detailed_report(self) -> Dict:
        """
        获取详细评估报告
        
        Returns:
            详细评估结果字典
        """
        univariate_results = self.evaluate_univariate_distributions()
        correlation_results = self.evaluate_correlation_similarity()
        tvd_results = self.evaluate_tvd()
        fidelity_score = self.compute_fidelity_score()
        
        return {
            'fidelity_score': fidelity_score,
            'univariate_distributions': univariate_results,
            'correlation_similarity': correlation_results,
            'tvd_results': tvd_results,
            'tvd_score': tvd_results['tvd_score']  # 方便访问
        }

