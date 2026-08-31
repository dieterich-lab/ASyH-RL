"""
可视化工具
生成评估结果的各种图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """评估结果可视化器"""
    
    def __init__(self, save_dir: str = 'results'):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存图表的目录
        """
        self.save_dir = save_dir
        sns.set_style('whitegrid')
        sns.set_palette('husl')
    
    def plot_univariate_distributions(self, real_data: pd.DataFrame, 
                                     synthetic_data: pd.DataFrame,
                                     columns: List[str] = None,
                                     max_cols: int = 6):
        """
        绘制单变量分布对比图
        
        Args:
            real_data: 真实数据
            synthetic_data: 合成数据
            columns: 要绘制的列（None表示所有列）
            max_cols: 最大绘制列数
        """
        if columns is None:
            columns = list(real_data.columns)[:max_cols]
        else:
            columns = columns[:max_cols]
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            
            try:
                # 绘制核密度估计图
                real_data[col].dropna().plot(kind='density', ax=ax, label='Real', alpha=0.7)
                synthetic_data[col].dropna().plot(kind='density', ax=ax, label='Synthetic', alpha=0.7)
                
                ax.set_title(f'Distribution: {col}', fontsize=10)
                ax.set_xlabel('')
                ax.legend()
            except:
                ax.text(0.5, 0.5, f'{col}\n(无法绘制)', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 隐藏多余的子图
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/univariate_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"单变量分布图已保存至: {self.save_dir}/univariate_distributions.png")
    
    def plot_correlation_heatmap(self, real_data: pd.DataFrame, 
                                synthetic_data: pd.DataFrame,
                                continuous_columns: List[str] = None):
        """
        绘制相关性热图对比
        
        Args:
            real_data: 真实数据
            synthetic_data: 合成数据
            continuous_columns: 连续变量列表
        """
        if continuous_columns is None:
            continuous_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 限制列数以保证可读性
        if len(continuous_columns) > 20:
            continuous_columns = continuous_columns[:20]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 真实数据相关性
        real_corr = real_data[continuous_columns].corr()
        sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', center=0, 
                   vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
        axes[0].set_title('Real Data Correlation', fontsize=14, fontweight='bold')
        
        # 合成数据相关性
        syn_corr = synthetic_data[continuous_columns].corr()
        sns.heatmap(syn_corr, ax=axes[1], cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
        axes[1].set_title('Synthetic Data Correlation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相关性热图已保存至: {self.save_dir}/correlation_heatmap.png")
    
    def plot_tstr_results(self, tstr_results: Dict):
        """
        绘制TSTR评估结果
        
        Args:
            tstr_results: TSTR评估结果字典
        """
        models = list(tstr_results['baseline'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            baseline_scores = [tstr_results['baseline'][model][metric] for model in models]
            tstr_scores = [tstr_results['tstr'][model][metric] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[idx].bar(x - width/2, baseline_scores, width, label='Baseline (Real)', alpha=0.8)
            axes[idx].bar(x + width/2, tstr_scores, width, label='TSTR (Synthetic)', alpha=0.8)
            
            axes[idx].set_xlabel('Model', fontsize=11)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=11)
            axes[idx].set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models, rotation=15, ha='right')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/tstr_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"TSTR结果图已保存至: {self.save_dir}/tstr_results.png")
    
    def plot_utility_scores(self, tstr_results: Dict):
        """
        绘制Utility分数条形图
        
        Args:
            tstr_results: TSTR评估结果字典
        """
        utility_scores = tstr_results['utility_scores']
        models = list(utility_scores.keys())
        scores = list(utility_scores.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if s >= 0.8 else '#f39c12' if s >= 0.6 else '#e74c3c' for s in scores]
        bars = ax.barh(models, scores, color=colors, alpha=0.8)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=11)
        
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Score (1.0)')
        ax.axvline(x=0.8, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good Threshold (0.8)')
        
        ax.set_xlabel('Utility Score (TSTR / Baseline)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Machine Learning Utility Scores', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(scores) + 0.15)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/utility_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Utility分数图已保存至: {self.save_dir}/utility_scores.png")
    
    def plot_rl_training_history(self, training_history: Dict):
        """
        绘制强化学习训练历史
        
        Args:
            training_history: 训练历史字典
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 绘制奖励曲线
        episodes = training_history['episodes']
        rewards = training_history['rewards']
        
        axes[0].plot(episodes, rewards, alpha=0.6, label='Reward')
        # 绘制移动平均
        window = min(10, len(rewards) // 10)
        if window > 1:
            rewards_smooth = pd.Series(rewards).rolling(window=window).mean()
            axes[0].plot(episodes, rewards_smooth, linewidth=2, label=f'Moving Avg (window={window})')
        
        axes[0].set_xlabel('Episode', fontsize=11)
        axes[0].set_ylabel('Reward', fontsize=11)
        axes[0].set_title('RL Training: Reward over Episodes', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 绘制探索率衰减曲线
        epsilon = training_history['epsilon']
        axes[1].plot(episodes, epsilon, color='orange', linewidth=2)
        axes[1].set_xlabel('Episode', fontsize=11)
        axes[1].set_ylabel('Epsilon (Exploration Rate)', fontsize=11)
        axes[1].set_title('RL Training: Epsilon Decay', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/rl_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"RL训练历史图已保存至: {self.save_dir}/rl_training_history.png")
    
    def plot_comprehensive_scores(self, evaluation_result: Dict):
        """
        绘制综合评估得分雷达图
        
        Args:
            evaluation_result: 评估结果字典
        """
        scores = evaluation_result['scores']
        fidelity_details = evaluation_result['fidelity_details']
        
        # 准备雷达图数据
        categories = ['Fidelity\n(Overall)', 'Utility\n(TSTR)', 
                     'KS Score', 'JS Score', 'Correlation']
        values = [
            scores['fidelity_score'],
            scores['utility_score'],
            1.0 - fidelity_details['univariate_distributions']['overall_ks_score'],
            1.0 - fidelity_details['univariate_distributions']['overall_js_score'],
            fidelity_details['correlation_similarity']['correlation_correlation']
        ]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Scores', color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.set_title('Comprehensive Evaluation Scores\n(Closer to edge = Better)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/comprehensive_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"综合评分雷达图已保存至: {self.save_dir}/comprehensive_scores.png")
    
    def create_summary_report(self, evaluation_result: Dict, 
                            tstr_results: Dict = None):
        """
        创建文本摘要报告
        
        Args:
            evaluation_result: 评估结果
            tstr_results: TSTR结果（可选）
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("合成数据质量评估报告")
        report_lines.append("Synthetic Data Quality Evaluation Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 基本信息
        report_lines.append("【评估策略】")
        if evaluation_result['use_rl']:
            report_lines.append(f"  使用强化学习策略: 动作 {evaluation_result['action']}")
        else:
            report_lines.append("  使用默认平衡策略")
        
        weights = evaluation_result['weights']
        report_lines.append(f"  权重设置: Fidelity={weights['fidelity_weight']:.2f}, "
                          f"Utility={weights['utility_weight']:.2f}")
        report_lines.append("")
        
        # 核心得分
        scores = evaluation_result['scores']
        report_lines.append("【核心得分】")
        report_lines.append(f"  统计相似度 (Fidelity Score):     {scores['fidelity_score']:.4f}")
        report_lines.append(f"  机器学习有效性 (Utility Score):   {scores['utility_score']:.4f}")
        report_lines.append(f"  综合得分 (Combined Score):        {scores['combined_score']:.4f}")
        report_lines.append("")
        
        # Fidelity详情
        fidelity_details = evaluation_result['fidelity_details']
        report_lines.append("【统计相似度详情】")
        report_lines.append(f"  KS统计量均值: {fidelity_details['univariate_distributions']['overall_ks_score']:.4f}")
        report_lines.append(f"  JS散度均值:   {fidelity_details['univariate_distributions']['overall_js_score']:.4f}")
        corr_sim = fidelity_details['correlation_similarity']
        report_lines.append(f"  相关性相似度: {corr_sim['correlation_correlation']:.4f}")
        report_lines.append(f"  Frobenius距离: {corr_sim['frobenius_distance']:.4f}")
        report_lines.append("")
        
        # Utility详情
        if tstr_results:
            report_lines.append("【机器学习有效性详情 (TSTR)】")
            report_lines.append("  模型性能对比 (Synthetic vs Real):")
            
            for model_name in tstr_results['baseline'].keys():
                baseline_acc = tstr_results['baseline'][model_name]['accuracy']
                tstr_acc = tstr_results['tstr'][model_name]['accuracy']
                utility_score = tstr_results['utility_scores'][model_name]
                
                report_lines.append(f"    {model_name}:")
                report_lines.append(f"      Baseline准确率: {baseline_acc:.4f}")
                report_lines.append(f"      TSTR准确率:     {tstr_acc:.4f}")
                report_lines.append(f"      Utility得分:    {utility_score:.4f}")
            
            report_lines.append(f"  平均Utility得分: {tstr_results['average_utility_score']:.4f}")
            report_lines.append("")
        
        # 评估结论
        report_lines.append("【评估结论】")
        if scores['combined_score'] >= 0.8:
            conclusion = "优秀 - 合成数据质量很高"
        elif scores['combined_score'] >= 0.6:
            conclusion = "良好 - 合成数据质量较好"
        elif scores['combined_score'] >= 0.4:
            conclusion = "中等 - 合成数据质量一般"
        else:
            conclusion = "较差 - 建议改进GAN模型"
        
        report_lines.append(f"  {conclusion}")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        with open(f'{self.save_dir}/evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n评估报告已保存至: {self.save_dir}/evaluation_report.txt")
        print("\n" + report_text)
        
        return report_text