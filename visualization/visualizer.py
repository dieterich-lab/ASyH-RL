import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# general plot settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    The class with the collection of functions for plotting 
    essential info about produced data:
        * a heatmap of features, their pair-wise correlations
        * distribution curves
        * radar plots with 5D comprehensive score
    """
    
    def __init__(self, save_dir: str = 'results'):
        """
        The constructor, the main argument is `save_dir` 
        where results are to be stored
        
        Args:
            save_dir: str,  the folder where results will be saved
        """
        self.save_dir = save_dir
        sns.set_style('whitegrid')
        sns.set_palette('husl')
    
    def plot_univariate_distributions(self, real_data: pd.DataFrame, 
                                     synthetic_data: pd.DataFrame,
                                     columns: List[str] = None,
                                     max_cols: int = 6):
        """
        Make a plot of one-variable distribution
        
        Args:
            real_data: pd.DataFrmame, the original dataset
            synthetic_data: pd.DataFrmame, synth. dataset
            columns: List, it contains column names which can be characterized by univar. distribution
            max_cols: int, max number of columns
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
                # visualize distr. density of real and synth. data 
                real_data[col].dropna().plot(kind='density', ax=ax, label='Real', alpha=0.7)
                synthetic_data[col].dropna().plot(kind='density', ax=ax, label='Synthetic', alpha=0.7)
                
                ax.set_title(f'Distribution: {col}', fontsize=10)
                ax.set_xlabel('')
                ax.legend()
            except:
                ax.text(0.5, 0.5, f'{col}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/univariate_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved the figure on the path: {self.save_dir}/univariate_distributions.png")
    
    def plot_correlation_heatmap(self, real_data: pd.DataFrame, 
                                synthetic_data: pd.DataFrame,
                                continuous_columns: List[str] = None):
        """
        Produce a correlation heatmap
        
        Args:
            real_data: pd.DataFrame
            synthetic_data: pd.DataFrame
            continuous_columns: List[str]
        """
        if continuous_columns is None:
            continuous_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit the number of continuous columns for the plot for better readability
        if len(continuous_columns) > 20:
            continuous_columns = continuous_columns[:20]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        real_corr = real_data[continuous_columns].corr()
        sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', center=0, 
                   vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
        axes[0].set_title('Real Data Correlation', fontsize=14, fontweight='bold')

        # produce the heatmap with seaborn library's method
        syn_corr = synthetic_data[continuous_columns].corr()
        sns.heatmap(syn_corr, ax=axes[1], cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
        axes[1].set_title('Synthetic Data Correlation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"The correlation heatmap has been saved to: {self.save_dir}/correlation_heatmap.png")
    
    def plot_tstr_results(self, tstr_results: Dict):
        """
        Demonstrate visually the result of TSTR evaluation
        
        Args:
            tstr_results: Dict
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
        print(f"Saved the visual result of TSTR eval. on the path: {self.save_dir}/tstr_results.png")
    
    def plot_utility_scores(self, tstr_results: Dict):
        """
        The plotting function for visualizing utility scores, the plot is saved on the disk

        Args:
            tstr_results: Dict,  (with tstr results)
        """
        utility_scores = tstr_results['utility_scores']
        models = list(utility_scores.keys())
        scores = list(utility_scores.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if s >= 0.8 else '#f39c12' if s >= 0.6 else '#e74c3c' for s in scores]
        bars = ax.barh(models, scores, color=colors, alpha=0.8)
        
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
        print(f"The figure with utility scores is saved on the path: {self.save_dir}/utility_scores.png")
    
    def plot_rl_training_history(self, training_history: Dict):
        """
        Visualize the history of gained rewards
        
        Args:
            training_history: Dict
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # get number of episodes and the list of rewards (X and Y axis)
        episodes = training_history['episodes']
        rewards = training_history['rewards']
        
        axes[0].plot(episodes, rewards, alpha=0.6, label='Reward')
        
        # smoothen reward values, get the rolling average
        window = min(10, len(rewards) // 10)
        if window > 1:
            rewards_smooth = pd.Series(rewards).rolling(window=window).mean()
            axes[0].plot(episodes, rewards_smooth, linewidth=2, label=f'Moving Avg (window={window})')
        
        axes[0].set_xlabel('Episode', fontsize=11)
        axes[0].set_ylabel('Reward', fontsize=11)
        axes[0].set_title('RL Training: Reward over Episodes', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # demonstrate how epsilon exploration rate changed throughout the rl-agent training
        epsilon = training_history['epsilon']
        axes[1].plot(episodes, epsilon, color='orange', linewidth=2)
        axes[1].set_xlabel('Episode', fontsize=11)
        axes[1].set_ylabel('Epsilon (Exploration Rate)', fontsize=11)
        axes[1].set_title('RL Training: Epsilon Decay', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/rl_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved the reward history plot on: {self.save_dir}/rl_training_history.png")
    
    def plot_comprehensive_scores(self, evaluation_result: Dict, 
                                  plot_name: str = 'comprehensive_scores.png'):
        """
        It plots a circular diagram showing scores comprehensively
        
        Args:
            evaluation_result: Dict
        """
        scores = evaluation_result['scores']
        fidelity_details = evaluation_result['fidelity_details']
        
        # Main metrics 
        categories = ['Fidelity\n(Overall)', 'Utility\n(TSTR)', 
                     'KS Score', 'JS Score', 'Correlation']
        values = [
            scores['fidelity_score'],
            scores['utility_score'],
            1.0 - fidelity_details['univariate_distributions']['overall_ks_score'],
            1.0 - fidelity_details['univariate_distributions']['overall_js_score'],
            fidelity_details['correlation_similarity']['correlation_correlation']
        ]
        
        # customizing angles on the plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1] 
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
        plt.savefig(f'{self.save_dir}/{plot_name}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"The plot is save on the path: {self.save_dir}/{plot_name}")

    def plot_comprehensive_scores_on_ax(self, evaluation_result: Dict, ax=None,
                                        label: str = 'Scores', color: str = None):
        """
        Same radar chart as plot_comprehensive_scores, but drawn on a given
        polar Axes instead of creating/saving its own figure, so several
        calls can overlay multiple score profiles on the same plane.

        Args:
            evaluation_result: assessment results dictionary
            ax: existing polar-projection Axes to draw on; created if None
            label: legend label for this series
            color: line/fill color (None lets matplotlib auto-cycle)

        Returns:
            The polar Axes used (pass it back in to keep overlaying).
        """
        scores = evaluation_result['scores']
        fidelity_details = evaluation_result['fidelity_details']

        categories = ['Fidelity\n(Overall)', 'Utility\n(TSTR)',
                     'KS Score', 'JS Score', 'Correlation']
        values = [
            scores['fidelity_score'],
            scores['utility_score'],
            1.0 - fidelity_details['univariate_distributions']['overall_ks_score'],
            1.0 - fidelity_details['univariate_distributions']['overall_js_score'],
            fidelity_details['correlation_similarity']['correlation_correlation']
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        line, = ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=line.get_color())

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_comprehensive_scores_overlay(self, evaluation_results: Dict[str, Dict],
                                          plot_name: str = 'comprehensive_scores_overlay.png'):
        """
        Overlays several plot_comprehensive_scores_on_ax radar charts on one
        shared polar Axes, e.g. to compare scores across models/runs.

        Args:
            evaluation_results: mapping of series label -> evaluation_result dict
            plot_name: output file name
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        for label, evaluation_result in evaluation_results.items():
            self.plot_comprehensive_scores_on_ax(evaluation_result, ax=ax, label=label)

        ax.set_title('Comprehensive Evaluation Scores\n(Closer to edge = Better)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{plot_name}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"The overlay plot is saved on the path: {self.save_dir}/{plot_name}")

    def create_summary_report(self, evaluation_result: Dict, 
                            tstr_results: Dict = None):
        """
        Produce a final summary report based on evaluation results
        
        Args:
            evaluation_result: Dict
            tstr_results: str, the summary with tstr eval. results
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Synthetic Data Quality Evaluation Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        if evaluation_result['use_rl']:
            report_lines.append(f"Using Reinforcement Learning Strategies: {evaluation_result['action']}")
        else:
            report_lines.append("Use the default balancing strategy")
        
        weights = evaluation_result['weights']
        report_lines.append(f"Weight settings: Fidelity={weights['fidelity_weight']:.2f}, "
                          f"Utility={weights['utility_weight']:.2f}")
        report_lines.append("")
        
        scores = evaluation_result['scores']
        report_lines.append(f"  Fidelity Score:     {scores['fidelity_score']:.4f}")
        report_lines.append(f"  Utility Score:   {scores['utility_score']:.4f}")
        report_lines.append(f"  Combined Score:        {scores['combined_score']:.4f}")
        report_lines.append("")
        
        # Fidelity
        fidelity_details = evaluation_result['fidelity_details']
        report_lines.append("Statistical Similarity Details")
        report_lines.append(f"KS-test: {fidelity_details['univariate_distributions']['overall_ks_score']:.4f}")
        report_lines.append(f"JS-score:   {fidelity_details['univariate_distributions']['overall_js_score']:.4f}")
        corr_sim = fidelity_details['correlation_similarity']
        report_lines.append(f"Correlation Similarity: {corr_sim['correlation_correlation']:.4f}")
        report_lines.append(f"Frobenius distance: {corr_sim['frobenius_distance']:.4f}")
        report_lines.append("")
        
        # Utility
        if tstr_results:
            report_lines.append("Details on Machine Learning Effectiveness (TSTR)")
            report_lines.append("Model Performance Comparison (Synthetic vs Real):")
            
            for model_name in tstr_results['baseline'].keys():
                baseline_acc = tstr_results['baseline'][model_name]['accuracy']
                tstr_acc = tstr_results['tstr'][model_name]['accuracy']
                utility_score = tstr_results['utility_scores'][model_name]
                
                report_lines.append(f"    {model_name}:")
                report_lines.append(f"      Baseline accuracy: {baseline_acc:.4f}")
                report_lines.append(f"      TSTR Accuracy Rate:     {tstr_acc:.4f}")
                report_lines.append(f"      Utility Score:    {utility_score:.4f}")
            
            report_lines.append(f"  Utility score: {tstr_results['average_utility_score']:.4f}")
            report_lines.append("")
        
        # conclude the evaluation results
        report_lines.append("Evaluation Conclusion")
        if scores['combined_score'] >= 0.8:
            conclusion = "Excellent - The synthetic data is of very high quality"
        elif scores['combined_score'] >= 0.6:
            conclusion = "Good - The synthetic data is of good quality"
        elif scores['combined_score'] >= 0.4:
            conclusion = "Moderate - Synthetic data quality is average"
        else:
            conclusion = "Poor - Recommendations for Improving the GAN Model"
        
        report_lines.append(f"  {conclusion}")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        with open(f'{self.save_dir}/evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nThe evaluation report has been saved to: {self.save_dir}/evaluation_report.txt")
        print("\n" + report_text)
        
        return report_text