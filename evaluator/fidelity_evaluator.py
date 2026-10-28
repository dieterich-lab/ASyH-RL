import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FidelityEvaluator:
    """
    The class with methods for assessing synth. data fidelity.
    """
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                 discrete_columns: List[str] = None):
        """
        The constructor which takes as arguments real_data, synthetic_data,
        discrete_columns
        """
        common_cols = [col for col in real_data.columns if col in synthetic_data.columns]
        if len(common_cols) == 0:
            raise ValueError("Zero columns in the table! Please check if it has any content or if you chose a wrong delimiter between fields")
        
        # import ipdb; ipdb.set_trace()
        
        # assign to class attributes copied sub-frames of real data, 
        # synth. data, and column lists (columns containing discrete variables or continuous ones)
        self.real_data = real_data[common_cols].copy()
        self.synthetic_data = synthetic_data[common_cols].copy()
        self.discrete_columns = [col for col in (discrete_columns if discrete_columns else []) 
                                if col in common_cols]
        self.continuous_columns = [col for col in common_cols 
                                   if col not in self.discrete_columns]
        
        # iterate over continuous columns and fill missing values with medians
        for col in self.continuous_columns:
            if col in self.real_data.columns and self.real_data[col].isnull().any():
                fill_val = self.real_data[col].median()
                self.real_data[col] = self.real_data[col].fillna(fill_val)

            if col in self.synthetic_data.columns and self.synthetic_data[col].isnull().any():
                fill_val = self.synthetic_data[col].median()
                self.synthetic_data[col] = self.synthetic_data[col].fillna(fill_val)
        
    def evaluate_univariate_distributions(self) -> Dict[str, float]:
        """
        Carry out evaluation of a one-variable distribution
        """
        results = {
            'ks_statistics': {},
            'js_divergences': {},
            'overall_ks_score': 0.0,
            'overall_js_score': 0.0
        }
        
        ks_scores = []
        js_scores = []
        
        # calculate  Kolmogorov-Smirnov test on  continous features
        # and find Jensen-Shannon divergence - to characterize distance between synth. and real distributions
        for col in self.continuous_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                # KS
                ks_stat, _ = stats.ks_2samp(
                    self.real_data[col], # .fillna(self.real_data[col].median()), 
                    self.synthetic_data[col] # .fillna(self.synthetic_data[col].median())
                )
                results['ks_statistics'][col] = ks_stat
                ks_scores.append(ks_stat)
                
                # JS
                try:
                    real_hist, bins = np.histogram(self.real_data[col], # .fillna(self.real_data[col].median()), 
                                                   bins=50, density=True)
                    syn_hist, _ = np.histogram(self.synthetic_data[col], # .fillna(self.synthetic_data[col].median()), 
                                              bins=bins, density=True)
                    
                    real_hist = real_hist / (real_hist.sum() + 1e-10)
                    syn_hist = syn_hist / (syn_hist.sum() + 1e-10)
                    
                    js_div = jensenshannon(real_hist + 1e-10, syn_hist + 1e-10)
                    results['js_divergences'][col] = js_div
                    js_scores.append(js_div)
                except:
                    pass
        
        for col in self.discrete_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                try:
                    real_counts = self.real_data[col].value_counts(normalize=True)
                    syn_counts = self.synthetic_data[col].value_counts(normalize=True)
                    
                    all_categories = set(real_counts.index) | set(syn_counts.index)
                    real_probs = [real_counts.get(cat, 0) for cat in all_categories]
                    syn_probs = [syn_counts.get(cat, 0) for cat in all_categories]
                    
                    real_probs = np.array(real_probs)
                    syn_probs = np.array(syn_probs)
                    real_probs = real_probs / (real_probs.sum() + 1e-10)
                    syn_probs = syn_probs / (syn_probs.sum() + 1e-10)
                    
                    js_div = jensenshannon(real_probs + 1e-10, syn_probs + 1e-10)
                    results['js_divergences'][col] = js_div
                    js_scores.append(js_div)
                except:
                    pass
        
        # drop NaN per-column values so one bad column doesn't poison the mean
        ks_scores = [s for s in ks_scores if not np.isnan(s)]
        js_scores = [s for s in js_scores if not np.isnan(s)]
        results['overall_ks_score'] = np.mean(ks_scores) if ks_scores else 1.0
        results['overall_js_score'] = np.mean(js_scores) if js_scores else 1.0
        
        return results
    
    def evaluate_tvd(self) -> Dict[str, float]:
        """
        The method finds the value of Total Variation Distance (TVD)
        
        TVD(P, Q) = 0.5 * sum(|P(x) - Q(x)|)
        Values lay in the range [0, 1]
        
        Returns:
            results: dict,
            it has the key `tvd_score` that contains the computed TVD value
        """
        results = {
            'tvd_by_column': {},
            'overall_tvd': 0.0,
            'tvd_score': 0.0  # (1 - TVD)
        }
        
        tvd_values = []
        
        # iterate through continuous features
        for col in self.continuous_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                try:
                    # fill in nans with median values
                    real_values = self.real_data[col].fillna(self.real_data[col].median()).values
                    syn_values = self.synthetic_data[col].fillna(self.synthetic_data[col].median()).values
                    
                    if len(real_values) > 0 and len(syn_values) > 0:
                        # find min anx max values
                        min_val = min(real_values.min(), syn_values.min())
                        max_val = max(real_values.max(), syn_values.max())
                        
                        # prepare bins for the histogram
                        bins = np.linspace(min_val, max_val, 51)
                        
                        # build histograms
                        real_hist, _ = np.histogram(real_values, bins=bins, density=False)
                        syn_hist, _ = np.histogram(syn_values, bins=bins, density=False)
                        
                        # find probabilities
                        real_probs = real_hist / (real_hist.sum() + 1e-10)
                        syn_probs = syn_hist / (syn_hist.sum() + 1e-10)
                        
                        # TVD: 0.5 * sum(|P(x) - Q(x)|)
                        tvd = 0.5 * np.sum(np.abs(real_probs - syn_probs))
                        
                        results['tvd_by_column'][col] = tvd
                        tvd_values.append(tvd)
                except Exception as e:
                    pass
        
        # iterate through discrete columns
        for col in self.discrete_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                try:
                    # find unique values countrs in real and synth. discrete features
                    real_counts = self.real_data[col].value_counts(normalize=True)
                    syn_counts = self.synthetic_data[col].value_counts(normalize=True)
                    
                    # find probabilities (for each unique value)
                    all_categories = set(real_counts.index) | set(syn_counts.index)
                    real_probs = np.array([real_counts.get(cat, 0) for cat in all_categories])
                    syn_probs = np.array([syn_counts.get(cat, 0) for cat in all_categories])
                    
                    # normalize probab-s
                    real_probs = real_probs / (real_probs.sum() + 1e-10)
                    syn_probs = syn_probs / (syn_probs.sum() + 1e-10)
                    
                    tvd = 0.5 * np.sum(np.abs(real_probs - syn_probs))
                    
                    results['tvd_by_column'][col] = tvd
                    tvd_values.append(tvd)
                except Exception as e:
                    pass
        
        results['overall_tvd'] = np.mean(tvd_values) if tvd_values else 1.0
        
        results['tvd_score'] = 1.0 - results['overall_tvd']
        
        return results
    
    def evaluate_correlation_similarity(self) -> Dict[str, float]:
        """
        Assess the similarity between datasets by means of calculating their correlation (Pearson's corr.)
        """
        results = {}
        
        # missing values are filled with zeros
        if len(self.continuous_columns) > 0:
            # try:
            # constant columns have undefined correlation (0/0 -> NaN); treat them as uncorrelated
            real_corr = self.real_data[self.continuous_columns].corr().fillna(0.0)
            syn_corr = self.synthetic_data[self.continuous_columns].corr().fillna(0.0)
            np.fill_diagonal(real_corr.values, 1.0)
            np.fill_diagonal(syn_corr.values, 1.0)
            
            # Frobenius distance
            frobenius_dist = np.linalg.norm(real_corr - syn_corr, 'fro')
            results['frobenius_distance'] = frobenius_dist
            
            # flattening
            real_corr_flat = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
            syn_corr_flat = syn_corr.values[np.triu_indices_from(syn_corr.values, k=1)]
            
            if len(real_corr_flat) >= 2 and real_corr_flat.std() > 0 and syn_corr_flat.std() > 0:
                corr_corr, _ = stats.pearsonr(real_corr_flat, syn_corr_flat)
                results['correlation_correlation'] = corr_corr
            elif len(real_corr_flat) >= 2:
                # pearsonr is undefined for a constant vector
                results['correlation_correlation'] = 1.0 if np.allclose(real_corr_flat, syn_corr_flat) else 0.0
            else:
                results['correlation_correlation'] = 1.0
            
            # find the corr. between MAEs (moving averages)
            mae = np.mean(np.abs(real_corr - syn_corr))
            results['correlation_mae'] = mae
                
        else:
            results['frobenius_distance'] = 0.0
            results['correlation_correlation'] = 1.0
            results['correlation_mae'] = 0.0
        
        return results
    
    def compute_fidelity_score(self) -> float:
        """
        With this method we find a fidelity score - the weighed average
        of: KS-score, JS-score, correlation, and TVD
        """
        univariate_results = self.evaluate_univariate_distributions()
        correlation_results = self.evaluate_correlation_similarity()
        tvd_results = self.evaluate_tvd()
        
        ks_score = 1.0 - univariate_results['overall_ks_score']
        
        js_score = 1.0 - univariate_results['overall_js_score']
        
        corr_score = correlation_results['correlation_correlation']
        
        # TVD
        tvd_score = tvd_results['tvd_score']
        
        weights = [0.3, 0.3, 0.2, 0.2]  # KS, JS, Correlation, TVD
        fidelity_score = (weights[0] * ks_score + 
                         weights[1] * js_score + 
                         weights[2] * corr_score +
                         weights[3] * tvd_score)
        
        return max(0.0, min(1.0, fidelity_score))
    
    def get_detailed_report(self) -> Dict:
        """
        Returns:
            Dict
            that contains several keys, including fidelity score,
            correlation similiarity, and tvd score
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
            'tvd_score': tvd_results['tvd_score']
        }

