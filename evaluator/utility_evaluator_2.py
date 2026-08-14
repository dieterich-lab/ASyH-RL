# -*- coding: utf-8 -*-
"""
Utility Evaluator - Corrected Indentation Version
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class UtilityEvaluator:
    
    def __init__(self, real_train: pd.DataFrame, real_test: pd.DataFrame, 
                 synthetic_data: pd.DataFrame, target_column: str, discrete_columns: list = None):
        """
        Initializes the evaluator with pre-split training and testing sets.
        """
        self.real_train = real_train.copy()
        self.real_test = real_test.copy()
        self.synthetic_data = synthetic_data.copy()
        self.target_column = target_column
        self.discrete_columns = discrete_columns if discrete_columns else []
        
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Unified encoding for all datasets."""
        self.label_encoders = {}
        
        for col in self.discrete_columns:
            if col in self.real_train.columns and col != self.target_column:
                le = LabelEncoder()
                all_values = pd.concat([
                    self.real_train[col].astype(str), 
                    self.real_test[col].astype(str),
                    self.synthetic_data[col].astype(str)
                ]).unique()
                le.fit(all_values)
                
                self.real_train[col] = le.transform(self.real_train[col].astype(str))
                self.real_test[col] = le.transform(self.real_test[col].astype(str))
                self.synthetic_data[col] = le.transform(self.synthetic_data[col].astype(str))
                self.label_encoders[col] = le
        
        if self.target_column in self.real_train.columns:
            le_target = LabelEncoder()
            all_targets = pd.concat([
                self.real_train[self.target_column].astype(str),
                self.real_test[self.target_column].astype(str),
                self.synthetic_data[self.target_column].astype(str)
            ]).unique()
            le_target.fit(all_targets)
            
            self.real_train[self.target_column] = le_target.transform(self.real_train[self.target_column].astype(str))
            self.real_test[self.target_column] = le_target.transform(self.real_test[self.target_column].astype(str))
            self.synthetic_data[self.target_column] = le_target.transform(self.synthetic_data[self.target_column].astype(str))
            self.label_encoders[self.target_column] = le_target
    
    def tstr_evaluation(self, random_state: int = 42) -> Dict[str, Any]:
        """TSTR evaluation logic."""
        X_train_real = self.real_train.drop(columns=[self.target_column])
        y_train_real = self.real_train[self.target_column]
        
        X_test_real = self.real_test.drop(columns=[self.target_column])
        y_test_real = self.real_test[self.target_column]
    
        X_train_syn = self.synthetic_data.drop(columns=[self.target_column])
        y_train_syn = self.synthetic_data[self.target_column]

        # order columns in X_test_real and X_train_syn in the same order as they're in X_train_real
        X_test_real = X_test_real[X_train_real.columns]
        X_train_syn = X_train_syn[X_train_real.columns]
    
        imputer = SimpleImputer(strategy='mean')
        X_train_real = pd.DataFrame(imputer.fit_transform(X_train_real), columns=X_train_real.columns)
        X_test_real = pd.DataFrame(imputer.transform(X_test_real), columns=X_test_real.columns)
        X_train_syn = pd.DataFrame(imputer.transform(X_train_syn), columns=X_train_syn.columns)
    
        scaler = StandardScaler()
        X_train_real_scaled = scaler.fit_transform(X_train_real)
        X_test_real_scaled = scaler.transform(X_test_real)
        X_train_syn_scaled = scaler.transform(X_train_syn)
    
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state)
        }
    
        results = {'baseline': {}, 'tstr': {}, 'utility_scores': {}}
    
        for model_name, model in models.items():
            model_baseline = type(model)(**model.get_params())
            model_baseline.fit(X_train_real_scaled, y_train_real)
            y_pred_baseline = model_baseline.predict(X_test_real_scaled)
            baseline_metrics = self._compute_metrics(y_test_real, y_pred_baseline, model_baseline, X_test_real_scaled)
        
            model_tstr = type(model)(**model.get_params())
            model_tstr.fit(X_train_syn_scaled, y_train_syn)
            y_pred_tstr = model_tstr.predict(X_test_real_scaled)
            tstr_metrics = self._compute_metrics(y_test_real, y_pred_tstr, model_tstr, X_test_real_scaled)
        
            results['baseline'][model_name] = baseline_metrics
            results['tstr'][model_name] = tstr_metrics
            results['utility_scores'][model_name] = tstr_metrics['auc'] / (baseline_metrics['auc'] + 1e-10)
    
        results['average_utility_score'] = np.mean(list(results['utility_scores'].values()))
        return results
    
    def _compute_metrics(self, y_true, y_pred, model, X_test) -> Dict[str, float]:
        """Calculates classification metrics with corrected indentation."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        # Fixed indentation for the try block
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_true)) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            else:
                metrics['auc'] = 0.0
        except:
            metrics['auc'] = 0.0
        return metrics
    
    def get_detailed_report(self, random_state: int = 42) -> Dict:
        """Detailed report interface."""
        tstr_results = self.tstr_evaluation(random_state)
        return {
            'utility_score': tstr_results['average_utility_score'],
            'model_results': tstr_results
        }