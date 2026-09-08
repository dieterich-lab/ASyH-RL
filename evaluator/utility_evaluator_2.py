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

    # baseline (real-train-vs-real-test) results only depend on real_train/real_test,
    # so cache them across episodes instead of retraining on every call
    _baseline_cache: Dict[Any, Dict[str, Any]] = {}

    def __init__(self, real_train: pd.DataFrame, real_test: pd.DataFrame, 
                 synthetic_data: pd.DataFrame, target_column: str, discrete_columns: list = None):
        """
        Initializes the evaluator with pre-split training and testing sets.
        """
        self._baseline_cache_key = (id(real_train), id(real_test), target_column)
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
    
        imputer = SimpleImputer(strategy='mean', keep_empty_features=True)

        # import ipdb; ipdb.set_trace()

        X_train_real = pd.DataFrame(imputer.fit_transform(X_train_real), columns=X_train_real.columns)

        X_test_real = pd.DataFrame(imputer.transform(X_test_real), columns=X_test_real.columns)
        X_train_syn = pd.DataFrame(imputer.transform(X_train_syn), columns=X_train_syn.columns)
    
        scaler = StandardScaler()
        X_train_real_scaled = scaler.fit_transform(X_train_real)
        X_test_real_scaled = scaler.transform(X_test_real)
        X_train_syn_scaled = scaler.transform(X_train_syn)
    
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state, verbose=1),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state, verbose=1)
        }
    
        results = {'baseline': {}, 'tstr': {}, 'utility_scores': {}}
        cached_baseline = self._baseline_cache.get(self._baseline_cache_key)

        # full set of target classes across train/test/synthetic (see _preprocess_data);
        # a model trained on synthetic data can easily miss rare classes, so every AUC call
        # must be scored against this fixed label set instead of the model's own classes_
        target_encoder = self.label_encoders.get(self.target_column)
        if target_encoder is not None:
            all_classes = np.arange(len(target_encoder.classes_))
        else:
            all_classes = np.unique(np.concatenate([y_train_real, y_test_real, y_train_syn]))

        # import ipdb; ipdb.set_trace()
    
        for model_name, model in models.items():
            if cached_baseline is not None and model_name in cached_baseline:
                baseline_metrics = cached_baseline[model_name]
            else:
                model_baseline = type(model)(**model.get_params())
                # model_baseline.fit(X_train_real_scaled, y_train_real)
                # call `model_baseline.fit()` in a way so its progress  would be visible in stdout
                model_baseline.fit(X_train_real_scaled, y_train_real)
                y_pred_baseline = model_baseline.predict(X_test_real_scaled)
                baseline_metrics = self._compute_metrics(y_test_real, y_pred_baseline, model_baseline, X_test_real_scaled, all_classes)
                self._baseline_cache.setdefault(self._baseline_cache_key, {})[model_name] = baseline_metrics
        
            model_tstr = type(model)(**model.get_params())
            model_tstr.fit(X_train_syn_scaled, y_train_syn)
            y_pred_tstr = model_tstr.predict(X_test_real_scaled)
            tstr_metrics = self._compute_metrics(y_test_real, y_pred_tstr, model_tstr, X_test_real_scaled, all_classes)
        
            results['baseline'][model_name] = baseline_metrics
            results['tstr'][model_name] = tstr_metrics
            results['utility_scores'][model_name] = tstr_metrics['auc'] / (baseline_metrics['auc'] + 1e-10)
    
        results['average_utility_score'] = np.mean(list(results['utility_scores'].values()))
        return results
    
    def _compute_metrics(self, y_true, y_pred, model, X_test, all_classes=None) -> Dict[str, float]:
        """Calculates classification metrics with corrected indentation."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if all_classes is None:
                all_classes = model.classes_
            # re-index predict_proba's columns onto the full label set (zero-fill classes
            # the model never saw during training) so AUC stays computable and meaningful
            # instead of raising/falling back to 0.0 whenever a class is missing.
            aligned_proba = np.zeros((y_proba.shape[0], len(all_classes)))
            class_to_col = {cls: idx for idx, cls in enumerate(all_classes)}
            for col_idx, cls in enumerate(model.classes_):
                if cls in class_to_col:
                    aligned_proba[:, class_to_col[cls]] = y_proba[:, col_idx]
            try:
                if len(all_classes) == 2:
                    metrics['auc'] = roc_auc_score(y_true, aligned_proba[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(
                        y_true, aligned_proba, multi_class='ovr', average='weighted', labels=all_classes
                    )
            except ValueError as e:
                print(f"[UtilityEvaluator] Could not compute AUC ({e}); falling back to 0.0.")
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
        return metrics
    
    def get_detailed_report(self, random_state: int = 42) -> Dict:
        """Detailed report interface."""
        tstr_results = self.tstr_evaluation(random_state)
        return {
            'utility_score': tstr_results['average_utility_score'],
            'model_results': tstr_results
        }