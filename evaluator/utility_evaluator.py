import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class UtilityEvaluator:
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                 target_column: str, discrete_columns: list = None):
        """
        The constructor of utility evaluator class, 
        it receives as arg-s  real, synthetic data, and 
        a string with name of a target column, and a list of discrete columns
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.target_column = target_column
        self.discrete_columns = discrete_columns if discrete_columns else []
        
        # call data preprocessing method
        self._preprocess_data()
        
    def _preprocess_data(self):
        """The internal method for preprocessing input data"""
        # initialize the dictionary of label encoders
        self.label_encoders = {}
        
        for col in self.discrete_columns:
            if col in self.real_data.columns and col != self.target_column:
                le = LabelEncoder()
                all_values = pd.concat([
                    self.real_data[col].astype(str), 
                    self.synthetic_data[col].astype(str)
                ]).unique()
                le.fit(all_values)
                
                self.real_data[col] = le.transform(self.real_data[col].astype(str))
                self.synthetic_data[col] = le.transform(self.synthetic_data[col].astype(str))
                self.label_encoders[col] = le
        
        if self.target_column in self.real_data.columns:
            le_target = LabelEncoder()
            all_targets = pd.concat([
                self.real_data[self.target_column].astype(str),
                self.synthetic_data[self.target_column].astype(str)
            ]).unique()
            le_target.fit(all_targets)
            
            self.real_data[self.target_column] = le_target.transform(
                self.real_data[self.target_column].astype(str)
            )
            self.synthetic_data[self.target_column] = le_target.transform(
                self.synthetic_data[self.target_column].astype(str)
            )
            self.label_encoders[self.target_column] = le_target
    
    def tstr_evaluation(self, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """
        The eval. of machine learning utility of data, based on TSTR principle
        """
        X_real = self.real_data.drop(columns=[self.target_column])
        y_real = self.real_data[self.target_column]
        
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state, stratify=y_real
        )
        
        X_train_syn = self.synthetic_data.drop(columns=[self.target_column])
        y_train_syn = self.synthetic_data[self.target_column]
        
        imputer_real = SimpleImputer(strategy='mean')
        imputer_syn = SimpleImputer(strategy='mean')

        X_train_real = pd.DataFrame(
            imputer_real.fit_transform(X_train_real),
            columns=X_train_real.columns,
            index=X_train_real.index
        )
        X_test_real = pd.DataFrame(
            imputer_real.transform(X_test_real),
            columns=X_test_real.columns,
            index=X_test_real.index
        )
        
        X_train_syn = pd.DataFrame(
            imputer_syn.fit_transform(X_train_syn),
            columns=X_train_syn.columns,
            index=X_train_syn.index
        )
        
        # standardize real and synth. data
        scaler_real = StandardScaler()
        scaler_syn = StandardScaler()
        
        X_train_real_scaled = scaler_real.fit_transform(X_train_real)
        X_test_real_scaled = scaler_real.transform(X_test_real)
        
        X_train_syn_scaled = scaler_syn.fit_transform(X_train_syn)
        X_test_syn_scaled = scaler_syn.transform(X_test_real)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state)
        }
        
        results = {
            'baseline': {}, 
            'tstr': {},      
            'utility_scores': {}
        }
        
        for model_name, model in models.items():
            model_baseline = type(model)(**model.get_params())
            model_baseline.fit(X_train_real_scaled, y_train_real)
            y_pred_baseline = model_baseline.predict(X_test_real_scaled)
            
            # TSTR
            model_tstr = type(model)(**model.get_params())
            model_tstr.fit(X_train_syn_scaled, y_train_syn)
            y_pred_tstr = model_tstr.predict(X_test_syn_scaled)
            
            baseline_metrics = self._compute_metrics(y_test_real, y_pred_baseline, model_baseline, X_test_real_scaled)
            tstr_metrics = self._compute_metrics(y_test_real, y_pred_tstr, model_tstr, X_test_syn_scaled)
            
            results['baseline'][model_name] = baseline_metrics
            results['tstr'][model_name] = tstr_metrics
            
            # compute utility score
            utility_score = tstr_metrics['accuracy'] / (baseline_metrics['accuracy'] + 1e-10)
            results['utility_scores'][model_name] = utility_score
        
        results['average_utility_score'] = np.mean(list(results['utility_scores'].values()))
        
        return results
    
    def _compute_metrics(self, y_true, y_pred, model, X_test) -> Dict[str, float]:
        """
        Here the method calculates main metrics: f1, recall, precision, and AUC
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # find the area under curve - AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_true)) == 2:  # 二分类
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # 多分类
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.0
        
        return metrics
    
    def compute_utility_score(self, test_size: float = 0.3, random_state: int = 42) -> float:
        """
        Find an averaged utility score, on test subset of given size `test_size`  
        """
        tstr_results = self.tstr_evaluation(test_size, random_state)
        return tstr_results['average_utility_score']
    
    def get_detailed_report(self, test_size: float = 0.3, random_state: int = 42) -> Dict:
        """
        It returns TSTR based detailed report
        """
        tstr_results = self.tstr_evaluation(test_size, random_state)
        
        return {
            'utility_score': tstr_results['average_utility_score'],
            'model_results': tstr_results
        }

