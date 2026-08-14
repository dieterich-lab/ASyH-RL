"""
机器学习有效性评估器 (Utility Evaluator)
"""

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
        初始化评估器
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.target_column = target_column
        self.discrete_columns = discrete_columns if discrete_columns else []
        
        # 数据预处理
        self._preprocess_data()
        
    def _preprocess_data(self):
        """数据预处理"""
        # 对离散列进行编码
        self.label_encoders = {}
        
        for col in self.discrete_columns:
            if col in self.real_data.columns and col != self.target_column:
                le = LabelEncoder()
                # 合并真实和合成数据的类别
                all_values = pd.concat([
                    self.real_data[col].astype(str), 
                    self.synthetic_data[col].astype(str)
                ]).unique()
                le.fit(all_values)
                
                self.real_data[col] = le.transform(self.real_data[col].astype(str))
                self.synthetic_data[col] = le.transform(self.synthetic_data[col].astype(str))
                self.label_encoders[col] = le
        
        # 编码目标列
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
        TSTR 评估
        """
        # 划分真实数据为训练集和测试集
        X_real = self.real_data.drop(columns=[self.target_column])
        y_real = self.real_data[self.target_column]
        
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state, stratify=y_real
        )
        
        # 准备合成数据作为训练集
        X_train_syn = self.synthetic_data.drop(columns=[self.target_column])
        y_train_syn = self.synthetic_data[self.target_column]
        
        # 处理NaN值：使用均值填充
        # 创建填充器
        imputer_real = SimpleImputer(strategy='mean')
        imputer_syn = SimpleImputer(strategy='mean')
        
        # 填充训练集和测试集
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
        
        # 填充合成数据
        X_train_syn = pd.DataFrame(
            imputer_syn.fit_transform(X_train_syn),
            columns=X_train_syn.columns,
            index=X_train_syn.index
        )
        
        # 数据标准化
        scaler_real = StandardScaler()
        scaler_syn = StandardScaler()
        
        X_train_real_scaled = scaler_real.fit_transform(X_train_real)
        X_test_real_scaled = scaler_real.transform(X_test_real)
        
        X_train_syn_scaled = scaler_syn.fit_transform(X_train_syn)
        X_test_syn_scaled = scaler_syn.transform(X_test_real)  # 使用合成数据的scaler
        
        # 定义多个模型
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
            # 基准：在真实训练集上训练
            model_baseline = type(model)(**model.get_params())
            model_baseline.fit(X_train_real_scaled, y_train_real)
            y_pred_baseline = model_baseline.predict(X_test_real_scaled)
            
            # TSTR：在合成训练集上训练
            model_tstr = type(model)(**model.get_params())
            model_tstr.fit(X_train_syn_scaled, y_train_syn)
            y_pred_tstr = model_tstr.predict(X_test_syn_scaled)
            
            # 计算评估指标
            baseline_metrics = self._compute_metrics(y_test_real, y_pred_baseline, model_baseline, X_test_real_scaled)
            tstr_metrics = self._compute_metrics(y_test_real, y_pred_tstr, model_tstr, X_test_syn_scaled)
            
            results['baseline'][model_name] = baseline_metrics
            results['tstr'][model_name] = tstr_metrics
            
            # 计算utility score（TSTR相对于基准的性能比率）
            utility_score = tstr_metrics['accuracy'] / (baseline_metrics['accuracy'] + 1e-10)
            results['utility_scores'][model_name] = utility_score
        
        # 计算平均utility score
        results['average_utility_score'] = np.mean(list(results['utility_scores'].values()))
        
        return results
    
    def _compute_metrics(self, y_true, y_pred, model, X_test) -> Dict[str, float]:
        """
        计算分类评估指标
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # 计算AUC
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
        计算综合机器学习有效性得分 
        """
        tstr_results = self.tstr_evaluation(test_size, random_state)
        return tstr_results['average_utility_score']
    
    def get_detailed_report(self, test_size: float = 0.3, random_state: int = 42) -> Dict:
        """
        TSTR评估报告
        """
        tstr_results = self.tstr_evaluation(test_size, random_state)
        
        return {
            'utility_score': tstr_results['average_utility_score'],
            'model_results': tstr_results
        }

