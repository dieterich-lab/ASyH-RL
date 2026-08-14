# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import os
import warnings
from typing import Dict, List
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import KNNImputer
# from sklearn.impute import MissingIndicator

from data import Data

# 内部导入
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
from evaluator.utility_evaluator_2 import UtilityEvaluator
from rl_agent.q_learning_agent import QLearningAgent
from evaluator.fidelity_evaluator import FidelityEvaluator

warnings.filterwarnings('ignore')


class RLOptimizer:
    def __init__(self, train_data: Data, 
                 val_data: Data,
                 modelParamSpace: Dict[str, List] = None,
                 discrete_columns: List[str] = None,
                 target_feature: str = 'score-eq5d-5l_start'):
        """
        初始化 RL 优化器
        train_path: 80% 训练集
        val_path: 10% 验证集 (用于计算 RL 奖励)
        """

        self.train_data = train_data
        self.val_data = val_data

        self.train_df = self.train_data.data
        self.val_df = self.val_data.data
        self.discrete_columns = discrete_columns

        self.target_feature = target_feature
        
        # 定义搜索空间
        self.modelParamSpace = modelParamSpace
        
        # self.action_size = len(self.n_components_options) * len(self.max_iter_options) * len(self.n_init_options)
        assert self.model_params is not None, "model_params cannot be None"
        self.action_size = np.abs(np.prod([len(v) for v in self.modelParamSpace.values()]))

        self.state_size = 4 
        self.last_eval = {'tvd_score': 0.5, 'utility_score': 0.5, 
                          'ks_score': 0.5, 'js_score': 0.5}
        
        # self._load_and_preprocess_data()
        
        self.agent = QLearningAgent(
            state_size=self.state_size, action_size=self.action_size,
            learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
            # epsilon_decay=0.995, 
            # epsilon_min=0.01
        )
        
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = {'rewards': []}


    def _generate_synthetic_data(self,
                                 modelClass: type,
                                 params: Dict, 
                                 n_samples: int = 100) -> pd.DataFrame:
        # import ipdb; ipdb.set_trace()
        n_samples = n_samples if n_samples else len(self.df_train_model)

        model = modelClass(**params)
        model.fit(self.train_data_transformed)
        samples, _ = model.sample(n_samples)
        
        num_c = len(self.continuous_cols)
        inv_num = self.preprocessor.named_transformers_['num'].inverse_transform(samples[:, :num_c])
        inv_cat = self.preprocessor.named_transformers_['cat'].inverse_transform(samples[:, num_c:])
        
        df_syn = pd.concat([pd.DataFrame(inv_num, columns=self.continuous_cols),
                            pd.DataFrame(inv_cat, columns=self.discrete_cols)], axis=1)
        # return df_syn[self.df_train_model.columns]
        # order columns of `df_syn` in the same way as they're ordered in `self.train_data_transformed` dataframe
        return df_syn[self.df_train_model.columns]
        # except: return None

    def _evaluate_synthetic_data(self, synthetic_data: pd.DataFrame) -> Dict:
        """核心评估：对比验证集"""
        # try:
        fid = FidelityEvaluator(self.df_train_model, synthetic_data, self.discrete_cols)
        fid_rep = fid.get_detailed_report()

        # import ipdb; ipdb.set_trace()
        
        # 修正语法错误：去掉参数名前的 self.
        util = UtilityEvaluator(
            real_train=self.train_df, 
            real_test=self.val_df, 
            synthetic_data=synthetic_data, 
            target_column=self.target_feature, 
            discrete_columns=self.discrete_cols
        )
        util_rep = util.get_detailed_report()
        
        return {
            'tvd_score': float(fid_rep.get('tvd_score', 0.0)),
            'utility_score': float(util_rep.get('utility_score', 0.0)),
            'ks_score': 1.0 - fid_rep['univariate_distributions']['overall_ks_score'],
            'js_score': 1.0 - fid_rep['univariate_distributions']['overall_js_score']   
        }
        # except: return {'tvd_score': 0.0, 'utility_score': 0.0, 'ks_score': 0.0, 'js_score': 0.0}
    
    def evaluate_synthetic_data(self, synthetic_data: pd.DataFrame) -> Dict:
            """核心评估：对比验证集"""
            # try:
            fid = FidelityEvaluator(self.df_train_model, synthetic_data, self.discrete_cols)
            fid_rep = fid.get_detailed_report()
    
            # import ipdb; ipdb.set_trace()
            
            # 修正语法错误：去掉参数名前的 self.
            util = UtilityEvaluator(
                real_train=self.df_train_model, 
                real_test=self.df_val_model, 
                synthetic_data=synthetic_data, 
                target_column=self.target_feature, 
                discrete_columns=self.discrete_cols
            )
            util_rep = util.get_detailed_report()
            
            return {
                'scores': {'fidelity_score':fid_rep['fidelity_score'], 'utility_score':util_rep['utility_score']},
                'fidelity_details':fid_rep,
                'tvd_score': float(fid_rep.get('tvd_score', 0.0)),
                'utility_score': float(util_rep.get('utility_score', 0.0)),
                'ks_score': 1.0 - fid_rep['univariate_distributions']['overall_ks_score'],
                'js_score': 1.0 - fid_rep['univariate_distributions']['overall_js_score']
            }

    def optimize(self, episodes=30, verbose=True):
        print("开始基于验证集的 RL 优化...")
        for ep in range(episodes):
            print(f"Episode {ep+1}/{episodes} - Last Eval: {self.last_eval}")
            state = np.clip([self.last_eval['tvd_score'], self.last_eval['utility_score'], 
                             self.last_eval['ks_score'], self.last_eval['js_score']], 0, 1)
            print(f"  State: {state}")
            action = self.agent.choose_action(state)
            print(f"  Action: {action}")
            params = self._action_to_params(action)
            print(f"  Params: {params}")

            # import ipdb; ipdb.set_trace()  # Debugging point to inspect the parameters and state before generating synthetic data

            syn = self._generate_synthetic_data(params)
            print(f"  Generated Synthetic Data Shape: {syn.shape if syn is not None else 'None'}")
            if syn is None: continue
                
            res = self.evaluate_synthetic_data(syn)
            print(f"  Evaluation Results: {res}")
            self.last_eval = res
            reward = 0.5 * np.mean([res['tvd_score'], res['ks_score'], res['js_score']]) + 0.5 * res['utility_score']
            print(f"  Reward: {reward:.4f}")
            
            self.agent.update(state, action, reward, state, True)
            if reward > self.best_score:
                self.best_score, self.best_params = reward, params
                if verbose: print(f"  Ep {ep+1}: New Best Reward {reward:.4f} (Util on Val: {res['utility_score']:.3f})")
        return self.best_params

    # def _action_to_params(self, action: int) -> Dict:
    #     n_idx = action % len(self.n_components_options)
    #     m_idx = (action // len(self.n_components_options)) % len(self.max_iter_options)
    #     i_idx = action // (len(self.n_components_options) * len(self.max_iter_options))
    #     return {'n_components': self.n_components_options[n_idx], 
    #             'max_iter': self.max_iter_options[m_idx], 
    #             'n_init': self.n_init_options[min(i_idx, len(self.n_init_options)-1)]}
    def _action_to_params(self, action: int) -> Dict:
        idxs = {}
        for i, (k,v) in enumerate(self.modelParamSpace.items()):
            if i == 0:
                idxs[k] = action % len(v)
            else:
                prev_product = np.prod([len(self.modelParamSpace[list(self.modelParamSpace.keys())[j]]) for j in range(i)])
                idxs[k] = (action // prev_product) % len(v)

        new_params = {k: self.modelParamSpace[k][idxs[k]] for k in self.modelParamSpace}
        return new_params

    def generate_with_best_params(self, output_path, n_samples=1000):
        syn = self._generate_synthetic_data(self.best_params, n_samples)
        syn.to_csv(output_path, index=False)
        return output_path
    def get_best_params(self): return self.best_params
    def save_agent(self, path): self.agent.save(path)
