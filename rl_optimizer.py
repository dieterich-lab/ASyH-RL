# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import os
import warnings
from typing import Dict, List
import inspect

from data import Data
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
from evaluator.utility_evaluator_2 import UtilityEvaluator
from rl_agent.q_learning_agent import QLearningAgent
from evaluator.fidelity_evaluator import FidelityEvaluator
# import SingleTableMetadata
from sdv.metadata import SingleTableMetadata

warnings.filterwarnings('ignore')

class RLOptimizer:
    def __init__(self, train_data: Data, 
                 val_data: Data,
                 modelParamSpace: Dict[str, List] = None,
                 discrete_columns: List[str] = None,
                 target_feature: str = 'score-eq5d-5l_start',
                 subsample: bool = False):
        """
        The class describes the behavious of RL-agent that acts as a model h-parameters optimizer
        train_path: 80% of initial data
        val_path: 10% of initial data
        """

        self.train_data = train_data
        self.val_data = val_data

        self.train_df = self.train_data.data
        self.val_df = self.val_data.data

        # REVIEW,  if it makes more sense to use sdv metadata instead of pandas api?
        col_types_dict = self.train_data.detect_column_types(self.train_df)
        self.discrete_columns = col_types_dict['categorical']
        self.continuous_cols = self.numerical_cols = col_types_dict['numerical']
        self.date_columns = col_types_dict['date']

        # self.train_data.metadata.metadata = self.train_data.metadata.metadata.remove_columns(self.date_columns)
        # self.val_data.metadata.metadata = self.val_data.metadata.metadata.remove_columns(self.date_columns)

        # ensure continuous columns are numeric everywhere evaluators consume them
        self.train_df = self._coerce_continuous_numeric(self.train_df)
        self.val_df = self._coerce_continuous_numeric(self.val_df)

        # self.train_data_transformed = self.train_data.preprocess_data(inplace=False, intact_cols=intact_cols)
        self.train_data.preprocess_data(inplace=True, subsample=subsample)
        self.val_data.preprocess_data(inplace=True, subsample=subsample,
                                      reuse_preprocessor=self.train_data.preprocessor)

        # wrap it as an instance of the class Data
        # self.train_data_transformed = Data(data = self.train_data_transformed, 
        #                                    metadata = None) # self.train_data.metadata)
        # self.train_data = Data(data = self.train_data, metadata = train_data.metadata)
        # self.val_data = Data(data = self.val_data, metadata = val_data.metadata)

        # rewrite values of keys present in both 'self.train_data_transformed.metadata' and 'self.train_data.metadata'
        # with values in the latter
        # dict_meta_transformed = self.train_data_transformed.metadata.metadata.to_dict()
        dict_meta = self.train_data.metadata.metadata.to_dict()

        # dict_meta_transformed.update(self.train_data.metadata.metadata.to_dict())
        dict_meta_src = self.train_data.metadata.metadata.to_dict()

        # update primary keys
        dict_meta['primary_key'] = 'row_id'

        for k,v in dict_meta['columns'].items():
            if k in dict_meta_src['columns'].keys():
                dict_meta['columns'][k].update(dict_meta_src['columns'][k])

            if 'pii' in dict_meta['columns'][k].keys():
                del dict_meta['columns'][k]['pii']
                # dict_meta['columns'][k]['sdtype']

            # ensure if sdtype is either numerical or categorical or id
            if 'sdtype' in dict_meta['columns'][k].keys():
                if dict_meta['columns'][k]['sdtype'] not in ['numerical', 'categorical', 'datetime', 'id']:
                    dict_meta['columns'][k]['sdtype'] = 'categorical'

            # DEBUG ; TODO: remove it later, find a better solution
            # override 'datetime' columns, switch their sdtype to 'numerical'
            if dict_meta['columns'][k]['sdtype'] == 'datetime':
                dict_meta['columns'][k]['sdtype'] = 'numerical'
                # and delete the key `datetime_format`
                del dict_meta['columns'][k]['datetime_format']

        # change sdtype of `row_id` to `id`
        if 'row_id' in dict_meta['columns'].keys():
            dict_meta['columns']['row_id']['sdtype'] = 'id'

        # dump the dictionary with metadata to the disk in the current folder
        with open('metadata_dict.json', 'w') as f:
            import json
            json.dump(dict_meta, f, indent=4)

        STM = SingleTableMetadata()
        self.train_data.metadata.metadata = STM.load_from_dict(dict_meta)
        self.val_data.metadata.metadata = STM.load_from_dict(dict_meta)

        self.target_feature = target_feature

        self.modelParamSpace = modelParamSpace

        if isinstance(self.modelParamSpace, dict):
            self.action_size = np.abs(np.prod([len(v) for v in self.modelParamSpace.values()]))
            self.state_size = 4 
            # self._load_and_preprocess_data()
            self.agent = QLearningAgent(
                state_size=self.state_size, action_size=self.action_size,
                learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
                # epsilon_decay=0.995, 
                # epsilon_min=0.01
            )

        self.last_eval = {'tvd_score': 0.5, 'utility_score': 0.5, 
                                  'ks_score': 0.5, 'js_score': 0.5}
        
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = {'rewards': []}
    

    def _coerce_continuous_numeric(self, df: pd.DataFrame, all_columns=False) -> pd.DataFrame:
        """Coerce continuous columns (mixed str/float object dtype, e.g. '65.0') to numeric."""
        df = df.copy()
        if not all_columns:
            for col in self.continuous_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # In rl_optimizer.py

    def _generate_synthetic_data(self, modelClass: type, params: Dict, n_samples: int = 100) -> pd.DataFrame:

        def enforce_ordinal_valid_range(N: int, max_N: int):
                """ Helper function that replaces ordinal integer with valid value if it is too big """
                if N >= max_N:
                    return N % max_N
                else:
                    return N

        n_samples = n_samples if n_samples else len(self.df_train_model)

        if hasattr(modelClass, '__init__'):
            listModelClassArgs = list(inspect.signature(modelClass.__init__).parameters)
        else:
            raise TypeError(f"{modelClass} does not have an __init__ method.")

        if 'override_args' in listModelClassArgs:
            model = modelClass(override_args=params)
        else:
            model = modelClass(**params)

        # self.train_data_transformed.metadata.metadata.primary_key = ''

        print("** Inside the method `_generate_synthetic_data **`")

        # import ipdb; ipdb.set_trace()

        # convert all object dtypes to float in the `self.train_data.data`
        self.train_data.data = self._coerce_continuous_numeric(self.train_data.data, all_columns=True)

        # import ipdb; ipdb.set_trace()

        # subsample train_data down to `n_samples` and save in the var. sub_train_data
        sub_train_data: pd.DataFrame = self._stratified_subsample(self.train_data.data, n_rows=n_samples)
        sub_train_data: Data = Data(sub_train_data, metadata=self.train_data.metadata)

        # model.fit(self.train_data)
        model.fit(sub_train_data)
        model_output = model.sample(n_samples)

        if isinstance(model_output, list):
            samples = np.array(model_output[0])
        else:
            samples = np.array(model_output)
        
        preprocessor = self.train_data.preprocessor

        # Convert array samples back into a DataFrame matching transformed column names
        full_feature_names = list(preprocessor.get_feature_names_out())
        df_samples = pd.DataFrame(samples, columns=[name.split("__")[-1] for name in full_feature_names])

        # Inverse transform using the class method of Data - `inverse_transform_data`
        df_syn = self.train_data.inverse_transform_data(df_samples)

        return df_syn[self.train_df.columns]


    def _evaluate_synthetic_data(self, synthetic_data: pd.DataFrame) -> Dict:
        """ Evaluatin pipelines """
        # try:
        fid = FidelityEvaluator(self.df_train_model, synthetic_data, self.discrete_cols)
        fid_rep = fid.get_detailed_report()

        # import ipdb; ipdb.set_trace()
        
        # Instantiate the utility evaluator class
        util = UtilityEvaluator(
            real_train=self.train_df, 
            real_test=self.val_df, 
            synthetic_data=synthetic_data, 
            target_column=self.target_feature, 
            discrete_columns=self.discrete_cols,
            n_target_bins=4
        )
        util_rep = util.get_detailed_report()
        
        return {
            'tvd_score': float(fid_rep.get('tvd_score', 0.0)),
            'utility_score': float(util_rep.get('utility_score', 0.0)),
            'ks_score': 1.0 - fid_rep['univariate_distributions']['overall_ks_score'],
            'js_score': 1.0 - fid_rep['univariate_distributions']['overall_js_score']   
        }
        
    
    def _stratified_subsample(self, df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
        """Subsample n_rows from df, guaranteeing every target class present in df keeps at least one row.

        One of method limitations - it cannot handle properly a dataframe if 
        the number of rows is lower than the number of classes. 
        Use it cautiously in case of features with high cardinality and not so long table.

        A plain `.sample()` can drop rare target classes entirely, which later makes
        roc_auc_score raise ValueError because the trained model's class count no
        longer matches the number of classes seen in the (separate) test split.
        """

        # send a warning if the number of classes exceeds the total number of rows in the dataframe `df`
        if self.target_feature not in df.columns:
            raise ValueError(f"Target feature '{self.target_feature}' not found in the dataframe columns.")

        # if df[self.target_feature].nunique() > len(df):
        #     warnings.warn(f"Number of unique classes in target feature '{self.target_feature}' "
        #                   f"({df[self.target_feature].nunique()}) exceeds the number of rows in the dataframe ({len(df)}). "
        #                   "This may lead to unexpected behavior during stratified subsampling.")

        if self.target_feature not in df.columns or len(df) <= n_rows:
            return df.sample(n=min(n_rows, len(df)), random_state=random_state)

        one_per_class = df.groupby(self.target_feature, group_keys=False).sample(n=1, random_state=random_state)
        remaining_needed = n_rows - len(one_per_class)
        if remaining_needed <= 0:
            return one_per_class.sample(n=n_rows, random_state=random_state)

        remaining_pool = df.drop(index=one_per_class.index)
        extra = remaining_pool.sample(n=min(remaining_needed, len(remaining_pool)), random_state=random_state)
        return pd.concat([one_per_class, extra]).sample(frac=1, random_state=random_state)


    def evaluate_synthetic_data(self, synthetic_data: pd.DataFrame, n_rows: int = 100) -> Dict:
            """ Evaluate fidelity and utility of synthetic data """

            # TODO: replace  a subsample of train_data with a subsample of validation data for sticking to good practice

            # try:
            synthetic_data = self._coerce_continuous_numeric(synthetic_data)

            # sumbsample `n_rows` from self.train_df and synthetic_data, they should be the rows from same positions but chosen randomly
            sub_val_data = self._stratified_subsample(self.val_data.data, n_rows)
            sub_train_data = self._stratified_subsample(self.train_data.data, n_rows)

            for c in self.val_data.date_cols:
                synthetic_data[c] = pd.to_datetime(synthetic_data[c], errors='coerce').dt.strftime("%Y-%m-%d")

            sub_synthetic_data = synthetic_data.sample(n=min(n_rows, len(synthetic_data)), 
                                                       random_state=42)

            # make a Data instance based on sub_synthetic_data
            sub_synthetic_data_ = Data(sub_synthetic_data, metadata=self.train_data.metadata)

            # transform data
            sub_synthetic_data_transformed = sub_synthetic_data_.preprocess_data(
                inplace=False, subsample=False, reuse_preprocessor=self.train_data.preprocessor)

            # fid = FidelityEvaluator(self.train_df, synthetic_data, self.discrete_columns)
            fid = FidelityEvaluator(sub_val_data, sub_synthetic_data_transformed, 
                                    self.discrete_columns)

            # import ipdb;  ipdb.set_trace() # TODO: delete this line later
            
            fid_rep = fid.get_detailed_report()
    
            util = UtilityEvaluator(
                real_train=sub_train_data,
                real_test=self.val_data.data,  # val data must be transformed too!
                synthetic_data=sub_synthetic_data_transformed,
                target_column=self.target_feature,
                discrete_columns=self.discrete_columns,
            )

            util_rep = util.get_detailed_report()

            # throw a watning if utility score is not within the range  [0, 1]
            if not (0.0 <= util_rep.get('utility_score') <= 1.0):
                warnings.warn(f"Utility score {util_rep.get('utility_score')} is not within the range [0, 1]. "
                              "This may indicate an issue with the evaluation process or the synthetic data quality.")

            # overwrite utility score to 1.0 if it's higher
            util_rep['utility_score'] = min(util_rep.get('utility_score', 0.0), 0.99)
            
            return {
                'scores': {'fidelity_score':fid_rep['fidelity_score'], 'utility_score':util_rep['utility_score']},
                'fidelity_details':fid_rep,
                'tvd_score': float(fid_rep.get('tvd_score', 0.0)),
                'utility_score': float(util_rep.get('utility_score', 0.0)),
                # 'utility_score': min(float(util_rep.get('utility_score', 0.0)), 1.0),
                'ks_score': 1.0 - fid_rep['univariate_distributions']['overall_ks_score'],
                'js_score': 1.0 - fid_rep['univariate_distributions']['overall_js_score']
            }
    

    def optimize(self, episodes=30, verbose=True, 
                 modelClass: type=None, n_rows: int = 200):
        self.modelClass = modelClass
        print(f"Running the optimization for {episodes} episodes ...")
        for ep in range(episodes):
            print(f"Episode {ep+1}/{episodes} - Last Eval: {self.last_eval}")
            state = np.clip([self.last_eval['tvd_score'], self.last_eval['utility_score'], 
                             self.last_eval['ks_score'], self.last_eval['js_score']], 0, 1)
            print(f"  State: {state}")
            action = self.agent.choose_action(state)
            print(f"  Action: {action}")
            params = self._action_to_params(action)
            print(f"  Params: {params}")
            print(f"  Generating synthetic data with params: {params} ...")

            syn = self._generate_synthetic_data(modelClass, params, n_samples=n_rows)
            print(f"  Generated Synthetic Data Shape: {syn.shape if syn is not None else 'None'}")
            if syn is None: continue

            # import ipdb; ipdb.set_trace()
            res = self.evaluate_synthetic_data(syn, n_rows=n_rows)

            # check if any values in the dict res contain nans, if yes - throw a warning
            scalar_scores = [res[k] for k in ('tvd_score', 'utility_score', 'ks_score', 'js_score')]
            if any(np.isnan(scalar_scores)):
                warnings.warn(f"Evaluation results contain NaN values: {res}. This may affect the reward calculation.")
                # replace nans with 0.0 in res
                # res = {k: (0.0 if np.isnan(v) else v) for k, v in res.items()} 

            print(f"  Evaluation Results: {res}")

            self.last_eval = res
            reward = 0.3 * np.nanmean([res['tvd_score'], res['ks_score'], res['js_score']]) + 0.7 * res['utility_score'] # TODO: test this formula and change later if needed
            print(f"  Reward: {reward:.4f}") 
            self.optimization_history['rewards'].append(reward)
            
            self.agent.update(state, action, reward, state, True)
            if reward > self.best_score:
                self.best_score, self.best_params = reward, params
                if verbose: print(f"  Ep {ep+1}: New Best Reward {reward:.4f} (Util on Val: {res['utility_score']:.3f})")
                
        # dump the optimization history as json file
        from datetime import date    
        today = date.today().isoformat()
        history_path = f'optimization_history_{modelClass.__qualname__}_{today}.json'
        with open(history_path, 'w') as f:
            import json
            json.dump(self.optimization_history, f)

        return self.best_params


    def _action_to_params(self, action: int) -> Dict:
        idxs = {}
        prev_product = 1

        for k,v in self.modelParamSpace.items():
            idxs[k] = (action // prev_product) % len(v)
            prev_product *= len(v)

        new_params = {k: self.modelParamSpace[k][idxs[k]] for k in self.modelParamSpace}
        return new_params


    def generate_with_best_params(self, output_path, n_samples=1000):
        syn = self._generate_synthetic_data(self.modelClass, self.best_params, n_samples)
        syn.to_csv(output_path, index=False)
        return output_path
    def get_best_params(self): return self.best_params
    def save_agent(self, path): self.agent.save(path)