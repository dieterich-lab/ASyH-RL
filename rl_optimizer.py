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
                 target_feature: str = 'score-eq5d-5l_start'):
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
        # self.discrete_columns = Data.detect_discrete_columns(self.train_df) if discrete_columns is None else discrete_columns
        # self.date_columns = Data.detect_date_columns(self.train_df)
        # self.continuous_cols = list(set(self.train_df.columns) - set(self.discrete_columns) - set(self.date_columns))
        col_types_dict = self.train_data.detect_column_types(self.train_df)
        self.discrete_columns = col_types_dict['categorical']
        self.continuous_cols = self.numerical_cols = col_types_dict['numerical']
        self.date_columns = col_types_dict['date']

        # TODO: remove the following lines later upon finishing debugging  DONE
        # DEBUG: delete all columns present in self.date_columns from self.train_data and self.train_df
        # self.train_df.drop(columns=self.date_columns, inplace=True, errors='ignore')
        # self.train_data.data.drop(columns=self.date_columns, inplace=True, errors='ignore')
        # # DEBUG
        # self.val_df.drop(columns=self.date_columns, inplace=True, errors='ignore')
        # self.val_data.data.drop(columns=self.date_columns, inplace=True, errors='ignore')

        # # update metadata accordingly
        # from sdv.metadata import SingleTableMetadata
        # STM = SingleTableMetadata()
        # STM.detect_from_dataframe(self.train_df)
        # self.train_data.metadata.metadata = STM.load_from_dict(STM.to_dict())
        # self.val_data.metadata.metadata = STM.load_from_dict(STM.to_dict())

        # self.train_data.metadata.metadata = self.train_data.metadata.metadata.remove_columns(self.date_columns)
        # self.val_data.metadata.metadata = self.val_data.metadata.metadata.remove_columns(self.date_columns)

        # ensure continuous columns are numeric everywhere evaluators consume them
        self.train_df = self._coerce_continuous_numeric(self.train_df)
        self.val_df = self._coerce_continuous_numeric(self.val_df)

        # self.train_data_transformed = self.train_data.preprocess_data(inplace=False, intact_cols=intact_cols)
        self.train_data.preprocess_data(inplace=True, subsample=True)
        self.val_data.preprocess_data(inplace=True, subsample=True)

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
        # dict_meta_transformed['primary_key'] = dict_meta_src.get('primary_key', None)
        # dict_meta['primary_key'] = dict_meta_src.get('primary_key', None)
        # assign 'row_id' as primary_key in the dict_meta
        dict_meta['primary_key'] = 'row_id'

        # for k,v in dict_meta_src['columns'].items():
        #     if k in dict_meta_transformed['columns'].keys():
        #         dict_meta_transformed['columns'][k].update(v)
        #         if 'pii' in dict_meta_transformed['columns'][k].keys():
        #             if 'pii' in v.keys():
        #                 dict_meta_transformed['columns'][k]['pii'] = v['pii']
        #             else:
        #                 del dict_meta_transformed['columns'][k]['pii']

        for k,v in dict_meta['columns'].items():
            if k in dict_meta_src['columns'].keys():
                dict_meta['columns'][k].update(dict_meta_src['columns'][k])

            if 'pii' in dict_meta['columns'][k].keys():
                # if k in dict_meta_src['columns'].keys() and 'pii' in dict_meta_src['columns'][k].keys():
                    # if 'pii' in dict_meta_src['columns'][k].keys():
                    # dict_meta['columns'][k]['pii'] = dict_meta_src['columns'][k]['pii']
                # else:
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
        dict_meta['columns']['row_id']['sdtype'] = 'id'

        # dump the dictionary with metadata to the disk in the current folder
        with open('metadata_dict.json', 'w') as f:
            import json
            json.dump(dict_meta, f, indent=4)

        # import ipdb; ipdb.set_trace()
        # self.train_data_transformed.metadata.metadata = self.train_data_transformed.metadata.metadata.load_from_dict(dict_meta)
        STM = SingleTableMetadata()
        self.train_data.metadata.metadata = STM.load_from_dict(dict_meta)
        self.val_data.metadata.metadata = STM.load_from_dict(dict_meta)

        # import ipdb; ipdb.set_trace()

        self.target_feature = target_feature

        self.modelParamSpace = modelParamSpace
        
        # self.action_size = len(self.n_components_options) * len(self.max_iter_options) * len(self.n_init_options)
        # assert self.modelParamSpace is not None, "model_params cannot be None"

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

        model.fit(self.train_data)
        model_output = model.sample(n_samples)

        if isinstance(model_output, list):
            samples = np.array(model_output[0])
        else:
            samples = np.array(model_output)
        
        preprocessor = self.train_data.preprocessor
        # num_transformer = preprocessor.named_transformers_['num']
        # cat_transformer = preprocessor.named_transformers_['cat']

        # num_cols = list(preprocessor.transformers_[0][2])
        # cat_cols = list(preprocessor.transformers_[1][2])

        # Convert array samples back into a DataFrame matching transformed column names
        full_feature_names = list(preprocessor.get_feature_names_out())
        df_samples = pd.DataFrame(samples, columns=[name.split("__")[-1] for name in full_feature_names])

        # # scaler/ordinal were fit on their branch's *full* output (incl. own missingindicator_* cols),
        # # so we must feed all of those back in before slicing out the real columns
        # num_branch_cols = [name.split("__")[-1] for name in full_feature_names if name.startswith("num__")]
        # cat_branch_cols = [name.split("__")[-1] for name in full_feature_names if name.startswith("cat__")]

        # # 1. Inverse transform numeric features using the scaler step, then drop indicator columns

        # # check if `scaler` is present in named_steps of num_transformer
        # if 'scaler' not in num_transformer.named_steps:
        #     # then take df_samples as it is
        #     inv_num = df_samples[num_branch_cols].values
        #     inv_num = inv_num[:, :len(num_cols)]
        # else:
        #     scaler = num_transformer.named_steps['scaler']
        #     # import ipdb; ipdb.set_trace()
        #     inv_num_full = scaler.inverse_transform(df_samples[num_branch_cols].values)
        #     inv_num = inv_num_full[:, :len(num_cols)]

        # # 2. Inverse transform categorical features, then drop indicator columns
        # ordinal = cat_transformer.named_steps['ordinal']
        # # onehot = cat_transformer.named_steps['onehot']

        # # check if the number of categories stored inside `ordinal` is not less than `df_samples[cat_branch_cols].nunique()`
        # # if len(ordinal.categories_) < df_samples[cat_branch_cols].nunique():
        # #     # using nearest neighbor algorithm replace values in df_samples[cat_branch_cols] with nearest ones present in ordinal
        # #     from sklearn.neighbors import NearestNeighbors
        # #     nn = NearestNeighbors(n_neighbors=1)
        # #     # fit nn on ordinal.categories_
        # #     nn.fit(np.array(ordinal.categories_).reshape(-1, 1))
        # #     # find nearest neighbors for df_samples[cat_branch_cols]
        # #     nearest = nn.kneighbors(df_samples[cat_branch_cols].values.reshape(-1, 1), return_distance=False)

        # # apply the helper function `self.move_ordinal_valid_range(N, max_N)` on `df_samples[num_branch_cols].values`
        # # where max_N is the number of categories in ordinal
        # for idx,col in enumerate(cat_branch_cols):
        #     df_samples[col] = df_samples[col].apply(lambda x: enforce_ordinal_valid_range(x, len(ordinal.categories_[idx])))

        # inv_cat_full = ordinal.inverse_transform(df_samples[cat_branch_cols].values)
        # # inv_cat_full = onehot.inverse_transform(df_samples[cat_branch_cols].values)
        # inv_cat = inv_cat_full[:, :len(cat_cols)]

        # df_syn = pd.concat([
        #     pd.DataFrame(inv_num, columns=num_cols),
        #     pd.DataFrame(inv_cat, columns=cat_cols)
        # ], axis=1)

        # return df_syn[self.train_df.columns]

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

        if df[self.target_feature].nunique() > len(df):
            warnings.warn(f"Number of unique classes in target feature '{self.target_feature}' "
                          f"({df[self.target_feature].nunique()}) exceeds the number of rows in the dataframe ({len(df)}). "
                          "This may lead to unexpected behavior during stratified subsampling.")

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
            sub_train_data = self._stratified_subsample(self.train_data.data, n_rows)

            for c in self.train_data.date_cols:
                synthetic_data[c] = synthetic_data[c].dt.strftime("%Y-%m-%d")

            sub_synthetic_data = synthetic_data.sample(n=min(n_rows, len(synthetic_data)), 
                                                       random_state=42)

            # make a Data instance based on sub_synthetic_data
            sub_synthetic_data_ = Data(sub_synthetic_data, metadata=self.train_data.metadata)

            # transform data
            sub_synthetic_data_transformed = sub_synthetic_data_.preprocess_data(inplace=False, subsample=False)

            # fid = FidelityEvaluator(self.train_df, synthetic_data, self.discrete_columns)
            fid = FidelityEvaluator(sub_train_data, sub_synthetic_data_transformed, 
                                    self.discrete_columns)

            # import ipdb;  ipdb.set_trace() # TODO: delete this line later
            
            fid_rep = fid.get_detailed_report()
    
            # import ipdb; ipdb.set_trace()
            
            # util = UtilityEvaluator(
            #     real_train=self.train_df, 
            #     real_test=self.val_df, 
            #     synthetic_data=synthetic_data, 
            #     target_column=self.target_feature, 
            #     discrete_columns=self.discrete_columns
            # )
            util = UtilityEvaluator(
                real_train=sub_train_data,
                real_test=self.val_data.data,  # val data must be transformed too!
                synthetic_data=sub_synthetic_data_transformed,
                target_column=self.target_feature,
                discrete_columns=self.discrete_columns,
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
    

    def optimize(self, episodes=30, verbose=True, modelClass: type=None):
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
            # import ipdb; ipdb.set_trace()  # Debugging point to inspect the parameters and state before generating synthetic data

            syn = self._generate_synthetic_data(modelClass, params)
            print(f"  Generated Synthetic Data Shape: {syn.shape if syn is not None else 'None'}")
            if syn is None: continue
                
            res = self.evaluate_synthetic_data(syn)
            print(f"  Evaluation Results: {res}")
            self.last_eval = res
            reward = 0.5 * np.mean([res['tvd_score'], res['ks_score'], res['js_score']]) + 0.5 * res['utility_score']
            print(f"  Reward: {reward:.4f}")
            self.optimization_history['rewards'].append(reward)
            
            self.agent.update(state, action, reward, state, True)
            if reward > self.best_score:
                self.best_score, self.best_params = reward, params
                if verbose: print(f"  Ep {ep+1}: New Best Reward {reward:.4f} (Util on Val: {res['utility_score']:.3f})")
                
        # dump the optimization history as json file
        history_path = f'optimization_history_{modelClass.__qualname__}.json'
        with open(history_path, 'w') as f:
            import json
            json.dump(self.optimization_history, f)

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
