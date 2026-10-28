import sys
import os
from typing import Dict, List
import numpy as np
import pandas as pd
# -------------------------------------------------
base_dir = os.path.dirname(__file__)
# sys.path.insert(0, base_dir)
sys.path.append(os.path.abspath(os.path.join(base_dir, '..')))
# --------------------------------------------------
from visualizer import Visualizer
# from evaluator.utility_evaluator_2 import UtilityEvaluator
# from rl_agent.q_learning_agent import QLearningAgent
# from evaluator.fidelity_evaluator import FidelityEvaluator
from rl_optimizer import RLOptimizer
# import class data from the unit `data.py` located in the parent folder
from data import Data


def main():
    train_path = 'puf_wide/train.csv'
    val_path = 'puf_wide/val.csv'

    syn_paths_names = {
        'GMM-RL':'./synthetic_data_GaussianMixtureModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv',
        'Gaussian Copula':'./synthetic_data_GaussianCopulaModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv',
        'TVAE':'./synthetic_data_TVAEModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv',
        'CTGAN':'./synthetic_data_CTGANModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv'
        }

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    train_data = Data(train_data)
    val_data = Data(val_data)

    columns_to_drop = ['row_id']

    # cols_drop = [col for col in columns_to_drop if col in train_data.data.columns and col in val_data.data.columns]
    # # drop columns in place
    # train_data.data.drop(columns=cols_drop, inplace=True)
    # val_data.data.drop(columns=cols_drop, inplace=True)

    cont_columns = []
    for col in train_data.data.columns:
        if train_data.data[col].nunique() > 20:
            cont_columns.append(col)
    
    discrete_columns = list([col for col in train_data.data.columns if col not in cont_columns])

    # import ipdb; ipdb.set_trace()

    columns_to_drop += train_data.date_cols
    cols_drop = [col for col in columns_to_drop if col in train_data.data.columns]
    # drop the columns in `cols_to_drop`, in train_data and val_data, do it in-place
    train_data.data.drop(columns=cols_drop, inplace=True)
    cols_drop = [col for col in columns_to_drop if col in val_data.data.columns]
    val_data.data.drop(columns=cols_drop, inplace=True)

    optimizer = RLOptimizer(train_data, val_data,
                            discrete_columns=discrete_columns,
                            target_feature='pcss') # TODO: compare later the result from pcss vs. eq5d-5l
                            # target_feature='score-eq5d-5l')

    eval_results = {}

    # import ipdb; ipdb.set_trace()

    for name,path in syn_paths_names.items():
        synthetic_data = pd.read_csv(path)

        print(f"\nEvaluating synthetic data from {name} at path: {path}")

        synthetic_data = Data(synthetic_data)

        cols_drop = [col for col in columns_to_drop if col in synthetic_data.data.columns]

        synthetic_data.data.drop(columns=cols_drop, inplace=True)

        eval_results[name] = optimizer.evaluate_synthetic_data(synthetic_data=synthetic_data.data, n_rows=1000) # (len(train_data.data) // 10))

        print(f"\nEvaluation results for {name}:")

        # import ipdb; ipdb.set_trace()

        del synthetic_data

    # eval_result: Dict = optimizer.evaluate_synthetic_data(synthetic_data=synthetic_data.data)

    print(eval_results)

    # import ipdb; ipdb.set_trace()

    visualizer = Visualizer()

    # visualizer.plot_comprehensive_scores(eval_result, plot_name='comprehensive_score_gmm_rl_wide_40ep.png')
    visualizer.plot_comprehensive_scores_overlay(eval_results, plot_name='comprehensive_score_overlay_rl_wide_40ep_2026-10-24.png')


if __name__ == '__main__':
    main()
