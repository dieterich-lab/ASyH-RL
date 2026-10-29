# -*- coding: utf-8 -*-
"""
Main entry point for generative model optimization.
Synchronized with 80/10/10 data splitting.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from rl_optimizer import RLOptimizer
from sklearn.compose import ColumnTransformer
from models import GaussianMixtureModel
from models import GaussianCopulaModel
from models import CopulaGANModel
from models import TVAEModel
from models import CTGANModel
from data import Data


SHARE_TRAIN: float = 0.8
NUM_EPISODES: int = 50
POSTFIX: str = 'pcs_score_2026-10-19-1830-long' # add a date here and short label to distinguish the experiment results
MODELS_AVAIL = [CTGANModel, CopulaGANModel, TVAEModel, GaussianCopulaModel, GaussianMixtureModel]


# the function for parsing command-line arguments
def get_args():
    # add the argument `--model`, the type - string and it must be an element of `MODELS_AVAIL`
    parser = argparse.ArgumentParser(description="Run RL-driven hyperparameter optimization for a given model.")
    # parser.add_argument('--model', type=str, choices=[m.__name__ for m in MODELS_AVAIL], required=True, help="The model to optimize. Must be one of: " + ", ".join([m.__name__ for m in MODELS_AVAIL]))
    # add the argument `--num-model` the number of used model in a list `MODELS_AVAIL`
    parser.add_argument('--num-model', type=int, choices=range(len(MODELS_AVAIL)), required=True, dest="model_num",
                        help="The index of the model to optimize. Must be one of: " + ", ".join([f"{i}: {m.__name__}" for i, m in enumerate(MODELS_AVAIL)]))
    return parser.parse_args()


def invoke_model(MODEL):
    # MODEL = TVAEModel # CopulaGANModel # GaussianCopulaModel
    print("=" * 80)
    print(f"{MODEL.__qualname__} Hyperparameter Optimization (RL-driven with Val Set)")
    print("=" * 80)
    
    # 1. Provide valid paths
    train_data_path = 'puf_wide/train.csv' # 80%
    val_data_path = 'puf_wide/val.csv'     # 10%
    
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        print("Invalid path to - train.csv or val！")
        sys.exit(1)

    # 2. Open the csv table with data for training
    df = pd.read_csv(train_data_path, sep=',', on_bad_lines='skip')
    # discrete_columns = ['agegroup', 'sex', 'location', 'status', 'SAHfh', 'drink', 'smoke',
                        # 'HBP', 'DM', 'IS', 'CAD', 'number', 'shape1', 'maxgroup', 'shape2']
    # discrete_columns = ['cohort']
    cont_columns = []
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric or df[col].nunique() > 20:
            cont_columns.append(col)

    discrete_columns = list([col for col in df.columns if col not in cont_columns])

    train_df = pd.read_csv(train_data_path, sep=',', on_bad_lines='skip')
    val_df = pd.read_csv(val_data_path, sep=',', on_bad_lines='skip')
    # import ipdb; ipdb.set_trace()
    num_rows_all = int(train_df.shape[0] / SHARE_TRAIN)
    print(f"Total number of rows expected in a synthesized table is: {num_rows_all}")

    # TODO: REVIEW if we still really need the column "row_id"
    # introduce a new column `row_id`, where each element is a string of format "pXvY", 
    # `X` is the value in `Pat_ID.numeric` on the same row
    #`Y`is the value in `Visite.numeric` on the same row
    new_col_name = 'row_id'
    discrete_columns.append(new_col_name)
    train_df[new_col_name] = train_df.apply(lambda row: f"p{row['Pat_ID.numeric']}v{row['Visite.numeric']}c{row['Cohort.numeric']}dexam{row['date_exam']}dinfect{row['date_infection']}", axis=1)
    val_df[new_col_name] = val_df.apply(lambda row: f"p{row['Pat_ID.numeric']}v{row['Visite.numeric']}c{row['Cohort.numeric']}dexam{row['date_exam']}dinfect{row['date_infection']}", axis=1)

    # drop duplicate rows where unique values of `row_id` column are repeated
    train_df = train_df.drop_duplicates(subset=new_col_name, keep='first')
    val_df = val_df.drop_duplicates(subset=new_col_name, keep='first')

    train_data = Data(train_df)
    val_data = Data(val_df)


    # TODO: once the bug is resolved, switch parameter ranges back to original values
    modelParamSpace = {'default_distribution':['beta', 'norm', 'gamma'], 
                       'enforce_min_max_values':[True,False], 
                       'epochs':[25, 50, 100], 
                       'verbose':[True],
                       'n_components':[5, 10],
                       'n_init':[2, 5],
                       'max_iter':[50, 100]}

    if hasattr(MODEL(), '_sdv_model_class'):
        valid_param_names = MODEL()._sdv_model_class.__init__.__code__.co_varnames
    else:
        valid_param_names = MODEL.__init__.__code__.co_varnames

    # check each key of modelParamSpace, if it's present in `valid_param_names`
    # if not - delete the key
    modelParamSpace = {k: v for k, v in modelParamSpace.items() if k in valid_param_names}
    print(f"\n[RL-Asyh] Valid hyperparameter search space for {MODEL.__qualname__}: {modelParamSpace}")
    
    optimizer = RLOptimizer(
        train_data = train_data, 
        val_data = val_data, 
        discrete_columns=discrete_columns,
        modelParamSpace=modelParamSpace,
        target_feature = 'pcss' # 'score-eq5d-5l'
        # target_feature='score-eq5d-5l'
    )

    # add modelParamSpace to the dictionary `optimizer.optimization_history`
    optimizer.optimization_history['modelParamSpace'] = modelParamSpace
    # calculate the product of all value lengths, for each value which is a list or sequence
    sizeParamSpace = np.prod([len(v) for v in modelParamSpace.values() if isinstance(v, (list, tuple, np.ndarray))])
    print(f"\n[RL-Asyh] Total number of hyperparameter combinations to explore: {sizeParamSpace}")
    optimizer.optimization_history['sizeParamSpace'] = str(sizeParamSpace)
    
    # 3. Optimize model h-params with rl-agent
    print("\n[RL-Asyh] running the optimization ...")
    # TODO: once the bug is resolved, switch number of episodes back to 30
    optimizer.optimize(episodes=NUM_EPISODES, modelClass=MODEL, n_rows=500) # TODO: REVIEW, what's an optimal value of the arg. `n_rows`?
    
    # 4. Collect the best h-params
    best_params = optimizer.get_best_params()
    print(f"\n Best parameters: {best_params}")
    
    output_csv = f'synthetic_data_{MODEL.__qualname__}_wide_rl_{NUM_EPISODES}ep_{POSTFIX}.csv'
    optimizer.generate_with_best_params(output_csv, n_samples=num_rows_all)
    print(f"✓ The synthesis has been completed: {output_csv}")

    # open the output csv
    with open(output_csv, 'r') as f:
        synth_data = pd.read_csv(f, sep=',', on_bad_lines='skip')

    # evaluate quality with sdv built-in methods
    from sdv.evaluation.single_table import evaluate_quality

    quality_report = evaluate_quality(train_data, synth_data, metadata=train_data.metadata)

    # save it on disk
    quality_report_path = f'quality_report_{MODEL.__qualname__}_wide_rl_{NUM_EPISODES}ep_{POSTFIX}.csv'
    quality_report.to_csv(quality_report_path, index=False)

    # evaluate data fidelity with sdv built-in methods

    from sdv.evaluation.single_table import run_diagnostic

    diagnostic = run_diagnostic(
        real_data=train_data,
        synthetic_data=synth_data,
        metadata=train_data.metadata
    )

    # save it on disk
    diagnostic_report_path = f'diagnostic_report_{MODEL.__qualname__}_wide_rl_{NUM_EPISODES}ep_{POSTFIX}.csv'
    diagnostic.to_csv(diagnostic_report_path, index=False)


def main():
    args = get_args()
    MODEL = MODELS_AVAIL[args.model_num]
    invoke_model(MODEL)


if __name__ == "__main__":
    main()