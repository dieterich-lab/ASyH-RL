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
    train_path = 'puf_narrow/train.csv'
    val_path = 'puf_narrow/val.csv'
    # syn_path = 'tables4comparison/synthetic_data_gmm_optimized_fixed.csv'
    syn_path = 'tables4comparison/synthetic_data_gaussian_copula_optimized.csv'
    discrete_cols = ...

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    synthetic_data = pd.read_csv(syn_path)

    train_data = Data(train_data)
    val_data = Data(val_data)
    synthetic_data = Data(synthetic_data)

    cont_columns = []
    for col in train_data.data.columns:
        if train_data.data[col].nunique() > 20:
            cont_columns.append(col)

    discrete_columns = list([col for col in train_data.data.columns if col not in cont_columns])

    optimizer = RLOptimizer(train_data, val_data, discrete_columns=discrete_columns)

    eval_result: Dict = optimizer.evaluate_synthetic_data(synthetic_data=synthetic_data.data)

    print(eval_result)

    visualizer = Visualizer()

    visualizer.plot_comprehensive_scores(eval_result)


if __name__ == '__main__':
    main()
