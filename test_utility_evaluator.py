# test_utility_evaluator.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from evaluator.utility_evaluator_2 import UtilityEvaluator


@pytest.fixture
def classification_frames():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    # split per class so every class is guaranteed to appear in both train and
    # test (an uneven split previously caused roc_auc_score to raise ValueError)
    train_parts, test_parts = [], []
    for cls in df["target"].unique():
        cls_df = df[df["target"] == cls]
        cut = int(len(cls_df) * 0.7)
        train_parts.append(cls_df.iloc[:cut])
        test_parts.append(cls_df.iloc[cut:])

    real_train = pd.concat(train_parts).reset_index(drop=True)
    real_test = pd.concat(test_parts).reset_index(drop=True)
    # shuffled copy of the training data stands in for "synthetic" data with a learnable signal
    synthetic = real_train.sample(frac=1.0, random_state=1).reset_index(drop=True)

    return real_train, real_test, synthetic


def test_utility_evaluator_returns_non_zero_utility_score(classification_frames):
    real_train, real_test, synthetic = classification_frames

    evaluator = UtilityEvaluator(
        real_train=real_train,
        real_test=real_test,
        synthetic_data=synthetic,
        target_column="target",
    )

    report = evaluator.get_detailed_report(random_state=42)
    utility_score = report["utility_score"]

    assert np.isfinite(utility_score)
    assert utility_score != 0.0, "utility_score is 0 - evaluator likely failed internally and fell back to 0.0"

    # each model's utility score should have been computed, not silently zeroed out
    for model_name, score in report["model_results"]["utility_scores"].items():
        assert score != 0.0, f"{model_name} utility score fell back to 0.0"
