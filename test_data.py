# tests/test_data.py
import numpy as np
import pandas as pd
import pytest

from data import Data


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, None, 35, 45],
            "sex": ["M", "F", None, "M"],
            "income": [50000, 60000, 70000, None],
            "city": ["A", "B", "A", "C"],
        }
    )


def test_data_init_and_metadata(sample_df):
    meta = {"columns": ["age", "sex", "income", "city"]}
    d = Data(data=sample_df, metadata=meta)

    assert d.data is sample_df
    assert d.metadata == meta


def test_set_metadata(sample_df):
    d = Data(sample_df)
    d.set_metadata({"version": 1})

    assert d.metadata == {"version": 1}


def test_read_csv_creates_dataframe(tmp_path):
    path = tmp_path / "sample.csv"
    expected = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    expected.to_csv(path, index=False)

    d = Data()
    d.read(str(path))

    pd.testing.assert_frame_equal(d.data, expected)


def test_preprocess_data_returns_array_and_records_original_columns(sample_df):
    d = Data()

    X = d._preprocess_data(sample_df)

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(sample_df)
    assert hasattr(d, "preprocessor")
    assert hasattr(d, "original_columns_")
    assert list(d.original_columns_) == list(sample_df.columns)


def test_inverse_transform_data_restores_feature_columns(sample_df):
    d = Data()

    X = d._preprocess_data(sample_df)

    # import ipdb; ipdb.set_trace()

    X_inv = d._inverse_transform_data(X)

    assert isinstance(X_inv, pd.DataFrame)
    assert list(X_inv.columns) == list(sample_df.columns)
    assert X_inv.shape[0] == len(sample_df)


def test_preprocess_data_handles_empty_category_or_numeric_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, None], "b": [3.0, 4.0, 5.0]})

    d = Data()
    X = d._preprocess_data(df)

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(df)


def run_example():
    example_df = pd.DataFrame(
            {
                "age": [25, None, 35, 45],
                "sex": ["M", "F", None, "M"],
                "income": [50000, 60000, 70000, None],
                "city": ["A", "B", "A", "C"],
            }
        )

    test_inverse_transform_data_restores_feature_columns(example_df)


run_example()