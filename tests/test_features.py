"""Tests du module de feature engineering."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.data.features import (
    add_lag_features,
    add_rolling_features,
    build_sequences, 
    normalize,
)

@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((100, 5)), columns=[f"f{i}" for i in range(5)])


def test_rolling_features_shape(sample_df: pd.DataFrame) -> None:
    result = add_rolling_features(sample_df)
    # 5 cols originales + 5 cols × 3 fenêtres × 4 stats = 65
    assert result.shape[1] > sample_df.shape[1]
    assert result.shape[0] == sample_df.shape[0]


def test_rolling_features_no_nan(sample_df: pd.DataFrame) -> None:
    result = add_rolling_features(sample_df)
    assert not result.isnull().any().any()


def test_lag_features_shape(sample_df: pd.DataFrame) -> None:
    result = add_lag_features(sample_df)
    assert result.shape[1] > sample_df.shape[1]
    assert result.shape[0] == sample_df.shape[0]


def test_normalize_range() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, 10)).astype(np.float32)
    X_test = rng.standard_normal((50, 10)).astype(np.float32)

    X_train_s, X_test_s, scaler = normalize(X_train, X_test)

    assert X_train_s.shape == X_train.shape
    assert X_test_s is not None
    assert X_test_s.shape == X_test.shape


def test_build_sequences() -> None:
    X = np.arange(50 * 4).reshape(50, 4).astype(np.float32)
    seqs = build_sequences(X, seq_len=10)
    assert seqs.shape == (41, 10, 4)
    np.testing.assert_array_equal(seqs[0], X[:10])
    np.testing.assert_array_equal(seqs[1], X[1:11])
