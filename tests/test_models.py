# tests / test_models.py

import threading
import tempfile
import os
import numpy as np
import pandas as pd
import pytest

from src.volatility_surface.models.svr_model import SVRModel
from src.volatility_surface.models.mlp_model import MLPModel

# Minimal synthetic dataset for testing
def make_sample_df(n=100):
    np.random.seed(42)
    df = pd.DataFrame({
        'moneyness': np.random.uniform(0.8, 1.2, n),
        'log_moneyness': np.random.uniform(-0.2, 0.2, n),
        'time_to_maturity': np.random.uniform(0.01, 1.0, n),
        'ttm_squared': np.random.uniform(0.0001, 1.0, n),
        'risk_free_rate': np.full(n, 0.01),
        'historical_volatility': np.random.uniform(0.1, 0.4, n),
        'volatility_skew': np.random.uniform(-0.05, 0.05, n),
        'implied_volatility': np.random.uniform(0.1, 0.5, n)
    })
    return df

@pytest.fixture
def sample_data():
    return make_sample_df()

def test_svrmodel_train_predict_save_load(sample_data):
    model = SVRModel()
    metrics = model.train(sample_data)
    assert metrics['train_rmse'] > 0
    preds = model.predict_volatility(sample_data)
    assert preds.shape[0] == sample_data.shape[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = model.save_model(tmpdir)
        assert os.path.exists(paths['model'])
        assert os.path.exists(paths['scaler'])

        model2 = SVRModel()
        model2.load_model(paths['model'], paths['scaler'])
        preds2 = model2.predict_volatility(sample_data)
        np.testing.assert_allclose(preds, preds2, rtol=1e-5)

def test_mlpmodel_train_predict_save_load(sample_data):
    model = MLPModel(epochs=5)  # keep epochs low for test speed
    metrics = model.train(sample_data)
    assert metrics['train_loss'] > 0
    preds = model.predict_volatility(sample_data, mc_samples=3)
    assert preds.shape[0] == sample_data.shape[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = model.save_model(tmpdir)
        assert os.path.exists(paths['model'])
        assert os.path.exists(paths['scaler'])

        model2 = MLPModel()
        model2.load_model(paths['model'], paths['scaler'])
        preds2 = model2.predict_volatility(sample_data)
        np.testing.assert_allclose(preds, preds2, rtol=1e-4)

def test_thread_safety(sample_data):
    model = SVRModel()
    model.train(sample_data)

    preds_results = []
    def worker():
        preds_results.append(model.predict_volatility(sample_data))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(preds_results) == 5
    for preds in preds_results:
        assert preds.shape[0] == sample_data.shape[0]
