# tests/test_data_loader.py
"""
Unit tests for data loading infrastructure.
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_loader import (
    OptionChainDataset,
    OptionChainLoader,
    load_option_data,
)


# =============================================================================
# Test OptionChainDataset
# =============================================================================


class TestOptionChainDataset:
    """Tests for OptionChainDataset container."""

    @pytest.fixture
    def sample_data(self):
        """Create sample option chain data."""
        return pd.DataFrame({
            "strike": [90, 95, 100, 105, 110],
            "expiry": ["2024-03-15"] * 5,
            "type": ["call"] * 5,
            "implied_vol": [0.25, 0.22, 0.20, 0.22, 0.25],
            "volume": [500, 1000, 2000, 1000, 500],
        })

    def test_init(self, sample_data):
        dataset = OptionChainDataset(data=sample_data, underlying_price=100.0)
        assert dataset.n_options == 5
        assert dataset.underlying_price == 100.0

    def test_column_standardization(self):
        """Test that column names are standardized."""
        df = pd.DataFrame({
            "Strike": [100],
            "Expiry": ["2024-03-15"],
            "Type": ["C"],
            "IV": [0.2],
        })
        dataset = OptionChainDataset(data=df)
        
        assert "strike" in dataset.data.columns
        assert "expiry" in dataset.data.columns
        assert "type" in dataset.data.columns
        assert "implied_vol" in dataset.data.columns

    def test_type_standardization(self):
        """Test that option types are standardized."""
        df = pd.DataFrame({
            "strike": [100, 100],
            "type": ["C", "P"],
        })
        dataset = OptionChainDataset(data=df)
        
        assert dataset.data["type"].iloc[0] == "call"
        assert dataset.data["type"].iloc[1] == "put"

    def test_filter_liquid(self, sample_data):
        dataset = OptionChainDataset(data=sample_data)
        filtered = dataset.filter_liquid(min_volume=800)
        
        assert filtered.n_options == 3  # Only options with volume >= 800

    def test_filter_moneyness(self, sample_data):
        dataset = OptionChainDataset(data=sample_data, underlying_price=100.0)
        filtered = dataset.filter_moneyness(min_moneyness=0.95, max_moneyness=1.05)
        
        # Should keep strikes 95, 100, 105
        assert filtered.n_options == 3

    def test_compute_log_moneyness(self, sample_data):
        dataset = OptionChainDataset(data=sample_data, underlying_price=100.0)
        df = dataset.compute_log_moneyness()
        
        assert "log_moneyness" in df.columns
        assert "T" in df.columns
        
        # ATM strike should have log_moneyness ≈ 0
        atm_row = df[df["strike"] == 100]
        assert abs(atm_row["log_moneyness"].iloc[0]) < 0.01

    def test_to_model_input(self, sample_data):
        dataset = OptionChainDataset(data=sample_data, underlying_price=100.0)
        model_input = dataset.to_model_input()
        
        assert "log_moneyness" in model_input.columns
        assert "T" in model_input.columns
        assert "implied_volatility" in model_input.columns
        assert len(model_input) == 5


# =============================================================================
# Test OptionChainLoader
# =============================================================================


class TestOptionChainLoader:
    """Tests for OptionChainLoader factory methods."""

    def test_from_synthetic(self):
        dataset = OptionChainLoader.from_synthetic(
            n_strikes=30,
            maturities=[0.25, 0.5],
            spot=100.0,
            seed=42,
        )
        
        assert dataset.n_options == 60  # 30 strikes × 2 maturities
        assert dataset.underlying_price == 100.0
        assert dataset.source == "synthetic"

    def test_from_synthetic_reproducibility(self):
        ds1 = OptionChainLoader.from_synthetic(seed=123)
        ds2 = OptionChainLoader.from_synthetic(seed=123)
        
        pd.testing.assert_frame_equal(ds1.data, ds2.data)

    def test_from_csv(self):
        """Test loading from CSV file."""
        # Create temp CSV
        df = pd.DataFrame({
            "strike": [100, 105, 110],
            "T": [0.5, 0.5, 0.5],
            "type": ["call", "call", "call"],
            "implied_vol": [0.2, 0.21, 0.22],
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            dataset = OptionChainLoader.from_csv(
                temp_path,
                underlying_price=100.0,
            )
            
            assert dataset.n_options == 3
            assert "csv:" in dataset.source
        finally:
            os.unlink(temp_path)

    def test_from_parquet(self):
        """Test loading from Parquet file."""
        df = pd.DataFrame({
            "strike": [100, 105, 110],
            "T": [0.5, 0.5, 0.5],
            "implied_vol": [0.2, 0.21, 0.22],
        })
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name
        
        df.to_parquet(temp_path)
        
        try:
            dataset = OptionChainLoader.from_parquet(
                temp_path,
                underlying_price=100.0,
            )
            
            assert dataset.n_options == 3
            assert "parquet:" in dataset.source
        finally:
            os.unlink(temp_path)

    def test_synthetic_smile_shape(self):
        """Verify synthetic data has realistic smile shape."""
        dataset = OptionChainLoader.from_synthetic(
            n_strikes=50,
            maturities=[1.0],
            atm_vol=0.2,
            skew=-0.3,
            smile=0.05,
            seed=42,
        )
        
        df = dataset.compute_log_moneyness()
        
        # OTM puts (negative log-moneyness) should have higher vol
        otm_puts = df[df["log_moneyness"] < -0.1]["implied_vol"].mean()
        atm = df[abs(df["log_moneyness"]) < 0.05]["implied_vol"].mean()
        
        assert otm_puts > atm


# =============================================================================
# Test load_option_data Convenience Function
# =============================================================================


class TestLoadOptionData:
    """Tests for load_option_data convenience function."""

    def test_load_synthetic(self):
        dataset = load_option_data("synthetic", n_strikes=20, seed=42)
        
        assert dataset.source == "synthetic"
        assert dataset.n_options > 0

    def test_load_csv(self):
        df = pd.DataFrame({
            "strike": [100],
            "implied_vol": [0.2],
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            dataset = load_option_data(temp_path)
            assert dataset.n_options == 1
        finally:
            os.unlink(temp_path)

    def test_unsupported_format(self):
        with pytest.raises(ValueError):
            load_option_data("file.xyz")


# =============================================================================
# Test Data Preprocessing
# =============================================================================


class TestDataPreprocessing:
    """Tests for data preprocessing pipeline."""

    def test_full_pipeline(self):
        """Test complete data loading and preprocessing pipeline."""
        # Load synthetic data
        dataset = OptionChainLoader.from_synthetic(
            n_strikes=40,
            maturities=[0.25, 0.5, 1.0],
            seed=42,
        )
        
        # Apply filters
        filtered = (
            dataset
            .filter_liquid(min_volume=200)
            .filter_moneyness(min_moneyness=0.85, max_moneyness=1.15)
        )
        
        # Convert to model input
        model_input = filtered.to_model_input()
        
        # Verify output
        assert len(model_input) > 0
        assert "log_moneyness" in model_input.columns
        assert "implied_volatility" in model_input.columns
        assert model_input["implied_volatility"].min() > 0

    def test_model_input_no_nans(self):
        """Verify model input has no NaN values."""
        dataset = OptionChainLoader.from_synthetic(seed=42)
        model_input = dataset.to_model_input()
        
        assert not model_input.isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
