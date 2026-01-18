"""Tests for ML model."""
import numpy as np
import pytest
import tempfile
import os

from src.ml.models import RugPullDetector


class TestRugPullDetector:
    """Tests for the RugPullDetector model."""

    @pytest.fixture
    def feature_names(self):
        """Feature names for the model."""
        return [
            'token_age_hours',
            'liquidity_usd',
            'market_cap',
            'price_usd',
            'volume_5min',
            'volume_1hour',
            'volume_24hour',
            'buy_count_5min',
            'sell_count_5min',
            'buy_count_1hour',
            'sell_count_1hour',
            'price_change_5min',
            'price_change_1hour',
            'price_change_24hour',
            'buy_sell_ratio_5min',
            'buy_sell_ratio_1hour',
            'liquidity_to_mcap_ratio',
            'volume_acceleration',
            'launch_phase_risk',
            'safety_score',
            'volume_to_liquidity_ratio'
        ]

    @pytest.fixture
    def synthetic_data(self, feature_names):
        """Generate synthetic training data."""
        np.random.seed(42)

        # Generate scam tokens (50 samples)
        scam_samples = []
        for _ in range(50):
            scam_samples.append([
                np.random.uniform(1, 48),  # Very young
                np.random.uniform(100, 5000),  # Low liquidity
                np.random.uniform(1000, 50000),  # Low market cap
                np.random.uniform(0.0001, 0.01),  # Low price
                np.random.uniform(10, 500),  # Low volume
                np.random.uniform(100, 2000),
                np.random.uniform(1000, 20000),
                np.random.randint(1, 20),  # Few buys
                np.random.randint(5, 50),  # Many sells
                np.random.randint(10, 100),
                np.random.randint(50, 200),
                np.random.uniform(-50, 20),  # Negative price change
                np.random.uniform(-70, 10),
                np.random.uniform(-95, -20),
                np.random.uniform(0.1, 0.8),  # Low buy/sell ratio
                np.random.uniform(0.2, 0.9),
                np.random.uniform(0.01, 0.05),  # Low liquidity ratio
                np.random.uniform(0.5, 3),  # Normal acceleration
                np.random.uniform(0.5, 1.0),  # High launch risk
                np.random.randint(0, 2),  # Low safety
                np.random.uniform(5, 50),  # High vol/liq
            ])

        # Generate legitimate tokens (50 samples)
        legit_samples = []
        for _ in range(50):
            legit_samples.append([
                np.random.uniform(168, 720),  # Older (1+ weeks)
                np.random.uniform(50000, 500000),  # Good liquidity
                np.random.uniform(500000, 5000000),  # Good market cap
                np.random.uniform(0.01, 10),  # Reasonable price
                np.random.uniform(1000, 10000),  # Good volume
                np.random.uniform(10000, 100000),
                np.random.uniform(100000, 1000000),
                np.random.randint(20, 100),  # Many buys
                np.random.randint(10, 60),  # Balanced sells
                np.random.randint(100, 500),
                np.random.randint(80, 400),
                np.random.uniform(-10, 30),  # Mixed price change
                np.random.uniform(-20, 50),
                np.random.uniform(-30, 100),
                np.random.uniform(1.0, 3.0),  # Good buy/sell ratio
                np.random.uniform(1.0, 2.5),
                np.random.uniform(0.1, 0.3),  # Good liquidity ratio
                np.random.uniform(0.8, 2),  # Normal acceleration
                np.random.uniform(0, 0.2),  # Low launch risk
                np.random.randint(3, 5),  # High safety
                np.random.uniform(1, 10),  # Normal vol/liq
            ])

        X = np.array(scam_samples + legit_samples)
        y = np.array([0] * 50 + [1] * 50)  # 0=scam, 1=legitimate

        return X, y, feature_names

    @pytest.fixture
    def trained_model(self, synthetic_data):
        """Create a trained model."""
        X, y, feature_names = synthetic_data
        model = RugPullDetector(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.train(X, y, feature_names)
        return model

    def test_model_training(self, synthetic_data):
        """Test that model trains successfully."""
        X, y, feature_names = synthetic_data

        model = RugPullDetector(n_estimators=100, max_depth=5, learning_rate=0.1)
        metrics = model.train(X, y, feature_names)

        assert model.is_trained
        assert 'train_accuracy' in metrics
        assert metrics['train_accuracy'] > 0.5  # Should be better than random

    def test_predict_scam_token(self, trained_model, scam_token_features):
        """Test predicting a scam-like token."""
        risk_score = trained_model.predict_risk(scam_token_features)

        assert 0 <= risk_score <= 1
        # Scam tokens should have higher risk scores
        assert risk_score > 0.5

    def test_predict_legit_token(self, trained_model, legit_token_features):
        """Test predicting a legitimate-like token."""
        risk_score = trained_model.predict_risk(legit_token_features)

        assert 0 <= risk_score <= 1
        # Legitimate tokens should have lower risk scores
        assert risk_score < 0.5

    def test_feature_importance(self, trained_model):
        """Test that feature importance is available."""
        importance = trained_model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Values should be numeric
        assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_save_and_load(self, trained_model, scam_token_features):
        """Test saving and loading the model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")

            # Save
            trained_model.save(model_path)
            assert os.path.exists(model_path)

            # Get prediction before loading
            original_risk = trained_model.predict_risk(scam_token_features)

            # Load into new model
            loaded_model = RugPullDetector()
            loaded_model.load(model_path)

            # Verify loaded model works
            loaded_risk = loaded_model.predict_risk(scam_token_features)

            assert abs(loaded_risk - original_risk) < 0.001

    def test_score_to_recommendation(self, trained_model):
        """Test score to recommendation conversion."""
        # Low risk
        rec = trained_model._score_to_recommendation(0.1)
        assert rec in ['PASS', 'LOW_RISK', 'SAFE']

        # High risk
        rec = trained_model._score_to_recommendation(0.9)
        assert rec in ['REJECT', 'HIGH_RISK', 'AVOID', 'SCAM']
