"""ML models for detecting scam tokens"""
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class RugPullDetector:
    """Binary classifier to detect scam tokens (rug pulls, pump&dumps, wash trading)"""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize rug pull detector

        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of individual trees
            learning_rate: Learning rate shrinks contribution of each tree
            random_state: Random seed for reproducibility
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=0
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Train the model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=scam, 1=legitimate)
            feature_names: Names of features for interpretability

        Returns:
            Training metrics dict
        """
        print("Training RugPullDetector...")
        print(f"  Training samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Scam tokens: {np.sum(y == 0)}")
        print(f"  Legitimate tokens: {np.sum(y == 1)}")

        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate training accuracy
        train_score = self.model.score(X_scaled, y)

        print(f"✓ Training complete! Accuracy: {train_score:.3f}")

        return {
            'train_accuracy': train_score,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

    def predict_risk(self, features: Dict[str, float]) -> float:
        """
        Predict scam probability for a token

        Args:
            features: Dict of feature name -> value

        Returns:
            Risk score 0-1 (higher = more likely to be scam)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert dict to array in correct order
        X = self._dict_to_array(features)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict probability of class 0 (scam)
        risk_score = self.model.predict_proba(X_scaled)[0, 0]

        return float(risk_score)

    def predict_batch(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Predict risk scores for multiple tokens

        Args:
            features_list: List of feature dicts

        Returns:
            Array of risk scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to array
        X = np.array([[f[name] for name in self.feature_names] for f in features_list])

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        risk_scores = self.model.predict_proba(X_scaled)[:, 0]

        return risk_scores

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores

        Returns:
            Dict of feature name -> importance score
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        importances = self.model.feature_importances_

        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }

    def explain_prediction(self, features: Dict[str, float], top_n: int = 10) -> Dict:
        """
        Explain a prediction using feature importance

        Args:
            features: Feature dict for a token
            top_n: Number of top features to include

        Returns:
            Dict with risk score and top contributing features
        """
        risk_score = self.predict_risk(features)

        # Get feature importances
        importance_dict = self.get_feature_importance()

        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Build explanation
        explanation = {
            'risk_score': risk_score,
            'recommendation': self._score_to_recommendation(risk_score),
            'top_features': [
                {
                    'name': name,
                    'importance': importance,
                    'value': features.get(name, None)
                }
                for name, importance in sorted_features
            ]
        }

        return explanation

    def _score_to_recommendation(self, risk_score: float) -> str:
        """Convert risk score to recommendation"""
        if risk_score >= 0.70:
            return 'REJECT'
        elif risk_score >= 0.40:
            return 'CAUTION'
        elif risk_score >= 0.20:
            return 'MONITOR'
        else:
            return 'PASS'

    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in correct order"""
        return np.array([[features.get(name, 0.0) for name in self.feature_names]])

    def save(self, filepath: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Model not trained. Train before saving.")

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']

        print(f"✓ Model loaded from {filepath}")


class MultiModelDetector:
    """Ensemble of specialized detectors for different scam types"""

    def __init__(self):
        """Initialize multi-model detector"""
        self.rug_pull_detector = None
        self.pump_dump_detector = None
        self.wash_trading_detector = None

        self.weights = {
            'rug_pull': 0.50,
            'pump_dump': 0.30,
            'wash_trading': 0.20
        }

    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Train all specialized detectors

        Args:
            X: Feature matrix
            y: Multi-class labels (0=rug_pull, 1=pump_dump, 2=wash_trading, 3=legitimate)
            feature_names: Feature names
        """
        print("\n" + "=" * 60)
        print("Training Multi-Model Detector")
        print("=" * 60)

        # Train rug pull detector (0 vs rest)
        print("\n1. Rug Pull Detector")
        y_rug = (y == 0).astype(int)  # 1 if rug pull, 0 otherwise
        self.rug_pull_detector = RugPullDetector()
        self.rug_pull_detector.train(X, y_rug, feature_names)

        # Train pump & dump detector (1 vs rest)
        print("\n2. Pump & Dump Detector")
        y_pump = (y == 1).astype(int)
        self.pump_dump_detector = RugPullDetector()
        self.pump_dump_detector.train(X, y_pump, feature_names)

        # Train wash trading detector (2 vs rest)
        print("\n3. Wash Trading Detector")
        y_wash = (y == 2).astype(int)
        self.wash_trading_detector = RugPullDetector()
        self.wash_trading_detector.train(X, y_wash, feature_names)

        print("\n✓ All models trained!")

    def comprehensive_risk_assessment(self, features: Dict[str, float]) -> Dict:
        """
        Comprehensive risk assessment using all detectors

        Args:
            features: Token features

        Returns:
            Dict with individual and combined risk scores
        """
        rug_risk = self.rug_pull_detector.predict_risk(features)
        pump_risk = self.pump_dump_detector.predict_risk(features)
        wash_risk = self.wash_trading_detector.predict_risk(features)

        # Weighted combination
        combined_risk = (
            rug_risk * self.weights['rug_pull'] +
            pump_risk * self.weights['pump_dump'] +
            wash_risk * self.weights['wash_trading']
        )

        return {
            'rug_pull_risk': rug_risk,
            'pump_dump_risk': pump_risk,
            'wash_trading_risk': wash_risk,
            'combined_risk': combined_risk,
            'recommendation': self._categorize_risk(combined_risk),
            'primary_threat': self._identify_primary_threat(rug_risk, pump_risk, wash_risk)
        }

    def _categorize_risk(self, score: float) -> str:
        """Categorize combined risk score"""
        if score >= 0.70:
            return 'REJECT'
        elif score >= 0.40:
            return 'CAUTION'
        elif score >= 0.20:
            return 'MONITOR'
        else:
            return 'PASS'

    def _identify_primary_threat(
        self,
        rug_risk: float,
        pump_risk: float,
        wash_risk: float
    ) -> str:
        """Identify the highest risk category"""
        risks = {
            'rug_pull': rug_risk,
            'pump_dump': pump_risk,
            'wash_trading': wash_risk
        }

        primary = max(risks.items(), key=lambda x: x[1])

        if primary[1] < 0.3:
            return 'none'

        return primary[0]

    def save(self, directory: str):
        """Save all models"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.rug_pull_detector.save(str(dir_path / 'rug_pull_detector.pkl'))
        self.pump_dump_detector.save(str(dir_path / 'pump_dump_detector.pkl'))
        self.wash_trading_detector.save(str(dir_path / 'wash_trading_detector.pkl'))

        print(f"✓ All models saved to {directory}")

    def load(self, directory: str):
        """Load all models"""
        dir_path = Path(directory)

        self.rug_pull_detector = RugPullDetector()
        self.rug_pull_detector.load(str(dir_path / 'rug_pull_detector.pkl'))

        self.pump_dump_detector = RugPullDetector()
        self.pump_dump_detector.load(str(dir_path / 'pump_dump_detector.pkl'))

        self.wash_trading_detector = RugPullDetector()
        self.wash_trading_detector.load(str(dir_path / 'wash_trading_detector.pkl'))

        print(f"✓ All models loaded from {directory}")
