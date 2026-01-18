"""ML model training pipeline"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import json

from .models import RugPullDetector, MultiModelDetector
from ..utils.database import Database


class ModelTrainer:
    """Pipeline for training ML models"""

    def __init__(self, db: Database):
        """
        Initialize trainer

        Args:
            db: Database instance with training data
        """
        self.db = db
        self.feature_names = None

    def load_training_data(self, min_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load training data from database

        Args:
            min_samples: Minimum number of samples required

        Returns:
            Tuple of (X, y, feature_names)
        """
        print("\n" + "=" * 60)
        print("Loading Training Data")
        print("=" * 60)

        # Get labeled dataset
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, l.label
                FROM token_snapshots s
                JOIN token_labels l ON s.token_address = l.token_address
                WHERE s.snapshot_type = 'initial'
                AND l.label IN (0, 1, 2, 3)
                ORDER BY s.snapshot_time
            """)

            rows = cursor.fetchall()

        if len(rows) < min_samples:
            raise ValueError(
                f"Insufficient training data. Need {min_samples}, have {len(rows)}. "
                f"Collect more data using collect_training_data.py"
            )

        print(f"\nTotal samples: {len(rows)}")

        # Define feature columns (in specific order)
        self.feature_names = [
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
            'price_change_24hour'
        ]

        # Extract features and labels
        X_list = []
        y_list = []

        for row in rows:
            row_dict = dict(row)

            # Extract label
            label = row_dict['label']
            y_list.append(label)

            # Extract features
            features = [row_dict.get(name, 0.0) for name in self.feature_names]
            X_list.append(features)

        X = np.array(X_list)
        y = np.array(y_list)

        # Add derived features
        X, derived_names = self._add_derived_features(X)
        self.feature_names.extend(derived_names)

        print(f"Features: {len(self.feature_names)}")
        print(f"\nClass distribution:")
        label_names = {0: 'Rug Pull', 1: 'Pump & Dump', 2: 'Wash Trading', 3: 'Legitimate'}
        for label_id in range(4):
            count = np.sum(y == label_id)
            pct = count / len(y) * 100
            print(f"  {label_names[label_id]}: {count} ({pct:.1f}%)")

        return X, y, self.feature_names

    def _add_derived_features(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Add derived features to feature matrix

        Args:
            X: Original feature matrix

        Returns:
            Tuple of (augmented X, new feature names)
        """
        # Extract columns by index
        token_age = X[:, 0]
        liquidity = X[:, 1]
        market_cap = X[:, 2]
        price = X[:, 3]
        volume_5m = X[:, 4]
        volume_1h = X[:, 5]
        volume_24h = X[:, 6]
        buy_5m = X[:, 7]
        sell_5m = X[:, 8]
        buy_1h = X[:, 9]
        sell_1h = X[:, 10]

        # Derived features
        derived = []
        derived_names = []

        # Buy/sell ratios
        buy_sell_5m = np.where(sell_5m > 0, buy_5m / sell_5m, buy_5m)
        derived.append(buy_sell_5m)
        derived_names.append('buy_sell_ratio_5min')

        buy_sell_1h = np.where(sell_1h > 0, buy_1h / sell_1h, buy_1h)
        derived.append(buy_sell_1h)
        derived_names.append('buy_sell_ratio_1hour')

        # Liquidity to market cap ratio
        liq_mcap = np.where(market_cap > 0, liquidity / market_cap, 0)
        derived.append(liq_mcap)
        derived_names.append('liquidity_to_mcap_ratio')

        # Volume acceleration
        rate_5m = volume_5m / 5
        rate_1h = volume_1h / 60
        vol_accel = np.where(rate_1h > 0, rate_5m / rate_1h, 0)
        derived.append(vol_accel)
        derived_names.append('volume_acceleration')

        # Launch phase risk (higher for very new tokens)
        launch_risk = np.where(token_age < 24, 1 - (token_age / 24), 0)
        derived.append(launch_risk)
        derived_names.append('launch_phase_risk')

        # Safety score (0-4 based on multiple factors)
        safety = np.zeros(len(X))
        safety += (token_age > 168).astype(int)  # >1 week old
        safety += (liquidity > 10000).astype(int)  # >$10k liquidity
        safety += (market_cap > 100000).astype(int)  # >$100k mcap
        safety += (liq_mcap > 0.1).astype(int)  # Good liquidity ratio
        derived.append(safety)
        derived_names.append('safety_score')

        # Volume to liquidity ratio
        vol_liq = np.where(liquidity > 0, volume_24h / liquidity, 0)
        derived.append(vol_liq)
        derived_names.append('volume_to_liquidity_ratio')

        # Stack derived features
        derived_matrix = np.column_stack(derived)
        X_augmented = np.hstack([X, derived_matrix])

        return X_augmented, derived_names

    def train_binary_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Tuple[RugPullDetector, Dict]:
        """
        Train binary classifier (scam vs legitimate)

        Args:
            X: Feature matrix
            y: Multi-class labels (0-3)
            test_size: Test set proportion
            cv_folds: Cross-validation folds

        Returns:
            Tuple of (trained_model, results_dict)
        """
        print("\n" + "=" * 60)
        print("Training Binary Classifier (Scam vs Legitimate)")
        print("=" * 60)

        # Convert to binary: 0-2 = scam (0), 3 = legitimate (1)
        y_binary = (y == 3).astype(int)

        print(f"\nBinary distribution:")
        print(f"  Scam (0-2): {np.sum(y_binary == 0)}")
        print(f"  Legitimate (3): {np.sum(y_binary == 1)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary,
            test_size=test_size,
            random_state=42,
            stratify=y_binary
        )

        print(f"\nTrain/Test Split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")

        # Train model
        model = RugPullDetector(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )

        model.train(X_train, y_train, self.feature_names)

        # Evaluate on test set
        X_test_scaled = model.scaler.transform(X_test)
        test_score = model.model.score(X_test_scaled, y_test)

        print(f"\n✓ Test Accuracy: {test_score:.3f}")

        # Cross-validation
        print(f"\nRunning {cv_folds}-fold cross-validation...")
        X_scaled_full = model.scaler.transform(X)
        cv_scores = cross_val_score(model.model, X_scaled_full, y_binary, cv=cv_folds)

        print(f"CV Scores: {cv_scores}")
        print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Get predictions for test set
        y_pred = model.model.predict(X_test_scaled)

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Scam', 'Legitimate']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"               Predicted")
        print(f"             Scam  Legit")
        print(f"Actual Scam   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"      Legit   {cm[1,0]:4d}  {cm[1,1]:4d}")

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'train_accuracy': model.model.score(model.scaler.transform(X_train), y_train),
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1]
        }

        return model, results

    def train_multiclass_detector(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[MultiModelDetector, Dict]:
        """
        Train multi-model detector for different scam types

        Args:
            X: Feature matrix
            y: Multi-class labels (0-3)
            test_size: Test set proportion

        Returns:
            Tuple of (trained_detector, results_dict)
        """
        print("\n" + "=" * 60)
        print("Training Multi-Model Detector")
        print("=" * 60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # Train detector
        detector = MultiModelDetector()
        detector.train_all(X_train, y_train, self.feature_names)

        # Evaluate each detector
        results = {}

        for name, model in [
            ('rug_pull', detector.rug_pull_detector),
            ('pump_dump', detector.pump_dump_detector),
            ('wash_trading', detector.wash_trading_detector)
        ]:
            X_test_scaled = model.scaler.transform(X_test)
            y_binary = (y_test == {'rug_pull': 0, 'pump_dump': 1, 'wash_trading': 2}[name]).astype(int)
            score = model.model.score(X_test_scaled, y_binary)

            results[f'{name}_test_accuracy'] = score
            print(f"{name} test accuracy: {score:.3f}")

        return detector, results

    def get_feature_importance_report(self, model: RugPullDetector, top_n: int = 15) -> pd.DataFrame:
        """
        Generate feature importance report

        Args:
            model: Trained model
            top_n: Number of top features to show

        Returns:
            DataFrame with feature importance
        """
        importance = model.get_feature_importance()

        df = pd.DataFrame([
            {'feature': name, 'importance': score}
            for name, score in importance.items()
        ])

        df = df.sort_values('importance', ascending=False).head(top_n)

        return df
