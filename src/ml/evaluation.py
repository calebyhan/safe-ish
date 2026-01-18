"""ML model evaluation and metrics"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from pathlib import Path

from .models import RugPullDetector


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, model: RugPullDetector):
        """
        Initialize evaluator

        Args:
            model: Trained model to evaluate
        """
        self.model = model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation

        Args:
            X_test: Test features
            y_test: Test labels (0=scam, 1=legitimate)

        Returns:
            Dict of evaluation metrics
        """
        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)

        # Scale features
        X_test_scaled = self.model.scaler.transform(X_test)

        # Get predictions
        y_pred = self.model.model.predict(X_test_scaled)
        y_pred_proba = self.model.model.predict_proba(X_test_scaled)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n📊 Performance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1_score']:.3f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")

        print(f"\n📋 Confusion Matrix:")
        print(f"               Predicted")
        print(f"             Scam  Legit")
        print(f"Actual Scam   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"      Legit   {cm[1,0]:4d}  {cm[1,1]:4d}")

        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()

        # False positive rate (legitimate tokens marked as scam)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # False negative rate (scams marked as legitimate)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\n⚠️  Error Analysis:")
        print(f"  False Positive Rate: {fpr:.3f} (legit marked as scam)")
        print(f"  False Negative Rate: {fnr:.3f} (scam marked as legit)")

        metrics['false_positive_rate'] = fpr
        metrics['false_negative_rate'] = fnr
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    def cost_benefit_analysis(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        false_negative_cost: float = 1000,
        false_positive_cost: float = 50
    ) -> Dict[str, float]:
        """
        Calculate expected value of the ML filter

        Args:
            X_test: Test features
            y_test: Test labels
            false_negative_cost: Cost of missing a scam ($)
            false_positive_cost: Cost of rejecting a legitimate token ($)

        Returns:
            Dict with cost-benefit analysis
        """
        print("\n" + "=" * 60)
        print("Cost-Benefit Analysis")
        print("=" * 60)

        # Get predictions
        X_test_scaled = self.model.scaler.transform(X_test)
        y_pred = self.model.model.predict(X_test_scaled)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate costs
        fn_total_cost = fn * false_negative_cost  # Missed scams
        fp_total_cost = fp * false_positive_cost  # Rejected good tokens

        total_cost = fn_total_cost + fp_total_cost

        # Calculate benefit (avoided scam costs)
        benefit = tp * false_negative_cost

        # Net value
        net_value = benefit - total_cost

        # Expected value per prediction
        ev_per_prediction = net_value / len(y_test)

        print(f"\n💰 Cost Parameters:")
        print(f"  False Negative Cost: ${false_negative_cost:,.0f} (missing a scam)")
        print(f"  False Positive Cost: ${false_positive_cost:,.0f} (rejecting good token)")

        print(f"\n📊 Results:")
        print(f"  True Positives:  {tp} scams detected")
        print(f"  False Negatives: {fn} scams missed")
        print(f"  False Positives: {fp} good tokens rejected")
        print(f"  True Negatives:  {tn} good tokens passed")

        print(f"\n💵 Financial Impact:")
        print(f"  Benefit (scams caught): ${benefit:,.0f}")
        print(f"  Cost (missed scams):    ${fn_total_cost:,.0f}")
        print(f"  Cost (rejected good):   ${fp_total_cost:,.0f}")
        print(f"  Total Cost:             ${total_cost:,.0f}")
        print(f"  Net Value:              ${net_value:,.0f}")
        print(f"  EV per Prediction:      ${ev_per_prediction:,.2f}")

        return {
            'true_positives': int(tp),
            'false_negatives': int(fn),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'benefit': float(benefit),
            'fn_cost': float(fn_total_cost),
            'fp_cost': float(fp_total_cost),
            'total_cost': float(total_cost),
            'net_value': float(net_value),
            'ev_per_prediction': float(ev_per_prediction)
        }

    def feature_importance_analysis(self, top_n: int = 15) -> pd.DataFrame:
        """
        Analyze feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        print("\n" + "=" * 60)
        print("Feature Importance Analysis")
        print("=" * 60)

        importance = self.model.get_feature_importance()

        df = pd.DataFrame([
            {'feature': name, 'importance': score}
            for name, score in importance.items()
        ])

        df = df.sort_values('importance', ascending=False)

        print(f"\n🔝 Top {top_n} Features:")
        for i, row in df.head(top_n).iterrows():
            print(f"  {i+1}. {row['feature']:<30} {row['importance']:.4f}")

        return df

    def threshold_analysis(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze model performance at different decision thresholds

        Args:
            X_test: Test features
            y_test: Test labels
            thresholds: List of thresholds to test

        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        print("\n" + "=" * 60)
        print("Threshold Analysis")
        print("=" * 60)

        X_test_scaled = self.model.scaler.transform(X_test)
        y_pred_proba = self.model.model.predict_proba(X_test_scaled)[:, 0]  # Probability of scam

        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Note: y_pred is now 1=scam, 0=legitimate
            # We need to flip for metrics calculation
            y_pred_flipped = 1 - y_pred

            accuracy = accuracy_score(y_test, y_pred_flipped)
            precision = precision_score(y_test, y_pred_flipped, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred_flipped, pos_label=1, zero_division=0)

            # Calculate false negative rate (scams passed through)
            cm = confusion_matrix(y_test, y_pred_flipped)
            tn, fp, fn, tp = cm.ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'false_negative_rate': fnr,
                'scams_rejected': int(np.sum(y_pred[y_test == 0] == 1)),
                'legit_rejected': int(np.sum(y_pred[y_test == 1] == 1))
            })

        df = pd.DataFrame(results)

        print("\n📊 Threshold Performance:")
        print(df.to_string(index=False))

        print("\n💡 Recommendation:")
        print("  Use threshold 0.40 as configured (balanced)")
        print("  Higher threshold (0.6+) = more strict, fewer false positives")
        print("  Lower threshold (0.3-) = more lenient, fewer false negatives")

        return df

    def generate_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_path: str = "data/models/evaluation_report.txt"
    ):
        """
        Generate comprehensive evaluation report

        Args:
            X_test: Test features
            y_test: Test labels
            output_path: Path to save report
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ML MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Basic metrics
            metrics = self.evaluate(X_test, y_test)
            f.write("\nPerformance Metrics:\n")
            for key, value in metrics.items():
                if key != 'confusion_matrix':
                    f.write(f"  {key}: {value:.4f}\n")

            # Cost-benefit analysis
            cb = self.cost_benefit_analysis(X_test, y_test)
            f.write("\nCost-Benefit Analysis:\n")
            for key, value in cb.items():
                if isinstance(value, float):
                    f.write(f"  {key}: ${value:,.2f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

            # Feature importance
            importance_df = self.feature_importance_analysis()
            f.write("\nTop 15 Features:\n")
            f.write(importance_df.head(15).to_string(index=False))

            # Threshold analysis
            threshold_df = self.threshold_analysis(X_test, y_test)
            f.write("\n\nThreshold Analysis:\n")
            f.write(threshold_df.to_string(index=False))

        print(f"\n✓ Evaluation report saved to {output_path}")
