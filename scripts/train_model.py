#!/usr/bin/env python3
"""CLI script for training ML models"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.training import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from src.utils.database import Database


def train_binary_model(args):
    """Train binary classifier (scam vs legitimate)"""
    print("\n" + "=" * 60)
    print("TRAINING BINARY CLASSIFIER")
    print("=" * 60)

    # Initialize
    db = Database(args.db_path)
    trainer = ModelTrainer(db)

    # Load data
    try:
        X, y, feature_names = trainer.load_training_data(min_samples=args.min_samples)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return

    # Train model
    model, results = trainer.train_binary_classifier(
        X, y,
        test_size=args.test_size,
        cv_folds=args.cv_folds
    )

    # Show feature importance
    print("\n" + "=" * 60)
    print("Feature Importance")
    print("=" * 60)
    importance_df = trainer.get_feature_importance_report(model, top_n=15)
    print(importance_df.to_string(index=False))

    # Save model
    model.save(args.output_path)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"✓ Model trained successfully!")
    print(f"✓ Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"✓ Precision: {results['precision']:.3f}")
    print(f"✓ Recall: {results['recall']:.3f}")
    print(f"✓ F1 Score: {results['f1_score']:.3f}")
    print(f"✓ Model saved to: {args.output_path}")

    # Optionally evaluate
    if args.evaluate:
        print("\n" + "=" * 60)
        print("Running Full Evaluation")
        print("=" * 60)

        # Split data again for evaluation
        from sklearn.model_selection import train_test_split
        y_binary = (y == 3).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary,
            test_size=args.test_size,
            random_state=42,
            stratify=y_binary
        )

        evaluator = ModelEvaluator(model)

        # Metrics
        metrics = evaluator.evaluate(X_test, y_test)

        # Cost-benefit
        cost_benefit = evaluator.cost_benefit_analysis(
            X_test, y_test,
            false_negative_cost=args.fn_cost,
            false_positive_cost=args.fp_cost
        )

        # Threshold analysis
        threshold_df = evaluator.threshold_analysis(X_test, y_test)

        # Generate report
        if args.report:
            evaluator.generate_report(X_test, y_test, args.report_path)

    print("\n✅ Training complete!")
    print("\nNext steps:")
    print("  1. Review model performance metrics")
    print("  2. Test model with: python test_ml_model.py")
    print("  3. If satisfied, proceed to Week 3: Trading Strategies")


def train_multimodel(args):
    """Train multi-model detector for specific scam types"""
    print("\n" + "=" * 60)
    print("TRAINING MULTI-MODEL DETECTOR")
    print("=" * 60)

    # Initialize
    db = Database(args.db_path)
    trainer = ModelTrainer(db)

    # Load data
    try:
        X, y, feature_names = trainer.load_training_data(min_samples=args.min_samples)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return

    # Train detector
    detector, results = trainer.train_multiclass_detector(X, y, test_size=args.test_size)

    # Save models
    output_dir = Path(args.output_path).parent / "multimodel"
    detector.save(str(output_dir))

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"✓ All models trained successfully!")
    for key, value in results.items():
        print(f"✓ {key}: {value:.3f}")
    print(f"✓ Models saved to: {output_dir}")

    print("\n✅ Training complete!")


def compare_models(args):
    """Compare different model configurations"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    db = Database(args.db_path)
    trainer = ModelTrainer(db)

    # Load data
    try:
        X, y, feature_names = trainer.load_training_data(min_samples=args.min_samples)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    # Convert to binary
    y_binary = (y == 3).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to compare
    models = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42
        )
    }

    results = []

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"\n{name}...")
        model.fit(X_train_scaled, y_train)

        train_score = accuracy_score(y_train, model.predict(X_train_scaled))
        test_score = accuracy_score(y_test, model.predict(X_test_scaled))
        f1 = f1_score(y_test, model.predict(X_test_scaled))

        results.append({
            'model': name,
            'train_acc': train_score,
            'test_acc': test_score,
            'f1_score': f1,
            'overfit': train_score - test_score
        })

        print(f"  Train: {train_score:.3f}")
        print(f"  Test:  {test_score:.3f}")
        print(f"  F1:    {f1:.3f}")

    # Print comparison table
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.sort_values('test_acc', ascending=False)

    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n💡 Recommendation:")
    best_model = df.iloc[0]['model']
    print(f"  Best performing model: {best_model}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ML models for scam token detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train binary classifier (default)
  python scripts/train_model.py --mode binary

  # Train with evaluation and report
  python scripts/train_model.py --mode binary --evaluate --report

  # Train multi-model detector
  python scripts/train_model.py --mode multimodel

  # Compare different algorithms
  python scripts/train_model.py --mode compare
        """
    )

    parser.add_argument(
        '--mode',
        choices=['binary', 'multimodel', 'compare'],
        default='binary',
        help='Training mode (default: binary)'
    )

    parser.add_argument(
        '--db-path',
        default='data/trading.db',
        help='Database path (default: data/trading.db)'
    )

    parser.add_argument(
        '--output-path',
        default='data/models/rug_detector.pkl',
        help='Output model path (default: data/models/rug_detector.pkl)'
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum training samples required (default: 100)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Cross-validation folds (default: 5)'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run full evaluation after training'
    )

    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate evaluation report'
    )

    parser.add_argument(
        '--report-path',
        default='data/models/evaluation_report.txt',
        help='Evaluation report path'
    )

    parser.add_argument(
        '--fn-cost',
        type=float,
        default=1000,
        help='False negative cost (default: 1000)'
    )

    parser.add_argument(
        '--fp-cost',
        type=float,
        default=50,
        help='False positive cost (default: 50)'
    )

    args = parser.parse_args()

    # Run appropriate mode
    if args.mode == 'binary':
        train_binary_model(args)
    elif args.mode == 'multimodel':
        train_multimodel(args)
    elif args.mode == 'compare':
        compare_models(args)


if __name__ == '__main__':
    main()
