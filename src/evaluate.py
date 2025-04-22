import argparse

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, average_precision_score)
import numpy as np
from utils.visualization import Visualizer
import joblib

class ModelEvaluator:
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def evaluate(self, model, X, y):
        """Comprehensive model evaluation"""
        # Cross-Validated metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42 )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        metrics = {
            "CV ROC AUC": f"{np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba),
            "Average Precision": average_precision_score(y_test, y_proba)
        }

        # Generate reports and visualizations
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.visualizer.plot_confusion_matrix(y_test, y_pred)
        self.visualizer.plot_feature_importance(model, X)

        return metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Titanic survival prediction model")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to processed data (PKL file)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (PKL file)')
    parser.add_argument('--no-vis',action='store_true',
                        help='Disable visualization outputs')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load data and model
    try:
        df = joblib.load(args.data)
        model = joblib.load(args.model)

    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Prepare features
    X = df.drop('survived', axis=1)
    y = df['survived']

    # Initialize components
    visualizer = Visualizer(model, X, y) if not args.no_vis else None
    evaluator = ModelEvaluator(visualizer)

    # Run evaluation
    metrics = evaluator.evaluate(model, X, y)

    # Print results
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name:>20}: {value}")

if __name__ == "__main__":
    main()