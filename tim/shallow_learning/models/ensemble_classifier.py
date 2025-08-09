"""
Ensemble Classifier - Hate Speech Classification
Model: Combines multiple trained models for improved performance
Performance: Expected to outperform individual models
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.ensemble import VotingClassifier
import warnings

warnings.filterwarnings("ignore")


class EnsembleHateSpeechClassifier:
    def __init__(self, ensemble_method="weighted_average"):
        """
        Initialize Ensemble Classifier

        Args:
            ensemble_method: 'voting', 'weighted_average', 'stacking'
        """
        self.ensemble_method = ensemble_method
        self.models = {}
        self.vectorizers = {}
        self.model_weights = {}
        self.validation_metrics = {}

        # Define which models to include based on performance
        self.selected_models = [
            "extra_trees",
            "catboost",
            "ridge",
            "lda",
            "lightgbm",  # Top 5 performers
        ]

        # TF-IDF Vectorizer (unified preprocessing)
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            stop_words="english",
            lowercase=True,
            strip_accents="ascii",
            max_df=0.9,
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )

    def load_trained_models(self):
        """Load all trained models and their components"""
        print("Loading trained models for ensemble...")

        model_configs = {
            "extra_trees": {
                "model_file": "extra_trees_outputs/extra_trees_model.pkl",
                "vectorizer_file": "extra_trees_outputs/tfidf_vectorizer.pkl",
                "metrics_file": "extra_trees_outputs/metrics.json",
            },
            "catboost": {
                "model_file": "catboost_outputs/catboost_model.cbm",
                "vectorizer_file": "catboost_outputs/tfidf_vectorizer.pkl",
                "metrics_file": "catboost_outputs/metrics.json",
            },
            "ridge": {
                "model_file": "ridge_outputs/ridge_model.pkl",
                "vectorizer_file": "ridge_outputs/tfidf_vectorizer.pkl",
                "metrics_file": "ridge_outputs/metrics.json",
            },
            "lda": {
                "model_file": "lda_outputs/lda_model.pkl",
                "vectorizer_file": "lda_outputs/tfidf_vectorizer.pkl",
                "metrics_file": "lda_outputs/metrics.json",
                "svd_file": "lda_outputs/svd_reducer.pkl",
            },
            "lightgbm": {
                "model_file": "lightgbm_outputs/lightgbm_model.pkl",
                "vectorizer_file": "lightgbm_outputs/tfidf_vectorizer.pkl",
                "metrics_file": "lightgbm_outputs/metrics.json",
            },
        }

        loaded_models = 0
        for model_name in self.selected_models:
            try:
                config = model_configs[model_name]

                # Load model
                if model_name == "catboost":
                    # CatBoost uses its own save format
                    from catboost import CatBoostClassifier

                    model = CatBoostClassifier()
                    model.load_model(config["model_file"])
                else:
                    with open(config["model_file"], "rb") as f:
                        model = pickle.load(f)

                # Load vectorizer
                with open(config["vectorizer_file"], "rb") as f:
                    vectorizer = pickle.load(f)

                # Load SVD if needed (for LDA)
                svd = None
                if "svd_file" in config and os.path.exists(config["svd_file"]):
                    with open(config["svd_file"], "rb") as f:
                        svd = pickle.load(f)

                # Load metrics for weighting
                if os.path.exists(config["metrics_file"]):
                    with open(config["metrics_file"], "r") as f:
                        metrics = json.load(f)
                        # Use F1 score as weight (better than accuracy for imbalanced data)
                        self.model_weights[model_name] = metrics.get("F1", 0.5)
                else:
                    self.model_weights[model_name] = 0.5

                self.models[model_name] = {
                    "model": model,
                    "vectorizer": vectorizer,
                    "svd": svd,
                }
                loaded_models += 1
                print(
                    f"✓ Loaded {model_name} (weight: {self.model_weights[model_name]:.4f})"
                )

            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                if model_name in self.selected_models:
                    self.selected_models.remove(model_name)

        print(
            f"\nSuccessfully loaded {loaded_models}/{len(self.selected_models)} models"
        )

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {
                k: v / total_weight for k, v in self.model_weights.items()
            }
            print("Model weights (normalized):")
            for model, weight in self.model_weights.items():
                print(f"  {model}: {weight:.4f}")

    def load_data(self, train_path, test_path=None):
        """Load training and testing data"""
        print("Loading data...")
        self.train_data = pd.read_csv(train_path)

        if test_path:
            self.test_data = pd.read_csv(test_path)
            print(
                f"Loaded {len(self.train_data)} training samples and {len(self.test_data)} test samples"
            )
        else:
            print(f"Loaded {len(self.train_data)} training samples")

        return self.train_data, self.test_data if test_path else None

    def preprocess_for_model(self, texts, model_name):
        """Preprocess texts for a specific model"""
        model_info = self.models[model_name]
        vectorizer = model_info["vectorizer"]
        svd = model_info.get("svd")

        # Transform using the model's original vectorizer
        X = vectorizer.transform(texts)

        # Apply SVD if the model uses it (LDA, QDA, KNN)
        if svd is not None:
            X = svd.transform(X)

        return X

    def predict_single_model(self, texts, model_name):
        """Get predictions from a single model"""
        try:
            X = self.preprocess_for_model(texts, model_name)
            model = self.models[model_name]["model"]

            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                return proba[:, 1]  # Return probability of class 1 (hate speech)
            else:
                # For models without predict_proba, use decision function or predictions
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(X)
                    # Normalize to [0,1] range
                    return (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    return model.predict(X).astype(float)
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            return np.zeros(len(texts))

    def predict(self, texts, threshold=0.5):
        """Make ensemble predictions"""
        if not self.models:
            raise ValueError("No models loaded. Call load_trained_models() first.")

        print(f"Making ensemble predictions using {len(self.models)} models...")

        # Get predictions from all models
        all_predictions = {}
        for model_name in self.models.keys():
            predictions = self.predict_single_model(texts, model_name)
            all_predictions[model_name] = predictions

        if self.ensemble_method == "weighted_average":
            # Weighted average of probabilities
            ensemble_proba = np.zeros(len(texts))
            for model_name, predictions in all_predictions.items():
                weight = self.model_weights.get(model_name, 1.0)
                ensemble_proba += weight * predictions

            # Convert probabilities to binary predictions
            return (ensemble_proba > threshold).astype(int), ensemble_proba

        elif self.ensemble_method == "voting":
            # Simple majority voting
            vote_matrix = np.array([pred > 0.5 for pred in all_predictions.values()]).T
            ensemble_pred = (vote_matrix.sum(axis=1) > len(self.models) / 2).astype(int)

            # Also return average probabilities for consistency
            ensemble_proba = np.mean(list(all_predictions.values()), axis=0)
            return ensemble_pred, ensemble_proba

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance"""
        predictions, probabilities = self.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )

        # Store metrics
        self.validation_metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

        print(f"Ensemble Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        return self.validation_metrics

    def compare_with_individuals(self, texts, y_true):
        """Compare ensemble with individual model performance"""
        print("\n" + "=" * 80)
        print("ENSEMBLE vs INDIVIDUAL MODEL COMPARISON")
        print("=" * 80)

        results = {}

        # Individual model performance
        for model_name in self.models.keys():
            try:
                pred_proba = self.predict_single_model(texts, model_name)
                predictions = (pred_proba > 0.5).astype(int)

                accuracy = accuracy_score(y_true, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, predictions, average="weighted", zero_division=0
                )

                results[model_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")

        # Ensemble performance
        ensemble_pred, _ = self.predict(texts)
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred)
        ensemble_precision, ensemble_recall, ensemble_f1, _ = (
            precision_recall_fscore_support(
                y_true, ensemble_pred, average="weighted", zero_division=0
            )
        )

        results["ensemble"] = {
            "Accuracy": ensemble_accuracy,
            "Precision": ensemble_precision,
            "Recall": ensemble_recall,
            "F1": ensemble_f1,
        }

        # Display results
        print(
            f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}"
        )
        print("-" * 60)

        for model_name, metrics in results.items():
            print(
                f"{model_name:<15} {metrics['Accuracy']:<10.4f} {metrics['Precision']:<10.4f} "
                f"{metrics['Recall']:<10.4f} {metrics['F1']:<10.4f}"
            )

        # Check if ensemble is best
        best_f1 = max(results.values(), key=lambda x: x["F1"])["F1"]
        ensemble_f1 = results["ensemble"]["F1"]

        if ensemble_f1 >= best_f1:
            print(f"\n🎉 Ensemble achieves BEST performance with F1: {ensemble_f1:.4f}")
        else:
            improvement = ensemble_f1 - max(
                [v["F1"] for k, v in results.items() if k != "ensemble"]
            )
            print(
                f"\n📊 Ensemble F1: {ensemble_f1:.4f} (improvement: {improvement:+.4f})"
            )

        return results

    def save_model(self, model_path=None):
        """Save ensemble configuration and metrics"""
        # Create model-specific directory
        model_dir = "ensemble_outputs"
        os.makedirs(model_dir, exist_ok=True)

        if model_path is None:
            model_path = os.path.join(model_dir, "ensemble_config.pkl")

        # Save ensemble configuration
        ensemble_config = {
            "ensemble_method": self.ensemble_method,
            "selected_models": self.selected_models,
            "model_weights": self.model_weights,
        }

        with open(model_path, "wb") as f:
            pickle.dump(ensemble_config, f)

        print(f"Saving ensemble config to {model_path}")

        # Save metrics if available
        if hasattr(self, "validation_metrics") and self.validation_metrics:
            metrics_path = os.path.join(model_dir, "metrics.json")
            print(f"Saving metrics to {metrics_path}")
            with open(metrics_path, "w") as f:
                json.dump(self.validation_metrics, f, indent=2)

    def create_submission(self, predictions, test_ids, output_path=None):
        """Create submission file for Kaggle"""
        model_dir = "ensemble_outputs"
        os.makedirs(model_dir, exist_ok=True)

        if output_path is None:
            output_path = os.path.join(model_dir, "submission_ensemble.csv")

        submission_df = pd.DataFrame({"id": test_ids, "label": predictions})

        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to: {output_path}")
        return submission_df


def main():
    """Main execution function"""
    # Initialize ensemble classifier
    ensemble = EnsembleHateSpeechClassifier(ensemble_method="weighted_average")

    # Load trained models
    ensemble.load_trained_models()

    if not ensemble.models:
        print("❌ No models loaded. Please train individual models first.")
        return

    # Load data for evaluation
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    train_data, test_data = ensemble.load_data(train_path, test_path)

    # Use validation split to evaluate ensemble performance
    X_train_text = train_data["post"].values
    y_train = train_data["label"].values

    # Split for evaluation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_text, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Evaluate ensemble on validation set
    print("\nEvaluating ensemble performance...")
    ensemble.evaluate_ensemble(X_val_split, y_val_split)

    # Compare with individual models
    ensemble.compare_with_individuals(X_val_split, y_val_split)

    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_predictions, test_probabilities = ensemble.predict(test_data["post"].values)

    # Create submission file
    test_ids = test_data["id"].values
    submission = ensemble.create_submission(test_predictions, test_ids)

    # Save ensemble model
    ensemble.save_model()

    print(f"\n✅ Ensemble training and prediction completed!")
    print(f"Final prediction distribution: {np.bincount(test_predictions)}")

    # Summary
    hate_ratio = np.sum(test_predictions) / len(test_predictions)
    print(f"Predicted hate speech ratio: {hate_ratio:.4f}")


if __name__ == "__main__":
    main()
