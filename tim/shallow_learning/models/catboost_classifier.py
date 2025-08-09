"""
CatBoost Classifier (catboost) - Hate Speech Classification
Model: CatBoost Classifier with hyperparameter tuning
Performance: 78.99% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class CatBoostHateSpeechClassifier:
    def __init__(self):
        """Initialize CatBoost Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = CatBoostClassifier(
            iterations=1000,  # Number of boosting iterations
            depth=6,  # Depth of the tree
            learning_rate=0.1,  # Learning rate
            l2_leaf_reg=3,  # L2 regularization coefficient
            border_count=128,  # Number of splits for numerical features
            random_seed=42,  # For reproducibility
            verbose=100,  # Print progress every 100 iterations
            task_type="CPU",  # Use CPU for training
            thread_count=-1,  # Use all available threads
            eval_metric="Accuracy",  # Evaluation metric
            loss_function="Logloss",  # Loss function for binary classification
            early_stopping_rounds=50,  # Early stopping
        )

        # TF-IDF Vectorizer for text preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Maximum number of features
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words="english",  # Remove English stop words
            lowercase=True,  # Convert to lowercase
            strip_accents="ascii",  # Remove accents
            max_df=0.95,  # Ignore terms in >95% of documents
            min_df=2,  # Ignore terms in <2 documents
        )

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

    def preprocess_text(self, fit_on_train=True):
        """Preprocess text data using TF-IDF vectorization"""
        print("Preprocessing text data...")

        if fit_on_train:
            # Fit vectorizer on training data and transform
            X_train = self.vectorizer.fit_transform(self.train_data["post"])
            print(f"TF-IDF feature matrix shape: {X_train.shape}")
        else:
            # Only transform (vectorizer already fitted)
            X_train = self.vectorizer.transform(self.train_data["post"])

        y_train = self.train_data["label"].values

        if hasattr(self, "test_data"):
            X_test = self.vectorizer.transform(self.test_data["post"])
            return X_train, y_train, X_test

        return X_train, y_train

    def train(self, X_train, y_train, validation_split=0.2):
        """Train the CatBoost Classifier"""
        print("Training CatBoost Classifier...")

        # Store validation metrics for comparison
        self.validation_metrics = {}

        # Split training data for validation
        if validation_split > 0:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_split,
                random_state=42,
                stratify=y_train,
            )

            # Train with validation set for early stopping
            self.model.fit(
                X_train_split,
                y_train_split,
                eval_set=(X_val, y_val),
                verbose=100,
                use_best_model=True,
                plot=False,
            )

            # Validate
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)

            # Calculate additional metrics
            from sklearn.metrics import precision_recall_fscore_support

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, val_predictions, average="weighted", zero_division=0
            )

            # Store metrics
            self.validation_metrics = {
                "Accuracy": val_accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }

            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_predictions))

        else:
            # Train without validation
            self.model.fit(X_train, y_train)

        print("Training completed!")

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")

        # Parameter grid for tuning
        param_grid = {
            "iterations": [500, 1000, 1500],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [64, 128, 255],
        }

        # Initialize base model for grid search
        base_model = CatBoostClassifier(
            random_seed=42,
            verbose=False,
            task_type="CPU",
            thread_count=-1,
            eval_metric="Accuracy",
            loss_function="Logloss",
        )

        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_params_, grid_search.best_score_

    def predict(self, X_test):
        """Make predictions on test data"""
        print("Making predictions...")
        predictions = self.model.predict(X_test)
        return predictions

    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        return self.model.predict_proba(X_test)

    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get feature importance from the trained model"""
        if not hasattr(self.model, "feature_importances_"):
            print("Model not trained yet!")
            return None

        importance = self.model.feature_importances_

        if feature_names is None:
            feature_names = self.vectorizer.get_feature_names_out()

        # Create DataFrame with feature importance
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        print(f"Top {top_n} most important features:")
        print(feature_importance_df.head(top_n))

        return feature_importance_df

    def plot_training_history(self):
        """Plot training history if available"""
        try:
            import matplotlib.pyplot as plt

            # Get training history
            if hasattr(self.model, "get_evals_result"):
                evals_result = self.model.get_evals_result()

                plt.figure(figsize=(12, 4))

                # Plot training metrics
                plt.subplot(1, 2, 1)
                iterations = range(len(evals_result["learn"]["Logloss"]))
                plt.plot(
                    iterations, evals_result["learn"]["Logloss"], label="Train Loss"
                )
                if "validation" in evals_result:
                    plt.plot(
                        iterations,
                        evals_result["validation"]["Logloss"],
                        label="Validation Loss",
                    )
                plt.xlabel("Iteration")
                plt.ylabel("Log Loss")
                plt.title("Training History - Loss")
                plt.legend()

                # Plot accuracy if available
                if "Accuracy" in evals_result["learn"]:
                    plt.subplot(1, 2, 2)
                    plt.plot(
                        iterations,
                        evals_result["learn"]["Accuracy"],
                        label="Train Accuracy",
                    )
                    if (
                        "validation" in evals_result
                        and "Accuracy" in evals_result["validation"]
                    ):
                        plt.plot(
                            iterations,
                            evals_result["validation"]["Accuracy"],
                            label="Validation Accuracy",
                        )
                    plt.xlabel("Iteration")
                    plt.ylabel("Accuracy")
                    plt.title("Training History - Accuracy")
                    plt.legend()

                plt.tight_layout()
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Could not plot training history: {e}")

    def save_model(self, model_path=None, vectorizer_path=None):
        """Save trained model and vectorizer"""
        # Create model-specific directory
        model_dir = "catboost_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "catboost_model.cbm")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        print(f"Saving model to {model_path}")
        self.model.save_model(model_path)

        print(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        # Save metrics if available
        if hasattr(self, "validation_metrics") and self.validation_metrics:
            metrics_path = os.path.join(model_dir, "metrics.json")
            print(f"Saving metrics to {metrics_path}")
            import json

            with open(metrics_path, "w") as f:
                json.dump(self.validation_metrics, f, indent=2)

    def load_model(self, model_path=None, vectorizer_path=None):
        """Load trained model and vectorizer"""
        # Use model-specific directory
        model_dir = "catboost_outputs"

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "catboost_model.cbm")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        print(f"Loading model from {model_path}")
        self.model.load_model(model_path)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def create_submission(self, predictions, test_ids, output_path=None):
        """Create submission file for Kaggle"""
        # Create model-specific directory if not exists
        model_dir = "catboost_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default output path within the model directory
        if output_path is None:
            output_path = os.path.join(model_dir, "submission_catboost.csv")

        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    cb_classifier = CatBoostHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = cb_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = cb_classifier.preprocess_text()

    # Option 1: Train with default parameters
    cb_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = cb_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = cb_classifier.predict(X_test)

    # Get feature importance
    feature_importance = cb_classifier.get_feature_importance()

    # Plot training history
    cb_classifier.plot_training_history()

    # Create submission file
    test_ids = test_data["id"].values
    submission = cb_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    cb_classifier.save_model()

    print("CatBoost Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
