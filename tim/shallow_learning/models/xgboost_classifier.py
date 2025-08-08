"""
XGBoost Classifier (xgboost) - Hate Speech Classification
Model: Extreme Gradient Boosting
Performance: Third with 81.44% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class XGBoostHateSpeechClassifier:
    def __init__(self):
        """Initialize XGBoost Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = xgb.XGBClassifier(
            n_estimators=200,  # Number of boosting rounds
            max_depth=6,  # Maximum depth of trees
            learning_rate=0.1,  # Step size shrinkage
            subsample=0.8,  # Subsample ratio of training instances
            colsample_bytree=0.8,  # Subsample ratio of columns
            min_child_weight=3,  # Minimum sum of instance weight needed in child
            gamma=0,  # Minimum loss reduction for split
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1,  # L2 regularization
            random_state=42,  # For reproducibility
            n_jobs=-1,  # Use all available processors
            verbosity=1,  # Print messages
            objective="binary:logistic",  # Binary classification
            eval_metric="logloss",  # Evaluation metric
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

    def train(self, X_train, y_train, validation_split=0.2, early_stopping_rounds=10):
        """Train the XGBoost Classifier"""
        print("Training XGBoost Classifier...")

        # Split training data for validation
        if validation_split > 0:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_split,
                random_state=42,
                stratify=y_train,
            )

            # Train with early stopping
            self.model.fit(
                X_train_split,
                y_train_split,
                eval_set=[(X_train_split, y_train_split), (X_val, y_val)],
                eval_metric="logloss",
                early_stopping_rounds=early_stopping_rounds,
                verbose=True,
            )

            # Validate
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
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
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [1, 1.5, 2],
        }

        # Initialize base model for grid search
        base_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
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

            # Get evaluation results
            results = self.model.evals_result()

            plt.figure(figsize=(12, 4))

            # Plot training loss
            plt.subplot(1, 2, 1)
            plt.plot(results["validation_0"]["logloss"], label="Train")
            if "validation_1" in results:
                plt.plot(results["validation_1"]["logloss"], label="Validation")
            plt.xlabel("Boosting Round")
            plt.ylabel("Log Loss")
            plt.title("Training History")
            plt.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Could not plot training history: {e}")

    def save_model(
        self, model_path="xgboost_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"
    ):
        """Save trained model and vectorizer"""
        print(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load_model(
        self, model_path="xgboost_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"
    ):
        """Load trained model and vectorizer"""
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def create_submission(
        self, predictions, test_ids, output_path="submission_xgboost.csv"
    ):
        """Create submission file for Kaggle"""
        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    xgb_classifier = XGBoostHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = xgb_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = xgb_classifier.preprocess_text()

    # Option 1: Train with default parameters
    xgb_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = xgb_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = xgb_classifier.predict(X_test)

    # Get feature importance
    feature_importance = xgb_classifier.get_feature_importance()

    # Plot training history
    xgb_classifier.plot_training_history()

    # Create submission file
    test_ids = test_data["id"].values
    submission = xgb_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    xgb_classifier.save_model()

    print("XGBoost Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
