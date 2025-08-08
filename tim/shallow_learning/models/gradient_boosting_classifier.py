"""
Gradient Boosting Classifier (gbc) - Hate Speech Classification
Model: Gradient Boosting Classifier
Performance: 72.88% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class GradientBoostingHateSpeechClassifier:
    def __init__(self):
        """Initialize Gradient Boosting Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = GradientBoostingClassifier(
            n_estimators=200,  # Number of boosting stages
            learning_rate=0.1,  # Learning rate shrinks contribution of each tree
            max_depth=6,  # Maximum depth of individual trees
            min_samples_split=5,  # Minimum samples required to split
            min_samples_leaf=2,  # Minimum samples required at leaf
            max_features="sqrt",  # Number of features to consider for best split
            subsample=0.8,  # Fraction of samples for fitting trees
            random_state=42,  # For reproducibility
            verbose=1,  # Print progress
            warm_start=False,  # Don't reuse solution of previous call
            validation_fraction=0.1,  # Fraction of training data for early stopping
            n_iter_no_change=10,  # Number of iterations with no improvement for early stopping
            tol=1e-4,  # Tolerance for early stopping
        )

        # TF-IDF Vectorizer for text preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=8000,  # Maximum number of features
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
        """Train the Gradient Boosting Classifier"""
        print("Training Gradient Boosting Classifier...")

        # Split training data for validation
        if validation_split > 0:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_split,
                random_state=42,
                stratify=y_train,
            )
        else:
            X_train_split, y_train_split = X_train, y_train

        # Train the model
        self.model.fit(X_train_split, y_train_split)

        # Print training information
        print(f"Number of estimators used: {self.model.n_estimators_}")
        print(f"Training score: {self.model.train_score_[-1]:.4f}")

        # Validate if validation split provided
        if validation_split > 0:
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_predictions))

        print("Training completed!")

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")

        # Parameter grid for tuning
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 9],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "subsample": [0.8, 0.9, 1.0],
        }

        # Initialize base model for grid search
        base_model = GradientBoostingClassifier(
            random_state=42, warm_start=False, verbose=0
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
        """Plot training history (loss curve)"""
        try:
            import matplotlib.pyplot as plt

            if hasattr(self.model, "train_score_"):
                plt.figure(figsize=(10, 6))

                # Plot training loss
                train_scores = self.model.train_score_
                iterations = range(1, len(train_scores) + 1)

                plt.plot(iterations, train_scores, "b-", label="Training Loss")

                # Plot validation loss if available
                if (
                    hasattr(self.model, "validation_score_")
                    and self.model.validation_score_ is not None
                ):
                    val_scores = self.model.validation_score_
                    plt.plot(iterations, val_scores, "r-", label="Validation Loss")

                plt.xlabel("Boosting Iteration")
                plt.ylabel("Loss")
                plt.title("Gradient Boosting Training History")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

                print(f"Final training loss: {train_scores[-1]:.4f}")
                if (
                    hasattr(self.model, "validation_score_")
                    and self.model.validation_score_ is not None
                ):
                    print(f"Final validation loss: {val_scores[-1]:.4f}")

        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Could not plot training history: {e}")

    def get_staged_predictions(self, X_test, stages=None):
        """Get predictions at different boosting stages"""
        if stages is None:
            stages = [10, 50, 100, 150, 200]

        staged_predictions = {}

        for stage in stages:
            if stage <= self.model.n_estimators_:
                # Get predictions at specific stage
                proba = list(self.model.staged_predict_proba(X_test))
                if len(proba) >= stage:
                    staged_predictions[stage] = np.argmax(proba[stage - 1], axis=1)

        return staged_predictions

    def save_model(
        self,
        model_path="gradient_boosting_model.pkl",
        vectorizer_path="tfidf_vectorizer.pkl",
    ):
        """Save trained model and vectorizer"""
        print(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load_model(
        self,
        model_path="gradient_boosting_model.pkl",
        vectorizer_path="tfidf_vectorizer.pkl",
    ):
        """Load trained model and vectorizer"""
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def create_submission(
        self, predictions, test_ids, output_path="submission_gradient_boosting.csv"
    ):
        """Create submission file for Kaggle"""
        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    gb_classifier = GradientBoostingHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = gb_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = gb_classifier.preprocess_text()

    # Option 1: Train with default parameters
    gb_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = gb_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = gb_classifier.predict(X_test)

    # Get feature importance
    feature_importance = gb_classifier.get_feature_importance()

    # Plot training history
    gb_classifier.plot_training_history()

    # Create submission file
    test_ids = test_data["id"].values
    submission = gb_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    gb_classifier.save_model()

    print("Gradient Boosting Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
