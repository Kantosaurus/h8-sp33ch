"""
Ridge Classifier (ridge) - Hate Speech Classification
Model: Ridge Classifier
Performance: 71.39% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class RidgeHateSpeechClassifier:
    def __init__(self):
        """Initialize Ridge Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = RidgeClassifier(
            alpha=1.0,  # Regularization strength
            fit_intercept=True,  # Whether to fit intercept
            copy_X=True,  # Copy X or use in-place
            max_iter=None,  # Maximum number of iterations
            tol=1e-3,  # Tolerance for stopping criteria
            class_weight=None,  # Weights associated with classes
            solver="auto",  # Solver to use
            random_state=42,  # For reproducibility
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
            norm="l2",  # L2 normalization
            use_idf=True,  # Enable inverse-document-frequency reweighting
            smooth_idf=True,  # Smooth idf weights
            sublinear_tf=True,  # Apply sublinear tf scaling
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
        """Train the Ridge Classifier"""
        print("Training Ridge Classifier...")

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

        # Store validation metrics for comparison
        self.validation_metrics = {}

        # Validate if validation split provided
        if validation_split > 0:
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

        print("Training completed!")

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")

        # Parameter grid for tuning
        param_grid = {
            "alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            "fit_intercept": [True, False],
            "tol": [1e-4, 1e-3, 1e-2],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            "class_weight": [None, "balanced"],
        }

        # Initialize base model for grid search
        base_model = RidgeClassifier(random_state=42, copy_X=True)

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

    def decision_function(self, X_test):
        """Get decision function scores"""
        return self.model.decision_function(X_test)

    def get_model_coefficients(self, feature_names=None, top_n=20):
        """Get model coefficients (feature weights)"""
        if not hasattr(self.model, "coef_"):
            print("Model not trained yet!")
            return None

        coefficients = self.model.coef_[0]  # For binary classification

        if feature_names is None:
            feature_names = self.vectorizer.get_feature_names_out()

        # Create DataFrame with coefficients
        coef_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False)

        print(f"Top {top_n} features with highest absolute coefficients:")
        print(coef_df.head(top_n))

        # Show top positive and negative coefficients
        print(f"\nTop {top_n//2} features indicating HATE (positive coefficients):")
        positive_coef = coef_df[coef_df["coefficient"] > 0].head(top_n // 2)
        print(positive_coef[["feature", "coefficient"]])

        print(f"\nTop {top_n//2} features indicating NON-HATE (negative coefficients):")
        negative_coef = coef_df[coef_df["coefficient"] < 0].tail(top_n // 2)
        print(negative_coef[["feature", "coefficient"]])

        return coef_df

    def get_model_info(self):
        """Get information about the trained model"""
        if hasattr(self.model, "coef_"):
            print(f"Number of features: {len(self.model.coef_[0])}")
            print(f"Alpha (regularization): {self.model.alpha}")
            print(f"Intercept: {self.model.intercept_[0]:.4f}")
            print(f"Solver used: {self.model.solver}")
            print(f"Number of iterations: {getattr(self.model, 'n_iter_', 'N/A')}")

            # Coefficient statistics
            coef = self.model.coef_[0]
            print(f"\nCoefficient statistics:")
            print(f"  Mean: {np.mean(coef):.6f}")
            print(f"  Std: {np.std(coef):.6f}")
            print(f"  Min: {np.min(coef):.6f}")
            print(f"  Max: {np.max(coef):.6f}")
            print(f"  L1 norm: {np.sum(np.abs(coef)):.6f}")
            print(f"  L2 norm: {np.sqrt(np.sum(coef**2)):.6f}")
        else:
            print("Model not trained yet!")

    def save_model(self, model_path=None, vectorizer_path=None):
        """Save trained model and vectorizer"""
        # Create model-specific directory
        model_dir = "ridge_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "ridge_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        print(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

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
        model_dir = "ridge_outputs"

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "ridge_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def create_submission(self, predictions, test_ids, output_path=None):
        """Create submission file for Kaggle"""
        # Create model-specific directory if not exists
        model_dir = "ridge_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default output path within the model directory
        if output_path is None:
            output_path = os.path.join(model_dir, "submission_ridge.csv")

        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    ridge_classifier = RidgeHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = ridge_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = ridge_classifier.preprocess_text()

    # Option 1: Train with default parameters
    ridge_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = ridge_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = ridge_classifier.predict(X_test)

    # Get model coefficients
    coefficients = ridge_classifier.get_model_coefficients()

    # Get model information
    ridge_classifier.get_model_info()

    # Create submission file
    test_ids = test_data["id"].values
    submission = ridge_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    ridge_classifier.save_model()

    print("Ridge Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
