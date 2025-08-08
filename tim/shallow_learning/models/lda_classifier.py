"""
Linear Discriminant Analysis (lda) - Hate Speech Classification
Model: Linear Discriminant Analysis
Performance: 74.60% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
import pickle
import os


class LDAHateSpeechClassifier:
    def __init__(self):
        """Initialize LDA Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = LinearDiscriminantAnalysis(
            solver="svd",  # SVD solver for numerical stability
            shrinkage=None,  # No shrinkage for SVD solver
            store_covariance=False,  # Don't store covariance matrices (memory efficient)
            tol=1e-4,  # Threshold for rank estimation
        )

        # TF-IDF Vectorizer for text preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced features for LDA
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words="english",  # Remove English stop words
            lowercase=True,  # Convert to lowercase
            strip_accents="ascii",  # Remove accents
            max_df=0.95,  # Ignore terms in >95% of documents
            min_df=2,  # Ignore terms in <2 documents
        )

        # SVD for dimensionality reduction (LDA works better with reduced dimensions)
        self.svd = TruncatedSVD(
            n_components=500, random_state=42  # Reduce to 500 dimensions
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
        """Preprocess text data using TF-IDF vectorization and SVD"""
        print("Preprocessing text data...")

        if fit_on_train:
            # Fit vectorizer on training data and transform
            X_train_tfidf = self.vectorizer.fit_transform(self.train_data["post"])
            print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

            # Apply SVD for dimensionality reduction
            X_train = self.svd.fit_transform(X_train_tfidf)
            print(f"SVD reduced feature matrix shape: {X_train.shape}")
        else:
            # Only transform (vectorizer and SVD already fitted)
            X_train_tfidf = self.vectorizer.transform(self.train_data["post"])
            X_train = self.svd.transform(X_train_tfidf)

        y_train = self.train_data["label"].values

        if hasattr(self, "test_data"):
            X_test_tfidf = self.vectorizer.transform(self.test_data["post"])
            X_test = self.svd.transform(X_test_tfidf)
            return X_train, y_train, X_test

        return X_train, y_train

    def train(self, X_train, y_train, validation_split=0.2):
        """Train the LDA Classifier"""
        print("Training Linear Discriminant Analysis...")

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
            "solver": ["svd", "lsqr", "eigen"],
            "tol": [1e-6, 1e-4, 1e-3, 1e-2],
        }

        # For lsqr solver, we can also tune shrinkage
        param_grid_shrinkage = {
            "solver": ["lsqr"],
            "shrinkage": [None, "auto", 0.1, 0.3, 0.5, 0.7, 0.9],
            "tol": [1e-6, 1e-4, 1e-3],
        }

        # Initialize base model for grid search
        base_model = LinearDiscriminantAnalysis(store_covariance=False)

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

    def get_model_info(self):
        """Get information about the trained model"""
        if hasattr(self.model, "classes_"):
            print(f"Classes: {self.model.classes_}")
            print(f"Number of features: {self.model.means_.shape[1]}")
            print(f"Solver: {self.model.solver}")
            print(f"SVD components: {self.svd.n_components}")
            print(
                f"SVD explained variance ratio sum: {self.svd.explained_variance_ratio_.sum():.4f}"
            )

            # Print class means (first few features)
            print("\nClass means (first 10 features):")
            for i, class_name in enumerate(["Non-Hate", "Hate"]):
                print(f"{class_name}: {self.model.means_[i][:10]}")

        else:
            print("Model not trained yet!")

    def get_discriminant_coefficients(self, feature_names=None, top_n=20):
        """Get discriminant coefficients (linear combination weights)"""
        if not hasattr(self.model, "coef_"):
            print("Model not trained yet!")
            return None

        # Get coefficients
        coefficients = self.model.coef_[0]  # For binary classification

        if feature_names is None:
            feature_names = [f"SVD_Component_{i}" for i in range(len(coefficients))]

        # Create DataFrame with coefficients
        coef_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False)

        print(f"Top {top_n} most important discriminant coefficients:")
        print(coef_df.head(top_n))

        return coef_df

    def save_model(
        self,
        model_path=None,
        vectorizer_path=None,
        svd_path=None,
    ):
        """Save trained model, vectorizer, and SVD reducer"""
        # Create model-specific directory
        model_dir = "lda_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "lda_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if svd_path is None:
            svd_path = os.path.join(model_dir, "svd_reducer.pkl")

        print(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Saving SVD reducer to {svd_path}")
        with open(svd_path, "wb") as f:
            pickle.dump(self.svd, f)

        # Save metrics if available
        if hasattr(self, "validation_metrics") and self.validation_metrics:
            metrics_path = os.path.join(model_dir, "metrics.json")
            print(f"Saving metrics to {metrics_path}")
            import json

            with open(metrics_path, "w") as f:
                json.dump(self.validation_metrics, f, indent=2)

    def load_model(
        self,
        model_path=None,
        vectorizer_path=None,
        svd_path=None,
    ):
        """Load trained model, vectorizer, and SVD reducer"""
        # Use model-specific directory
        model_dir = "lda_outputs"

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "lda_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if svd_path is None:
            svd_path = os.path.join(model_dir, "svd_reducer.pkl")

        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        print(f"Loading SVD reducer from {svd_path}")
        with open(svd_path, "rb") as f:
            self.svd = pickle.load(f)

    def create_submission(self, predictions, test_ids, output_path=None):
        """Create submission file for Kaggle"""
        # Create model-specific directory if not exists
        model_dir = "lda_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default output path within the model directory
        if output_path is None:
            output_path = os.path.join(model_dir, "submission_lda.csv")

        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    lda_classifier = LDAHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = lda_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = lda_classifier.preprocess_text()

    # Option 1: Train with default parameters
    lda_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = lda_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = lda_classifier.predict(X_test)

    # Get model information
    lda_classifier.get_model_info()

    # Get discriminant coefficients
    lda_classifier.get_discriminant_coefficients()

    # Create submission file
    test_ids = test_data["id"].values
    submission = lda_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    lda_classifier.save_model()

    print("LDA Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
