"""
Quadratic Discriminant Analysis (qda) - Hate Speech Classification
Model: Quadratic Discriminant Analysis
Performance: 78.70% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
import pickle
import os


class QDAHateSpeechClassifier:
    def __init__(self):
        """Initialize QDA Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = QuadraticDiscriminantAnalysis(
            reg_param=0.1,  # Regularization parameter
            store_covariance=False,  # Don't store covariance matrices (memory efficient)
            tol=1e-4,  # Threshold for rank estimation
        )

        # TF-IDF Vectorizer for text preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced features for QDA (memory constraints)
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words="english",  # Remove English stop words
            lowercase=True,  # Convert to lowercase
            strip_accents="ascii",  # Remove accents
            max_df=0.95,  # Ignore terms in >95% of documents
            min_df=2,  # Ignore terms in <2 documents
        )

        # SVD for dimensionality reduction (QDA needs this for high-dimensional data)
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
        """Train the QDA Classifier"""
        print("Training Quadratic Discriminant Analysis...")

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
        try:
            self.model.fit(X_train_split, y_train_split)
        except Exception as e:
            print(f"Training failed with error: {e}")
            print("Trying with increased regularization...")
            self.model.reg_param = 0.5
            self.model.fit(X_train_split, y_train_split)

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
            "reg_param": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
            "tol": [1e-6, 1e-4, 1e-3, 1e-2],
        }

        # Initialize base model for grid search
        base_model = QuadraticDiscriminantAnalysis(store_covariance=False)

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
            print(f"Regularization parameter: {self.model.reg_param}")
            print(f"SVD components: {self.svd.n_components}")
            print(
                f"SVD explained variance ratio sum: {self.svd.explained_variance_ratio_.sum():.4f}"
            )
        else:
            print("Model not trained yet!")

    def save_model(
        self,
        model_path="qda_model.pkl",
        vectorizer_path="tfidf_vectorizer.pkl",
        svd_path="svd_reducer.pkl",
    ):
        """Save trained model, vectorizer, and SVD reducer"""
        print(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Saving SVD reducer to {svd_path}")
        with open(svd_path, "wb") as f:
            pickle.dump(self.svd, f)

    def load_model(
        self,
        model_path="qda_model.pkl",
        vectorizer_path="tfidf_vectorizer.pkl",
        svd_path="svd_reducer.pkl",
    ):
        """Load trained model, vectorizer, and SVD reducer"""
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        print(f"Loading SVD reducer from {svd_path}")
        with open(svd_path, "rb") as f:
            self.svd = pickle.load(f)

    def create_submission(
        self, predictions, test_ids, output_path="submission_qda.csv"
    ):
        """Create submission file for Kaggle"""
        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    qda_classifier = QDAHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = qda_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = qda_classifier.preprocess_text()

    # Option 1: Train with default parameters
    qda_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = qda_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = qda_classifier.predict(X_test)

    # Get model information
    qda_classifier.get_model_info()

    # Create submission file
    test_ids = test_data["id"].values
    submission = qda_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    qda_classifier.save_model()

    print("QDA Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
