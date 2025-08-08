"""
Random Forest Classifier (rf) - Hate Speech Classification
Model: Random Forest Classifier with hyperparameter tuning
Performance: Close second with 83.49% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class RandomForestHateSpeechClassifier:
    def __init__(self):
        """Initialize Random Forest Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = RandomForestClassifier(
            n_estimators=200,  # Number of trees in the forest
            max_depth=20,  # Maximum depth of trees
            min_samples_split=5,  # Minimum samples required to split
            min_samples_leaf=2,  # Minimum samples required at leaf
            max_features="sqrt",  # Number of features to consider for best split
            bootstrap=True,  # Use bootstrap samples (Random Forest characteristic)
            random_state=42,  # For reproducibility
            n_jobs=-1,  # Use all available processors
            verbose=1,  # Show progress
            oob_score=True,  # Calculate out-of-bag score
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
        """Train the Random Forest Classifier"""
        print("Training Random Forest Classifier...")

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

        # Print OOB score if available
        if hasattr(self.model, "oob_score_"):
            print(f"Out-of-bag score: {self.model.oob_score_:.4f}")

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
            "max_depth": [15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

        # Initialize base model for grid search
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)

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

    def save_model(
        self,
        model_path="random_forest_model.pkl",
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
        model_path="random_forest_model.pkl",
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
        self, predictions, test_ids, output_path="submission_random_forest.csv"
    ):
        """Create submission file for Kaggle"""
        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    rf_classifier = RandomForestHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = rf_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = rf_classifier.preprocess_text()

    # Option 1: Train with default parameters
    rf_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = rf_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = rf_classifier.predict(X_test)

    # Get feature importance
    feature_importance = rf_classifier.get_feature_importance()

    # Create submission file
    test_ids = test_data["id"].values
    submission = rf_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    rf_classifier.save_model()

    print("Random Forest Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
