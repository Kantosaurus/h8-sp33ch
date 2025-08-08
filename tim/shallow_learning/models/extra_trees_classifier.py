"""
Extra Trees Classifier (et) - Hate Speech Classification
Model: Extra Trees Classifier with hyperparameter tuning
Performance: Best overall performance with 83.84% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class ExtraTreesHateSpeechClassifier:
    def __init__(self):
        """Initialize Extra Trees Classifier with optimized hyperparameters"""
        # Key hyperparameters optimized for hate speech classification
        self.model = ExtraTreesClassifier(
            n_estimators=500,  # Increased trees for better performance
            max_depth=None,  # Allow deeper trees
            min_samples_split=2,  # Lower threshold for splitting
            min_samples_leaf=1,  # Lower threshold for leaf nodes
            max_features="sqrt",  # Number of features to consider for best split
            bootstrap=False,  # Use all samples for each tree (Extra Trees characteristic)
            class_weight="balanced",  # CRITICAL: Handle class imbalance
            random_state=42,  # For reproducibility
            n_jobs=-1,  # Use all available processors
            verbose=1,  # Show progress
        )

        # Enhanced TF-IDF Vectorizer for better text preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=20000,  # Increased features for better representation
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            stop_words="english",  # Remove English stop words
            lowercase=True,  # Convert to lowercase
            strip_accents="ascii",  # Remove accents
            max_df=0.9,  # Slightly more restrictive on common terms
            min_df=2,  # Ignore terms in <2 documents
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm="l2",  # L2 normalization
            use_idf=True,  # Use inverse document frequency
            smooth_idf=True,  # Smooth idf weights
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
        """Train the Extra Trees Classifier"""
        print("Training Extra Trees Classifier...")

        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class distribution: {class_dist}")
        print(f"Class imbalance ratio: {counts[0]/counts[1]:.2f}:1 (Non-hate:Hate)")

        # Split training data for validation with stratification
        if validation_split > 0:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_split,
                random_state=42,
                stratify=y_train,
            )

            # Check validation split class distribution
            val_unique, val_counts = np.unique(y_val, return_counts=True)
            val_class_dist = dict(zip(val_unique, val_counts))
            print(f"Validation class distribution: {val_class_dist}")
        else:
            X_train_split, y_train_split = X_train, y_train

        # Train the model
        print("Fitting model with balanced class weights...")
        self.model.fit(X_train_split, y_train_split)

        # Validate if validation split provided
        if validation_split > 0:
            val_predictions = self.model.predict(X_val)
            val_probabilities = self.model.predict_proba(X_val)

            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            # Additional metrics
            from sklearn.metrics import (
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
            )

            precision = precision_score(y_val, val_predictions, average="weighted")
            recall = recall_score(y_val, val_predictions, average="weighted")
            f1 = f1_score(y_val, val_predictions, average="weighted")
            auc = roc_auc_score(y_val, val_probabilities[:, 1])

            print(f"Validation Precision: {precision:.4f}")
            print(f"Validation Recall: {recall:.4f}")
            print(f"Validation F1-Score: {f1:.4f}")
            print(f"Validation AUC: {auc:.4f}")

            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_predictions))

            # Check prediction distribution
            val_pred_unique, val_pred_counts = np.unique(
                val_predictions, return_counts=True
            )
            val_pred_dist = dict(zip(val_pred_unique, val_pred_counts))
            print(f"Validation prediction distribution: {val_pred_dist}")

        print("Training completed!")

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")

        # Enhanced parameter grid for better hate speech classification
        param_grid = {
            "n_estimators": [300, 500, 700],
            "max_depth": [None, 15, 25],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample"],
        }

        # Initialize base model for grid search
        base_model = ExtraTreesClassifier(
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
            verbose=0,  # Reduce verbosity during grid search
        )

        # Perform grid search with better scoring
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring="f1_weighted",  # Better metric for imbalanced data
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        print("Running grid search (this may take a while)...")
        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (F1-weighted): {grid_search.best_score_:.4f}")

        # Additional info about the search
        print(f"Total combinations tested: {len(grid_search.cv_results_['params'])}")

        return grid_search.best_params_, grid_search.best_score_

    def predict(self, X_test):
        """Make predictions on test data"""
        print("Making predictions...")
        predictions = self.model.predict(X_test)

        # Analyze prediction distribution
        pred_unique, pred_counts = np.unique(predictions, return_counts=True)
        pred_dist = dict(zip(pred_unique, pred_counts))
        print(f"Test prediction distribution: {pred_dist}")

        return predictions

    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        return self.model.predict_proba(X_test)

    def analyze_predictions(self, X_test, threshold=0.5):
        """Analyze predictions with different thresholds"""
        print("Analyzing predictions...")

        # Get probabilities
        probabilities = self.model.predict_proba(X_test)
        hate_probs = probabilities[:, 1]  # Probabilities for hate class

        # Default predictions
        default_preds = self.model.predict(X_test)

        # Try different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        print("\nPrediction distribution by threshold:")
        print("Threshold | Class 0 | Class 1 | Total")
        print("-" * 40)

        for thresh in thresholds:
            thresh_preds = (hate_probs >= thresh).astype(int)
            unique, counts = np.unique(thresh_preds, return_counts=True)

            count_0 = counts[0] if 0 in unique else 0
            count_1 = counts[1] if 1 in unique else 0

            print(
                f"{thresh:8.1f} | {count_0:7d} | {count_1:7d} | {len(thresh_preds):5d}"
            )

        # Statistics about probabilities
        print(f"\nHate speech probability statistics:")
        print(f"Min: {hate_probs.min():.4f}")
        print(f"Max: {hate_probs.max():.4f}")
        print(f"Mean: {hate_probs.mean():.4f}")
        print(f"Std: {hate_probs.std():.4f}")
        print(f"Median: {np.median(hate_probs):.4f}")

        return default_preds, probabilities

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
        self, model_path="extra_trees_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"
    ):
        """Save trained model and vectorizer"""
        print(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load_model(
        self, model_path="extra_trees_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"
    ):
        """Load trained model and vectorizer"""
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def create_submission(
        self, predictions, test_ids, output_path="submission_extra_trees.csv"
    ):
        """Create submission file for Kaggle"""
        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    et_classifier = ExtraTreesHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = et_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = et_classifier.preprocess_text()

    # Option 1: Train with improved parameters (commenting out to use hyperparameter tuning)
    et_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer but better results)
    # print("\nStarting hyperparameter tuning...")
    # best_params, best_score = et_classifier.hyperparameter_tuning(X_train, y_train)
    # print("Hyperparameter tuning completed. Retraining with best parameters...")
    # et_classifier.train(X_train, y_train, validation_split=0.2)

    # Analyze predictions with different thresholds
    test_predictions, test_probabilities = et_classifier.analyze_predictions(X_test)

    # Get feature importance
    feature_importance = et_classifier.get_feature_importance()

    # Create submission file
    test_ids = test_data["id"].values
    submission = et_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    et_classifier.save_model()

    print("\nExtra Trees Classifier training and prediction completed!")
    print(f"Final prediction distribution: {np.bincount(test_predictions)}")

    # Provide recommendations
    hate_ratio = np.sum(test_predictions) / len(test_predictions)
    print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    if hate_ratio < 0.05:
        print("\n⚠️  WARNING: Very low hate speech detection rate!")
        print("💡 Consider:")
        print("   - Running hyperparameter tuning (uncomment line in main)")
        print("   - Using a lower probability threshold")
        print("   - Adding more sophisticated text preprocessing")
    else:
        print("✅ Reasonable hate speech detection rate achieved")


if __name__ == "__main__":
    main()
