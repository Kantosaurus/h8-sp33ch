"""
Bagging Classifier (Random Forest with PCA) - Hate Speech Classification
Model: Bootstrap Aggregating with dimensionality reduction
Performance: Bagging ensemble method with PCA preprocessing
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")


class BaggingHateSpeechClassifier:
    def __init__(
        self,
        base_estimator="decision_tree",  # 'decision_tree', 'random_forest'
        n_estimators=100,
        max_samples=1.0,
        max_features=1.0,
        use_pca=True,
        n_components=400,
    ):
        """
        Bagging Classifier with PCA

        Args:
            base_estimator: Type of base estimator ('decision_tree', 'random_forest')
            n_estimators: Number of base estimators
            max_samples: Fraction of samples to draw for each base estimator
            max_features: Fraction of features to draw for each base estimator
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
        """
        self.base_estimator_type = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.use_pca = use_pca
        self.n_components = n_components

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents="ascii",
        )

        # PCA for dimensionality reduction
        if use_pca:
            self.pca = PCA(n_components=n_components, random_state=42)
        else:
            self.pca = None

        # Base estimator
        if base_estimator == "decision_tree":
            base_est = DecisionTreeClassifier(
                max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
            )
        elif base_estimator == "random_forest":
            base_est = RandomForestClassifier(
                n_estimators=10,  # Smaller for bagging
                max_depth=8,
                random_state=42,
                n_jobs=1,  # Will be parallelized at bagging level
            )
        else:
            raise ValueError(f"Unknown base estimator: {base_estimator}")

        # Bagging Classifier
        self.classifier = BaggingClassifier(
            base_estimator=base_est,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=True,  # Out-of-bag score estimation
            warm_start=False,
            n_jobs=-1,
            random_state=42,
        )

        # Store training metrics
        self.training_metrics = {}

    def preprocess_data(self, texts):
        """Apply TF-IDF vectorization and PCA"""
        X_tfidf = self.vectorizer.transform(texts)

        if self.pca is not None:
            X_pca = self.pca.transform(X_tfidf.toarray())
            return X_pca
        else:
            return X_tfidf

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Bagging classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("Training Bagging Classifier...")
        start_time = time.time()

        # Transform texts to TF-IDF
        print("Applying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Apply PCA if specified
        if self.pca is not None:
            print(f"Applying PCA (reducing to {self.n_components} components)...")
            X_train_pca = self.pca.fit_transform(X_train_tfidf.toarray())

            print(
                f"Dimensionality reduced from {X_train_tfidf.shape[1]} to {X_train_pca.shape[1]} features"
            )
            print(
                f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}"
            )

            X_train_processed = X_train_pca
        else:
            X_train_processed = X_train_tfidf
            print(f"No PCA applied. Using {X_train_tfidf.shape[1]} features")

        # Train Bagging classifier
        print(
            f"Training Bagging with {self.n_estimators} {self.base_estimator_type} estimators..."
        )
        self.classifier.fit(X_train_processed, y_train)

        # Out-of-bag score
        if hasattr(self.classifier, "oob_score_"):
            print(f"Out-of-bag score: {self.classifier.oob_score_:.4f}")

        # Validation performance
        if X_val is not None and y_val is not None:
            X_val_processed = self.preprocess_data(X_val)
            val_predictions = self.classifier.predict(X_val_processed)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation accuracy: {val_accuracy:.4f}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self

    def predict(self, X):
        """Make predictions on new data"""
        X_processed = self.preprocess_data(X)
        return self.classifier.predict(X_processed)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_processed = self.preprocess_data(X)
        return self.classifier.predict_proba(X_processed)

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")

        # Transform data
        X_tfidf = self.vectorizer.fit_transform(X)

        if self.pca is not None:
            X_processed = self.pca.fit_transform(X_tfidf.toarray())
        else:
            X_processed = X_tfidf

        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier,
            X_processed,
            y,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
        )

        print(f"Cross-validation scores: {cv_scores}")
        print(
            f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        return cv_scores

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning"""
        print("Starting hyperparameter tuning...")

        # Transform data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        if self.pca is not None:
            X_train_processed = self.pca.fit_transform(X_train_tfidf.toarray())
        else:
            X_train_processed = X_train_tfidf

        # Base estimator for grid search
        if self.base_estimator_type == "decision_tree":
            base_est = DecisionTreeClassifier(random_state=42)
        else:
            base_est = RandomForestClassifier(
                n_estimators=10, random_state=42, n_jobs=1
            )

        # Parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_samples": [0.7, 0.8, 1.0],
            "max_features": [0.7, 0.8, 1.0],
        }

        # Add base estimator specific parameters
        if self.base_estimator_type == "decision_tree":
            param_grid["base_estimator__max_depth"] = [5, 10, 15]
            param_grid["base_estimator__min_samples_split"] = [2, 5, 10]

        grid_search = GridSearchCV(
            BaggingClassifier(
                base_estimator=base_est,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
            ),
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train_processed, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update classifier with best parameters
        self.classifier = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get feature importance from base estimators"""
        if not hasattr(self.classifier, "estimators_"):
            print("Model not trained yet!")
            return None

        # Collect feature importances from all estimators
        importances_list = []
        for estimator in self.classifier.estimators_:
            if hasattr(estimator, "feature_importances_"):
                importances_list.append(estimator.feature_importances_)
            elif hasattr(estimator, "estimators_"):  # For RandomForest base estimator
                # Average importance across trees in the RF
                rf_importance = np.zeros(estimator.n_features_in_)
                for tree in estimator.estimators_:
                    rf_importance += tree.feature_importances_
                rf_importance /= len(estimator.estimators_)
                importances_list.append(rf_importance)

        if not importances_list:
            print("No feature importances available from base estimators")
            return None

        # Average importance across all base estimators
        avg_importances = np.mean(importances_list, axis=0)

        if feature_names is None:
            if self.pca is not None:
                feature_names = [f"PC_{i+1}" for i in range(len(avg_importances))]
            else:
                feature_names = self.vectorizer.get_feature_names_out()

        # Get top important features
        indices = np.argsort(avg_importances)[::-1][:top_n]

        print(f"\nTop {top_n} most important features (averaged across estimators):")
        print("-" * 60)
        for i, idx in enumerate(indices):
            print(
                f"{i+1:2d}. {feature_names[idx][:35]:<35} : {avg_importances[idx]:8.4f}"
            )

        return dict(zip([feature_names[i] for i in indices], avg_importances[indices]))

    def get_estimator_info(self):
        """Get information about the ensemble"""
        if not hasattr(self.classifier, "estimators_"):
            print("Model not trained yet!")
            return None

        print(f"\nBagging Ensemble Information:")
        print(f"Number of estimators: {len(self.classifier.estimators_)}")
        print(f"Base estimator type: {self.base_estimator_type}")

        if hasattr(self.classifier, "oob_score_"):
            print(f"Out-of-bag score: {self.classifier.oob_score_:.4f}")

        print(f"Bootstrap samples: {self.classifier.bootstrap}")
        print(f"Max samples per estimator: {self.max_samples}")
        print(f"Max features per estimator: {self.max_features}")

        return {
            "n_estimators": len(self.classifier.estimators_),
            "oob_score": getattr(self.classifier, "oob_score_", None),
            "base_estimator": self.base_estimator_type,
        }

    def save_model(self, model_path, vectorizer_path, pca_path=None):
        """Save the trained model components"""
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        if self.pca is not None and pca_path is not None:
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        if pca_path:
            print(f"PCA transformer saved to {pca_path}")


def load_data(train_path, test_path):
    """Load training and testing data"""
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, test_data


def main():
    """Main execution function"""
    print("=" * 60)
    print("BAGGING HATE SPEECH CLASSIFICATION")
    print("=" * 60)

    # Set up paths
    data_dir = "../../data"
    output_dir = "bagging_outputs"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Load data
    train_data, test_data = load_data(train_path, test_path)

    # Prepare training data
    X_train_full = train_data["post"].fillna("")
    y_train_full = train_data["label"]

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Initialize and train model
    print(f"\nInitializing Bagging Classifier...")
    print(f"Configuration:")
    print(f"  - Base estimator: Decision Tree")
    print(f"  - Number of estimators: 100")
    print(f"  - PCA components: 400")
    print(f"  - Max samples: 1.0")
    print(f"  - Max features: 1.0")
    print(f"  - Bootstrap: True")
    print(f"  - Out-of-bag scoring: True")

    model = BaggingHateSpeechClassifier(
        base_estimator="decision_tree",
        n_estimators=100,
        max_samples=1.0,
        max_features=1.0,
        use_pca=True,
        n_components=400,
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Cross-validation on full training data
    cv_scores = model.cross_validate(X_train_full, y_train_full, cv_folds=5)

    # Get model insights
    feature_importance = model.get_feature_importance(top_n=15)
    estimator_info = model.get_estimator_info()

    # Validation predictions
    print("\nEvaluating on validation set...")
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, val_predictions, average="weighted"
    )

    print(f"\nValidation Results:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1-score: {val_f1:.4f}")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_val, val_predictions))

    # Test predictions
    print(f"\nMaking predictions on test set...")
    X_test = test_data["post"].fillna("")
    test_predictions = model.predict(X_test)

    # Create submission file
    submission_df = pd.DataFrame({"id": test_data["id"], "label": test_predictions})

    submission_path = os.path.join(output_dir, "submission_bagging.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    # Save model components
    model_path = os.path.join(output_dir, "bagging_model.pkl")
    vectorizer_path = os.path.join(output_dir, "bagging_vectorizer.pkl")
    pca_path = os.path.join(output_dir, "bagging_pca.pkl")

    model.save_model(model_path, vectorizer_path, pca_path)

    # Save metrics
    metrics = {
        "Model": "bagging",
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "F1": float(val_f1),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "Base_Estimator": model.base_estimator_type,
        "N_Estimators": model.n_estimators,
        "Max_Samples": model.max_samples,
        "Max_Features": model.max_features,
        "PCA_Components": model.n_components,
    }

    if model.pca is not None:
        metrics["Explained_Variance"] = float(model.pca.explained_variance_ratio_.sum())

    if estimator_info and estimator_info.get("oob_score"):
        metrics["OOB_Score"] = float(estimator_info["oob_score"])

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Print final summary
    print(f"\n" + "=" * 60)
    print("BAGGING CLASSIFIER SUMMARY")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    if estimator_info and estimator_info.get("oob_score"):
        print(f"Out-of-bag Score: {estimator_info['oob_score']:.4f}")
    print(f"Base Estimator: {model.base_estimator_type}")
    print(f"Number of Estimators: {model.n_estimators}")
    if model.pca:
        print(f"PCA Components: {model.n_components}")
        print(
            f"PCA Explained Variance: {model.pca.explained_variance_ratio_.sum():.4f}"
        )
    print(f"Test Predictions: {len(test_predictions)} samples")

    # Print prediction distribution
    unique, counts = np.unique(test_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    if len(unique) == 2:
        hate_ratio = counts[1] / len(test_predictions) if 1 in unique else 0
        print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    print(f"\n✅ Bagging classifier training and prediction completed!")


if __name__ == "__main__":
    main()
