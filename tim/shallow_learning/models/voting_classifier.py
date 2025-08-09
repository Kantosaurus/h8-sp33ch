"""
Voting Classifier - Hate Speech Classification
Model: Ensemble voting combining multiple diverse base estimators
Performance: Soft/hard voting with optimized base estimators
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")


class VotingHateSpeechClassifier:
    def __init__(
        self,
        voting="soft",  # 'soft' or 'hard'
        use_dimensionality_reduction=True,
        reduction_method="truncatedsvd",  # 'pca', 'truncatedsvd'
        n_components=500,
        estimator_weights=None,
    ):
        """
        Voting Classifier combining multiple diverse estimators

        Args:
            voting: Voting strategy ('soft' for probabilities, 'hard' for predictions)
            use_dimensionality_reduction: Whether to apply dimensionality reduction
            reduction_method: Type of reduction ('pca', 'truncatedsvd')
            n_components: Number of components for dimensionality reduction
            estimator_weights: Weights for each estimator (None for equal weights)
        """
        self.voting = voting
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.estimator_weights = estimator_weights

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

        # Dimensionality reduction
        self.reducer = None
        if use_dimensionality_reduction:
            if reduction_method == "pca":
                self.reducer = PCA(n_components=n_components, random_state=42)
            elif reduction_method == "truncatedsvd":
                self.reducer = TruncatedSVD(n_components=n_components, random_state=42)
            else:
                raise ValueError(f"Unknown reduction method: {reduction_method}")

        # Base estimators for voting
        self.base_estimators = self._create_base_estimators()

        # Voting Classifier
        self.classifier = VotingClassifier(
            estimators=self.base_estimators,
            voting=voting,
            weights=estimator_weights,
            n_jobs=-1,
        )

        # Store training metrics
        self.training_metrics = {}

    def _create_base_estimators(self):
        """Create diverse base estimators for voting"""
        estimators = [
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",  # Works well with sparse data
                    random_state=42,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,  # Required for soft voting
                    random_state=42,
                ),
            ),
            ("naive_bayes", MultinomialNB(alpha=0.5, fit_prior=True)),
        ]

        return estimators

    def preprocess_data(self, texts):
        """Apply TF-IDF vectorization and optional dimensionality reduction"""
        X_tfidf = self.vectorizer.transform(texts)

        if self.reducer is not None:
            if self.reduction_method == "pca":
                X_reduced = self.reducer.transform(X_tfidf.toarray())
            else:  # truncatedsvd
                X_reduced = self.reducer.transform(X_tfidf)
            return X_reduced
        else:
            return X_tfidf

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Voting classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("Training Voting Classifier...")
        start_time = time.time()

        # Transform texts to TF-IDF
        print("Applying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Apply dimensionality reduction if specified
        if self.reducer is not None:
            print(
                f"Applying {self.reduction_method.upper()} (reducing to {self.n_components} components)..."
            )
            if self.reduction_method == "pca":
                X_train_reduced = self.reducer.fit_transform(X_train_tfidf.toarray())
            else:  # truncatedsvd
                X_train_reduced = self.reducer.fit_transform(X_train_tfidf)

            print(
                f"Dimensionality reduced from {X_train_tfidf.shape[1]} to {X_train_reduced.shape[1]} features"
            )

            if hasattr(self.reducer, "explained_variance_ratio_"):
                print(
                    f"Explained variance ratio: {self.reducer.explained_variance_ratio_.sum():.4f}"
                )

            X_train_processed = X_train_reduced
        else:
            X_train_processed = X_train_tfidf
            print(
                f"No dimensionality reduction applied. Using {X_train_tfidf.shape[1]} features"
            )

        # Train Voting classifier
        print(
            f"Training Voting Classifier with {len(self.base_estimators)} estimators..."
        )
        print(f"Base estimators: {[name for name, _ in self.base_estimators]}")
        print(f"Voting strategy: {self.voting}")

        self.classifier.fit(X_train_processed, y_train)

        # Individual estimator training info
        print("\nIndividual estimator details:")
        for name, estimator in self.classifier.named_estimators_.items():
            print(f"  {name}: {type(estimator).__name__}")

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
        """Get prediction probabilities (only for soft voting)"""
        if self.voting == "hard":
            raise ValueError("predict_proba is only available for soft voting")

        X_processed = self.preprocess_data(X)
        return self.classifier.predict_proba(X_processed)

    def get_individual_predictions(self, X):
        """Get predictions from each individual estimator"""
        X_processed = self.preprocess_data(X)

        individual_preds = {}
        for name, estimator in self.classifier.named_estimators_.items():
            preds = estimator.predict(X_processed)
            individual_preds[name] = preds

        return individual_preds

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")

        # Transform data
        X_tfidf = self.vectorizer.fit_transform(X)

        if self.reducer is not None:
            if self.reduction_method == "pca":
                X_processed = self.reducer.fit_transform(X_tfidf.toarray())
            else:  # truncatedsvd
                X_processed = self.reducer.fit_transform(X_tfidf)
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
        print("Starting hyperparameter tuning for individual estimators...")

        # Transform data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        if self.reducer is not None:
            if self.reduction_method == "pca":
                X_train_processed = self.reducer.fit_transform(X_train_tfidf.toarray())
            else:  # truncatedsvd
                X_train_processed = self.reducer.fit_transform(X_train_tfidf)
        else:
            X_train_processed = X_train_tfidf

        # Parameter grid for voting classifier
        param_grid = {
            "logistic__C": [0.1, 1.0, 10.0],
            "random_forest__n_estimators": [50, 100, 150],
            "random_forest__max_depth": [10, 15, 20],
            "svm__C": [1.0, 10.0, 100.0],
            "svm__gamma": ["scale", "auto", 0.001, 0.01],
            "naive_bayes__alpha": [0.1, 0.5, 1.0],
        }

        grid_search = GridSearchCV(
            self.classifier,
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
        """Get feature importance from estimators that support it"""
        if not hasattr(self.classifier, "named_estimators_"):
            print("Model not trained yet!")
            return None

        # Collect feature importances from supporting estimators
        importance_dict = {}

        # Random Forest importance
        if "random_forest" in self.classifier.named_estimators_:
            rf_estimator = self.classifier.named_estimators_["random_forest"]
            if hasattr(rf_estimator, "feature_importances_"):
                importance_dict["random_forest"] = rf_estimator.feature_importances_

        # Logistic Regression coefficients
        if "logistic" in self.classifier.named_estimators_:
            lr_estimator = self.classifier.named_estimators_["logistic"]
            if hasattr(lr_estimator, "coef_"):
                # Use absolute values of coefficients
                importance_dict["logistic"] = np.abs(lr_estimator.coef_[0])

        # SVM coefficients (if linear kernel)
        if "svm" in self.classifier.named_estimators_:
            svm_estimator = self.classifier.named_estimators_["svm"]
            if hasattr(svm_estimator, "coef_") and svm_estimator.kernel == "linear":
                importance_dict["svm"] = np.abs(svm_estimator.coef_[0])

        if not importance_dict:
            print("No feature importance information available from any estimator")
            return None

        # Average importance across available estimators
        if len(importance_dict) > 1:
            avg_importances = np.mean(list(importance_dict.values()), axis=0)
            importance_source = "averaged"
        else:
            avg_importances = list(importance_dict.values())[0]
            importance_source = list(importance_dict.keys())[0]

        if feature_names is None:
            if self.reducer is not None:
                if self.reduction_method == "pca":
                    feature_names = [f"PC_{i+1}" for i in range(len(avg_importances))]
                else:
                    feature_names = [f"SVD_{i+1}" for i in range(len(avg_importances))]
            else:
                feature_names = self.vectorizer.get_feature_names_out()

        # Get top important features
        indices = np.argsort(avg_importances)[::-1][:top_n]

        print(f"\nTop {top_n} most important features ({importance_source}):")
        print("-" * 70)
        for i, idx in enumerate(indices):
            print(
                f"{i+1:2d}. {feature_names[idx][:40]:<40} : {avg_importances[idx]:8.4f}"
            )

        return dict(zip([feature_names[i] for i in indices], avg_importances[indices]))

    def get_estimator_agreement(self, X, y=None):
        """Analyze agreement between estimators"""
        if not hasattr(self.classifier, "named_estimators_"):
            print("Model not trained yet!")
            return None

        individual_preds = self.get_individual_predictions(X)

        # Calculate pairwise agreement
        estimator_names = list(individual_preds.keys())
        n_estimators = len(estimator_names)

        agreement_matrix = np.zeros((n_estimators, n_estimators))

        for i, name1 in enumerate(estimator_names):
            for j, name2 in enumerate(estimator_names):
                if i <= j:
                    if i == j:
                        agreement_matrix[i, j] = 1.0
                    else:
                        agreement = np.mean(
                            individual_preds[name1] == individual_preds[name2]
                        )
                        agreement_matrix[i, j] = agreement
                        agreement_matrix[j, i] = agreement

        print(f"\nEstimator Agreement Matrix:")
        print("=" * 60)
        print(f"{'':>12s}", end="")
        for name in estimator_names:
            print(f"{name[:10]:>10s}", end="")
        print()

        for i, name in enumerate(estimator_names):
            print(f"{name[:10]:>10s}: ", end="")
            for j in range(n_estimators):
                print(f"{agreement_matrix[i, j]:8.3f} ", end="")
            print()

        # Overall agreement statistics
        off_diagonal = agreement_matrix[np.triu_indices(n_estimators, k=1)]
        print(f"\nOverall Agreement Statistics:")
        print(f"Mean pairwise agreement: {off_diagonal.mean():.4f}")
        print(f"Std pairwise agreement: {off_diagonal.std():.4f}")
        print(f"Min pairwise agreement: {off_diagonal.min():.4f}")
        print(f"Max pairwise agreement: {off_diagonal.max():.4f}")

        return {
            "agreement_matrix": agreement_matrix,
            "estimator_names": estimator_names,
            "mean_agreement": float(off_diagonal.mean()),
            "std_agreement": float(off_diagonal.std()),
        }

    def save_model(self, model_path, vectorizer_path, reducer_path=None):
        """Save the trained model components"""
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        if self.reducer is not None and reducer_path is not None:
            with open(reducer_path, "wb") as f:
                pickle.dump(self.reducer, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        if reducer_path:
            print(f"Dimensionality reducer saved to {reducer_path}")


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
    print("VOTING ENSEMBLE HATE SPEECH CLASSIFICATION")
    print("=" * 60)

    # Set up paths
    data_dir = "../../data"
    output_dir = "voting_outputs"
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
    print(f"\nInitializing Voting Classifier...")
    print(f"Configuration:")
    print(f"  - Voting strategy: Soft")
    print(f"  - Dimensionality reduction: TruncatedSVD")
    print(f"  - Components: 500")
    print(f"  - Base estimators: LogisticRegression, RandomForest, SVM, NaiveBayes")

    model = VotingHateSpeechClassifier(
        voting="soft",
        use_dimensionality_reduction=True,
        reduction_method="truncatedsvd",
        n_components=500,
        estimator_weights=None,  # Equal weights
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Cross-validation on full training data
    cv_scores = model.cross_validate(X_train_full, y_train_full, cv_folds=5)

    # Get model insights
    feature_importance = model.get_feature_importance(top_n=15)
    agreement_info = model.get_estimator_agreement(X_val)

    # Individual estimator predictions on validation set
    print(f"\nAnalyzing individual estimator performance...")
    individual_preds = model.get_individual_predictions(X_val)

    print(f"Individual estimator accuracies on validation set:")
    for name, preds in individual_preds.items():
        acc = accuracy_score(y_val, preds)
        print(f"  {name}: {acc:.4f}")

    # Validation predictions
    print("\nEvaluating ensemble on validation set...")
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, val_predictions, average="weighted"
    )

    print(f"\nEnsemble Validation Results:")
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

    submission_path = os.path.join(output_dir, "submission_voting.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    # Save model components
    model_path = os.path.join(output_dir, "voting_model.pkl")
    vectorizer_path = os.path.join(output_dir, "voting_vectorizer.pkl")
    reducer_path = os.path.join(output_dir, "voting_reducer.pkl")

    model.save_model(model_path, vectorizer_path, reducer_path)

    # Save metrics
    metrics = {
        "Model": "voting_ensemble",
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "F1": float(val_f1),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "Voting_Strategy": model.voting,
        "Reduction_Method": model.reduction_method,
        "N_Components": model.n_components,
        "Base_Estimators": [name for name, _ in model.base_estimators],
    }

    if model.reducer and hasattr(model.reducer, "explained_variance_ratio_"):
        metrics["Explained_Variance"] = float(
            model.reducer.explained_variance_ratio_.sum()
        )

    if agreement_info:
        metrics["Mean_Estimator_Agreement"] = agreement_info["mean_agreement"]
        metrics["Std_Estimator_Agreement"] = agreement_info["std_agreement"]

    # Individual estimator accuracies
    individual_accs = {}
    for name, preds in individual_preds.items():
        individual_accs[f"{name}_accuracy"] = float(accuracy_score(y_val, preds))
    metrics.update(individual_accs)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Print final summary
    print(f"\n" + "=" * 60)
    print("VOTING ENSEMBLE SUMMARY")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"Voting Strategy: {model.voting}")
    print(f"Base Estimators: {len(model.base_estimators)}")
    if model.reducer:
        print(f"Dimensionality Reduction: {model.reduction_method.upper()}")
        print(f"Components: {model.n_components}")
        if hasattr(model.reducer, "explained_variance_ratio_"):
            print(
                f"Explained Variance: {model.reducer.explained_variance_ratio_.sum():.4f}"
            )
    if agreement_info:
        print(f"Mean Estimator Agreement: {agreement_info['mean_agreement']:.4f}")
    print(f"Test Predictions: {len(test_predictions)} samples")

    # Print prediction distribution
    unique, counts = np.unique(test_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    if len(unique) == 2:
        hate_ratio = counts[1] / len(test_predictions) if 1 in unique else 0
        print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    print(f"\n✅ Voting ensemble training and prediction completed!")


if __name__ == "__main__":
    main()
