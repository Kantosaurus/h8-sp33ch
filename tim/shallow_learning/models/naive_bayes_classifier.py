"""
Naive Bayes Classifiers - Hate Speech Classification
Model: Multinomial, Complement, and Bernoulli Naive Bayes
Performance: Probabilistic classifiers for text classification
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")


class NaiveBayesHateSpeechClassifier:
    def __init__(
        self,
        nb_type="multinomial",  # 'multinomial', 'complement', 'bernoulli'
        vectorizer_type="tfidf",  # 'tfidf', 'count'
        alpha=1.0,
    ):
        """
        Naive Bayes Classifier variants

        Args:
            nb_type: Type of Naive Bayes ('multinomial', 'complement', 'bernoulli')
            vectorizer_type: Type of vectorizer ('tfidf', 'count')
            alpha: Smoothing parameter
        """
        self.nb_type = nb_type
        self.vectorizer_type = vectorizer_type
        self.alpha = alpha

        # Choose vectorizer based on Naive Bayes type
        if vectorizer_type == "count" or nb_type == "multinomial":
            # Count vectorizer is more traditional for Naive Bayes
            self.vectorizer = CountVectorizer(
                max_features=15000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True,
                strip_accents="ascii",
                binary=(nb_type == "bernoulli"),  # Binary features for Bernoulli NB
            )
        else:
            # TF-IDF can also work well
            self.vectorizer = TfidfVectorizer(
                max_features=15000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True,
                strip_accents="ascii",
            )

        # Choose Naive Bayes classifier
        if nb_type == "multinomial":
            self.classifier = MultinomialNB(alpha=alpha)
        elif nb_type == "complement":
            self.classifier = ComplementNB(alpha=alpha)
        elif nb_type == "bernoulli":
            self.classifier = BernoulliNB(alpha=alpha)
        else:
            raise ValueError(f"Unknown Naive Bayes type: {nb_type}")

        # Store training metrics
        self.training_metrics = {}

    def preprocess_data(self, texts):
        """Apply vectorization"""
        return self.vectorizer.transform(texts)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Naive Bayes classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print(f"Training {self.nb_type.title()} Naive Bayes Classifier...")
        start_time = time.time()

        # Transform texts to vectors
        print(f"Applying {self.vectorizer_type.upper()} vectorization...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"Feature matrix shape: {X_train_vec.shape}")
        print(
            f"Sparsity: {1.0 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]):.4f}"
        )

        # Train Naive Bayes
        print(f"Training {self.nb_type.title()} Naive Bayes...")
        self.classifier.fit(X_train_vec, y_train)

        # Validation performance
        if X_val is not None and y_val is not None:
            X_val_vec = self.preprocess_data(X_val)
            val_predictions = self.classifier.predict(X_val_vec)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation accuracy: {val_accuracy:.4f}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self

    def predict(self, X):
        """Make predictions on new data"""
        X_vec = self.preprocess_data(X)
        return self.classifier.predict(X_vec)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_vec = self.preprocess_data(X)
        return self.classifier.predict_proba(X_vec)

    def predict_log_proba(self, X):
        """Get log prediction probabilities"""
        X_vec = self.preprocess_data(X)
        return self.classifier.predict_log_proba(X_vec)

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")

        # Transform data
        X_vec = self.vectorizer.fit_transform(X)

        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier,
            X_vec,
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
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Parameter grid
        param_grid = {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}

        # Additional parameters for specific NB types
        if self.nb_type == "complement":
            param_grid["norm"] = [True, False]
        elif self.nb_type == "bernoulli":
            param_grid["binarize"] = [0.0, 0.5, 1.0]

        grid_search = GridSearchCV(
            type(self.classifier)(alpha=self.alpha),
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train_vec, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update classifier with best parameters
        self.classifier = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def get_feature_log_prob(self, feature_names=None, top_n=20):
        """Get feature log probabilities for interpretation"""
        if not hasattr(self.classifier, "feature_log_prob_"):
            print("Model not trained yet!")
            return None

        feature_log_prob = self.classifier.feature_log_prob_

        if feature_names is None:
            feature_names = self.vectorizer.get_feature_names_out()

        # Get features with highest log probability for each class
        results = {}
        for class_idx, class_probs in enumerate(feature_log_prob):
            # Get top features for this class
            top_indices = np.argsort(class_probs)[::-1][:top_n]
            top_features = [(feature_names[i], class_probs[i]) for i in top_indices]
            results[f"class_{class_idx}"] = top_features

            print(f"\nTop {top_n} features for class {class_idx}:")
            print("-" * 50)
            for i, (feature, log_prob) in enumerate(top_features):
                print(f"{i+1:2d}. {feature:<25} : {log_prob:8.4f}")

        return results

    def get_class_priors(self):
        """Get class prior probabilities"""
        if hasattr(self.classifier, "class_log_prior_"):
            priors = np.exp(self.classifier.class_log_prior_)
            print(f"\nClass prior probabilities:")
            for i, prior in enumerate(priors):
                print(f"Class {i}: {prior:.4f}")
            return priors
        return None

    def save_model(self, model_path, vectorizer_path):
        """Save the trained model components"""
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")


def load_data(train_path, test_path):
    """Load training and testing data"""
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, test_data


def run_naive_bayes_variant(nb_type, train_data, test_data, output_base_dir):
    """Run a specific Naive Bayes variant"""
    print(f"\n{'='*80}")
    print(f"TRAINING {nb_type.upper()} NAIVE BAYES")
    print(f"{'='*80}")

    # Create output directory
    output_dir = os.path.join(output_base_dir, f"{nb_type}_outputs")
    os.makedirs(output_dir, exist_ok=True)

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

    # Choose vectorizer type based on NB type
    vectorizer_type = "count" if nb_type in ["multinomial", "bernoulli"] else "tfidf"

    print(f"Configuration:")
    print(f"  - Naive Bayes type: {nb_type}")
    print(f"  - Vectorizer: {vectorizer_type}")
    print(f"  - Alpha (smoothing): 1.0")

    # Initialize and train model
    model = NaiveBayesHateSpeechClassifier(
        nb_type=nb_type,
        vectorizer_type=vectorizer_type,
        alpha=1.0,
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Cross-validation on full training data
    cv_scores = model.cross_validate(X_train_full, y_train_full, cv_folds=5)

    # Get class priors and feature probabilities
    class_priors = model.get_class_priors()
    feature_probs = model.get_feature_log_prob(top_n=10)

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

    submission_path = os.path.join(output_dir, f"submission_{nb_type}_nb.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    # Save model components
    model_path = os.path.join(output_dir, f"{nb_type}_nb_model.pkl")
    vectorizer_path = os.path.join(output_dir, f"{nb_type}_nb_vectorizer.pkl")

    model.save_model(model_path, vectorizer_path)

    # Save metrics
    metrics = {
        "Model": f"{nb_type}_nb",
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "F1": float(val_f1),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "NB_Type": nb_type,
        "Vectorizer_Type": vectorizer_type,
        "Alpha": model.alpha,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Print prediction distribution
    unique, counts = np.unique(test_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    if len(unique) == 2:
        hate_ratio = counts[1] / len(test_predictions) if 1 in unique else 0
        print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    return {
        "model_type": nb_type,
        "accuracy": val_accuracy,
        "f1": val_f1,
        "cv_mean": cv_scores.mean(),
    }


def main():
    """Main execution function"""
    print("=" * 80)
    print("NAIVE BAYES CLASSIFIERS HATE SPEECH CLASSIFICATION")
    print("=" * 80)

    # Set up paths
    data_dir = "../../data"
    output_base_dir = "naive_bayes_outputs"
    os.makedirs(output_base_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Load data
    train_data, test_data = load_data(train_path, test_path)

    # Run all three Naive Bayes variants
    nb_variants = ["multinomial", "complement", "bernoulli"]
    results = []

    for nb_type in nb_variants:
        result = run_naive_bayes_variant(
            nb_type, train_data, test_data, output_base_dir
        )
        results.append(result)

    # Print comparison summary
    print(f"\n{'='*80}")
    print("NAIVE BAYES VARIANTS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Variant':<15} {'Accuracy':<10} {'F1-Score':<10} {'CV Mean':<10}")
    print("-" * 50)

    for result in results:
        print(
            f"{result['model_type']:<15} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['cv_mean']:<10.4f}"
        )

    # Find best variant
    best_result = max(results, key=lambda x: x["f1"])
    print(
        f"\n🏆 Best Naive Bayes variant: {best_result['model_type']} (F1: {best_result['f1']:.4f})"
    )

    print(f"\n✅ All Naive Bayes classifiers training and prediction completed!")


if __name__ == "__main__":
    main()
