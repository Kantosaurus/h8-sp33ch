"""
K Neighbors Classifier (knn) - Hate Speech Classification
Model: K-Nearest Neighbors Classifier
Performance: Not specified in benchmark (listed with dashes)
"""

import pandas as pd
import numpy as np
import os
import warnings

# Set environment variable to silence joblib CPU warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

# Suppress the specific joblib warning
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pickle


class KNNHateSpeechClassifier:
    def __init__(self):
        """Initialize KNN Classifier with optimized hyperparameters"""
        # Key hyperparameters for KNN
        self.model = KNeighborsClassifier(
            n_neighbors=5,  # Number of neighbors to use
            weights="uniform",  # Weight function ('uniform' or 'distance')
            algorithm="auto",  # Algorithm to compute nearest neighbors
            leaf_size=30,  # Leaf size for BallTree or KDTree
            p=2,  # Power parameter for Minkowski metric
            metric="minkowski",  # Distance metric
            n_jobs=-1,  # Use all available processors
        )

        # TF-IDF Vectorizer for text preprocessing (reduced features for KNN)
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced features for KNN (memory/speed)
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

        # SVD for dimensionality reduction (important for KNN performance)
        self.svd = TruncatedSVD(
            n_components=200, random_state=42  # Reduce to 200 dimensions
        )

        # StandardScaler for feature scaling (important for KNN)
        self.scaler = StandardScaler()

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
        """Preprocess text data using TF-IDF vectorization, SVD, and scaling"""
        print("Preprocessing text data...")

        if fit_on_train:
            # Fit vectorizer on training data and transform
            X_train_tfidf = self.vectorizer.fit_transform(self.train_data["post"])
            print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

            # Apply SVD for dimensionality reduction
            X_train_svd = self.svd.fit_transform(X_train_tfidf)
            print(f"SVD reduced feature matrix shape: {X_train_svd.shape}")

            # Apply standard scaling
            X_train = self.scaler.fit_transform(X_train_svd)
            print(f"Scaled feature matrix shape: {X_train.shape}")
        else:
            # Only transform (all preprocessing already fitted)
            X_train_tfidf = self.vectorizer.transform(self.train_data["post"])
            X_train_svd = self.svd.transform(X_train_tfidf)
            X_train = self.scaler.transform(X_train_svd)

        y_train = self.train_data["label"].values

        if hasattr(self, "test_data"):
            X_test_tfidf = self.vectorizer.transform(self.test_data["post"])
            X_test_svd = self.svd.transform(X_test_tfidf)
            X_test = self.scaler.transform(X_test_svd)
            return X_train, y_train, X_test

        return X_train, y_train

    def train(self, X_train, y_train, validation_split=0.2):
        """Train the KNN Classifier"""
        print("Training K-Nearest Neighbors Classifier...")

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

        # Train the model (KNN is lazy learning - just stores training data)
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
        print(f"Training data stored: {X_train_split.shape[0]} samples")

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")

        # Parameter grid for tuning
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 50],
            "p": [1, 2],  # 1: Manhattan, 2: Euclidean
            "metric": ["minkowski", "cosine", "manhattan"],
        }

        # Initialize base model for grid search
        base_model = KNeighborsClassifier(n_jobs=-1)

        # Perform grid search (warning: can be slow for KNN)
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

    def get_neighbors(self, X_test, n_neighbors=None):
        """Get k-nearest neighbors for test samples"""
        if n_neighbors is None:
            n_neighbors = self.model.n_neighbors

        distances, indices = self.model.kneighbors(X_test, n_neighbors=n_neighbors)
        return distances, indices

    def analyze_predictions(self, X_test, y_test=None, sample_indices=None):
        """Analyze predictions by examining nearest neighbors"""
        if sample_indices is None:
            sample_indices = [0, 1, 2]  # Analyze first few samples

        predictions = self.model.predict(X_test[sample_indices])
        probabilities = self.model.predict_proba(X_test[sample_indices])
        distances, neighbor_indices = self.get_neighbors(X_test[sample_indices])

        for i, sample_idx in enumerate(sample_indices):
            print(f"\n--- Sample {sample_idx} ---")
            print(f"Prediction: {predictions[i]}")
            print(f"Probability: {probabilities[i]}")

            if y_test is not None:
                print(f"True label: {y_test[sample_idx]}")

            print(
                f"Distances to {self.model.n_neighbors} nearest neighbors: {distances[i]}"
            )
            print(f"Neighbor indices: {neighbor_indices[i]}")

    def get_model_info(self):
        """Get information about the trained model"""
        print(f"Number of neighbors (k): {self.model.n_neighbors}")
        print(f"Weight function: {self.model.weights}")
        print(f"Algorithm: {self.model.algorithm}")
        print(f"Distance metric: {self.model.metric}")
        print(f"SVD components: {self.svd.n_components}")
        print(
            f"SVD explained variance ratio sum: {self.svd.explained_variance_ratio_.sum():.4f}"
        )
        print(f"Feature scaling: StandardScaler applied")

        if hasattr(self.model, "_fit_X"):
            print(f"Training data shape: {self.model._fit_X.shape}")

    def save_model(
        self,
        model_path=None,
        vectorizer_path=None,
        svd_path=None,
        scaler_path=None,
    ):
        """Save trained model, vectorizer, SVD reducer, and scaler"""
        # Create model-specific directory
        model_dir = "knn_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "knn_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if svd_path is None:
            svd_path = os.path.join(model_dir, "svd_reducer.pkl")
        if scaler_path is None:
            scaler_path = os.path.join(model_dir, "standard_scaler.pkl")

        print(f"Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Saving SVD reducer to {svd_path}")
        with open(svd_path, "wb") as f:
            pickle.dump(self.svd, f)

        print(f"Saving scaler to {scaler_path}")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

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
        scaler_path=None,
    ):
        """Load trained model, vectorizer, SVD reducer, and scaler"""
        # Use model-specific directory
        model_dir = "knn_outputs"

        # Set default paths within the model directory
        if model_path is None:
            model_path = os.path.join(model_dir, "knn_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if svd_path is None:
            svd_path = os.path.join(model_dir, "svd_reducer.pkl")
        if scaler_path is None:
            scaler_path = os.path.join(model_dir, "standard_scaler.pkl")

        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        print(f"Loading SVD reducer from {svd_path}")
        with open(svd_path, "rb") as f:
            self.svd = pickle.load(f)

        print(f"Loading scaler from {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def create_submission(self, predictions, test_ids, output_path=None):
        """Create submission file for Kaggle"""
        # Create model-specific directory if not exists
        model_dir = "knn_outputs"
        os.makedirs(model_dir, exist_ok=True)

        # Set default output path within the model directory
        if output_path is None:
            output_path = os.path.join(model_dir, "submission_knn.csv")

        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    knn_classifier = KNNHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = knn_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = knn_classifier.preprocess_text()

    # Option 1: Train with default parameters
    knn_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes much longer for KNN)
    # best_params, best_score = knn_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = knn_classifier.predict(X_test)

    # Get model information
    knn_classifier.get_model_info()

    # Create submission file
    test_ids = test_data["id"].values
    submission = knn_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    knn_classifier.save_model()

    print("KNN Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
