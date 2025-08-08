"""
Decision Tree Classifier (dt) - Hate Speech Classification
Model: Decision Tree Classifier with hyperparameter tuning
Performance: 75.67% accuracy according to benchmark
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class DecisionTreeHateSpeechClassifier:
    def __init__(self):
        """Initialize Decision Tree Classifier with optimized hyperparameters"""
        # Key hyperparameters based on benchmark performance
        self.model = DecisionTreeClassifier(
            max_depth=20,  # Maximum depth of the tree
            min_samples_split=10,  # Minimum samples required to split
            min_samples_leaf=5,  # Minimum samples required at leaf
            max_features="sqrt",  # Number of features to consider for best split
            criterion="gini",  # Measure of impurity
            splitter="best",  # Strategy to choose split at each node
            random_state=42,  # For reproducibility
            max_leaf_nodes=None,  # Maximum number of leaf nodes
            min_impurity_decrease=0.0,  # Minimum impurity decrease for split
        )

        # TF-IDF Vectorizer for text preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=8000,  # Maximum number of features (reduced for tree)
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
        """Train the Decision Tree Classifier"""
        print("Training Decision Tree Classifier...")

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

        # Validate if validation split provided
        if validation_split > 0:
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_predictions))

        print("Training completed!")
        print(f"Tree depth: {self.model.tree_.max_depth}")
        print(f"Number of leaves: {self.model.tree_.n_leaves}")
        print(f"Number of nodes: {self.model.tree_.node_count}")

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")

        # Parameter grid for tuning
        param_grid = {
            "max_depth": [10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy"],
            "min_impurity_decrease": [0.0, 0.001, 0.01],
        }

        # Initialize base model for grid search
        base_model = DecisionTreeClassifier(random_state=42, splitter="best")

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

    def visualize_tree(
        self, max_depth=3, output_path="decision_tree_visualization.png"
    ):
        """Visualize the decision tree (limited depth for readability)"""
        try:
            from sklearn.tree import plot_tree
            import matplotlib.pyplot as plt

            plt.figure(figsize=(20, 10))

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Plot tree with limited depth
            plot_tree(
                self.model,
                max_depth=max_depth,
                feature_names=feature_names,
                class_names=["Non-Hate", "Hate"],
                filled=True,
                rounded=True,
                fontsize=8,
            )

            plt.title(f"Decision Tree Visualization (Max Depth: {max_depth})")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.show()

            print(f"Tree visualization saved to {output_path}")

        except ImportError:
            print("Matplotlib not available for tree visualization")
        except Exception as e:
            print(f"Could not visualize tree: {e}")

    def get_tree_rules(self, max_rules=10):
        """Extract decision tree rules"""
        try:
            from sklearn.tree import export_text

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Export tree rules
            tree_rules = export_text(
                self.model,
                feature_names=feature_names,
                max_depth=5,  # Limit depth for readability
                show_weights=True,
            )

            print("Decision Tree Rules (truncated for readability):")
            print(tree_rules[:2000] + "..." if len(tree_rules) > 2000 else tree_rules)

            return tree_rules

        except Exception as e:
            print(f"Could not extract tree rules: {e}")
            return None

    def save_model(
        self,
        model_path="decision_tree_model.pkl",
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
        model_path="decision_tree_model.pkl",
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
        self, predictions, test_ids, output_path="submission_decision_tree.csv"
    ):
        """Create submission file for Kaggle"""
        submission = pd.DataFrame({"row ID": test_ids, "label": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission


def main():
    """Main execution function"""
    # Initialize classifier
    dt_classifier = DecisionTreeHateSpeechClassifier()

    # Define file paths
    train_path = "../../data/train.csv"
    test_path = "../../data/test.csv"

    # Load data
    train_data, test_data = dt_classifier.load_data(train_path, test_path)

    # Preprocess data
    X_train, y_train, X_test = dt_classifier.preprocess_text()

    # Option 1: Train with default parameters
    dt_classifier.train(X_train, y_train)

    # Option 2: Uncomment for hyperparameter tuning (takes longer)
    # best_params, best_score = dt_classifier.hyperparameter_tuning(X_train, y_train)

    # Make predictions
    test_predictions = dt_classifier.predict(X_test)

    # Get feature importance
    feature_importance = dt_classifier.get_feature_importance()

    # Visualize tree (optional)
    # dt_classifier.visualize_tree()

    # Get tree rules
    dt_classifier.get_tree_rules()

    # Create submission file
    test_ids = test_data["id"].values
    submission = dt_classifier.create_submission(test_predictions, test_ids)

    # Save model for future use
    dt_classifier.save_model()

    print("Decision Tree Classifier training and prediction completed!")
    print(f"Prediction distribution: {np.bincount(test_predictions)}")


if __name__ == "__main__":
    main()
