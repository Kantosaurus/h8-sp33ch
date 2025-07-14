import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class LogisticRegression:
    """
    Logistic Regression implementation from scratch
    """
    def __init__(self):
        self.weights = None
        self.bias = None
        self.cost_history = []

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def loss(y, y_hat):
        """
        Log loss function
        """
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        m = y.shape[0]
        return -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @staticmethod
    def gradients(X, y, y_hat):
        """
        Compute gradients for weights and bias
        """
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        return dw, db

    def train(self, X, y, bs=64, epochs=10, lr=0.01):
        """
        Train the logistic regression model using mini-batch gradient descent
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []

        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, m, bs):
                end = min(start + bs, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                z = np.dot(X_batch, self.weights) + self.bias
                y_hat = self.sigmoid(z)

                dw, db = self.gradients(X_batch, y_batch, y_hat)

                self.weights -= lr * dw
                self.bias -= lr * db

            # Compute loss for the whole dataset at the end of each epoch
            y_hat_full = self.sigmoid(np.dot(X, self.weights) + self.bias)
            cost = self.loss(y, y_hat_full)
            self.cost_history.append(cost)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {cost:.6f}")

        print(f"Training completed after {epochs} epochs.")
        return self

    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        """
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        return (probabilities >= threshold).astype(int)

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def plot_cost_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def load_and_prepare_data():
    """
    Load and prepare the training and test data
    """
    print("Loading training data...")
    train_data = pd.read_csv('data/train_tfidf_features.csv')
    
    print("Loading test data...")
    test_data = pd.read_csv('data/test_tfidf_features.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Separate features and labels
    X_train = train_data.drop(['id', 'label'], axis=1).values
    y_train = train_data['label'].values
    
    X_test = test_data.drop(['id'], axis=1).values
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Check for any missing values
    print(f"Missing values in X_train: {np.isnan(X_train).sum()}")
    print(f"Missing values in y_train: {np.isnan(y_train).sum()}")
    print(f"Missing values in X_test: {np.isnan(X_test).sum()}")
    
    # Handle any missing values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    return X_train, y_train, X_test, train_data['id'].values, test_data['id'].values

def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate the model and print metrics
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n{dataset_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Non-Hateful', 'Hateful']))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Hateful', 'Hateful'],
                yticklabels=['Non-Hateful', 'Hateful'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy, y_pred

def create_submission(predictions, test_ids, filename='submission.csv'):
    """
    Create submission file in the required format
    """
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    submission_df.to_csv(filename, index=False)
    print(f"Submission file saved as {filename}")
    
    # Show first few predictions
    print("\nFirst 10 predictions:")
    print(submission_df.head(10))

def main():
    print("=== Hate Speech Detection using Logistic Regression ===")
    X_train, y_train, X_test, train_ids, test_ids = load_and_prepare_data()
    print(f"\nClass distribution in training data:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples ({count/len(y_train)*100:.2f}%)")

    print("\n=== Training Logistic Regression Model ===")
    model = LogisticRegression()
    # You can adjust batch size, epochs, and learning rate as needed
    model.train(X_train, y_train, bs=64, epochs=10, lr=0.01)
    model.plot_cost_history()
    train_accuracy, train_predictions = evaluate_model(model, X_train, y_train, "Training")
    print("\n=== Making Predictions on Test Data ===")
    test_predictions = model.predict(X_test)
    create_submission(test_predictions, test_ids, 'logistic_regression_submission.csv')
    print("\n=== Model Summary ===")
    print(f"Final training accuracy: {train_accuracy:.4f}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")
    print("\n=== Example Predictions ===")
    sample_indices = np.random.choice(len(X_train), 10, replace=False)
    sample_X = X_train[sample_indices]
    sample_y = y_train[sample_indices]
    sample_pred = model.predict(sample_X)
    sample_proba = model.predict_proba(sample_X)
    for i, (true_label, pred_label, proba) in enumerate(zip(sample_y, sample_pred, sample_proba)):
        print(f"Sample {i+1}: True={true_label}, Pred={pred_label}, Probability={proba:.4f}")

if __name__ == "__main__":
    main() 