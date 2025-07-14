import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

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

def apply_pca_and_evaluate(X_train, y_train, X_test, n_components_list):
    """
    Apply PCA with different numbers of components and evaluate using KNN
    """
    results = {}
    
    for n_components in n_components_list:
        print(f"\n=== PCA with {n_components} components ===")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        print(f"Original feature space: {X_train.shape[1]} dimensions")
        print(f"Reduced feature space: {X_train_pca.shape[1]} dimensions")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Train KNN model
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X_train_pca, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test_pca)
        
        # Store results
        results[n_components] = {
            'pca': pca,
            'knn': knn,
            'X_train_pca': X_train_pca,
            'X_test_pca': X_test_pca,
            'predictions': y_pred,
            'explained_variance_ratio': pca.explained_variance_ratio_.sum()
        }
        
        print(f"PCA with {n_components} components completed")
    
    return results

def create_submission(predictions, test_ids, filename):
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
    print(f"\nFirst 10 predictions for {filename}:")
    print(submission_df.head(10))

def plot_explained_variance_ratio(results):
    """
    Plot explained variance ratio for different numbers of components
    """
    components = list(results.keys())
    explained_variances = [results[comp]['explained_variance_ratio'] for comp in components]
    
    plt.figure(figsize=(10, 6))
    plt.plot(components, explained_variances, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (comp, var) in enumerate(zip(components, explained_variances)):
        plt.annotate(f'{var:.3f}', (comp, var), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_cumulative_explained_variance(X_train, max_components=2000):
    """
    Plot cumulative explained variance to help understand the optimal number of components
    """
    # Fit PCA with maximum components to get all explained variance ratios
    pca_full = PCA(n_components=min(max_components, X_train.shape[1]), random_state=42)
    pca_full.fit(X_train)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    components_range = range(1, len(cumulative_variance) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(components_range, cumulative_variance, 'b-', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs Number of Components')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines for our target components
    target_components = [100, 500, 1000, 2000]
    colors = ['red', 'orange', 'green', 'purple']
    
    for comp, color in zip(target_components, colors):
        if comp <= len(cumulative_variance):
            var_ratio = cumulative_variance[comp - 1]
            plt.axhline(y=var_ratio, color=color, linestyle='--', alpha=0.7, 
                       label=f'{comp} components: {var_ratio:.3f}')
            plt.axvline(x=comp, color=color, linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("=== Task 2: Dimension Reduction using PCA ===")
    print("Loading and preparing data...")
    X_train, y_train, X_test, train_ids, test_ids = load_and_prepare_data()
    
    print(f"\nClass distribution in training data:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples ({count/len(y_train)*100:.2f}%)")
    
    # List of components to test
    n_components_list = [2000, 1000, 500, 100]
    
    print(f"\n=== Applying PCA with different numbers of components ===")
    print(f"Testing with components: {n_components_list}")
    
    # Apply PCA and evaluate
    results = apply_pca_and_evaluate(X_train, y_train, X_test, n_components_list)
    
    print(f"\n=== Creating submission files ===")
    # Create submission files for each number of components
    for n_components in n_components_list:
        filename = f'pca_{n_components}_components_submission.csv'
        create_submission(results[n_components]['predictions'], test_ids, filename)
    
    print(f"\n=== Results Summary ===")
    print("Number of Components | Explained Variance Ratio")
    print("-" * 45)
    for n_components in n_components_list:
        var_ratio = results[n_components]['explained_variance_ratio']
        print(f"{n_components:^19} | {var_ratio:^23.4f}")
    
    print(f"\n=== Visualization ===")
    # Plot explained variance ratio
    plot_explained_variance_ratio(results)
    
    # Plot cumulative explained variance
    plot_cumulative_explained_variance(X_train)
    
    print(f"\n=== Task 2 Complete ===")
    print("Submission files have been created for each number of components.")
    print("Please submit these files to Kaggle to get the Macro F1 scores:")
    for n_components in n_components_list:
        print(f"- pca_{n_components}_components_submission.csv")
    
    print(f"\nNote: The Macro F1 scores will be available after submitting to Kaggle.")
    print(f"Use these scores in your final report to compare the performance")
    print(f"of different numbers of PCA components.")

if __name__ == "__main__":
    main() 