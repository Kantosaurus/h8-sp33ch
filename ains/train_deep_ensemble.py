import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from deep_ensemble_classifier import DeepEnsembleClassifier, train_deep_ensemble


def evaluate_model_performance(model, X_text, y_true, cv_folds=5):
    """
    Comprehensive model evaluation with cross-validation
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Cross-validation evaluation
    print("Performing cross-validation...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_text, y_true)):
        print(f"Fold {fold + 1}/{cv_folds}...")
        
        X_train_fold = X_text[train_idx]
        X_val_fold = X_text[val_idx]
        y_train_fold = y_true[train_idx]
        y_val_fold = y_true[val_idx]
        
        # Create fresh model for this fold
        fold_model = DeepEnsembleClassifier(
            n_hidden_layers=3,
            models_per_layer=6,
            random_state=42 + fold
        )
        
        # Train and evaluate
        fold_model.fit(X_train_fold, y_train_fold)
        fold_pred = fold_model.predict(X_val_fold)
        fold_score = f1_score(y_val_fold, fold_pred, average='macro')
        cv_scores.append(fold_score)
        
        print(f"Fold {fold + 1} Macro F1: {fold_score:.4f}")
    
    print(f"\nCross-Validation Results:")
    print(f"Mean Macro F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    print(f"Individual fold scores: {cv_scores}")
    
    return cv_scores


def generate_detailed_analysis(model, test_data, predictions, probabilities):
    """
    Generate detailed analysis of predictions
    """
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    # Prediction statistics
    unique, counts = np.unique(predictions, return_counts=True)
    pred_dist = dict(zip(unique, counts))
    
    print(f"Prediction Distribution:")
    print(f"Class 0 (Non-Hateful): {pred_dist.get(0, 0)} ({pred_dist.get(0, 0)/len(predictions)*100:.1f}%)")
    print(f"Class 1 (Hateful): {pred_dist.get(1, 0)} ({pred_dist.get(1, 0)/len(predictions)*100:.1f}%)")
    
    # Confidence analysis
    max_probs = np.max(probabilities, axis=1)
    
    print(f"\nConfidence Analysis:")
    print(f"Mean confidence: {np.mean(max_probs):.4f}")
    print(f"Median confidence: {np.median(max_probs):.4f}")
    print(f"High confidence (>0.8): {np.sum(max_probs > 0.8)} samples ({np.sum(max_probs > 0.8)/len(max_probs)*100:.1f}%)")
    print(f"Low confidence (<0.6): {np.sum(max_probs < 0.6)} samples ({np.sum(max_probs < 0.6)/len(max_probs)*100:.1f}%)")
    
    # Create detailed analysis dataframe
    analysis_df = pd.DataFrame({
        'id': test_data['id'],
        'post': test_data['post'],
        'predicted_label': predictions,
        'confidence': max_probs,
        'prob_non_hateful': probabilities[:, 0],
        'prob_hateful': probabilities[:, 1]
    })
    
    # Show high confidence predictions
    print(f"\nHigh Confidence Predictions (confidence > 0.9):")
    high_conf = analysis_df[analysis_df['confidence'] > 0.9].head(5)
    for idx, row in high_conf.iterrows():
        print(f"ID: {row['id']}, Label: {row['predicted_label']}, Conf: {row['confidence']:.3f}")
        print(f"Text: {row['post'][:100]}...")
        print()
    
    # Show uncertain predictions
    print(f"Uncertain Predictions (0.4 < confidence < 0.6):")
    uncertain = analysis_df[(analysis_df['confidence'] > 0.4) & (analysis_df['confidence'] < 0.6)].head(5)
    for idx, row in uncertain.iterrows():
        print(f"ID: {row['id']}, Label: {row['predicted_label']}, Conf: {row['confidence']:.3f}")
        print(f"Text: {row['post'][:100]}...")
        print()
    
    return analysis_df


def save_model_and_results(model, submission, analysis_df, model_filename='deep_ensemble_model.pkl'):
    """
    Save trained model and results
    """
    print(f"\nSaving model and results...")
    
    # Save model
    model_path = model_filename
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save predictions
    submission_path = "deep_ensemble_predictions.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Predictions saved to: {submission_path}")
    
    # Save detailed analysis
    analysis_path = "deep_ensemble_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"Analysis saved to: {analysis_path}")
    
    # Save network architecture summary
    summary = model.get_network_summary()
    summary_path = "network_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Deep Neural Network-Like Ensemble Summary\n")
        f.write("="*50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Network summary saved to: {summary_path}")


def main():
    """
    Main training and evaluation pipeline
    """
    print("DEEP NEURAL NETWORK-LIKE ENSEMBLE FOR HATE SPEECH DETECTION")
    print("="*80)
    
    # Data paths
    train_path = "../data/combined.csv"
    test_path = "../data/test.csv"
    
    # Check if data exists
    if not os.path.exists(train_path):
        train_path = "data/combined.csv"
        test_path = "data/test.csv"
    
    if not os.path.exists(train_path):
        print("Error: Could not find training data file")
        print("Expected: data/combined.csv or ../data/combined.csv")
        return
    
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    
    # Load data
    print("\nLoading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Data preprocessing
    print("\nPreprocessing data...")
    train_data = train_data.dropna(subset=['post', 'label'])
    train_data = train_data[train_data['post'].str.strip() != '']
    
    # Clean and convert labels to integers
    train_data['label'] = pd.to_numeric(train_data['label'], errors='coerce')
    train_data = train_data.dropna(subset=['label'])
    train_data['label'] = train_data['label'].astype(int)
    
    X_train_text = train_data['post'].values
    y_train = train_data['label'].values
    X_test_text = test_data['post'].fillna('').values
    
    print(f"Final training samples: {len(X_train_text)}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Create and train the deep ensemble
    print(f"\nCreating Deep Ensemble Classifier...")
    deep_ensemble = DeepEnsembleClassifier(
        n_hidden_layers=3,  # 3 hidden layers like a deep network
        models_per_layer=6,  # 6 models per layer
        random_state=42
    )
    
    # Train the model
    print(f"\nTraining deep ensemble...")
    deep_ensemble.fit(X_train_text, y_train)
    
    # Cross-validation evaluation (on subset for speed)
    if len(X_train_text) > 5000:
        # Use subset for CV evaluation to save time
        subset_indices = np.random.choice(len(X_train_text), 5000, replace=False)
        X_cv = X_train_text[subset_indices]
        y_cv = y_train[subset_indices]
        print(f"\nPerforming cross-validation on subset of {len(X_cv)} samples...")
    else:
        X_cv = X_train_text
        y_cv = y_train
        print(f"\nPerforming cross-validation on full dataset...")
    
    cv_scores = evaluate_model_performance(deep_ensemble, X_cv, y_cv, cv_folds=3)
    
    # Generate test predictions
    print(f"\nGenerating test predictions...")
    test_predictions = deep_ensemble.predict(X_test_text)
    test_probabilities = deep_ensemble.predict_proba(X_test_text)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_data['id'],
        'label': test_predictions
    })
    
    # Detailed analysis
    analysis_df = generate_detailed_analysis(
        deep_ensemble, test_data, test_predictions, test_probabilities
    )
    
    # Save everything
    save_model_and_results(deep_ensemble, submission, analysis_df)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Cross-validation Macro F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    print(f"Network Architecture: Input → 3 Hidden Layers → Output")
    print(f"Total Models Trained: {3 * 6 + 5} (18 hidden + 5 output)")
    print(f"Test Predictions: {len(test_predictions)} samples")
    print(f"Files generated:")
    print(f"  - deep_ensemble_model.pkl")
    print(f"  - deep_ensemble_predictions.csv")
    print(f"  - deep_ensemble_analysis.csv")
    print(f"  - network_summary.txt")
    print("="*80)
    
    return deep_ensemble, submission, analysis_df


if __name__ == "__main__":
    model, submission, analysis = main()