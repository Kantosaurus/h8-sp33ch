import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HateSpeechEnsemble:
    """
    Simple ensemble system for hate speech detection
    Uses: Logistic Regression and Multinomial Naive Bayes
    Works with pre-computed TF-IDF features
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.voting_classifier = None
        self.is_trained = False
        self.results = {}
        
    def initialize_models(self):
        """
        Initialize the 2 base models: Logistic Regression and Multinomial Naive Bayes
        """
        print("Initializing base models...")
        
        self.base_models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0,
                class_weight='balanced'
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0
            )
        }
        
        print(f"Initialized {len(self.base_models)} base models:")
        for name in self.base_models.keys():
            print(f"  - {name.replace('_', ' ').title()}")
        
    def load_data(self, train_tfidf_path='/Users/sathish.k/Downloads/Git HUB/h8-sp33ch/data/train_tfidf_features.csv', 
                  test_tfidf_path='/Users/sathish.k/Downloads/Git HUB/h8-sp33ch/data/test_tfidf_features.csv',
                  train_labels_path='/Users/sathish.k/Downloads/Git HUB/h8-sp33ch/data/train.csv',
                  test_ids_path='/Users/sathish.k/Downloads/Git HUB/h8-sp33ch/data/test.csv'):
        """
        Load pre-computed TF-IDF features and labels
        """
        print("Loading pre-computed TF-IDF features...")
        
        try:
            # Load TF-IDF features
            train_tfidf = pd.read_csv(train_tfidf_path)
            test_tfidf = pd.read_csv(test_tfidf_path)
            
            # Load labels and IDs
            train_data = pd.read_csv(train_labels_path)
            test_data = pd.read_csv(test_ids_path)
            
            # Extract features and labels
            X_train = train_tfidf.values
            X_test = test_tfidf.values
            y_train = train_data['label'].values
            test_ids = test_data['id'].values
            
            print(f"Data loaded successfully:")
            print(f"  Training features shape: {X_train.shape}")
            print(f"  Test features shape: {X_test.shape}")
            print(f"  Training labels: {len(y_train)}")
            print(f"  Test IDs: {len(test_ids)}")
            
            # Validate data
            if len(y_train) != X_train.shape[0]:
                raise ValueError("Mismatch between labels and training features")
            
            print(f"  Label distribution: {np.bincount(y_train)}")
            
            return X_train, X_test, y_train, test_ids
            
        except FileNotFoundError as e:
            print(f"Error: Required files not found. {e}")
            print("Please ensure the following files exist:")
            print(f"  - {train_tfidf_path}")
            print(f"  - {test_tfidf_path}")
            print(f"  - {train_labels_path}")
            print(f"  - {test_ids_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def train_individual_models(self, X_train, y_train, cv_folds=5):
        """
        Train individual models with cross-validation
        """
        print("\n" + "=" * 50)
        print("TRAINING INDIVIDUAL MODELS")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.base_models.items():
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Store results
            self.results[name] = {
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"  CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def create_voting_ensemble(self):
        """
        Create voting classifier ensemble
        """
        print("\nCreating voting ensemble...")
        
        # Create voting classifier with both models
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        self.voting_classifier = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use predicted probabilities
        )
        
        print("Voting ensemble created with soft voting")
    
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None, cv_folds=5):
        """
        Train the complete ensemble system
        """
        print("=" * 60)
        print("TRAINING HATE SPEECH DETECTION ENSEMBLE")
        print("=" * 60)
        
        # Initialize models if not already done
        if not self.base_models:
            self.initialize_models()
        
        # Train individual models
        self.train_individual_models(X_train, y_train, cv_folds)
        
        # Create and train voting ensemble
        self.create_voting_ensemble()
        
        print("\nTraining voting ensemble...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        ensemble_cv_scores = cross_val_score(self.voting_classifier, X_train, y_train, cv=cv, scoring='f1')
        
        # Train ensemble on full training set
        self.voting_classifier.fit(X_train, y_train)
        
        # Store ensemble results
        self.results['ensemble'] = {
            'cv_f1_mean': ensemble_cv_scores.mean(),
            'cv_f1_std': ensemble_cv_scores.std(),
            'cv_scores': ensemble_cv_scores
        }
        
        print(f"Ensemble CV F1 Score: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        print("\nEnsemble training completed!")
        
        return self.results
    
    def predict(self, X):
        """
        Make predictions using the voting ensemble
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        return self.voting_classifier.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities from the voting ensemble
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        return self.voting_classifier.predict_proba(X)
    
    def evaluate_ensemble(self, X_test, y_test):
        """
        Evaluate the complete ensemble system
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        print("\n" + "=" * 60)
        print("ENSEMBLE EVALUATION")
        print("=" * 60)
        
        # Evaluate individual base models
        print("\nBase Model Performance:")
        base_results = {}
        
        for name, model in self.base_models.items():
            print(f"\n--- {name.replace('_', ' ').title()} ---")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            base_results[name] = results
            
            # Print results
            for metric, value in results.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        # Evaluate voting ensemble
        print(f"\n--- Voting Ensemble ---")
        y_pred_ensemble = self.voting_classifier.predict(X_test)
        y_pred_proba_ensemble = self.voting_classifier.predict_proba(X_test)[:, 1]
        
        ensemble_results = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble),
            'recall': recall_score(y_test, y_pred_ensemble),
            'f1': f1_score(y_test, y_pred_ensemble),
            'auc': roc_auc_score(y_test, y_pred_proba_ensemble)
        }
        
        for metric, value in ensemble_results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        # Store all results
        self.results['base_models'] = base_results
        self.results['ensemble_evaluation'] = ensemble_results
        
        return self.results
    
    def compare_models(self):
        """
        Compare performance of all models
        """
        if 'base_models' not in self.results:
            print("No evaluation results available. Run evaluate_ensemble() first.")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Add base model results
        for name, results in self.results['base_models'].items():
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'AUC': results['auc']
            })
        
        # Add ensemble results
        if 'ensemble_evaluation' in self.results:
            ensemble_results = self.results['ensemble_evaluation']
            comparison_data.append({
                'Model': 'Voting Ensemble',
                'Accuracy': ensemble_results['accuracy'],
                'Precision': ensemble_results['precision'],
                'Recall': ensemble_results['recall'],
                'F1-Score': ensemble_results['f1'],
                'AUC': ensemble_results['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nModel Performance Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def plot_model_comparison(self):
        """
        Plot model performance comparison
        """
        if 'base_models' not in self.results:
            print("No evaluation results available. Run evaluate_ensemble() first.")
            return
        
        comparison_df = self.compare_models()
        
        # Create subplots for different metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                bars = axes[i].bar(comparison_df['Model'], comparison_df[metric])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, comparison_df[metric]):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Hide the last subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_model_agreement(self, X):
        """
        Analyze agreement between the two models
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before analyzing model agreement")
        
        # Get predictions from both models
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
        
        # Calculate agreement
        model_names = list(self.base_models.keys())
        lr_pred = predictions['logistic_regression']
        nb_pred = predictions['naive_bayes']
        
        agreement = np.mean(lr_pred == nb_pred)
        
        print(f"\nModel Agreement Analysis:")
        print("=" * 40)
        print(f"Logistic Regression vs Naive Bayes Agreement: {agreement:.3f}")
        
        # Show disagreement examples
        disagreements = np.sum(lr_pred != nb_pred)
        print(f"Number of disagreements: {disagreements} out of {len(lr_pred)} samples")
        
        return agreement
    
    def create_submission(self, X_test, test_ids, filename='ensemble_submission.csv'):
        """
        Create submission file with predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before creating submission")
        
        predictions = self.predict(X_test)
        
        submission = pd.DataFrame({
            'id': test_ids,
            'label': predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission saved to {filename}")
        
        return submission
    
    def save_results(self, filename='ensemble_results.txt'):
        """
        Save detailed results to file
        """
        with open(filename, 'w') as f:
            f.write("HATE SPEECH DETECTION ENSEMBLE RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Cross-validation results
            f.write("CROSS-VALIDATION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for name, results in self.results.items():
                if 'cv_f1_mean' in results:
                    f.write(f"\n{name.replace('_', ' ').title()}:\n")
                    f.write(f"  CV F1 Mean: {results['cv_f1_mean']:.4f}\n")
                    f.write(f"  CV F1 Std: {results['cv_f1_std']:.4f}\n")
            
            # Test results if available
            if 'base_models' in self.results:
                f.write(f"\n\nTEST SET RESULTS:\n")
                f.write("-" * 20 + "\n")
                
                for name, results in self.results['base_models'].items():
                    f.write(f"\n{name.replace('_', ' ').title()}:\n")
                    for metric, value in results.items():
                        f.write(f"  {metric.capitalize()}: {value:.4f}\n")
                
                if 'ensemble_evaluation' in self.results:
                    f.write(f"\nVoting Ensemble:\n")
                    for metric, value in self.results['ensemble_evaluation'].items():
                        f.write(f"  {metric.capitalize()}: {value:.4f}\n")
        
        print(f"Results saved to {filename}")

def main():
    """
    Main function to run the simplified ensemble system with Logistic Regression and Naive Bayes
    """
    print("Hate Speech Detection - Simple Ensemble System")
    print("Models: Logistic Regression + Multinomial Naive Bayes")
    print("Using pre-computed TF-IDF features")
    print("=" * 60)
    
    # Initialize ensemble
    ensemble = HateSpeechEnsemble(random_state=42)
    
    # Load pre-computed TF-IDF features
    X_train, X_test, y_train, test_ids = ensemble.load_data()
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train ensemble
    results = ensemble.train_ensemble(X_train_split, y_train_split)
    
    # Evaluate on validation set
    evaluation_results = ensemble.evaluate_ensemble(X_val, y_val)
    
    # Compare models
    comparison_df = ensemble.compare_models()
    
    # Plot comparison
    ensemble.plot_model_comparison()
    
    # Analyze model agreement
    agreement = ensemble.analyze_model_agreement(X_val)
    
    # Create submission file
    submission = ensemble.create_submission(X_test, test_ids)
    
    # Save results
    ensemble.save_results()
    
    print("\nEnsemble system completed successfully!")
    
    # Print best model
    if not comparison_df.empty:
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_f1 = comparison_df['F1-Score'].max()
        print(f"\nBest performing model: {best_model} (F1-Score: {best_f1:.4f})")

if __name__ == "__main__":
    main()