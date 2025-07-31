import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Import all base models from the models directory (root level)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.logistic_regression_model import LogisticRegressionModel
from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.naive_bayes_model import NaiveBayesModel
from models.extra_trees_model import ExtraTreesModel
from models.adaboost_model import AdaBoostModel
from models.catboost_model import CatBoostModel
from models.meta_classifier import MetaClassifier

class HateSpeechEnsemble:
    """
    Main ensemble system for hate speech detection
    Combines multiple base models with a meta-classifier
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.meta_classifier = MetaClassifier(random_state=random_state)
        self.is_trained = False
        self.results = {}
        
    def initialize_models(self):
        """
        Initialize all base models
        """
        print("Initializing base models...")
        
        self.base_models = {
            'logistic_regression': LogisticRegressionModel(random_state=self.random_state),
            'svm': SVMModel(random_state=self.random_state),
            'random_forest': RandomForestModel(random_state=self.random_state),
            'xgboost': XGBoostModel(random_state=self.random_state),
            'naive_bayes': NaiveBayesModel(random_state=self.random_state),
            'extra_trees': ExtraTreesModel(random_state=self.random_state),
            'adaboost': AdaBoostModel(random_state=self.random_state),
            'catboost': CatBoostModel(random_state=self.random_state)
        }
        
        # Add all base models to meta-classifier
        for name, model in self.base_models.items():
            self.meta_classifier.add_base_model(name, model)
        
        print(f"Initialized {len(self.base_models)} base models")
        
    def load_data(self):
        """
        Load and prepare the dataset
        """
        print("Loading data...")
        
        # Load main data - fix paths to point to data directory
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')
        
        # Load TF-IDF features - fix paths to point to data directory
        train_tfidf = pd.read_csv('data/train_tfidf_features.csv')
        test_tfidf = pd.read_csv('data/test_tfidf_features.csv')
        
        # Prepare features and labels
        X_train_tfidf = train_tfidf.values
        X_test_tfidf = test_tfidf.values
        y_train = train_data['label'].values
        
        print(f"Training set shape: {X_train_tfidf.shape}")
        print(f"Test set shape: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf, y_train, test_data['id'].values, train_data['text'].values, test_data['text'].values
        
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
        
        # Train all base models
        print("\nTraining base models...")
        for name, model in self.base_models.items():
            print(f"\n--- {name.upper().replace('_', ' ')} ---")
            cv_scores = model.train(X_train, y_train, cv_folds=cv_folds)
            self.results[name] = cv_scores
        
        # Train meta-classifier
        print("\n" + "=" * 40)
        print("TRAINING META-CLASSIFIER")
        print("=" * 40)
        
        self.meta_classifier.train(X_train, y_train, X_val, y_val)
        self.is_trained = True
        
        print("\nEnsemble training completed!")
        
        return self.results
    
    def predict(self, X):
        """
        Make predictions using the meta-classifier
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        return self.meta_classifier.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities from the meta-classifier
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        return self.meta_classifier.predict_proba(X)
    
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
            print(f"\n--- {name.upper().replace('_', ' ')} ---")
            results = model.evaluate(X_test, y_test)
            base_results[name] = results
        
        # Evaluate meta-classifier
        print(f"\n" + "=" * 40)
        print("META-CLASSIFIER PERFORMANCE")
        print("=" * 40)
        meta_results = self.meta_classifier.evaluate(X_test, y_test)
        
        # Store all results
        self.results['base_models'] = base_results
        self.results['meta_classifier'] = meta_results
        
        return self.results
    
    def compare_models(self):
        """
        Compare performance of all models
        """
        if 'base_models' not in self.results:
            print("No evaluation results available. Run evaluate_ensemble() first.")
            return
        
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
        
        # Add meta-classifier results
        meta_results = self.results['meta_classifier']
        comparison_data.append({
            'Model': 'Meta-Classifier',
            'Accuracy': meta_results['accuracy'],
            'Precision': meta_results['precision'],
            'Recall': meta_results['recall'],
            'F1-Score': meta_results['f1'],
            'AUC': meta_results['auc']
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
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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
        
        # Hide the last subplot if there are more axes than metrics
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from all models
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before getting feature importance")
        
        importance_dict = {}
        
        # Get feature importance from base models
        for name, model in self.base_models.items():
            try:
                importance = model.get_feature_importance(feature_names)
                importance_dict[name] = importance
            except Exception as e:
                print(f"Could not get feature importance for {name}: {e}")
        
        # Get meta-feature importance
        try:
            meta_importance = self.meta_classifier.get_feature_importance()
            importance_dict['meta_classifier'] = meta_importance
        except Exception as e:
            print(f"Could not get meta-feature importance: {e}")
        
        return importance_dict
    
    def analyze_model_agreement(self, X):
        """
        Analyze agreement between base models
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before analyzing model agreement")
        
        agreement_matrix, model_names = self.meta_classifier.analyze_model_agreement(X)
        
        # Create DataFrame for better visualization
        agreement_df = pd.DataFrame(
            agreement_matrix,
            index=model_names,
            columns=model_names
        )
        
        print("\nModel Agreement Matrix:")
        print("=" * 50)
        print(agreement_df.to_string(float_format='%.3f'))
        
        # Plot agreement heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_df, annot=True, fmt='.3f', cmap='Blues', vmin=0, vmax=1)
        plt.title('Model Agreement Matrix')
        plt.tight_layout()
        plt.show()
        
        return agreement_df
    
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
            
            # Base model CV results
            f.write("BASE MODEL CROSS-VALIDATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            for name, results in self.results.items():
                if name != 'base_models' and name != 'meta_classifier':
                    f.write(f"\n{name.replace('_', ' ').title()}:\n")
                    f.write(f"  CV F1 Mean: {results['f1_mean']:.4f}\n")
                    f.write(f"  CV F1 Std: {results['f1_std']:.4f}\n")
            
            # Test results if available
            if 'base_models' in self.results:
                f.write(f"\n\nTEST SET RESULTS:\n")
                f.write("-" * 20 + "\n")
                
                for name, results in self.results['base_models'].items():
                    f.write(f"\n{name.replace('_', ' ').title()}:\n")
                    for metric, value in results.items():
                        f.write(f"  {metric.capitalize()}: {value:.4f}\n")
                
                f.write(f"\nMeta-Classifier:\n")
                for metric, value in self.results['meta_classifier'].items():
                    f.write(f"  {metric.capitalize()}: {value:.4f}\n")
        
        print(f"Results saved to {filename}")

def main():
    """
    Example usage of the ensemble system
    """
    print("Hate Speech Detection Ensemble System")
    print("=" * 50)
    
    # Initialize ensemble
    ensemble = HateSpeechEnsemble(random_state=42)
    
    # Load data
    X_train_tfidf, X_test_tfidf, y_train, test_ids, train_texts, test_texts = ensemble.load_data()
    
    # Split data for validation
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train ensemble
    results = ensemble.train_ensemble(X_train, y_train_split, X_val, y_val)
    
    # Evaluate ensemble
    evaluation_results = ensemble.evaluate_ensemble(X_test_tfidf, y_train)
    
    # Compare models
    comparison_df = ensemble.compare_models()
    
    # Create submission
    submission = ensemble.create_submission(X_test_tfidf, test_ids)
    
    # Save results
    ensemble.save_results()
    
    print("\nEnsemble system training and evaluation completed!")

if __name__ == "__main__":
    main() 