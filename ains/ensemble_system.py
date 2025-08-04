import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
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

# Import multi-view feature engineering
from feature_engineering import MultiViewFeatureEngineering, create_multi_view_features

# Import view-specialized models
from view_specialized_models import ViewSpecializedModels, create_view_specialized_ensemble

# Import Logistic Regression meta-classifier
from logistic_meta_classifier import LogisticMetaClassifier, create_logistic_meta_classifier

# Import advanced techniques
from boosted_stacking_ensemble import BoostedStackingEnsemble, create_boosted_stacking_ensemble
from catboost_text_model import CatBoostTextModel, create_catboost_text_model
from rule_augmented_ml import RuleAugmentedML, create_rule_augmented_ml

class HateSpeechEnsemble:
    """
    Main ensemble system for hate speech detection
    Combines multiple base models with a meta-classifier
    """
    
    def __init__(self, random_state=42, use_view_specialized=True, use_logistic_meta=True, 
                 use_boosted_stacking=True, use_catboost_text=True, use_rule_augmented=True):
        self.random_state = random_state
        self.use_view_specialized = use_view_specialized
        self.use_logistic_meta = use_logistic_meta
        self.use_boosted_stacking = use_boosted_stacking
        self.use_catboost_text = use_catboost_text
        self.use_rule_augmented = use_rule_augmented
        
        self.base_models = {}
        self.view_specialized_models = None
        self.logistic_meta_classifier = None
        self.boosted_stacking_ensemble = None
        self.catboost_text_model = None
        self.rule_augmented_ml = None
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
    
    def train_view_specialized_models(self, features_dict: Dict[str, np.ndarray], y: np.ndarray):
        """
        Train view-specialized models for each feature view
        
        Args:
            features_dict: Dictionary with features for each view
            y: Target labels
        """
        if not self.use_view_specialized:
            print("View-specialized models disabled")
            return
        
        print("\n" + "=" * 60)
        print("TRAINING VIEW-SPECIALIZED MODELS")
        print("=" * 60)
        
        # Create and train view-specialized models
        self.view_specialized_models = create_view_specialized_ensemble(
            features_dict, y, random_state=self.random_state
        )
        
        # Extract meta-features from view-specialized models
        view_meta_features = self.view_specialized_models.extract_view_meta_features(features_dict)
        
        print(f"\nView-specialized meta-features shape: {view_meta_features.shape}")
        print("Meta-features include:")
        print("  - Predicted probabilities from each view model")
        print("  - Confidence scores (margin between top two probabilities)")
        
        return view_meta_features
    
    def train_logistic_meta_classifier(self, features_dict: Dict[str, np.ndarray], y: np.ndarray):
        """
        Train Logistic Regression meta-classifier using view-specialized models' outputs
        
        Args:
            features_dict: Dictionary with features for each view
            y: Target labels
        """
        if not self.use_logistic_meta:
            print("Logistic meta-classifier disabled")
            return
        
        if self.view_specialized_models is None:
            print("View-specialized models not available. Train them first.")
            return
        
        print("\n" + "=" * 60)
        print("TRAINING LOGISTIC REGRESSION META-CLASSIFIER")
        print("=" * 60)
        
        # Create and train Logistic Regression meta-classifier
        self.logistic_meta_classifier = create_logistic_meta_classifier(
            self.view_specialized_models, features_dict, y, random_state=self.random_state
        )
        
        # Store results
        self.results['logistic_meta'] = {
            'trained': True,
            'meta_features_shape': self.logistic_meta_classifier.results['meta_features_shape']
        }
        
        return self.logistic_meta_classifier
    
    def get_view_specialized_predictions(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get predictions from view-specialized models
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.view_specialized_models is None:
            raise ValueError("View-specialized models not trained")
        
        predictions = self.view_specialized_models.predict_views(features_dict)
        confidence_scores = self.view_specialized_models.get_confidence_scores(features_dict)
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'meta_features': self.view_specialized_models.extract_view_meta_features(features_dict)
        }
        
    def load_data(self, use_multi_view=True, fasttext_model_path=None, use_feature_union=True):
        """
        Load and prepare the dataset with multi-view feature engineering
        
        Args:
            use_multi_view: Whether to use multi-view feature engineering
            fasttext_model_path: Path to FastText model (optional)
            use_feature_union: Whether to use FeatureUnion for fusion (recommended)
        """
        print("Loading data...")
        
        try:
            # Load main data
            train_data = pd.read_csv('data/train.csv')
            test_data = pd.read_csv('data/test.csv')
        except FileNotFoundError as e:
            print(f"Error: Data files not found. {e}")
            print("Please ensure 'data/train.csv' and 'data/test.csv' exist.")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        # Extract text and labels
        train_texts = train_data['post'].values
        test_texts = test_data['post'].values
        y_train = train_data['label'].values
        
        # Validate data
        if len(train_texts) == 0 or len(test_texts) == 0:
            raise ValueError("Empty text data found")
        
        if len(y_train) != len(train_texts):
            raise ValueError("Mismatch between labels and text data lengths")
        
        if not all(isinstance(text, str) for text in train_texts):
            raise ValueError("Non-string values found in text data")
        
        print(f"Data loaded successfully:")
        print(f"  Training samples: {len(train_texts)}")
        print(f"  Test samples: {len(test_texts)}")
        print(f"  Labels: {len(y_train)}")
        
        # Store text data for advanced models
        self.train_texts = train_texts
        self.test_texts = test_texts
        
        if use_multi_view:
            print("\n" + "=" * 60)
            print("MULTI-VIEW FEATURE ENGINEERING")
            print("=" * 60)
            
            # Create multi-view features with FeatureUnion fusion
            X_train, X_test = create_multi_view_features(
                train_texts, 
                test_texts, 
                fasttext_model_path=fasttext_model_path,
                use_feature_union=use_feature_union
            )
            
            print(f"\nMulti-view features created:")
            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            
            if use_feature_union:
                print(f"Feature fusion: FeatureUnion with proper normalization")
            else:
                print(f"Feature fusion: Simple concatenation")
            
            # For view-specialized models, we need individual view features
            if self.use_view_specialized:
                print("\nCreating individual view features for specialized models...")
                
                # Create feature engineering instance to get individual views
                fe = MultiViewFeatureEngineering(
                    max_tfidf_features=20000,
                    fasttext_model_path=fasttext_model_path,
                    use_feature_union=use_feature_union
                )
                
                # Get individual view features
                train_features_dict = fe.fit_transform(train_texts)
                test_features_dict = fe.transform(test_texts)
                
                # Store for later use
                self.train_features_dict = train_features_dict
                self.test_features_dict = test_features_dict
                
                print("Individual view features created for specialized models")
            
        else:
            print("\nUsing pre-computed TF-IDF features...")
            
            # Load pre-computed TF-IDF features
            train_tfidf = pd.read_csv('data/train_tfidf_features.csv')
            test_tfidf = pd.read_csv('data/test_tfidf_features.csv')
            
            X_train = train_tfidf.values
            X_test = test_tfidf.values
            
            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, test_data['id'].values, train_texts, test_texts
        
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
        
        # Train view-specialized models if enabled
        if self.use_view_specialized and hasattr(self, 'train_features_dict'):
            print("\nTraining view-specialized models...")
            
            # Split the features dictionary to match the training data split
            if X_val is not None and y_val is not None:
                # Calculate the split indices based on the training data size
                train_size = len(y_train)
                # Handle sparse matrices properly
                if hasattr(self.train_features_dict['lexical'], 'shape'):
                    total_size = self.train_features_dict['lexical'].shape[0]
                else:
                    total_size = len(self.train_features_dict['lexical'])
                
                if train_size < total_size:
                    # We need to split the features dictionary
                    print(f"Splitting features: {total_size} -> {train_size} for training")
                    
                    # Create split feature dictionary
                    split_features_dict = {}
                    for view_name, features in self.train_features_dict.items():
                        if hasattr(features, 'toarray'):
                            # Handle sparse matrices
                            split_features_dict[view_name] = features[:train_size]
                        else:
                            # Handle dense matrices
                            split_features_dict[view_name] = features[:train_size]
                    
                    view_meta_features = self.train_view_specialized_models(
                        split_features_dict, y_train
                    )
                else:
                    view_meta_features = self.train_view_specialized_models(
                        self.train_features_dict, y_train
                    )
            else:
                view_meta_features = self.train_view_specialized_models(
                    self.train_features_dict, y_train
                )
            
            self.results['view_specialized'] = {
                'meta_features_shape': view_meta_features.shape,
                'models_trained': True
            }
            
                    # Train Logistic Regression meta-classifier if enabled
        if self.use_logistic_meta:
            print("\nTraining Logistic Regression meta-classifier...")
            
            # Use the same split logic as view-specialized models
            if X_val is not None and y_val is not None:
                train_size = len(y_train)
                if hasattr(self.train_features_dict['lexical'], 'shape'):
                    total_size = self.train_features_dict['lexical'].shape[0]
                else:
                    total_size = len(self.train_features_dict['lexical'])
                
                if train_size < total_size:
                    # Create split feature dictionary
                    split_features_dict = {}
                    for view_name, features in self.train_features_dict.items():
                        if hasattr(features, 'toarray'):
                            split_features_dict[view_name] = features[:train_size]
                        else:
                            split_features_dict[view_name] = features[:train_size]
                    
                    self.train_logistic_meta_classifier(split_features_dict, y_train)
                else:
                    self.train_logistic_meta_classifier(self.train_features_dict, y_train)
            else:
                self.train_logistic_meta_classifier(self.train_features_dict, y_train)
        
        # Train boosted stacking ensemble if enabled
        if self.use_boosted_stacking and hasattr(self, 'train_features_dict'):
            print("\nTraining boosted stacking ensemble...")
            
            # Use the same split logic
            if X_val is not None and y_val is not None:
                train_size = len(y_train)
                if hasattr(self.train_features_dict['lexical'], 'shape'):
                    total_size = self.train_features_dict['lexical'].shape[0]
                else:
                    total_size = len(self.train_features_dict['lexical'])
                
                if train_size < total_size:
                    # Create split feature dictionary
                    split_features_dict = {}
                    for view_name, features in self.train_features_dict.items():
                        if hasattr(features, 'toarray'):
                            split_features_dict[view_name] = features[:train_size]
                        else:
                            split_features_dict[view_name] = features[:train_size]
                    
                    self.train_boosted_stacking_ensemble(split_features_dict, y_train)
                else:
                    self.train_boosted_stacking_ensemble(self.train_features_dict, y_train)
            else:
                self.train_boosted_stacking_ensemble(self.train_features_dict, y_train)
        
        # Train CatBoost text model if enabled
        if self.use_catboost_text and hasattr(self, 'train_texts'):
            print("\nTraining CatBoost text model...")
            
            # Split the text data to match the training split
            if X_val is not None and y_val is not None:
                train_size = len(y_train)
                total_size = len(self.train_texts)
                
                if train_size < total_size:
                    # Split text data
                    split_texts = self.train_texts[:train_size]
                    # Create additional features for CatBoost
                    additional_features = {
                        'text_length': np.array([len(text) for text in split_texts]),
                        'word_count': np.array([len(text.split()) for text in split_texts])
                    }
                    self.train_catboost_text_model(split_texts, y_train, additional_features)
                else:
                    # Create additional features for CatBoost
                    additional_features = {
                        'text_length': np.array([len(text) for text in self.train_texts]),
                        'word_count': np.array([len(text.split()) for text in self.train_texts])
                    }
                    self.train_catboost_text_model(self.train_texts, y_train, additional_features)
            else:
                # Create additional features for CatBoost
                additional_features = {
                    'text_length': np.array([len(text) for text in self.train_texts]),
                    'word_count': np.array([len(text.split()) for text in self.train_texts])
                }
                self.train_catboost_text_model(self.train_texts, y_train, additional_features)
        
        # Train rule-augmented ML if enabled
        if self.use_rule_augmented and hasattr(self, 'train_texts'):
            print("\nTraining rule-augmented ML system...")
            
            # Split the text data to match the training split
            if X_val is not None and y_val is not None:
                train_size = len(y_train)
                total_size = len(self.train_texts)
                
                if train_size < total_size:
                    # Split text data
                    split_texts = self.train_texts[:train_size]
                    # Use lexical features as ML features for rule-augmented system
                    ml_features = None
                    if hasattr(self, 'train_features_dict'):
                        if hasattr(self.train_features_dict['lexical'], 'toarray'):
                            ml_features = self.train_features_dict['lexical'][:train_size]
                        else:
                            ml_features = self.train_features_dict['lexical'][:train_size]
                    self.train_rule_augmented_ml(split_texts, y_train, ml_features)
                else:
                    # Use lexical features as ML features for rule-augmented system
                    ml_features = self.train_features_dict.get('lexical', None) if hasattr(self, 'train_features_dict') else None
                    self.train_rule_augmented_ml(self.train_texts, y_train, ml_features)
            else:
                # Use lexical features as ML features for rule-augmented system
                ml_features = self.train_features_dict.get('lexical', None) if hasattr(self, 'train_features_dict') else None
                self.train_rule_augmented_ml(self.train_texts, y_train, ml_features)
        
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
    
    def evaluate_view_specialized_models(self, X_test, y_test):
        """
        Evaluate view-specialized models separately
        """
        if not self.use_view_specialized or self.view_specialized_models is None:
            print("View-specialized models not available")
            return
        
        if not hasattr(self, 'test_features_dict'):
            print("Test features dictionary not available")
            return
        
        print("\n" + "=" * 60)
        print("EVALUATING VIEW-SPECIALIZED MODELS")
        print("=" * 60)
        
        # Evaluate view-specialized models
        evaluation_results = self.view_specialized_models.evaluate_models(
            self.test_features_dict, y_test
        )
        
        # Compare performance
        comparison_df = self.view_specialized_models.compare_view_performance(evaluation_results)
        
        # Store results
        self.results['view_specialized_evaluation'] = evaluation_results
        
        return evaluation_results
    
    def evaluate_logistic_meta_classifier(self, X_test, y_test):
        """
        Evaluate the Logistic Regression meta-classifier
        """
        if not self.use_logistic_meta or self.logistic_meta_classifier is None:
            print("Logistic meta-classifier not available")
            return
        
        if not hasattr(self, 'test_features_dict'):
            print("Test features dictionary not available")
            return
        
        print("\n" + "=" * 60)
        print("EVALUATING LOGISTIC REGRESSION META-CLASSIFIER")
        print("=" * 60)
        
        # Evaluate Logistic Regression meta-classifier
        evaluation_results = self.logistic_meta_classifier.evaluate(
            self.test_features_dict, y_test
        )
        
        # Analyze feature importance
        importance_df = self.logistic_meta_classifier.analyze_feature_importance()
        
        # Analyze threshold optimization
        threshold_df = self.logistic_meta_classifier.analyze_threshold_optimization()
        
        # Store results
        self.results['logistic_meta_evaluation'] = evaluation_results
        
        return evaluation_results
    
    def train_boosted_stacking_ensemble(self, features_dict: Dict[str, np.ndarray], y: np.ndarray):
        """
        Train boosted stacking ensemble with advanced meta-features
        
        Args:
            features_dict: Dictionary with features for each view
            y: Target labels
        """
        if not self.use_boosted_stacking:
            print("Boosted stacking ensemble disabled")
            return
        
        print("\n" + "=" * 60)
        print("TRAINING BOOSTED STACKING ENSEMBLE")
        print("=" * 60)
        
        # Create and train boosted stacking ensemble
        self.boosted_stacking_ensemble = create_boosted_stacking_ensemble(
            features_dict, y, random_state=self.random_state
        )
        
        # Store results
        self.results['boosted_stacking'] = {
            'trained': True,
            'meta_features_shape': self.boosted_stacking_ensemble.meta_features_train.shape
        }
        
        return self.boosted_stacking_ensemble
    
    def train_catboost_text_model(self, texts: List[str], y: np.ndarray, 
                                 additional_features: Dict[str, np.ndarray] = None):
        """
        Train CatBoost text model with built-in text transformer
        
        Args:
            texts: List of text strings
            y: Target labels
            additional_features: Additional numerical features
        """
        if not self.use_catboost_text:
            print("CatBoost text model disabled")
            return
        
        print("\n" + "=" * 60)
        print("TRAINING CATBOOST TEXT MODEL")
        print("=" * 60)
        
        # Create and train CatBoost text model
        self.catboost_text_model = create_catboost_text_model(
            texts, y, additional_features, random_state=self.random_state
        )
        
        # Store results
        self.results['catboost_text'] = {
            'trained': True,
            'text_features_processed': len(texts)
        }
        
        return self.catboost_text_model
    
    def train_rule_augmented_ml(self, texts: List[str], y: np.ndarray, 
                               ml_features: np.ndarray = None):
        """
        Train rule-augmented ML system
        
        Args:
            texts: List of text strings
            y: Target labels
            ml_features: ML features (optional)
        """
        if not self.use_rule_augmented:
            print("Rule-augmented ML disabled")
            return
        
        print("\n" + "=" * 60)
        print("TRAINING RULE-AUGMENTED ML SYSTEM")
        print("=" * 60)
        
        # Create and train rule-augmented ML system
        self.rule_augmented_ml = create_rule_augmented_ml(
            texts, y, ml_features, random_state=self.random_state
        )
        
        # Store results
        self.results['rule_augmented_ml'] = {
            'trained': True,
            'texts_processed': len(texts)
        }
        
        return self.rule_augmented_ml
    
    def evaluate_boosted_stacking_ensemble(self, X_test, y_test):
        """
        Evaluate boosted stacking ensemble separately
        """
        if not self.use_boosted_stacking or self.boosted_stacking_ensemble is None:
            print("Boosted stacking ensemble not available")
            return
        
        if not hasattr(self, 'test_features_dict'):
            print("Test features dictionary not available")
            return
        
        print("\n" + "=" * 60)
        print("EVALUATING BOOSTED STACKING ENSEMBLE")
        print("=" * 60)
        
        # Evaluate boosted stacking ensemble
        evaluation_results = self.boosted_stacking_ensemble.evaluate(
            self.test_features_dict, y_test
        )
        
        # Analyze meta-features
        meta_analysis = self.boosted_stacking_ensemble.analyze_meta_features()
        
        # Store results
        self.results['boosted_stacking_evaluation'] = evaluation_results
        
        return evaluation_results
    
    def evaluate_catboost_text_model(self, X_test, y_test):
        """
        Evaluate CatBoost text model separately
        """
        if not self.use_catboost_text or self.catboost_text_model is None:
            print("CatBoost text model not available")
            return
        
        if not hasattr(self, 'test_texts'):
            print("Test texts not available")
            return
        
        print("\n" + "=" * 60)
        print("EVALUATING CATBOOST TEXT MODEL")
        print("=" * 60)
        
        # Create additional features for evaluation
        additional_features = {
            'text_length': np.array([len(text) for text in self.test_texts]),
            'word_count': np.array([len(text.split()) for text in self.test_texts])
        }
        
        # Evaluate CatBoost text model
        evaluation_results = self.catboost_text_model.evaluate(
            self.test_texts, y_test, additional_features
        )
        
        # Store results
        self.results['catboost_text_evaluation'] = evaluation_results
        
        return evaluation_results
    
    def evaluate_rule_augmented_ml(self, X_test, y_test):
        """
        Evaluate rule-augmented ML system separately
        """
        if not self.use_rule_augmented or self.rule_augmented_ml is None:
            print("Rule-augmented ML system not available")
            return
        
        if not hasattr(self, 'test_texts'):
            print("Test texts not available")
            return
        
        print("\n" + "=" * 60)
        print("EVALUATING RULE-AUGMENTED ML SYSTEM")
        print("=" * 60)
        
        # Use lexical features as ML features for evaluation
        ml_features = self.test_features_dict.get('lexical', None) if hasattr(self, 'test_features_dict') else None
        
        # Evaluate rule-augmented ML system
        evaluation_results = self.rule_augmented_ml.evaluate(
            self.test_texts, y_test, ml_features
        )
        
        # Store results
        self.results['rule_augmented_ml_evaluation'] = evaluation_results
        
        return evaluation_results
    
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
    Example usage of the ensemble system with multi-view feature engineering and view-specialized models
    """
    print("Hate Speech Detection Ensemble System")
    print("=" * 50)
    
    # Initialize ensemble with all advanced techniques enabled
    ensemble = HateSpeechEnsemble(
        random_state=42, 
        use_view_specialized=True, 
        use_logistic_meta=True,
        use_boosted_stacking=True,
        use_catboost_text=True,
        use_rule_augmented=True
    )
    
    # Load data with multi-view feature engineering and FeatureUnion fusion
    # Set use_multi_view=True to use the new multi-view approach
    # Set use_feature_union=True for proper feature fusion with normalization
    # Set fasttext_model_path if you have a pre-trained FastText model
    X_train, X_test, y_train, test_ids, train_texts, test_texts = ensemble.load_data(
        use_multi_view=True,
        use_feature_union=True,  # Use FeatureUnion for proper fusion
        fasttext_model_path=None  # Set path to FastText model if available
    )
    
    # Split data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train ensemble (includes view-specialized models)
    results = ensemble.train_ensemble(X_train_split, y_train_split, X_val, y_val)
    
    # Evaluate view-specialized models separately
    view_results = ensemble.evaluate_view_specialized_models(X_test, y_train)
    
    # Evaluate Logistic Regression meta-classifier
    logistic_results = ensemble.evaluate_logistic_meta_classifier(X_test, y_train)
    
    # Evaluate boosted stacking ensemble
    boosted_results = ensemble.evaluate_boosted_stacking_ensemble(X_test, y_train)
    
    # Evaluate CatBoost text model
    catboost_results = ensemble.evaluate_catboost_text_model(X_test, y_train)
    
    # Evaluate rule-augmented ML system
    rule_results = ensemble.evaluate_rule_augmented_ml(X_test, y_train)
    
    # Evaluate ensemble
    evaluation_results = ensemble.evaluate_ensemble(X_test, y_train)
    
    # Compare models
    comparison_df = ensemble.compare_models()
    
    # Create submission
    submission = ensemble.create_submission(X_test, test_ids)
    
    # Save results
    ensemble.save_results()
    
    print("\nEnsemble system training and evaluation completed!")

if __name__ == "__main__":
    main() 