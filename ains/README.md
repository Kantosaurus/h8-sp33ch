# Hate Speech Detection Ensemble System

A modular machine learning ensemble system for detecting hate speech in online social media posts. Each model is implemented in its own file for better organization and maintainability.

## 🎯 Project Overview

This system implements a sophisticated ensemble approach to hate speech detection using multiple traditional classifiers combined with a meta-classifier that learns from base model outputs. The system addresses the challenge of automatically identifying hateful content in online social platforms.

## 🏗️ System Architecture

### Base Models (Each in its own .py file)

1. **`logistic_regression_model.py`** - Logistic Regression (high bias, good baseline)
2. **`svm_model.py`** - Support Vector Machine with linear kernel (LinearSVC)
3. **`random_forest_model.py`** - Random Forest Classifier
4. **`xgboost_model.py`** - XGBoost (Gradient Boosting with use_label_encoder=False)
5. **`naive_bayes_model.py`** - Naive Bayes (complements well with sparse data)
6. **`extra_trees_model.py`** - Extra Trees Classifier (introduces more randomness)

### Meta-Classifier

7. **`meta_classifier.py`** - Logistic Regression meta-classifier that learns from base model outputs

### Main System

8. **`ensemble_system.py`** - Main ensemble orchestrator that combines all components

## 📁 File Structure

```
ains/
├── logistic_regression_model.py  # Logistic Regression model
├── svm_model.py                  # Support Vector Machine model
├── random_forest_model.py        # Random Forest model
├── xgboost_model.py              # XGBoost model
├── naive_bayes_model.py          # Naive Bayes model
├── extra_trees_model.py          # Extra Trees model
├── meta_classifier.py            # Meta-classifier
├── ensemble_system.py            # Main ensemble system
├── test_ensemble.py              # Test script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Key Features

### Multi-View Feature Engineering

The system implements a comprehensive multi-view approach with three distinct feature views:

#### 📌 A. Lexical View (Sparse)
- **TF-IDF vectorizer**: unigram + bigram + trigram, max_features=20,000
- **Character-level TF-IDF**: Good for obfuscated hate (e.g., "b!tch")
- **Stop word removal** and sublinear TF scaling

#### 📌 B. Semantic View (Dense)
- **Pre-trained FastText word embeddings** with average pooling
- **Hate lexicon similarity**: Cosine similarity to Hatebase, Google's "toxic" word list, and common slurs
- **Semantic understanding** of hate speech patterns

#### 📌 C. Stylistic View
- **Text statistics**: uppercase ratio, punctuation frequency, word/char counts
- **Linguistic features**: average word length, unique word ratio, digit ratio
- **Sentiment analysis**: VADER polarity & subjectivity, TextBlob sentiment

### 🔗 Feature Fusion

The system implements advanced feature fusion using scikit-learn's FeatureUnion:

- **FeatureUnion pipeline**: Combines all three views seamlessly
- **Z-score normalization**: Applied to dense features (semantic, stylistic)
- **Sparse preservation**: Lexical features remain sparse (TF-IDF)
- **Hybrid matrix**: Sparse + dense combination for optimal performance

### 🎯 Model Specialization (Per-view Learners)

The system implements specialized models for each feature view:

#### 📌 Lexical View: Logistic Regression
- **Rationale**: Great on sparse binary features (TF-IDF)
- **Advantages**: Efficient with high-dimensional sparse data, fast training
- **Characteristics**: Linear model captures word presence/absence patterns

#### 📌 Semantic View: CatBoost
- **Rationale**: Handles dense features well (FastText embeddings), text-aware
- **Advantages**: Built-in text processing, captures complex semantic relationships
- **Characteristics**: Gradient boosting, robust to overfitting

#### 📌 Stylistic View: Random Forest
- **Rationale**: Nonlinear model handles discrete features well
- **Advantages**: Good for mixed data types (ratios, counts, sentiment)
- **Characteristics**: Captures interactions between stylistic features

#### 🔗 Meta-Features Extraction
- **Predicted class probabilities** from each view model
- **Confidence scores** (margin between top two probabilities)
- **Rich information** for meta-learning and ensemble combination

### 🧠 Logistic Regression Meta-Classifier

The system implements a specialized Logistic Regression meta-classifier that uses:

#### 📊 Meta-Features from View-Specialized Models
- **Predicted probabilities** from each view model (lexical, semantic, stylistic)
- **Confidence scores** (margin between top two probabilities)
- **Base meta-features** from view-specialized models

#### 📈 Additional Statistical Meta-Features (15 features)
- **Prediction statistics**: mean, std, max, min, range, variance
- **Confidence statistics**: mean, std, max, min, range, variance
- **Model agreement**: number of high/low confidence models, agreement indicator
- **Model disagreement**: standard deviation of predictions across views

#### 🔧 Technical Features
- **Feature scaling**: StandardScaler for optimal performance
- **Cross-validation**: Robust performance estimation
- **Feature importance**: Analysis of which meta-features contribute most
- **Comprehensive evaluation**: Accuracy, F1, Precision, Recall, AUC
- **Threshold optimization**: Custom threshold that maximizes F1-score

### Meta-Classifier Capabilities

The meta-classifier extracts and learns from:

1. **Predicted probabilities** from each base model
2. **Confidence score gaps** (margin between top-2 class probabilities)
3. **Model disagreement** measures (standard deviation of predictions)
4. **Prediction variance** across models
5. **Mean and median predictions**
6. **Prediction ranges**
7. **Number of models predicting above threshold**

### Advanced Ensemble Features

- **Cross-validation** for robust performance estimation
- **Model comparison** and visualization
- **Feature importance** analysis for all models
- **Model agreement** analysis
- **Comprehensive evaluation** metrics
- **Submission file generation**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the System

```bash
cd ains
python test_ensemble.py
```

### 3. Use with Real Data

```python
from ensemble_system import HateSpeechEnsemble
import pandas as pd

# Load your data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Initialize ensemble with multi-view feature engineering
ensemble = HateSpeechEnsemble(random_state=42)

# Load data with multi-view features and FeatureUnion fusion
X_train, X_test, y_train, test_ids, train_texts, test_texts = ensemble.load_data(
    use_multi_view=True,
    use_feature_union=True,  # Use FeatureUnion for proper fusion
    fasttext_model_path=None  # Set path to FastText model if available
)

# Train ensemble
ensemble.train_ensemble(X_train, y_train, X_val, y_val)

# Make predictions
predictions = ensemble.predict(X_test)

# Evaluate
results = ensemble.evaluate_ensemble(X_test, y_test)
```

## 🔥 Advanced Techniques

### 1. Boosted Stacking Ensemble

```python
from boosted_stacking_ensemble import create_boosted_stacking_ensemble

# Create boosted stacking ensemble with meta-features
ensemble = create_boosted_stacking_ensemble(features_dict, y_train)

# Advanced meta-features include:
# - Model outputs (predictions, probabilities)
# - Model disagreement (variance across predictions)
# - Confidence scores (prob_max - prob_2nd_max)
# - Agreement indicators and consistency measures
# - Decision function values (from LinearSVC and other models)

# Make predictions
predictions = ensemble.predict(features_dict)
probabilities = ensemble.predict_proba(features_dict)

# Analyze meta-features
importance_df = ensemble.get_feature_importance()
meta_analysis = ensemble.analyze_meta_features()
```

#### LinearSVC Integration
```python
# LinearSVC is automatically included as a base model
# It provides decision function values as additional meta-features

# Check LinearSVC base model
if 'svm_linear' in ensemble.base_models:
    svm_model = ensemble.base_models['svm_linear']['model']
    print(f"LinearSVC model: {type(svm_model).__name__}")

# Decision function values are converted to probability-like scores
# using sigmoid transformation: 1 / (1 + exp(-decision_score))

# Extract decision functions manually
decision_functions = ensemble._extract_decision_functions(features_dict)
print(f"Decision functions shape: {decision_functions.shape}")
```

### 2. CatBoost Text Mode

```python
from catboost_text_model import create_catboost_text_model

# Create CatBoost model with built-in text transformer
model = create_catboost_text_model(texts, y_train, additional_features)

# No TF-IDF needed - CatBoost handles text directly
# Additional features can include text statistics
additional_features = {
    'text_length': [len(text) for text in texts],
    'word_count': [len(text.split()) for text in texts]
}

# Make predictions
predictions = model.predict(texts, additional_features)
probabilities = model.predict_proba(texts, additional_features)

# Cross-validation
cv_results = model.cross_validate(texts, y_train, additional_features)
```

### 3. Rule-Augmented Machine Learning

```python
from rule_augmented_ml import create_rule_augmented_ml

# Create rule-augmented ML system
system = create_rule_augmented_ml(texts, y_train, ml_features)

# Rules detect:
# - Hate keywords (racial, gender, religious, disability, sexual orientation)
# - Threatening patterns
# - Dehumanizing language
# - Intensity modifiers
# - Obfuscation techniques (leetspeak, misspellings)

# Make predictions
predictions = system.predict(texts, ml_features)
probabilities = system.predict_proba(texts, ml_features)

# Analyze rule application
rule_analysis = system.analyze_rules(texts)
confidence_scores = system.get_rule_confidence(texts)
```

### 4. Semi-Supervised Learning

```python
# Train with limited labeled data
labeled_texts = texts[:100]  # Only 100 labeled examples
labeled_labels = y_train[:100]
unlabeled_texts = texts[100:]  # 900 unlabeled examples

# Rule-augmented ML can work with weak labels
system = create_rule_augmented_ml(
    labeled_texts + unlabeled_texts,  # All texts
    labeled_labels,  # Only labeled portion
    ml_features
)

# Weak labels are automatically generated for unlabeled data
# System combines human rules with ML for better generalization
```

### 4. Multi-View Feature Engineering

```python
from feature_engineering import MultiViewFeatureEngineering

# Initialize feature engineering
fe = MultiViewFeatureEngineering(
    max_tfidf_features=20000,
    fasttext_model_path='path/to/fasttext/model.bin'  # Optional
)

# Create features
features_dict = fe.fit_transform(texts)

# Access individual views
lexical_features = features_dict['lexical']      # TF-IDF features
semantic_features = features_dict['semantic']    # FastText + lexicon similarity
stylistic_features = features_dict['stylistic']  # Text statistics + sentiment

# Combine all views using FeatureUnion (recommended)
combined_features = fe.combine_views(features_dict, method='feature_union')

# Or use simple concatenation
combined_features = fe.combine_views(features_dict, method='concatenate')
```

### 5. Feature Fusion Options

```python
# Option 1: FeatureUnion with proper normalization (recommended)
fe = MultiViewFeatureEngineering(use_feature_union=True)
features_dict = fe.fit_transform(texts)
combined = fe.combine_views(features_dict, method='feature_union')

# Option 2: Legacy concatenation
fe = MultiViewFeatureEngineering(use_feature_union=False)
features_dict = fe.fit_transform(texts)
combined = fe.combine_views(features_dict, method='concatenate')

# Option 3: Convenience function with FeatureUnion
train_features, test_features = create_multi_view_features(
    train_texts, test_texts, use_feature_union=True
)
```

### 6. View-Specialized Models

```python
from view_specialized_models import ViewSpecializedModels, create_view_specialized_ensemble

# Option 1: Manual creation and training
vsm = ViewSpecializedModels(random_state=42)
training_results = vsm.train_models(features_dict, y_train)

# Get predictions and confidence scores
predictions = vsm.predict_views(features_dict)
confidence_scores = vsm.get_confidence_scores(features_dict)

# Extract meta-features
meta_features = vsm.extract_view_meta_features(features_dict)

# Option 2: Convenience function
vsm = create_view_specialized_ensemble(features_dict, y_train)

# Option 3: Integrated with ensemble system
ensemble = HateSpeechEnsemble(use_view_specialized=True)
# View-specialized models are automatically trained during ensemble training
```

### 7. Logistic Regression Meta-Classifier

```python
from logistic_meta_classifier import LogisticMetaClassifier, create_logistic_meta_classifier

# Option 1: Manual creation and training
meta_classifier = LogisticMetaClassifier(random_state=42)
meta_classifier.set_view_specialized_models(view_specialized_models)
training_results = meta_classifier.train(features_dict, y_train)

# Get predictions
predictions = meta_classifier.predict(features_dict)
probabilities = meta_classifier.predict_proba(features_dict)

# Analyze feature importance
importance_df = meta_classifier.analyze_feature_importance()

# Option 2: Convenience function
meta_classifier = create_logistic_meta_classifier(view_specialized_models, features_dict, y_train)

# Option 3: Integrated with ensemble system
ensemble = HateSpeechEnsemble(use_view_specialized=True, use_logistic_meta=True)
# Logistic meta-classifier is automatically trained during ensemble training
```

### 8. Threshold Optimization

```python
# Threshold optimization is automatically performed during training
# The optimal threshold maximizes F1-score on validation data

# Analyze threshold optimization results
threshold_df = meta_classifier.analyze_threshold_optimization()

# Plot threshold optimization (if matplotlib available)
meta_classifier.plot_threshold_optimization()

# Get optimal threshold
optimal_threshold = meta_classifier.optimal_threshold
print(f"Optimal threshold: {optimal_threshold:.3f}")

# Manual threshold optimization
from sklearn.metrics import f1_score
probabilities = meta_classifier.predict_proba(features_dict)[:, 1]
best_thresh = max([(thresh, f1_score(y_val, probabilities > thresh)) 
                   for thresh in np.arange(0.1, 0.9, 0.01)], key=lambda x: x[1])
```

## 📊 Model Performance

Each model provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve
- **Cross-validation scores**: Robust performance estimates

## 🎨 Visualization Features

The system includes extensive visualization capabilities:

1. **Model Performance Comparison**: Bar charts comparing all metrics
2. **Model Agreement Matrix**: Heatmap showing inter-model agreement
3. **Feature Importance**: Top features for each model
4. **Meta-feature Importance**: Which meta-features contribute most

## 📈 Usage Examples

### Basic Ensemble Training

```python
from ensemble_system import HateSpeechEnsemble

# Initialize ensemble
ensemble = HateSpeechEnsemble(random_state=42)

# Train the complete system
results = ensemble.train_ensemble(X_train, y_train, X_val, y_val)

# Make predictions
predictions = ensemble.predict(X_test)

# Create submission
submission = ensemble.create_submission(X_test, test_ids, 'submission.csv')
```

### Model Comparison

```python
# Compare all models
comparison_df = ensemble.compare_models()

# Plot comparison
ensemble.plot_model_comparison()
```

### Model Analysis

```python
# Analyze model agreement
agreement_df = ensemble.analyze_model_agreement(X_test)

# Get feature importance
importance_dict = ensemble.get_feature_importance()

# Save results
ensemble.save_results('results.txt')
```

## 🔍 Individual Model Usage

Each model can be used independently:

```python
from logistic_regression_model import LogisticRegressionModel

# Initialize model
lr_model = LogisticRegressionModel(random_state=42)

# Train with cross-validation
cv_scores = lr_model.train(X_train, y_train, cv_folds=5)

# Make predictions
predictions = lr_model.predict(X_test)
probabilities = lr_model.predict_proba(X_test)

# Evaluate
results = lr_model.evaluate(X_test, y_test)

# Get feature importance
importance = lr_model.get_feature_importance(feature_names)
```

## 🛠️ Customization

### Adding New Models

1. Create a new model file following the same interface
2. Add the model to `ensemble_system.py` in the `initialize_models()` method
3. The meta-classifier will automatically include it

Example new model structure:

```python
class NewModel:
    def __init__(self, random_state=42):
        self.model = YourModelClass()
        self.is_trained = False
        self.cv_scores = None
    
    def train(self, X_train, y_train, cv_folds=5):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
    
    def predict_proba(self, X):
        # Probability prediction logic
        pass
    
    def evaluate(self, X_test, y_test):
        # Evaluation logic
        pass
    
    def get_feature_importance(self, feature_names=None):
        # Feature importance logic
        pass
```

### Modifying Meta-Features

Edit the `extract_meta_features()` method in `meta_classifier.py` to add new meta-features:

```python
def extract_meta_features(self, X):
    # Existing meta-features...
    
    # Add your new meta-feature
    new_feature = your_calculation(predictions)
    meta_features.append(new_feature)
    
    return np.hstack(meta_features)
```

## 📋 Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- XGBoost >= 1.5.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## 🧪 Testing

Run the test script to verify all components work correctly:

```bash
python test_ensemble.py
```

This will:
- Create sample data
- Train all models
- Evaluate performance
- Generate visualizations
- Create sample submission
- Save detailed results

## 📄 Output Files

The system generates several output files:

- **`ensemble_submission.csv`**: Final predictions for submission
- **`ensemble_results.txt`**: Detailed performance results
- **`test_submission.csv`**: Sample submission from test run
- **`test_results.txt`**: Sample results from test run

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This system is designed for research and educational purposes. Always ensure compliance with relevant data protection and privacy regulations when using this system for real-world applications. 