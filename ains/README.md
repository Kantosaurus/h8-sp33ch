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

# Create features (TF-IDF, etc.)
# X_train, X_test = your_feature_engineering_function()

# Initialize and train ensemble
ensemble = HateSpeechEnsemble(random_state=42)
ensemble.train_ensemble(X_train, y_train, X_val, y_val)

# Make predictions
predictions = ensemble.predict(X_test)

# Evaluate
results = ensemble.evaluate_ensemble(X_test, y_test)
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