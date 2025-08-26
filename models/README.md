# Machine Learning Models for Hate Speech Classification

This directory contains a comprehensive collection of machine learning models designed for hate speech classification. Each model is implemented as a standalone class with consistent interfaces for training, prediction, and evaluation.

## Model Categories

### 1. Linear Models

#### Logistic Regression (`logistic_regression_model.py`)
- **How it works**: Uses sigmoid function to model probability of binary classification
- **Strengths**: Fast training, interpretable coefficients, good baseline performance
- **Weaknesses**: High bias, assumes linear relationship
- **Best for**: 
  - Quick baseline establishment
  - When interpretability is crucial
  - High-dimensional sparse data (text features)
  - Small to medium datasets

#### Ridge Classifier (`ridge_classifier_model.py`)
- **How it works**: Linear classification with L2 regularization
- **Strengths**: Handles multicollinearity, reduces overfitting
- **Best for**: 
  - When features are correlated
  - High-dimensional data with noise
  - When you need regularization

#### Lasso Classifier (`lasso_classifier_model.py`)
- **How it works**: Linear classification with L1 regularization
- **Strengths**: Automatic feature selection, sparse solutions
- **Best for**: 
  - Feature selection tasks
  - High-dimensional data with many irrelevant features
  - When you want sparse models

#### Elastic Net (`elastic_net_model.py`)
- **How it works**: Combines L1 and L2 regularization
- **Strengths**: Balance between Ridge and Lasso benefits
- **Best for**: 
  - When you have grouped features
  - High-dimensional data with both noise and correlation
  - When you need both regularization and feature selection

### 2. Support Vector Machines

#### SVM Model (`svm_model.py`)
- **How it works**: Finds optimal hyperplane using linear kernel
- **Strengths**: Effective in high dimensions, memory efficient
- **Weaknesses**: No probability estimates, sensitive to feature scaling
- **Best for**: 
  - High-dimensional text data
  - When you have more features than samples
  - Linear separable data

#### Linear SVC Model (`linear_svc_model.py`)
- **How it works**: Optimized linear SVM implementation
- **Strengths**: Faster than SVM for large datasets, scales well
- **Best for**: 
  - Large datasets
  - Text classification tasks
  - When speed is important

### 3. Tree-Based Models

#### Decision Tree (`decision_tree_model.py`)
- **How it works**: Creates binary splits based on feature values
- **Strengths**: Highly interpretable, handles non-linear relationships
- **Weaknesses**: Prone to overfitting, unstable
- **Best for**: 
  - When interpretability is critical
  - Non-linear relationships
  - Mixed data types
  - Rule extraction needs

#### Random Forest (`random_forest_model.py`)
- **How it works**: Ensemble of decision trees with voting
- **Strengths**: Reduces overfitting, handles missing values, feature importance
- **Weaknesses**: Less interpretable than single tree
- **Best for**: 
  - General-purpose classification
  - When you need feature importance
  - Mixed data types
  - Moderate to large datasets

#### Extra Trees (`extra_trees_model.py`)
- **How it works**: Extremely randomized trees with random splits
- **Strengths**: Faster training than Random Forest, reduces overfitting
- **Best for**: 
  - Very large datasets
  - When training speed is important
  - High-dimensional data

### 4. Boosting Models

#### AdaBoost (`adaboost_model.py`)
- **How it works**: Sequential learning focusing on misclassified examples
- **Strengths**: Good performance, adaptive to data
- **Weaknesses**: Sensitive to noise and outliers
- **Best for**: 
  - Clean datasets
  - Binary classification tasks
  - When you have weak learners

#### Gradient Boosting (`gradient_boosting_model.py`)
- **How it works**: Sequential optimization using gradient descent
- **Strengths**: Excellent performance, handles different data types
- **Weaknesses**: Prone to overfitting, slow training
- **Best for**: 
  - Competitions and high-performance needs
  - Complex non-linear relationships
  - When you have time for hyperparameter tuning

#### XGBoost (`xgboost_model.py`)
- **How it works**: Optimized gradient boosting with regularization
- **Strengths**: State-of-the-art performance, handles missing values
- **Weaknesses**: Complex hyperparameter tuning
- **Best for**: 
  - Maximum performance requirements
  - Structured/tabular data
  - Competition settings
  - When you have computational resources

#### LightGBM (`lightgbm_model.py`)
- **How it works**: Gradient boosting with leaf-wise tree growth
- **Strengths**: Faster than XGBoost, memory efficient
- **Weaknesses**: Can overfit on small datasets
- **Best for**: 
  - Large datasets
  - When speed is crucial
  - Memory-constrained environments
  - High-dimensional data

#### CatBoost (`catboost_model.py`)
- **How it works**: Gradient boosting with categorical feature handling
- **Strengths**: Handles categorical features natively, less overfitting
- **Best for**: 
  - Datasets with many categorical features
  - When you want minimal preprocessing
  - Robust performance out-of-the-box

### 5. Probabilistic Models

#### Naive Bayes (`naive_bayes_model.py`)
- **How it works**: Applies Bayes theorem with feature independence assumption
- **Strengths**: Fast training/prediction, works well with sparse data
- **Weaknesses**: Strong independence assumption
- **Best for**: 
  - Text classification
  - Sparse high-dimensional data
  - Real-time prediction needs
  - Small datasets

#### K-Nearest Neighbors (`knn_model.py`)
- **How it works**: Classification based on k nearest neighbors
- **Strengths**: Simple, no assumptions about data
- **Weaknesses**: Computationally expensive, sensitive to irrelevant features
- **Best for**: 
  - Small to medium datasets
  - When data has local patterns
  - Non-parametric problems

### 6. Ensemble Methods

#### Voting Classifier (`voting_classifier_model.py`)
- **How it works**: Combines multiple different algorithms with voting
- **Strengths**: Reduces overfitting, leverages different algorithm strengths
- **Weaknesses**: Increased complexity, slower prediction
- **Best for**: 
  - When you want to combine different algorithm types
  - Improving stability and robustness
  - When individual models perform similarly

#### Bagging Classifier (`bagging_classifier_model.py`)
- **How it works**: Bootstrap aggregation with same algorithm
- **Strengths**: Reduces overfitting, parallel training
- **Weaknesses**: May not improve bias
- **Best for**: 
  - High-variance models (like decision trees)
  - When you want to reduce overfitting
  - Parallel computing environments

#### Meta-Classifier (`meta_classifier.py`)
- **How it works**: Second-level model that learns from base model predictions
- **Strengths**: Sophisticated ensemble method, learns optimal combinations
- **Weaknesses**: Complex implementation, requires careful validation
- **Best for**: 
  - When you have multiple good base models
  - Competition settings
  - Maximum performance requirements
  - When base models have different strengths

## Usage Patterns

### Quick Start (Choose one):
1. **Logistic Regression** - Fast baseline
2. **Random Forest** - Reliable general-purpose
3. **XGBoost** - Maximum performance

### Interpretability Required:
1. **Decision Tree** - Most interpretable
2. **Logistic Regression** - Linear coefficients
3. **Naive Bayes** - Feature probabilities

### Large Dataset:
1. **LightGBM** - Speed and performance
2. **Linear SVC** - Scales well
3. **SGD variants** - Online learning

### Small Dataset:
1. **Naive Bayes** - Works with little data
2. **K-NN** - No assumptions
3. **Logistic Regression** - Simple and stable

### High-Dimensional Text Data:
1. **Naive Bayes** - Designed for sparse data
2. **Linear SVC** - Handles high dimensions well
3. **Logistic Regression** - Good baseline

### Maximum Performance:
1. **Meta-Classifier** - Ensemble of best models
2. **XGBoost/LightGBM** - Top individual performers
3. **Voting Classifier** - Combine different types

## Model Selection Guide

| Scenario | Primary Choice | Alternative | Backup |
|----------|---------------|-------------|---------|
| **Fast Prototype** | Logistic Regression | Naive Bayes | Random Forest |
| **Production System** | XGBoost | LightGBM | Random Forest |
| **Interpretable Model** | Decision Tree | Logistic Regression | Naive Bayes |
| **Large Dataset** | LightGBM | Linear SVC | SGD Classifier |
| **Small Dataset** | Naive Bayes | Logistic Regression | K-NN |
| **Text Classification** | Naive Bayes | Linear SVC | Logistic Regression |
| **Mixed Data Types** | Random Forest | XGBoost | Decision Tree |
| **Maximum Accuracy** | Meta-Classifier | XGBoost | Ensemble Methods |
| **Real-time Prediction** | Naive Bayes | Linear SVC | Logistic Regression |
| **Feature Selection** | Lasso | Elastic Net | Random Forest |

## Common Implementation Pattern

Each model follows this consistent interface:

```python
# Initialize
model = ModelClass(random_state=42)

# Train with cross-validation
cv_scores = model.train(X_train, y_train, cv_folds=5)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate performance
results = model.evaluate(X_test, y_test)

# Get feature importance (where available)
importance = model.get_feature_importance(feature_names)
```

## Performance Characteristics

### Training Speed (Fastest to Slowest):
1. Naive Bayes, Logistic Regression
2. Linear SVC, Ridge/Lasso
3. Decision Tree, K-NN
4. Random Forest, Extra Trees
5. AdaBoost, Gradient Boosting
6. XGBoost, LightGBM, CatBoost
7. Meta-Classifier, Voting Classifier

### Prediction Speed (Fastest to Slowest):
1. Linear models (Logistic, SVM, Ridge, Lasso)
2. Naive Bayes
3. Tree-based models
4. Boosting models
5. K-NN
6. Ensemble methods

### Memory Usage (Lowest to Highest):
1. Linear models
2. Naive Bayes
3. Decision Tree
4. Boosting models
5. Random Forest/Extra Trees
6. K-NN
7. Ensemble methods

Choose your model based on your specific requirements for accuracy, speed, interpretability, and computational resources.