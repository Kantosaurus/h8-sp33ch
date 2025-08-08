# Comprehensive Machine Learning Models for Hate Speech Classification

## Overview
This repository contains **19 different machine learning models** for hate speech classification, covering all major algorithm families including tree-based methods, linear models, probabilistic classifiers, boosting methods, and ensemble techniques.

## Model Architecture Summary

### Tree-Based Models (6 models)
1. **Random Forest** (`random_forest_classifier.py`)
   - Ensemble of decision trees with bootstrap sampling
   - Features: Class balancing, feature importance analysis, hyperparameter tuning
   - Performance: ~83.49% accuracy

2. **Extra Trees** (`extra_trees_classifier.py`) 
   - Extremely randomized trees with random splits
   - Features: Faster training, reduced overfitting
   - Performance: ~83.84% accuracy (best performer)

3. **Decision Tree** (`decision_tree_classifier.py`)
   - Single tree classifier with pruning
   - Features: Interpretability, feature importance
   - Performance: ~75.67% accuracy

4. **XGBoost** (`xgboost_classifier.py`)
   - Gradient boosting with advanced optimization
   - Features: GPU acceleration, early stopping, regularization
   - Performance: ~81.44% accuracy

5. **LightGBM** (`lightgbm_classifier.py`)
   - Fast gradient boosting with leaf-wise growth
   - Features: Memory efficiency, categorical feature handling
   - Performance: ~78.43% accuracy

6. **CatBoost** (`catboost_classifier.py`)
   - Gradient boosting optimized for categorical features
   - Features: Built-in overfitting protection, GPU support
   - Performance: ~78.99% accuracy

### Linear Models (3 models)
7. **Ridge Classifier** (`ridge_classifier.py`)
   - Linear classification with L2 regularization
   - Features: Regularization, fast training
   - Performance: ~71.39% accuracy

8. **Logistic Regression** (`logistic_regression_classifier.py`)
   - Linear classification with optional dimensionality reduction
   - Features: PCA/TruncatedSVD options, balanced class weights
   - Performance: TBD

9. **Linear/Quadratic Discriminant Analysis** (`lda_classifier.py`, `qda_classifier.py`)
   - Gaussian-based classification
   - LDA Performance: ~74.60% accuracy
   - QDA Performance: ~78.70% accuracy

### Probabilistic Models (1 model + variants)
10. **Naive Bayes Suite** (`naive_bayes_classifier.py`)
    - Three variants: Multinomial, Complement, Bernoulli
    - Features: Comparative analysis, algorithm-specific vectorization
    - Performance: TBD

### Support Vector Machines (1 model)
11. **SVM with PCA** (`svm_classifier.py`)
    - Support Vector Machine with PCA preprocessing
    - Features: RBF/linear kernels, probability estimation
    - Performance: TBD

### Boosting Methods (2 models)
12. **Gradient Boosting** (`gradient_boosting_classifier.py`)
    - Sequential boosting with decision trees
    - Performance: ~72.88% accuracy

13. **AdaBoost** (`adaboost_classifier.py`)
    - Adaptive boosting with decision tree base estimators
    - Features: SAMME.R algorithm, estimator weight analysis
    - Performance: TBD

### Bagging Methods (1 model)
14. **Bagging Classifier** (`bagging_classifier.py`)
    - Bootstrap aggregating with PCA preprocessing
    - Features: Out-of-bag scoring, flexible base estimators
    - Performance: TBD

### Dimensionality Reduction (1 model)
15. **PCA + Adam Optimizer** (`pca_classifier.py`)
    - Principal Component Analysis with custom neural network
    - Features: Adaptive optimization, dimensionality reduction
    - Performance: TBD

### Distance-Based Models (1 model)
16. **K-Nearest Neighbors** (`knn_classifier.py`)
    - Instance-based classification
    - Performance: TBD

### Ensemble Methods (3 models)
17. **Voting Ensemble** (`voting_classifier.py`)
    - Soft/hard voting combining diverse estimators
    - Features: Estimator agreement analysis, flexible weighting
    - Performance: TBD

18. **Custom Ensemble** (`ensemble_classifier.py`)
    - Combines top 5 performing models
    - Features: Weighted averaging, model selection
    - Performance: TBD

19. **Model Runner** (`model_runner.py`)
    - Comprehensive comparison framework
    - Features: Batch execution, performance comparison, metrics aggregation

## Data Preprocessing Pipeline

### Common Features Across All Models:
- **TF-IDF Vectorization**: Up to 15,000 features, 1-2 grams, English stop words
- **Text Preprocessing**: Lowercase conversion, ASCII accent stripping
- **Feature Selection**: Min/max document frequency filtering
- **Cross-Validation**: 5-fold stratified validation
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Class Balancing**: Various strategies (balanced weights, SMOTE, etc.)

### Advanced Preprocessing Options:
- **Dimensionality Reduction**: PCA, TruncatedSVD options
- **Feature Scaling**: StandardScaler for distance-based methods
- **Categorical Handling**: Built-in support for CatBoost

## Performance Monitoring

### Metrics Tracked:
- **Accuracy**: Primary evaluation metric
- **Precision/Recall/F1**: Comprehensive performance assessment
- **Cross-Validation**: Mean and standard deviation
- **Training Time**: Execution time tracking
- **Feature Importance**: Model interpretability
- **Hyperparameter Optimization**: Best parameter tracking

### Output Structure:
Each model creates its own output directory containing:
- `metrics.json`: Performance metrics and configuration
- `submission_[model].csv`: Test predictions
- `[model]_model.pkl`: Trained model
- `[model]_vectorizer.pkl`: TF-IDF vectorizer
- Additional model-specific artifacts (PCA, feature importance, etc.)

## Usage Instructions

### Running Individual Models:
```bash
cd shallow_learning/models
python model_runner.py --model [model_name]
```

### Running All Models:
```bash
python model_runner.py --all
```

### Comparing Model Performance:
```bash
python model_runner.py --compare
```

### Listing Available Models:
```bash
python model_runner.py --list
```

## Model Selection Strategy

### High Performance Models:
1. **Extra Trees**: Best overall accuracy (83.84%)
2. **Random Forest**: Consistent performance (83.49%)
3. **XGBoost**: Strong gradient boosting (81.44%)

### Specialized Models:
- **SVM**: Best for non-linear patterns
- **Naive Bayes**: Fast training, good baseline
- **Logistic Regression**: Interpretable linear model

### Ensemble Options:
- **Voting Ensemble**: Combines diverse algorithms
- **Custom Ensemble**: Top performer combination
- **Bagging**: Variance reduction through bootstrapping

## Technical Implementation

### Dependencies:
- scikit-learn: Core ML algorithms
- pandas/numpy: Data manipulation
- xgboost/lightgbm/catboost: Gradient boosting
- pickle: Model serialization

### Code Structure:
- Consistent class-based architecture
- Modular preprocessing pipeline
- Comprehensive error handling
- Extensible hyperparameter grids
- Automated metrics collection

### Key Features:
- **Memory Efficient**: Sparse matrix handling
- **Parallel Processing**: Multi-core utilization
- **GPU Support**: Where available (XGBoost, CatBoost)
- **Reproducible**: Fixed random seeds
- **Scalable**: Configurable feature limits

## Model Comparison Results

Current performance ranking (based on validation accuracy):
1. Extra Trees: 83.84%
2. Random Forest: 83.49% 
3. XGBoost: 81.44%
4. CatBoost: 78.99%
5. QDA: 78.70%
6. LightGBM: 78.43%
7. Decision Tree: 75.67%
8. LDA: 74.60%
9. Gradient Boosting: 72.88%
10. Ridge: 71.39%

*Note: New models (Logistic Regression, SVM, Naive Bayes, AdaBoost, Bagging, Voting) performance TBD*

## Future Enhancements

### Planned Improvements:
1. **Deep Learning Integration**: BERT, RoBERTa models
2. **Advanced Ensembles**: Stacking, meta-learning
3. **Feature Engineering**: N-gram analysis, sentiment features
4. **Hyperparameter Optimization**: Bayesian optimization
5. **Model Interpretability**: SHAP, LIME analysis

### Experimental Features:
- AutoML integration
- Multi-language support
- Real-time prediction API
- Model deployment pipeline

This comprehensive suite provides a robust foundation for hate speech classification with extensive algorithm coverage and performance monitoring capabilities.
