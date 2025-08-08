# Machine Learning Models for Hate Speech Classification

This directory contains implementations of 11 different machine learning models for the hate speech classification task (Task 3). Each model is implemented as a separate Python file with a consistent interface.

## Models Implemented

Based on the benchmark performance results:

1. **Extra Trees Classifier** (`extra_trees_classifier.py`) - 83.84% accuracy
2. **Random Forest Classifier** (`random_forest_classifier.py`) - 83.49% accuracy  
3. **XGBoost Classifier** (`xgboost_classifier.py`) - 81.44% accuracy
4. **CatBoost Classifier** (`catboost_classifier.py`) - 78.99% accuracy
5. **Quadratic Discriminant Analysis** (`qda_classifier.py`) - 78.70% accuracy
6. **LightGBM Classifier** (`lightgbm_classifier.py`) - 78.43% accuracy
7. **Decision Tree Classifier** (`decision_tree_classifier.py`) - 75.67% accuracy
8. **Linear Discriminant Analysis** (`lda_classifier.py`) - 74.60% accuracy
9. **Gradient Boosting Classifier** (`gradient_boosting_classifier.py`) - 72.88% accuracy
10. **Ridge Classifier** (`ridge_classifier.py`) - 71.39% accuracy
11. **K-Nearest Neighbors** (`knn_classifier.py`) - Performance TBD

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Models

Each model can be run independently:

```bash
python extra_trees_classifier.py
python random_forest_classifier.py
python xgboost_classifier.py
# ... etc
```

### Using the Model Runner

For convenience, use the model runner script:

```bash
# List available models
python model_runner.py --list

# Run a specific model
python model_runner.py --model extra_trees

# Run all models
python model_runner.py --all
```

## Model Features

Each model implementation includes:

- **Data Loading**: Loads train.csv and test.csv
- **Text Preprocessing**: TF-IDF vectorization with optimized parameters
- **Hyperparameter Tuning**: Grid search for optimal parameters (optional)
- **Training**: Model fitting with validation
- **Prediction**: Test set predictions
- **Evaluation**: Validation metrics and reports
- **Model Persistence**: Save/load trained models
- **Submission Generation**: Kaggle-ready CSV files

## Key Hyperparameters

### Tree-Based Models (Extra Trees, Random Forest, Decision Tree)
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples for split
- `max_features`: Features considered for best split

### Boosting Models (XGBoost, LightGBM, CatBoost, Gradient Boosting)
- `n_estimators`: Number of boosting rounds
- `learning_rate`: Step size shrinkage
- `max_depth`: Maximum tree depth
- `regularization`: L1/L2 regularization parameters

### Linear Models (Ridge, LDA, QDA)
- `alpha`: Regularization strength (Ridge)
- `solver`: Algorithm for optimization
- `reg_param`: Regularization parameter (QDA)

### Distance-Based Models (KNN)
- `n_neighbors`: Number of neighbors
- `weights`: Weight function (uniform/distance)
- `metric`: Distance metric
- `algorithm`: Nearest neighbor algorithm

## Text Preprocessing Pipeline

1. **TF-IDF Vectorization**:
   - Unigrams and bigrams (1,2)
   - English stop words removal
   - Case normalization
   - Document frequency filtering

2. **Dimensionality Reduction** (for some models):
   - Truncated SVD for QDA/LDA/KNN
   - Feature selection for memory efficiency

3. **Feature Scaling** (for distance-based models):
   - StandardScaler for KNN

## Output Files

Each model generates:
- `submission_<model_name>.csv`: Kaggle submission file
- `<model_name>_model.pkl`: Trained model
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer
- Additional preprocessing objects as needed

## Performance Optimization

- **Memory Efficiency**: Sparse matrices for TF-IDF features
- **Parallel Processing**: `n_jobs=-1` for multi-core utilization
- **Early Stopping**: Implemented for boosting models
- **Validation**: Built-in train/validation splits

## Model Selection Strategy

Models are ranked by benchmark accuracy:

1. **Top Performers** (>80%): Extra Trees, Random Forest, XGBoost
2. **Strong Performers** (75-80%): CatBoost, QDA, LightGBM, Decision Tree
3. **Baseline Performers** (<75%): LDA, Gradient Boosting, Ridge, KNN

## Task 3 Compliance

All implementations follow Task 3 requirements:

- ✅ **No Deep Learning**: Only traditional ML approaches
- ✅ **Commented Code**: Detailed hyperparameter explanations
- ✅ **Multiple Models**: 11 different algorithms implemented
- ✅ **Kaggle Ready**: Submission files in correct format
- ✅ **Reproducible**: Random seeds set for consistency

## Advanced Features

- **Feature Importance**: Available for tree-based models
- **Model Interpretation**: Coefficients for linear models
- **Training Visualization**: Loss curves for boosting models
- **Cross-Validation**: Grid search with CV
- **Model Comparison**: Consistent evaluation metrics

## Example Usage in Code

```python
# Initialize any classifier
from extra_trees_classifier import ExtraTreesHateSpeechClassifier

classifier = ExtraTreesHateSpeechClassifier()

# Load and preprocess data
train_data, test_data = classifier.load_data("../../data/train.csv", "../../data/test.csv")
X_train, y_train, X_test = classifier.preprocess_text()

# Train model
classifier.train(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Create submission
test_ids = test_data['id'].values
submission = classifier.create_submission(predictions, test_ids)
```

## Racing to the Top!

For the leaderboard competition:
1. Start with top performers (Extra Trees, Random Forest, XGBoost)
2. Experiment with hyperparameter tuning
3. Consider ensemble methods (combine multiple models)
4. Focus on feature engineering improvements
5. Monitor public leaderboard for feedback

Good luck with Task 3! 🚀
