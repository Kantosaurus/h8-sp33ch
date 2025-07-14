# Hate Speech Detection using Machine Learning

This repository contains implementations for hate speech detection using TF-IDF features and various machine learning techniques. The project includes both logistic regression from scratch and dimension reduction using PCA with KNN classification.

## 📋 Project Overview

This project addresses the challenge of detecting hate speech in text data using machine learning approaches. The dataset contains TF-IDF features extracted from text data, with 5000 features per sample.

### Tasks Implemented

1. **Task 1**: Logistic Regression from Scratch
2. **Task 2**: Dimension Reduction using PCA with KNN Classification

## 🗂️ Project Structure

```
h8-sp33ch/
├── data/                          # Dataset files (ignored by git)
│   ├── train.csv                  # Original training data
│   ├── test.csv                   # Original test data
│   ├── train_tfidf_features.csv   # Training data with TF-IDF features
│   ├── test_tfidf_features.csv    # Test data with TF-IDF features
│   └── sample_submission.csv      # Sample submission format
├── logistic_regression.py         # Task 1: Logistic Regression implementation
├── pca_dimension_reduction.py     # Task 2: PCA + KNN implementation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🚀 Setup and Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd h8-sp33ch
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**
   - Place your dataset files in the `data/` directory
   - Ensure you have the following files:
     - `train_tfidf_features.csv`
     - `test_tfidf_features.csv`

## 📊 Dataset Information

- **Training samples**: 17,184
- **Test samples**: 4,296
- **Features**: 5,000 TF-IDF features
- **Classes**: Binary classification (0: Non-Hateful, 1: Hateful)
- **Class distribution**: 
  - Class 0 (Non-Hateful): 61.88%
  - Class 1 (Hateful): 38.12%

## 🎯 Task 1: Logistic Regression from Scratch

### Implementation Details
- **Algorithm**: Mini-batch gradient descent
- **Loss function**: Log loss (binary cross-entropy)
- **Activation function**: Sigmoid
- **Features**: All 5000 TF-IDF features

### Key Features
- Custom implementation without using sklearn's LogisticRegression
- Mini-batch gradient descent for efficient training
- Cost function visualization
- Comprehensive evaluation metrics

### Usage
```bash
python logistic_regression.py
```

### Output
- Training progress with loss values
- Cost function history plot
- Confusion matrix visualization
- Classification report
- Submission file: `logistic_regression_submission.csv`

## 🎯 Task 2: Dimension Reduction using PCA

### Implementation Details
- **Dimension Reduction**: Principal Component Analysis (PCA)
- **Classification**: K-Nearest Neighbors (KNN) with n_neighbors=2
- **Components tested**: 2000, 1000, 500, and 100 components

### Key Features
- PCA implementation using sklearn
- Multiple component configurations
- Explained variance analysis
- Visualization of variance ratios

### Usage
```bash
python pca_dimension_reduction.py
```

### Output
- PCA results for each component count
- Explained variance ratios
- Visualization plots
- Submission files for each configuration:
  - `pca_2000_components_submission.csv`
  - `pca_1000_components_submission.csv`
  - `pca_500_components_submission.csv`
  - `pca_100_components_submission.csv`

## 📈 Results Summary

### Task 1: Logistic Regression
- **Features used**: 5000 (all TF-IDF features)
- **Model**: Custom logistic regression with mini-batch gradient descent
- **Evaluation**: Training accuracy and comprehensive metrics

### Task 2: PCA Dimension Reduction
| Components | Explained Variance Ratio |
|------------|-------------------------|
| 2000       | 82.47%                  |
| 1000       | 64.80%                  |
| 500        | 48.31%                  |
| 100        | 20.90%                  |

**Note**: Macro F1 scores for test set predictions are obtained by submitting the generated CSV files to Kaggle.

## 📁 Generated Files

### Submission Files
- `logistic_regression_submission.csv` - Task 1 predictions
- `pca_2000_components_submission.csv` - Task 2 with 2000 components
- `pca_1000_components_submission.csv` - Task 2 with 1000 components
- `pca_500_components_submission.csv` - Task 2 with 500 components
- `pca_100_components_submission.csv` - Task 2 with 100 components

### Visualizations
- Cost function history plots
- Confusion matrices
- Explained variance ratio plots
- Cumulative explained variance plots

## 🔧 Dependencies

The project requires the following Python packages (see `requirements.txt`):
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.4.0` - Plotting and visualization
- `scikit-learn>=1.0.0` - Machine learning utilities
- `seaborn>=0.11.0` - Statistical data visualization

## 📝 Usage Instructions

1. **Run Task 1 (Logistic Regression)**:
   ```bash
   python logistic_regression.py
   ```

2. **Run Task 2 (PCA Dimension Reduction)**:
   ```bash
   python pca_dimension_reduction.py
   ```

3. **Submit to Kaggle**:
   - Upload the generated CSV files to Kaggle
   - Record the Macro F1 scores for your report

## 🎓 Academic Context

This project demonstrates:
- Implementation of machine learning algorithms from scratch
- Application of dimension reduction techniques
- Comparison of different feature representations
- Evaluation of model performance using appropriate metrics

## 📄 License

This project is created for educational purposes as part of a machine learning course.

## 🤝 Contributing

This is an academic project. For questions or issues, please refer to the course materials or contact the course instructor.

---

**Note**: The `data/` directory is ignored by git to prevent large files from being uploaded. Ensure you have the required dataset files locally before running the scripts.