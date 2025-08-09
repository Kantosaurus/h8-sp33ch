# Deep Neural Network-Like Ensemble for Hate Speech Detection

This project implements a **Deep Neural Network-inspired ensemble classifier** that mimics the architecture and behavior of deep learning without actually using neural networks. Instead, it uses traditional machine learning models arranged in layers to create a deep learning-like system.

## 🧠 Architecture Overview

The system replicates key DNN concepts using ensemble methods:

### Network Architecture
```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Hidden Layer 3 → Output Layer
    ↓              ↓              ↓              ↓              ↓
Text Features → Transformations → Transformations → Transformations → Classification
```

### Key Components

1. **Input Layer (Feature Extraction)**
   - TF-IDF word n-grams (1-3)
   - TF-IDF character n-grams (2-5)
   - Count vectorization
   - Hashing vectorization
   - Statistical features (sentiment, length, punctuation, etc.)

2. **Hidden Layers (3 layers, 6 models each)**
   - Each layer contains diverse ML models acting as "neurons"
   - Models: Random Forest, XGBoost, Logistic Regression, SVM, Naive Bayes, KNN, etc.
   - Different feature transformations per model (PCA, SVD, feature selection)
   - Non-linear activation functions between layers

3. **Output Layer (Final Ensemble)**
   - Voting classifier with top-performing models
   - Soft voting using probability predictions

## 🔄 Deep Learning Concepts Replicated

### Forward Propagation
- Data flows through layers sequentially
- Each layer transforms input using multiple models
- Non-linear activations applied between layers

### Activation Functions
- ReLU-like: `max(0, x)`
- Sigmoid-like: `1 / (1 + exp(-x))`
- Tanh-like: `tanh(x)`
- Leaky ReLU-like: `max(0.01*x, x)`

### Regularization
- "Dropout": Random model selection per layer
- Feature transformation diversity
- Cross-validation for robust training

### Hierarchical Learning
- Each layer learns increasingly complex representations
- Layer 1: Basic patterns
- Layer 2: Combination patterns
- Layer 3: High-level abstractions
- Output: Final classification

## 🚀 Usage

### Installation
```bash
cd DNN
pip install -r requirements.txt
```

### Training the Model
```bash
python train_deep_ensemble.py
```

### Using the Trained Model
```python
from deep_ensemble_classifier import DeepEnsembleClassifier
import pickle

# Load trained model
with open('DNN/deep_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
texts = ["Sample text to classify"]
predictions = model.predict(texts)
probabilities = model.predict_proba(texts)
```

## 📊 Model Performance

The ensemble is optimized for **Macro F1 Score**:
```
F1 = TP / (TP + ½(FP + FN))
Macro F1 = (F1_hateful + F1_non_hateful) / 2
```

### Expected Performance
- **Cross-validation**: ~0.85+ Macro F1
- **Ensemble diversity**: 23+ models across all layers
- **Feature complexity**: 20,000+ input features
- **Training time**: 15-30 minutes

## 🏗️ Technical Implementation

### Layer Structure
Each hidden layer:
- 6 diverse ML models
- Different feature transformations
- Individual model training
- Activation function application
- Output combination

### Model Selection Per Layer
- **Linear**: Logistic Regression, SVM
- **Tree-based**: Random Forest, Extra Trees, Gradient Boosting
- **Probabilistic**: Naive Bayes
- **Instance-based**: K-Nearest Neighbors
- **Advanced**: XGBoost, CatBoost, LightGBM

### Feature Engineering Pipeline
1. **Text Preprocessing**: Cleaning, normalization
2. **Vectorization**: Multiple text representation methods
3. **Statistical Features**: Length, sentiment, punctuation analysis
4. **Dimensionality Reduction**: PCA, SVD for different model inputs
5. **Feature Selection**: SelectKBest for focused learning

## 📁 Output Files

After training:
- `deep_ensemble_model.pkl`: Trained ensemble model
- `deep_ensemble_predictions.csv`: Test set predictions
- `deep_ensemble_analysis.csv`: Detailed prediction analysis
- `network_summary.txt`: Architecture and performance summary

## 🎯 Innovation Features

### Deep Learning Simulation
- **Layer-wise learning**: Progressive feature abstraction
- **Non-linear transformations**: Multiple activation functions
- **Ensemble diversity**: Different model types per layer
- **Regularization**: Dropout-like model selection

### Optimization Techniques
- **Parallel processing**: Multi-core model training
- **Memory efficiency**: Sparse matrix operations
- **Cross-validation**: Robust performance estimation
- **Adaptive architecture**: Layer-wise performance monitoring

## 🔧 Customization

### Modify Architecture
```python
deep_ensemble = DeepEnsembleClassifier(
    n_hidden_layers=4,      # Add more layers
    models_per_layer=8,     # More models per layer
    random_state=42
)
```

### Add Custom Models
Extend the `HiddenLayer._create_models()` method to include new ML algorithms.

### Feature Engineering
Modify `DeepTextFeatureExtractor` to add domain-specific features.

## 🎯 Why This Approach?

1. **No Neural Networks**: Complies with restrictions while achieving DNN-like behavior
2. **Interpretability**: Individual models remain interpretable
3. **Robustness**: Ensemble approach reduces overfitting
4. **Flexibility**: Easy to modify and extend
5. **Performance**: Competitive with actual neural networks for text classification

## 📈 Performance Monitoring

The system tracks:
- Layer-wise performance improvements
- Cross-validation scores
- Feature importance across layers
- Model contribution analysis
- Prediction confidence distributions

This creates a comprehensive deep learning-like system using only traditional ML methods!