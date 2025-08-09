import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')

# Add root directory to sys.path so we can import models
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import base models from models/ folder at root
try:
    from models.svm_model import SVMModel
    from models.random_forest_model import RandomForestModel
    from models.lightgbm_model import LightGBMModel
except ImportError:
    # Fallback if custom models aren't available
    print("Custom models not found, using sklearn defaults")
    SVMModel = None
    RandomForestModel = None
    LightGBMModel = None


class OptimizedEnsembleClassifier:
    def __init__(self, optimization_method='optuna'):
        self.optimization_method = optimization_method
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.best_params = {}
        
    def load_features_and_labels(self):
        """Load TF-IDF features and labels from ../data/ directory"""
        print("Loading TF-IDF features and labels...")
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

        X_train = pd.read_csv(os.path.join(data_dir, 'train_tfidf_features.csv'))
        if 'label' in X_train.columns:
            X_train = X_train.drop(columns=['label'])

        y_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))['label'].values
        X_test = pd.read_csv(os.path.join(data_dir, 'test_tfidf_features.csv'))

        return X_train, y_train, X_test

    def optimize_with_optuna_fast(self, X, y, n_trials=100):
        """🚀 ULTRA-FAST Optuna optimization with aggressive speed optimizations"""
        print(f"🚀 Ultra-fast Optuna optimization ({n_trials} trials)...")
        
        # Use much smaller sample for hyperparameter tuning
        if len(X) > 2000:
            print("Using small subset (2000 samples) for ultra-fast tuning...")
            X_sample, _, y_sample, _ = train_test_split(X, y, train_size=2000, random_state=42, stratify=y)
        else:
            X_sample, y_sample = X, y
        
        def objective(trial):
            # Simplified search space with fewer options
            svm_c = trial.suggest_float('svm_c', 0.1, 10)
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 150)
            lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 50, 150)
            lgb_learning_rate = trial.suggest_float('lgb_learning_rate', 0.05, 0.2)
            
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from lightgbm import LGBMClassifier
            
            # Fixed configurations for speed
            svm_model = SVC(C=svm_c, kernel='linear', probability=True, random_state=42, max_iter=500)
            rf_model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=10, random_state=42, n_jobs=1)
            lgb_model = LGBMClassifier(n_estimators=lgb_n_estimators, learning_rate=lgb_learning_rate, 
                                     max_depth=5, random_state=42, verbose=-1, n_jobs=1)
            
            # Small validation split
            X_train_fast, X_val_fast, y_train_fast, y_val_fast = train_test_split(
                X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
            )
            
            try:
                # Train models
                svm_model.fit(X_train_fast, y_train_fast)
                rf_model.fit(X_train_fast, y_train_fast)
                lgb_model.fit(X_train_fast, y_train_fast)
                
                # Simple averaging ensemble instead of blending


                svm_pred = svm_model.predict_proba(X_val_fast)[:, 1]
                rf_pred = rf_model.predict_proba(X_val_fast)[:, 1]
                lgb_pred = lgb_model.predict_proba(X_val_fast)[:, 1]
                
                # Simple average
                ensemble_pred = (svm_pred + rf_pred + lgb_pred) / 3
                ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
                
                return f1_score(y_val_fast, ensemble_pred_binary, average='macro')
            except:
                return 0.0
        
        # Minimal study configuration
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.RandomSampler(seed=42),  # Fastest sampler
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Convert to full parameter format
        best = study.best_params
        self.best_params = {
            'svm_c': best['svm_c'],
            'svm_kernel': 'linear',
            'rf_n_estimators': best['rf_n_estimators'],
            'rf_max_depth': 10,
            'rf_min_samples_split': 2,
            'lgb_n_estimators': best['lgb_n_estimators'],
            'lgb_learning_rate': best['lgb_learning_rate'],
            'lgb_max_depth': 5,
            'meta_c': 1.0,
            'meta_solver': 'liblinear'
        }
        
        print(f"🚀 Ultra-fast optimization complete! Best F1: {study.best_value:.4f}")
        return self.best_params

    def optimize_base_models_grid_search(self, X, y):
        """Optimize base models using GridSearchCV"""
        print("Optimizing base models with GridSearchCV...")
        
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        
        # SVM optimization
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        svm_model = SVC(probability=True, random_state=42)
        svm_grid = GridSearchCV(svm_model, svm_params, cv=3, scoring='f1_macro', n_jobs=-1)
        svm_grid.fit(X, y)
        self.base_models['svm'] = svm_grid.best_estimator_
        self.best_params['svm'] = svm_grid.best_params_
        
        # Random Forest optimization
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_model = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='f1_macro', n_jobs=-1)
        rf_grid.fit(X, y)
        self.base_models['rf'] = rf_grid.best_estimator_
        self.best_params['rf'] = rf_grid.best_params_
        
        # LightGBM optimization
        lgb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        lgb_model = LGBMClassifier(random_state=42, verbose=-1)
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='f1_macro', n_jobs=-1)
        lgb_grid.fit(X, y)
        self.base_models['lgb'] = lgb_grid.best_estimator_
        self.best_params['lgb'] = lgb_grid.best_params_


    def optimize_with_optuna(self, X, y, n_trials=100):
        """Optimize ensemble using Optuna for more sophisticated hyperparameter tuning"""
        print(f"Optimizing with Optuna ({n_trials} trials)...")
        
        # 🚀 SPEED OPTIMIZATION 1: Use smaller data subset for hyperparameter tuning
        if len(X) > 5000:
            print("Using subset of data for faster tuning...")
            X_sample, _, y_sample, _ = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
        else:
            X_sample, y_sample = X, y
        
        def objective(trial):
            # 🚀 SPEED OPTIMIZATION 2: Reduced hyperparameter search space
            # SVM - Focus on most important parameters
            svm_c = trial.suggest_float('svm_c', 0.1, 10, log=True)  # Narrower range
            svm_kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf'])
            
            # Random Forest - Reduced ranges for speed
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 200)  # Smaller range
            rf_max_depth = trial.suggest_int('rf_max_depth', 5, 15)  # Smaller range
            rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 5)
            
            # LightGBM - Reduced ranges
            lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 50, 200)
            lgb_learning_rate = trial.suggest_float('lgb_learning_rate', 0.05, 0.2)
            lgb_max_depth = trial.suggest_int('lgb_max_depth', 3, 7)
            
            # Meta-model - Simplified
            meta_c = trial.suggest_float('meta_c', 0.1, 10, log=True)
            
            # Create models with suggested parameters
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from lightgbm import LGBMClassifier
            
            # 🚀 SPEED OPTIMIZATION 3: Use faster model configurations
            svm_model = SVC(
                C=svm_c, 
                kernel=svm_kernel,
                gamma='scale',  # Fixed to fastest option
                probability=True, 
                random_state=42,
                max_iter=1000  # Limit iterations for speed
            )
            rf_model = RandomForestClassifier(
                n_estimators=rf_n_estimators, 
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                random_state=42,
                n_jobs=2  # Limit parallelization to avoid overhead
            )
            lgb_model = LGBMClassifier(
                n_estimators=lgb_n_estimators, 
                learning_rate=lgb_learning_rate,
                max_depth=lgb_max_depth, 
                random_state=42, 
                verbose=-1,
                n_jobs=2,  # Limit parallelization
                force_col_wise=True  # Faster for wide datasets
            )
            
            # 🚀 SPEED OPTIMIZATION 4: Single train-validation split instead of CV
            X_base, X_blend, y_base, y_blend = train_test_split(
                X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample
            )
            
            try:
                # Train base models
                svm_model.fit(X_base, y_base)
                rf_model.fit(X_base, y_base)
                lgb_model.fit(X_base, y_base)
                
                # Generate meta-features
                svm_blend = svm_model.predict_proba(X_blend)
                rf_blend = rf_model.predict_proba(X_blend)
                lgb_blend = lgb_model.predict_proba(X_blend)
                X_meta_blend = np.hstack([svm_blend, rf_blend, lgb_blend])
                
                # Train and evaluate meta-model
                meta_model = LogisticRegression(
                    C=meta_c, 
                    solver='liblinear',  # Faster for small datasets
                    random_state=42,
                    max_iter=1000
                )
                meta_model.fit(X_meta_blend, y_blend)
                blend_preds = meta_model.predict(X_meta_blend)

                
                return f1_score(y_blend, blend_preds, average='macro')
            except Exception as e:
                return 0.0  # Silent failure for speed
        
        # 🚀 SPEED OPTIMIZATION 5: Enhanced pruning and sampling
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,  # Quick initial random trials
                n_ei_candidates=24,   # Fewer candidates for speed
                seed=42
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,   # Start pruning early
                n_warmup_steps=3,     # Quick warmup
                interval_steps=1      # Check every step
            )
        )
        
        # 🚀 SPEED OPTIMIZATION 6: Progress callback for monitoring
        def callback(study, trial):
            if trial.number % 10 == 0:
                print(f"Trial {trial.number}: Best so far = {study.best_value:.4f}")
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        
        self.best_params = study.best_params
        print(f"Best F1 score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study.best_params

    def train_optimized_ensemble(self, X, y, best_params=None):
        """Train the ensemble with optimized parameters"""
        if best_params is None:
            best_params = self.best_params
            
        if not best_params:
            raise ValueError("No best_params found! You must run optimization first.")
            
        print("Training optimized ensemble...")
        
        # Split data
        X_base, X_blend, y_base, y_blend = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Initialize optimized base models
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        
        if self.optimization_method == 'optuna':
            svm_model = SVC(
                C=best_params['svm_c'], 
                gamma=best_params['svm_gamma'],
                kernel=best_params.get('svm_kernel', 'rbf'),
                probability=True, 
                random_state=42
            )
            rf_model = RandomForestClassifier(
                n_estimators=best_params['rf_n_estimators'], 
                max_depth=best_params['rf_max_depth'],
                min_samples_split=best_params.get('rf_min_samples_split', 2),
                random_state=42
            )
            lgb_model = LGBMClassifier(
                n_estimators=best_params['lgb_n_estimators'], 
                learning_rate=best_params['lgb_learning_rate'],
                max_depth=best_params['lgb_max_depth'], 
                random_state=42, 
                verbose=-1
            )
        else:
            # Use GridSearchCV optimized models
            svm_model = self.base_models['svm']
            rf_model = self.base_models['rf']
            lgb_model = self.base_models['lgb']
        
        # Train base models
        svm_model.fit(X_base, y_base)
        rf_model.fit(X_base, y_base)
        lgb_model.fit(X_base, y_base)
        
        self.base_models = {
            'svm': svm_model,
            'rf': rf_model,
            'lgb': lgb_model
        }
        
        # Generate meta-features for blending
        svm_blend = svm_model.predict_proba(X_blend)
        rf_blend = rf_model.predict_proba(X_blend)
        lgb_blend = lgb_model.predict_proba(X_blend)
        X_meta_blend = np.hstack([svm_blend, rf_blend, lgb_blend])
        
        # Normalize meta-features
        X_meta_blend_scaled = self.scaler.fit_transform(X_meta_blend)
        
        # Train meta-classifier with optimized parameters
        if self.optimization_method == 'optuna':
            meta_c = best_params.get('meta_c', 1.0)
            meta_solver = best_params.get('meta_solver', 'lbfgs')
        else:
            meta_c = 1.0

            meta_solver = 'lbfgs'
            
        self.meta_model = LogisticRegression(C=meta_c, solver=meta_solver, random_state=42)
        self.meta_model.fit(X_meta_blend_scaled, y_blend)
        
        # Evaluate on blend set
        blend_preds = self.meta_model.predict(X_meta_blend_scaled)
        blend_f1 = f1_score(y_blend, blend_preds, average='macro')
        print(f"Blend set F1 score: {blend_f1:.4f}")
        
        return X_base, X_blend, y_base, y_blend

    def predict(self, X_test):
        """Make predictions on test set"""
        print("Generating predictions on test set...")
        
        if not self.base_models:
            raise ValueError("Models not trained yet!")
        
        # Generate base model predictions
        svm_test = self.base_models['svm'].predict_proba(X_test)
        rf_test = self.base_models['rf'].predict_proba(X_test)
        lgb_test = self.base_models['lgb'].predict_proba(X_test)
        X_meta_test = np.hstack([svm_test, rf_test, lgb_test])
        
        # Scale meta-features
        X_meta_test_scaled = self.scaler.transform(X_meta_test)
        
        # Generate final predictions
        final_preds = self.meta_model.predict(X_meta_test_scaled)
        return final_preds

    def evaluate_model(self, X, y, cv_folds=5):
        """Perform cross-validation evaluation"""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        scores = []
        for fold in range(cv_folds):
            print(f"Processing fold {fold + 1}/{cv_folds}...")
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                X, y, test_size=0.2, random_state=fold, stratify=y
            )
            
            # Create a temporary ensemble for this fold
            temp_ensemble = OptimizedEnsembleClassifier(self.optimization_method)
            temp_ensemble.best_params = self.best_params.copy()
            
            # Train on fold
            temp_ensemble.train_optimized_ensemble(X_train_fold, y_train_fold)
            
            # Predict on validation
            val_preds = temp_ensemble.predict(X_val_fold)
            fold_score = f1_score(y_val_fold, val_preds, average='macro')
            scores.append(fold_score)
            print(f"Fold {fold + 1} F1 score: {fold_score:.4f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Cross-validation F1 score: {mean_score:.4f} (+/- {std_score:.4f})")
        
        return mean_score, std_score


def main():
    print("== HATE SPEECH ENSEMBLE WITH FINETUNING ==")
    
    # Get user choice for optimization method
    print("Choose optimization method:")
    print("1: GridSearch (thorough but slow)")
    print("2: Optuna (balanced speed/quality)")
    print("3: Ultra-fast Optuna (fastest)")
    
    opt_choice = input("Enter choice (1/2/3): ")
    
    if opt_choice == '1':
        opt_method = 'grid_search'
    elif opt_choice == '3':
        opt_method = 'optuna_fast'
    else:
        opt_method = 'optuna'
    
    # Initialize ensemble classifier
    ensemble = OptimizedEnsembleClassifier(optimization_method=opt_method)
    
    # Load data
    X_train, y_train, X_test = ensemble.load_features_and_labels()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels distribution: {np.bincount(y_train)}")
    
    # Run optimization
    if opt_method == 'grid_search':
        ensemble.optimize_base_models_grid_search(X_train, y_train)
    elif opt_method == 'optuna_fast':
        n_trials = int(input("Enter number of trials (default: 50 for ultra-fast): ") or "50")
        ensemble.optimize_with_optuna_fast(X_train, y_train, n_trials=n_trials)
    else:
        n_trials = int(input("Enter number of Optuna trials (default: 50): ") or "50")
        ensemble.optimize_with_optuna(X_train, y_train, n_trials=n_trials)
    
    # Train the optimized ensemble
    ensemble.train_optimized_ensemble(X_train, y_train)
    
    # Cross-validation evaluation
    eval_choice = input("Perform cross-validation evaluation? (y/n): ")
    if eval_choice.lower() == 'y':
        ensemble.evaluate_model(X_train, y_train)
    
    # Generate final predictions
    predictions = ensemble.predict(X_test)
    
    # Save submission
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Try to load test IDs
    try:
        test_ids = pd.read_csv(os.path.join(data_dir, 'test.csv'))["id"]
    except:
        # If test.csv doesn't have ID column, create sequential IDs
        test_ids = range(len(predictions))
    
    submission = pd.DataFrame({
        "id": test_ids,
        "label": predictions
    })
    
    suffix = f"_{ensemble.optimization_method}" if hasattr(ensemble, 'optimization_method') else ""
    submission_path = os.path.join(data_dir, f"submission_optimized_blend{suffix}.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Optimized submission saved to {submission_path}")
    
    # Print best parameters if available
    if ensemble.best_params:
        print("\n=== BEST PARAMETERS ===")
        for param, value in ensemble.best_params.items():
            print(f"{param}: {value}")


if __name__ == "__main__":
    main()
