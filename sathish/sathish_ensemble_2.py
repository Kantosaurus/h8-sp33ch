import numpy as np
import pandas as pd
import os

# Logistic regression with bagging ensemble

def sigmoid(z):
    z = np.clip(z, -500, 500)  # numerical stability
    return 1 / (1 + np.exp(-z))

def bce_loss(y, y_hat, sample_weights=None):
    """Binary cross-entropy (optionally weighted). y, y_hat shape: (m,1)"""
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    if sample_weights is None:
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    else:
        sw = sample_weights.reshape(-1, 1)
        return -np.sum(sw * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / np.sum(sw)

def stratified_split(X, y, test_size=0.15, seed=42):
    """Return (X_train, y_train, X_val, y_val) with simple stratification."""
    rng = np.random.RandomState(seed)
    y = y.flatten()
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0_val = int(len(idx0) * test_size)
    n1_val = int(len(idx1) * test_size)
    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def compute_class_weights(y):
    n = len(y)
    n1 = np.sum(y == 1)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return 1.0, 1.0
    w1 = n / (2.0 * n1)
    w0 = n / (2.0 * n0)
    return w0, w1

def make_sample_weights(y, class_weight=None):
    """Return per-sample weights array shape (m,)."""
    if class_weight is None:
        return None
    if class_weight == 'balanced':
        w0, w1 = compute_class_weights(y)
        sw = np.where(y == 1, w1, w0).astype(float)
        return sw
    elif isinstance(class_weight, dict):
        sw = np.vectorize(lambda t: class_weight.get(int(t), 1.0))(y.astype(int))
        return sw.astype(float)
    else:
        return None

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def find_best_threshold(y_true, probs, metric='accuracy'):
    """Scan thresholds to maximize accuracy (default)."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        score = accuracy_score(y_true, preds) if metric == 'accuracy' else 0.0
        if score > best_score:
            best_thr, best_score = thr, score
    return best_thr, best_score

def train_lr_minibatch(
    X, y, batch_size=64, epochs=1000, lr=0.01, reg_lambda=0.0, sample_weights=None, seed=42
):
    """
    Train logistic regression via mini-batch GD with L2 reg and optional per-sample weights.
    X: (n, d), y: (n,) or (n,1)
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    y = y.reshape(n, 1)
    w = np.zeros((d, 1))
    b = 0.0
    sw = None if sample_weights is None else sample_weights.reshape(n, 1)

    print(f"Training: n={n}, d={d}, epochs={epochs}, bs={batch_size}, lr={lr}, L2={reg_lambda}")
    for epoch in range(epochs):
        idx = np.arange(n)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
        if sw is not None:
            sw = sw[idx]

        running_loss, batches = 0.0, 0
        for i in range(0, n, batch_size):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            yhat = sigmoid(Xb @ w + b)

            if sw is not None:
                swb = sw[i:i+batch_size]
                # Weighted residuals
                resid = (yhat - yb) * swb
                m = np.sum(swb)
                # Weighted BCE + L2 (for logging)
                batch_loss = bce_loss(yb, yhat, swb) + (reg_lambda * np.sum(w * w)) / (2.0 * max(m, 1.0))
            else:
                resid = (yhat - yb)
                m = len(yb)
                batch_loss = bce_loss(yb, yhat) + (reg_lambda * np.sum(w * w)) / (2.0 * max(m, 1.0))

            # Gradients with L2 on weights (not on bias)
            dw = (Xb.T @ resid) / max(m, 1.0) + (reg_lambda / max(m, 1.0)) * w
            db = np.sum(resid) / max(m, 1.0)

            # Update
            w -= lr * dw
            b -= lr * db

            running_loss += float(batch_loss)
            batches += 1

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  avg_loss={running_loss / max(batches,1):.4f}")

    return w, b

def predict_proba_lr(X, w, b):
    return sigmoid(X @ w + b).flatten()

# ============== Bagging ==============
def train_bagging(
    X_train, y_train, n_bags=10, batch_size=64, epochs=1200, lr=0.005, reg_lambda=0.1,
    class_weight='balanced', seed=42
):
    """Train multiple LR models on bootstrap samples of the TRAIN split."""
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    models = []
    sw_full = make_sample_weights(y_train, class_weight=class_weight)

    print("\n" + "="*60)
    print(f"TRAIN BAGGING: bags={n_bags}, bs={batch_size}, epochs={epochs}, lr={lr}, L2={reg_lambda}, cw={class_weight}")
    print("="*60)

    for b in range(n_bags):
        print(f"\n--- Bag {b+1}/{n_bags} ---")
        boot_idx = rng.choice(n, size=n, replace=True)
        Xb = X_train[boot_idx]
        yb = y_train[boot_idx]
        swb = None if sw_full is None else sw_full[boot_idx]
        w, bias = train_lr_minibatch(
            Xb, yb, batch_size=batch_size, epochs=epochs, lr=lr, reg_lambda=reg_lambda, sample_weights=swb, seed=rng.randint(0, 1_000_000)
        )
        models.append((w, bias))
    return models

def ensemble_predict_proba(models, X):
    """Average probabilities across all models."""
    probs = []
    for (w, b) in models:
        probs.append(predict_proba_lr(X, w, b))
    return np.mean(np.vstack(probs), axis=0)

# ============== Data IO ==============
def load_train(train_path):
    print(f"Loading train: {train_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    df = pd.read_csv(train_path)
    if 'label' not in df.columns:
        raise ValueError("'label' not found in train CSV")
    y = df['label'].values.astype(int)
    X = df.drop(columns=[c for c in ['label','id'] if c in df.columns]).values
    print(f"Train shapes: X={X.shape}, y={y.shape}")
    try:
        print("Label distribution:", np.bincount(y))
    except Exception:
        pass
    return X, y

def load_test(test_path):
    print(f"Loading test: {test_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)
    df = pd.read_csv(test_path)
    if 'id' not in df.columns:
        raise ValueError("'id' not found in test CSV")
    test_ids = df['id'].values
    X = df.drop(columns=['id']).values
    print(f"Test shapes: X={X.shape}, ids={test_ids.shape}")
    return X, test_ids

# ============== Main ==============
def main():
    # --- Paths ---
    TRAIN_PATH = '/Users/sathish.k/Downloads/Git HUB/h8-sp33ch/data/train_tfidf_features.csv'
    TEST_PATH  = '/Users/sathish.k/Downloads/Git HUB/h8-sp33ch/data/test_tfidf_features.csv'
    OUTPUT_CSV = 'submission_csv_ML.csv'

    # --- Hyperparameters you can tune ---
    N_BAGS = 20              # more bags can help smooth predictions
    BATCH_SIZE = 64
    EPOCHS = 1500            # train longer with smaller lr
    LR = 0.003               # smaller lr often helps generalization
    REG_LAMBDA = 0.2         # L2 penalty (try 0.05–0.5)
    CLASS_WEIGHT = 'balanced' # None or 'balanced'
    VAL_SIZE = 0.15
    SEED = 42

    np.random.seed(SEED)

    # --- Load data ---
    X_all, y_all = load_train(TRAIN_PATH)
    X_test, test_ids = load_test(TEST_PATH)

    # --- Validation split for threshold tuning ---
    X_tr, y_tr, X_val, y_val = stratified_split(X_all, y_all, test_size=VAL_SIZE, seed=SEED)
    print(f"\nSplit: train={X_tr.shape[0]}, val={X_val.shape[0]}")

    # --- Train ensemble on TRAIN split ---
    models = train_bagging(
        X_tr, y_tr,
        n_bags=N_BAGS, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, reg_lambda=REG_LAMBDA,
        class_weight=CLASS_WEIGHT, seed=SEED
    )

    # --- Tune threshold on VAL ---
    val_probs = ensemble_predict_proba(models, X_val)
    best_thr, best_acc = find_best_threshold(y_val, val_probs, metric='accuracy')
    print(f"\nBest threshold on validation: {best_thr:.3f}  (val accuracy = {best_acc:.4f})")

    # --- Retrain on FULL TRAIN with same hyperparams (to use all data) ---
    print("\nRetraining on FULL training data with tuned hyperparams...")
    full_models = train_bagging(
        X_all, y_all,
        n_bags=N_BAGS, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, reg_lambda=REG_LAMBDA,
        class_weight=CLASS_WEIGHT, seed=SEED
    )

    # --- Predict on TEST and write Kaggle CSV ---
    test_probs = ensemble_predict_proba(full_models, X_test)
    test_labels = (test_probs >= best_thr).astype(int)

    submission = pd.DataFrame({'id': test_ids, 'label': test_labels})
    submission = submission.sort_values('id').reset_index(drop=True)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  (rows={len(submission)})")
    print("Done.")

if __name__ == '__main__':
    main()
