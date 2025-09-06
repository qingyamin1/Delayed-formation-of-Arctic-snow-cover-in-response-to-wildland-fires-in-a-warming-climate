
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn import metrics
import shap
import joblib
import matplotlib.pyplot as plt
import os
import time
import torch

# ==============================
# User Configurations
# ==============================

DATA_FEATURES = "features.csv"     # Path to CSV file with features
DATA_LABELS = "labels.csv"         # Path to CSV file with labels
MODEL_SAVE_DIR = "xgb_models"      # Directory to save best model
SHAP_SAVE_PATH = "shap_summary.png" # SHAP plot save path
SEARCH_TYPE = "random"             # "random" or "grid"

# ==============================
# Define hyperparameter search space
# ==============================

def define_hyperparameters():
    """Return a smaller hyperparameter space for fast tuning."""
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

# ==============================
# Model training
# ==============================

def train_xgb_model(X, y, param_grid, search_type="random"):
    """Train XGBoost model with nested cross-validation."""
    start_time = time.time()
    gpu_status = torch.cuda.is_available()
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    all_f2_scores = []

    count = 1
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='gpu_hist' if gpu_status else 'hist',
            random_state=42
        )

        f2_scorer = metrics.make_scorer(metrics.fbeta_score, beta=2)

        if search_type == "random":
            search = RandomizedSearchCV(
                model, param_grid, scoring=f2_scorer, n_iter=10, cv=5, verbose=1
            )
        else:
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                model, param_grid, scoring=f2_scorer, cv=5, verbose=1
            )

        search.fit(X_train_cv, y_train_cv)

        best_model = search.best_estimator_
        y_pred_val = best_model.predict(X_val_cv)
        f2 = metrics.fbeta_score(y_val_cv, y_pred_val, beta=2)
        all_f2_scores.append(f2)

        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        joblib.dump(best_model, f"{MODEL_SAVE_DIR}/xgb_model_{count}.joblib")
        print(f"Fold {count} F2 score: {f2:.3f}")
        count += 1

    print(f"Average F2 score: {np.mean(all_f2_scores):.3f} ± {np.std(all_f2_scores):.3f}")
    print(f"Training completed in {(time.time()-start_time)/60:.2f} minutes")
    return best_model

# ==============================
# Model evaluation
# ==============================

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    f2 = metrics.fbeta_score(y_test, y_pred, beta=2)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print(f"F2 Score: {f2:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}")
    return cm, f2, recall, precision

# ==============================
# SHAP interpretation
# ==============================

def shap_summary_plot(model, X_test, save_path):
    """Generate and save SHAP summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP plot saved to {save_path}")

# ==============================
# Main flow
# ==============================

if __name__ == "__main__":
    # Load datasets
    X = pd.read_csv(DATA_FEATURES)
    y = pd.read_csv(DATA_LABELS).iloc[:, 0]  # assume label is first column

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train
    params = define_hyperparameters()
    best_model = train_xgb_model(X_train, y_train, params, search_type=SEARCH_TYPE)

    # Save final model
    joblib.dump(best_model, f"{MODEL_SAVE_DIR}/xgb_best_model.joblib")

