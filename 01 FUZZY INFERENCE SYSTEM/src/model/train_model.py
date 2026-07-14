# src/model/train_model.py
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle as pk
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from os import listdir as listdir
import os.path as path
import pickle as pk

from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, 
    log_loss, confusion_matrix
)

from src.preprocess import preprocessing as pp
from src.loader import data_loader as dl
from src.model import evaluate_model as em

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


def split_data(df: pd.DataFrame, target_col='bad_good', test_size=0.2, random_state=42):
    """Split DataFrame into training and testing sets.

    This function performs stratified train-test splitting on the input DataFrame,
    separating features from the target variable and displaying the resulting shapes.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_col (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test DataFrames/Series.
    """
    print("\n[INFO] Splitting data into training and testing sets")
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def split_data_base_cosi(X_train: pd.DataFrame, y_train: pd.Series,
                          cosi_size: float = 0.30,
                          random_state: int = 42):
    """Split the training set into two non-overlapping portions.

    This function divides the training data into two disjoint subsets to prevent
    data leakage in the RuleCOSI+ interpretability pipeline:

    - **Base portion**: used to train the base ensemble (CatBoost / RF / LightGBM).
    - **COSI portion**: used exclusively by RuleCOSI+ to extract interpretable rules
      from the already-trained base model. Because the base model has never seen
      these samples, the extracted rules reflect true generalisation rather than
      memorised training patterns.

    The split is stratified to preserve the class distribution in both subsets.

    Args:
        X_train (pd.DataFrame): Feature matrix from the original train/test split.
        y_train (pd.Series): Target vector corresponding to X_train.
        cosi_size (float): Fraction of X_train reserved for RuleCOSI+.
                           Defaults to 0.30 (30 % COSI / 70 % base).
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (X_train_base, X_train_cosi, y_train_base, y_train_cosi)
    """
    print("\n[INFO] Splitting training set into Base and COSI portions")
    print(f"       Base: {round((1 - cosi_size) * 100)}%  |  COSI: {round(cosi_size * 100)}%")

    X_train_base, X_train_cosi, y_train_base, y_train_cosi = train_test_split(
        X_train, y_train,
        test_size=cosi_size,
        stratify=y_train,
        random_state=random_state
    )

    print(f"  X_train_base : {X_train_base.shape}  —  class ratio: {y_train_base.mean():.3f}")
    print(f"  X_train_cosi : {X_train_cosi.shape}  —  class ratio: {y_train_cosi.mean():.3f}")

    return X_train_base, X_train_cosi, y_train_base, y_train_cosi


def save_model(
    model,
    metrics: Dict,
    X_train_encoded: pd.DataFrame,
    category_mappings: Dict,
    model_name: str,
    output_dir: str = "models"
) -> Dict:
    """Save trained model and associated artifacts to disk.

    This function saves the model, performance metrics, encoded training data, and
    categorical feature mappings to the specified directory. The function automatically
    detects the model type and uses the appropriate serialization method for
    RandomForest Pipeline, CatBoost, LightGBM, or sklearn models.

    Args:
        model: Trained model object (Pipeline, CatBoost, LightGBM, or sklearn model).
        metrics (Dict): Performance metrics dictionary from training.
        X_train_encoded (pd.DataFrame): Encoded training data for rule extraction.
        category_mappings (Dict): Category mappings in format {feature: {category: code}}.
        model_name (str): Name identifier for the model files.
        output_dir (str): Directory path to save files.

    Returns:
        Dict: Dictionary containing paths to all saved files including model, metrics,
              encoded data, category mappings, and feature importances if available.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{model_name}_{timestamp}"
    
    saved_paths = {}
    
    model_type = type(model).__name__
    
    if hasattr(model, 'named_steps'):
        model_path = output_path / f"{base_filename}_model.pkl"
        joblib.dump(model, model_path)
        saved_paths['model'] = str(model_path)
        
    elif model_type == 'CatBoostClassifier':
        model_path = output_path / f"{base_filename}_model.cbm"
        model.save_model(str(model_path))
        saved_paths['model'] = str(model_path)
        
    elif model_type == 'Booster':
        model_path = output_path / f"{base_filename}_model.txt"
        model.save_model(str(model_path))
        saved_paths['model'] = str(model_path)
        
    else:
        model_path = output_path / f"{base_filename}_model.pkl"
        joblib.dump(model, model_path)
        saved_paths['model'] = str(model_path)
        
        if metrics.get('encoder') is not None:
            encoder_path = output_path / f"{base_filename}_encoder.pkl"
            joblib.dump(metrics['encoder'], encoder_path)
            saved_paths['encoder'] = str(encoder_path)
    
    X_train_path = output_path / f"{base_filename}_X_train_encoded.parquet"
    X_train_encoded.to_parquet(X_train_path, index=False)
    saved_paths['X_train_encoded'] = str(X_train_path)
    
    mappings_path = output_path / f"{base_filename}_category_mappings.pkl"
    joblib.dump(category_mappings, mappings_path)
    saved_paths['category_mappings'] = str(mappings_path)
    
    metrics_json = {}
    for key, value in metrics.items():
        if key in ['feature_importances', 'encoder', 'category_mappings']:
            continue
        elif key in ['confusion_matrix', 'val_confusion_matrix']:
            metrics_json[key] = value.tolist() if hasattr(value, 'tolist') else value
        elif hasattr(value, 'tolist'):
            metrics_json[key] = value.tolist()
        else:
            metrics_json[key] = value
    
    metrics_json['model_name'] = model_name
    metrics_json['saved_timestamp'] = timestamp
    metrics_json['saved_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_json['has_category_mappings'] = len(category_mappings) > 0
    metrics_json['encoded_features'] = list(category_mappings.keys())
    
    metrics_path = output_path / f"{base_filename}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2, default=lambda o:
              int(o) if isinstance(o, (np.integer,))
              else float(o) if isinstance(o, (np.floating,))
              else o.tolist() if isinstance(o, np.ndarray)
              else o)
    saved_paths['metrics'] = str(metrics_path)
    
    if 'feature_importances' in metrics:
        fi_path = output_path / f"{base_filename}_feature_importances.csv"
        metrics['feature_importances'].to_csv(fi_path, index=False)
        saved_paths['feature_importances'] = str(fi_path)
    
    print(f"\n[INFO] Model saved successfully")
    print(f"  Model:              {saved_paths['model']}")
    print(f"  Metrics:            {saved_paths['metrics']}")
    print(f"  X_train_encoded:    {saved_paths['X_train_encoded']}")
    print(f"  Category mappings:  {saved_paths['category_mappings']}")
    if 'feature_importances' in saved_paths:
        print(f"  Feature importances: {saved_paths['feature_importances']}")
    
    return saved_paths


def load_model(
    model_path: str,
    load_metrics: bool = True,
    load_data: bool = True
) -> tuple:
    """Load trained model and associated artifacts from disk.

    This function automatically detects the model type based on file extension and
    loads all associated files including metrics, encoded training data, category
    mappings, feature importances, and encoders. Supports CatBoost (.cbm), LightGBM
    (.txt), and sklearn models (.pkl).

    Args:
        model_path (str): Path to the model file or base filename without extension.
                         Supports paths like 'models/rf_20250126_143022_model.pkl'
                         or 'models/rf_20250126_143022'.
        load_metrics (bool): Whether to load metrics and associated files.
        load_data (bool): Whether to load X_train_encoded data.

    Returns:
        tuple: A tuple containing:
            - model: Loaded model (Pipeline, CatBoost, LightGBM, or sklearn model)
            - metrics (Dict or None): Metrics dictionary if load_metrics=True
            - X_train_encoded (pd.DataFrame or None): Encoded training data if load_data=True
            - category_mappings (Dict or None): Category mappings if load_metrics=True
    """
    model_path = Path(model_path)
    
    if not model_path.exists() and model_path.suffix == '':
        for ext in ['.pkl', '.cbm', '.txt', '_model.pkl', '_model.cbm', '_model.txt']:
            test_path = Path(str(model_path) + ext)
            if test_path.exists():
                model_path = test_path
                break
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    base_path = str(model_path).replace('_model.pkl', '').replace('_model.cbm', '').replace('_model.txt', '')
    base_path = base_path.replace('.pkl', '').replace('.cbm', '').replace('.txt', '')
    
    if model_path.suffix == '.cbm':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        print(f"[INFO] CatBoost model loaded from: {model_path}")
        
    elif model_path.suffix == '.txt':
        model = lgb.Booster(model_file=str(model_path))
        print(f"[INFO] LightGBM model loaded from: {model_path}")
        
    elif model_path.suffix == '.pkl':
        model = joblib.load(model_path)
        
        if hasattr(model, 'named_steps'):
            model_type = "RandomForest Pipeline"
        elif hasattr(model, 'rules_'):
            model_type = "RuleFit"
        else:
            model_type = "Sklearn Model"
        
        print(f"[INFO] {model_type} loaded from: {model_path}")
    else:
        raise ValueError(f"Unsupported model file extension: {model_path.suffix}")
    
    metrics = None
    X_train_encoded = None
    category_mappings = None
    
    if load_metrics:
        metrics = {}
        
        metrics_path = Path(f"{base_path}_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            for cm_key in ['confusion_matrix', 'val_confusion_matrix']:
                if cm_key in metrics:
                    metrics[cm_key] = np.array(metrics[cm_key])
            
            print(f"[INFO] Metrics loaded from: {metrics_path}")
        else:
            print(f"[WARNING] Metrics file not found: {metrics_path}")
        
        mappings_path = Path(f"{base_path}_category_mappings.pkl")
        if mappings_path.exists():
            category_mappings = joblib.load(mappings_path)
            print(f"[INFO] Category mappings loaded from: {mappings_path}")
            print(f"  Encoded features: {list(category_mappings.keys())}")
        else:
            print(f"[WARNING] Category mappings file not found: {mappings_path}")
            category_mappings = {}
        
        fi_path = Path(f"{base_path}_feature_importances.csv")
        if fi_path.exists():
            metrics['feature_importances'] = pd.read_csv(fi_path)
            print(f"[INFO] Feature importances loaded from: {fi_path}")
        
        encoder_path = Path(f"{base_path}_encoder.pkl")
        if encoder_path.exists():
            metrics['encoder'] = joblib.load(encoder_path)
            print(f"[INFO] Encoder loaded from: {encoder_path}")
    
    if load_data:
        X_train_path = Path(f"{base_path}_X_train_encoded.parquet")
        if X_train_path.exists():
            X_train_encoded = pd.read_parquet(X_train_path)
            print(f"[INFO] X_train_encoded loaded from: {X_train_path}")
            print(f"  Shape: {X_train_encoded.shape}")
        else:
            print(f"[WARNING] X_train_encoded file not found: {X_train_path}")
    
    return model, metrics, X_train_encoded, category_mappings


def analyze_class_balance(y_train: pd.Series, verbose: bool = True) -> Dict:
    """Analyze class distribution in the target variable.

    This function computes class counts, proportions, and imbalance ratio for
    binary classification problems.

    Args:
        y_train (pd.Series): Target variable series.
        verbose (bool): Whether to print distribution statistics.

    Returns:
        Dict: Dictionary containing class_0_count, class_1_count, class_0_pct,
              class_1_pct, and imbalance_ratio.
    """
    class_counts = y_train.value_counts()
    class_0_count = class_counts.get(0, 0)
    class_1_count = class_counts.get(1, 0)
    total = len(y_train)
    
    imbalance_ratio = class_0_count / class_1_count if class_1_count > 0 else float('inf')
    
    if verbose:
        print(f"\n[INFO] Class Distribution Analysis:")
        print(f"  Class 0: {class_0_count:>8,} ({class_0_count/total:>6.1%})")
        print(f"  Class 1: {class_1_count:>8,} ({class_1_count/total:>6.1%})")
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    return {
        'class_0_count': class_0_count,
        'class_1_count': class_1_count,
        'class_0_pct': class_0_count / total,
        'class_1_pct': class_1_count / total,
        'imbalance_ratio': imbalance_ratio
    }

def calculate_class_weights(y: pd.Series, method: str = 'balanced') -> Dict[int, float]:
    """Calculate class weights for handling imbalanced datasets.

    This function computes weights inversely proportional to class frequencies
    using different balancing strategies.

    Args:
        y (pd.Series): Target variable series.
        method (str): Weighting method. Options are:
                     'balanced' - inversely proportional to class frequencies
                     'sqrt' - square root of balanced weights (less aggressive)
                     'moderate' - balanced weights capped at 1:3 ratio

    Returns:
        Dict[int, float]: Dictionary mapping class labels to their weights.
    """
    class_counts = y.value_counts()
    total = len(y)
    
    if method == 'balanced':
        weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    elif method == 'sqrt':
        balanced = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        weights = {cls: np.sqrt(w) for cls, w in balanced.items()}
    elif method == 'moderate':
        balanced = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        max_ratio = 3.0
        min_weight = min(balanced.values())
        weights = {cls: min(w, min_weight * max_ratio) for cls, w in balanced.items()}
    else:
        weights = {cls: 1.0 for cls in class_counts.index}
    
    return weights

def optimize_threshold_constrained(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_threshold: float = 0.40,
    max_threshold: float = 0.65,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """Find optimal classification threshold within specified constraints.

    This function searches for the threshold that maximizes the specified metric
    while remaining within the defined bounds.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        min_threshold (float): Minimum threshold value to consider.
        max_threshold (float): Maximum threshold value to consider.
        metric (str): Optimization metric. Options are:
                     'f1' - maximize F1 score
                     'balanced' - balance precision and recall (both > 0.5)
                     'recall_priority' - prioritize recall with minimum precision of 0.6

    Returns:
        Tuple[float, float]: A tuple containing (optimal_threshold, best_score).
    """
    thresholds = np.arange(min_threshold, max_threshold + 0.01, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if prec > 0.5 and rec > 0.5:
                score = 2 * (prec * rec) / (prec + rec)
            else:
                score = 0
        elif metric == 'recall_priority':
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if prec >= 0.60:
                score = rec
            else:
                score = 0
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score

def train_lightgbm_optimized(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    feature_fraction: float = 0.9,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 5,
    scale_pos_weight: Optional[float] = None,
    min_threshold: float = 0.40,
    max_threshold: float = 0.65,
    threshold_metric: str = 'f1',
    num_threads: int = -1,
    early_stopping_rounds: int = 30,
    validation_split: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> tuple[lgb.Booster, Dict, pd.DataFrame, Dict]:
    """Train LightGBM model with optimized configuration and threshold selection.

    This function trains a LightGBM gradient boosting model after encoding categorical
    features to numeric codes. The function performs validation splitting, optimizes
    the classification threshold within specified bounds, and computes comprehensive
    performance metrics including log loss.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        n_estimators (int): Number of boosting iterations.
        max_depth (int): Maximum tree depth.
        learning_rate (float): Learning rate for boosting.
        num_leaves (int): Maximum number of leaves per tree.
        feature_fraction (float): Fraction of features to use per iteration.
        bagging_fraction (float): Fraction of data to use for bagging.
        bagging_freq (int): Frequency for bagging.
        scale_pos_weight (Optional[float]): Weight for positive class.
        min_threshold (float): Minimum classification threshold.
        max_threshold (float): Maximum classification threshold.
        threshold_metric (str): Metric for threshold optimization.
        num_threads (int): Number of threads for parallel training.
        early_stopping_rounds (int): Number of rounds for early stopping.
        validation_split (float): Fraction of data for validation.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Whether to print training progress.

    Returns:
        tuple: A tuple containing:
            - model (lgb.Booster): Trained LightGBM model
            - metrics (Dict): Performance metrics dictionary
            - X_train_encoded (pd.DataFrame): Encoded training data
            - category_mappings (Dict): Categorical feature encodings
    """
    if verbose:
        print("\n[INFO] -------------------- LightGBM Training --------------------")
    
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if verbose and categorical_features:
        print(f"\n[INFO] Encoding {len(categorical_features)} categorical features:")
        for col in categorical_features:
            n_unique = X_train[col].nunique()
            print(f"  - {col}: {n_unique} unique values")
    
    X_train_encoded = X_train.copy()
    category_mappings = {}
    
    for col in categorical_features:
        cat_type = X_train_encoded[col].astype('category')
        category_mappings[col] = {
            cat: code 
            for code, cat in enumerate(cat_type.cat.categories)
        }
        
        X_train_encoded[col] = cat_type.cat.codes
    
    if verbose:
        print(f"\n[INFO] Configuration:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  categorical_feature: [] (all pre-encoded)")
        print(f"  threshold_range: [{min_threshold:.2f}, {max_threshold:.2f}]")
        print(f"  threshold_metric: {threshold_metric}")
        print(f"  num_threads: {num_threads if num_threads > 0 else 'all CPUs'}")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_encoded, y_train,
        test_size=validation_split,
        stratify=y_train,
        random_state=random_state
    )
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'verbose': -1,
        'num_threads': num_threads if num_threads > 0 else 0,
        'random_state': random_state
    }
    
    if verbose:
        print(f"\n[INFO] Training model (all features numeric)...")
    
    callbacks = []
    if verbose:
        callbacks.append(lgb.log_evaluation(period=50))
    callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    y_train_proba = model.predict(X_train_encoded, num_iteration=model.best_iteration)
    y_val_proba = model.predict(X_val, num_iteration=model.best_iteration)
    
    train_logloss = log_loss(y_train, y_train_proba)
    val_logloss = log_loss(y_val, y_val_proba)
    
    if verbose:
        print(f"\n[INFO] Optimizing threshold (range: [{min_threshold:.2f}, {max_threshold:.2f}])...")
    
    best_threshold, best_score = optimize_threshold_constrained(
        y_train.values,
        y_train_proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        metric=threshold_metric
    )
    
    if verbose:
        print(f"  Optimal threshold: {best_threshold:.3f} (score: {best_score:.4f})")
    
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    
    train_recall = recall_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    cm = confusion_matrix(y_train, y_train_pred)
    
    val_recall = recall_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_cm = confusion_matrix(y_val, y_val_pred)
    
    if verbose:
        print(f"\n[INFO] Training Performance (threshold={best_threshold:.3f}):")
        print(f"  Recall:    {train_recall:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  F1-Score:  {train_f1:.4f}")
        print(f"  ROC-AUC:   {train_auc:.4f}")
        print(f"  LogLoss:   {train_logloss:.4f}")
        
        print(f"\n[INFO] Validation Performance:")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  F1-Score:  {val_f1:.4f}")
        print(f"  ROC-AUC:   {val_auc:.4f}")
        print(f"  LogLoss:   {val_logloss:.4f}")
        print("\n[INFO] -------------------- Training Complete --------------------")
    
    metrics = {
        'model_type': 'LightGBM',
        'categorical_features_encoded': categorical_features,
        'category_mappings': category_mappings,
        'threshold_range': (min_threshold, max_threshold),
        'train_recall': train_recall,
        'train_precision': train_precision,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_logloss': train_logloss,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'val_logloss': val_logloss,
        'confusion_matrix': cm,
        'val_confusion_matrix': val_cm,
        'best_iteration': model.best_iteration,
        'optimal_threshold': best_threshold,
        'classification_threshold': best_threshold
    }
    
    return model, metrics, X_train_encoded, category_mappings

def train_random_forest_optimized_NOOOOOOOOOOO(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_split_ratio: float = 0.005,
    min_samples_leaf_ratio: float = 0.002,
    max_features: str = 'sqrt',
    max_samples: float = 0.8,
    min_threshold: float = 0.40,
    max_threshold: float = 0.65,
    threshold_metric: str = 'f1',
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True
) -> tuple[Pipeline, Dict, pd.DataFrame, Dict]:
    """Train Random Forest model with controlled thresholds.

    This function trains a Random Forest classifier after encoding categorical features
    to numeric codes. The function creates a pipeline, optimizes the classification
    threshold within specified bounds, and computes comprehensive performance metrics.
    This version does not use validation splitting.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum tree depth.
        min_samples_split_ratio (float): Minimum fraction of samples required to split.
        min_samples_leaf_ratio (float): Minimum fraction of samples required at leaf.
        max_features (str): Number of features to consider for best split.
        max_samples (float): Fraction of samples to draw for training each tree.
        min_threshold (float): Minimum classification threshold.
        max_threshold (float): Maximum classification threshold.
        threshold_metric (str): Metric for threshold optimization.
        n_jobs (int): Number of jobs for parallel training.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Whether to print training progress.

    Returns:
        tuple: A tuple containing:
            - pipeline (Pipeline): Trained Random Forest pipeline
            - metrics (Dict): Performance metrics dictionary
            - X_train_encoded (pd.DataFrame): Encoded training data
            - category_mappings (Dict): Categorical feature encodings
    """
    if verbose:
        print("\n[INFO] -------------------- Random Forest Training --------------------")
    
    n_samples = X_train.shape[0]
    min_samples_split = max(2, int(min_samples_split_ratio * n_samples))
    min_samples_leaf = max(1, int(min_samples_leaf_ratio * n_samples))
    
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if verbose and categorical_features:
        print(f"\n[INFO] Encoding {len(categorical_features)} categorical features:")
        for col in categorical_features:
            n_unique = X_train[col].nunique()
            print(f"  - {col}: {n_unique} unique values")
    
    X_train_encoded = X_train.copy()
    category_mappings = {}
    
    for col in categorical_features:
        cat_type = X_train_encoded[col].astype('category')
        category_mappings[col] = {
            cat: code 
            for code, cat in enumerate(cat_type.cat.categories)
        }
        
        X_train_encoded[col] = cat_type.cat.codes
    
    if verbose:
        print(f"\n[INFO] Configuration:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  categorical_features: [] (all pre-encoded)")
        print(f"  threshold_range: [{min_threshold:.2f}, {max_threshold:.2f}]")
        print(f"  threshold_metric: {threshold_metric}")
        print(f"  n_jobs: {n_jobs if n_jobs > 0 else 'all CPUs'}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        bootstrap=True,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0
    )
    
    pipeline = Pipeline([
        ('passthrough', 'passthrough'),
        ('classifier', model)
    ])
    
    if verbose:
        print(f"\n[INFO] Training model (all features numeric)...")
    
    pipeline.fit(X_train_encoded, y_train)
    
    y_train_proba = pipeline.predict_proba(X_train_encoded)[:, 1]
    
    train_logloss = log_loss(y_train, y_train_proba)
    
    if verbose:
        print(f"\n[INFO] Optimizing threshold (range: [{min_threshold:.2f}, {max_threshold:.2f}])...")
    
    best_threshold, best_score = optimize_threshold_constrained(
        y_train.values,
        y_train_proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        metric=threshold_metric
    )
    
    if verbose:
        print(f"  Optimal threshold: {best_threshold:.3f} (score: {best_score:.4f})")
    
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    
    train_recall = recall_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    cm = confusion_matrix(y_train, y_train_pred)
    
    if verbose:
        print(f"\n[INFO] Training Performance (threshold={best_threshold:.3f}):")
        print(f"  Recall:    {train_recall:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  F1-Score:  {train_f1:.4f}")
        print(f"  ROC-AUC:   {train_auc:.4f}")
        print(f"  LogLoss:   {train_logloss:.4f}")
        print(f"  OOB Score: {model.oob_score_:.4f}")
        print("\n[INFO] -------------------- Training Complete --------------------")
    
    metrics = {
        'model_type': 'RandomForest',
        'categorical_features_encoded': categorical_features,
        'category_mappings': category_mappings,
        'threshold_range': (min_threshold, max_threshold),
        'train_recall': train_recall,
        'train_precision': train_precision,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_logloss': train_logloss,
        'oob_score': model.oob_score_,
        'confusion_matrix': cm,
        'optimal_threshold': best_threshold,
        'classification_threshold': best_threshold
    }
    
    return pipeline, metrics, X_train_encoded, category_mappings

def train_random_forest_optimized(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_split_ratio: float = 0.005,
    min_samples_leaf_ratio: float = 0.002,
    max_features: str = 'sqrt',
    max_samples: float = 0.8,
    min_threshold: float = 0.40,
    max_threshold: float = 0.65,
    threshold_metric: str = 'f1',
    validation_split: float = 0.2,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True
) -> tuple[Pipeline, Dict, pd.DataFrame, Dict]:
    """Train Random Forest model with validation split and controlled thresholds.

    This function trains a Random Forest classifier after encoding categorical features
    to numeric codes. The function performs validation splitting, creates a pipeline,
    optimizes the classification threshold within specified bounds, and computes
    comprehensive performance metrics including validation set evaluation.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum tree depth.
        min_samples_split_ratio (float): Minimum fraction of samples required to split.
        min_samples_leaf_ratio (float): Minimum fraction of samples required at leaf.
        max_features (str): Number of features to consider for best split.
        max_samples (float): Fraction of samples to draw for training each tree.
        min_threshold (float): Minimum classification threshold.
        max_threshold (float): Maximum classification threshold.
        threshold_metric (str): Metric for threshold optimization.
        validation_split (float): Fraction of data for validation.
        n_jobs (int): Number of jobs for parallel training.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Whether to print training progress.

    Returns:
        tuple: A tuple containing:
            - pipeline (Pipeline): Trained Random Forest pipeline
            - metrics (Dict): Performance metrics dictionary
            - X_train_encoded (pd.DataFrame): Encoded training data
            - category_mappings (Dict): Categorical feature encodings
    """
    if verbose:
        print("\n[INFO] -------------------- Random Forest Training --------------------")
    
    n_samples = X_train.shape[0]
    min_samples_split = max(2, int(min_samples_split_ratio * n_samples))
    min_samples_leaf = max(1, int(min_samples_leaf_ratio * n_samples))
    
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if verbose and categorical_features:
        print(f"\n[INFO] Encoding {len(categorical_features)} categorical features:")
        for col in categorical_features:
            n_unique = X_train[col].nunique()
            print(f"  - {col}: {n_unique} unique values")
    
    X_train_encoded = X_train.copy()
    category_mappings = {}
    
    for col in categorical_features:
        cat_type = X_train_encoded[col].astype('category')
        category_mappings[col] = {
            cat: code 
            for code, cat in enumerate(cat_type.cat.categories)
        }
        
        X_train_encoded[col] = cat_type.cat.codes
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_encoded, y_train,
        test_size=validation_split,
        stratify=y_train,
        random_state=random_state
    )
    
    if verbose:
        print(f"\n[INFO] Configuration:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  categorical_features: [] (all pre-encoded)")
        print(f"  threshold_range: [{min_threshold:.2f}, {max_threshold:.2f}]")
        print(f"  threshold_metric: {threshold_metric}")
        print(f"  validation_split: {validation_split}")
        print(f"  n_jobs: {n_jobs if n_jobs > 0 else 'all CPUs'}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        bootstrap=True,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0
    )
    
    pipeline = Pipeline([
        ('passthrough', 'passthrough'),
        ('classifier', model)
    ])
    
    if verbose:
        print(f"\n[INFO] Training model (all features numeric)...")
    
    pipeline.fit(X_tr, y_tr)
    
    y_train_proba = pipeline.predict_proba(X_tr)[:, 1]
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    
    train_logloss = log_loss(y_tr, y_train_proba)
    val_logloss = log_loss(y_val, y_val_proba)
    
    if verbose:
        print(f"\n[INFO] Optimizing threshold (range: [{min_threshold:.2f}, {max_threshold:.2f}])...")
    
    best_threshold, best_score = optimize_threshold_constrained(
        y_tr.values,
        y_train_proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        metric=threshold_metric
    )
    
    if verbose:
        print(f"  Optimal threshold: {best_threshold:.3f} (score: {best_score:.4f})")
    
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    
    train_recall = recall_score(y_tr, y_train_pred)
    train_precision = precision_score(y_tr, y_train_pred)
    train_f1 = f1_score(y_tr, y_train_pred)
    train_auc = roc_auc_score(y_tr, y_train_proba)
    
    val_recall = recall_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    cm = confusion_matrix(y_tr, y_train_pred)
    
    if verbose:
        print(f"\n[INFO] Training Performance (threshold={best_threshold:.3f}):")
        print(f"  Recall:    {train_recall:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  F1-Score:  {train_f1:.4f}")
        print(f"  ROC-AUC:   {train_auc:.4f}")
        print(f"  LogLoss:   {train_logloss:.4f}")
        
        print(f"\n[INFO] Validation Performance:")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  F1-Score:  {val_f1:.4f}")
        print(f"  ROC-AUC:   {val_auc:.4f}")
        print(f"  LogLoss:   {val_logloss:.4f}")
        print("\n[INFO] -------------------- Training Complete --------------------")
    
    metrics = {
        'model_type': 'RandomForest',
        'categorical_features_encoded': categorical_features,
        'category_mappings': category_mappings,
        'threshold_range': (min_threshold, max_threshold),
        'train_recall': train_recall,
        'train_precision': train_precision,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_logloss': train_logloss,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'val_logloss': val_logloss,
        'confusion_matrix': cm,
        'optimal_threshold': best_threshold,
        'classification_threshold': best_threshold
    }
    
    return pipeline, metrics, X_train_encoded, category_mappings

def train_catboost_optimized(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    iterations: int = 600,
    depth: int = 8,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 1.0,
    rsm: float = 0.9,
    subsample: float = 0.8,
    bootstrap_type: str = 'Bernoulli',
    border_count: int = 254,
    min_threshold: float = 0.40,
    max_threshold: float = 0.65,
    threshold_metric: str = 'f1',
    thread_count: int = -1,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = True
) -> tuple[CatBoostClassifier, Dict, pd.DataFrame, Dict]:
    """Train CatBoost model with optimized configuration to prevent CTR generation.

    This function trains a CatBoost classifier after encoding categorical features
    to numeric codes to prevent Counter-Target-Rate (CTR) feature generation. The
    function performs validation splitting, optimizes the classification threshold
    within specified bounds, and computes comprehensive performance metrics.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        iterations (int): Number of boosting iterations.
        depth (int): Maximum tree depth.
        learning_rate (float): Learning rate for boosting.
        l2_leaf_reg (float): L2 regularization coefficient.
        rsm (float): Random subspace method ratio.
        subsample (float): Sample rate for training.
        bootstrap_type (str): Bootstrap type for bagging.
        border_count (int): Number of splits for numerical features.
        min_threshold (float): Minimum classification threshold.
        max_threshold (float): Maximum classification threshold.
        threshold_metric (str): Metric for threshold optimization.
        thread_count (int): Number of threads for parallel training.
        early_stopping_rounds (int): Number of rounds for early stopping.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Whether to print training progress.

    Returns:
        tuple: A tuple containing:
            - model (CatBoostClassifier): Trained CatBoost model
            - metrics (Dict): Performance metrics dictionary
            - X_train_encoded (pd.DataFrame): Encoded training data
            - category_mappings (Dict): Categorical feature encodings
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost not installed")
    
    if verbose:
        print("\n[INFO] -------------------- CatBoost Training --------------------")
    
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if verbose and categorical_features:
        print(f"\n[INFO] Encoding {len(categorical_features)} categorical features:")
        for col in categorical_features:
            n_unique = X_train[col].nunique()
            print(f"  - {col}: {n_unique} unique values")
    
    X_train_encoded = X_train.copy()
    category_mappings = {}
    
    for col in categorical_features:
        cat_type = X_train_encoded[col].astype('category')
        category_mappings[col] = {
            cat: code 
            for code, cat in enumerate(cat_type.cat.categories)
        }
        
        X_train_encoded[col] = cat_type.cat.codes
    
    if verbose:
        print(f"\n[INFO] Configuration:")
        print(f"  iterations: {iterations}")
        print(f"  depth: {depth}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  l2_leaf_reg: {l2_leaf_reg}")
        print(f"  rsm: {rsm}")
        print(f"  subsample: {subsample}")
        print(f"  bootstrap_type: {bootstrap_type}")
        print(f"  border_count: {border_count}")
        print(f"  cat_features: [] (all pre-encoded)")
        print(f"  threshold_range: [{min_threshold:.2f}, {max_threshold:.2f}]")
        print(f"  threshold_metric: {threshold_metric}")
        print(f"  thread_count: {thread_count if thread_count > 0 else 'all CPUs'}")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_encoded, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )
    
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        rsm=rsm,
        subsample=subsample,
        bootstrap_type=bootstrap_type,
        border_count=border_count,
        loss_function='Logloss',
        eval_metric='F1',
        random_seed=random_state,
        thread_count=thread_count,
        verbose=False,
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True
    )
    
    if verbose:
        print(f"\n[INFO] Training model (all features numeric)...")
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    y_train_proba = model.predict_proba(X_train_encoded)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    train_logloss = log_loss(y_train, y_train_proba)
    val_logloss = log_loss(y_val, y_val_proba)
    
    if verbose:
        print(f"\n[INFO] Optimizing threshold (range: [{min_threshold:.2f}, {max_threshold:.2f}])...")
    
    best_threshold, best_score = optimize_threshold_constrained(
        y_train.values,
        y_train_proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        metric=threshold_metric
    )
    
    if verbose:
        print(f"  Optimal threshold: {best_threshold:.3f} (score: {best_score:.4f})")
    
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    
    train_recall = recall_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    cm = confusion_matrix(y_train, y_train_pred)
    
    val_recall = recall_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_cm = confusion_matrix(y_val, y_val_pred)
    
    if verbose:
        print(f"\n[INFO] Training Performance (threshold={best_threshold:.3f}):")
        print(f"  Recall:    {train_recall:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  F1-Score:  {train_f1:.4f}")
        print(f"  ROC-AUC:   {train_auc:.4f}")
        print(f"  LogLoss:   {train_logloss:.4f}")
        
        print(f"\n[INFO] Validation Performance:")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  F1-Score:  {val_f1:.4f}")
        print(f"  ROC-AUC:   {val_auc:.4f}")
        print(f"  LogLoss:   {val_logloss:.4f}")
        print("\n[INFO] -------------------- Training Complete --------------------")
    
    metrics = {
        'model_type': 'CatBoost',
        'categorical_features_encoded': categorical_features,
        'category_mappings': category_mappings,
        'threshold_range': (min_threshold, max_threshold),
        'train_recall': train_recall,
        'train_precision': train_precision,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_logloss': train_logloss,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'val_logloss': val_logloss,
        'confusion_matrix': cm,
        'val_confusion_matrix': val_cm,
        'best_iteration': model.best_iteration_,
        'optimal_threshold': best_threshold,
        'classification_threshold': best_threshold
    }
    
    return model, metrics, X_train_encoded, category_mappings


def encode_data_like_training(
    X: pd.DataFrame,
    category_mappings: Dict[str, Dict]
) -> pd.DataFrame:
    """Encode categorical features using training category mappings.

    This function applies the same categorical encoding used during training to
    new data. Unseen categories are encoded as -1.

    Args:
        X (pd.DataFrame): Data to encode.
        category_mappings (Dict[str, Dict]): Category mappings from training in
                                             format {feature: {category: code}}.

    Returns:
        pd.DataFrame: Encoded DataFrame with categorical features as integers.
    """
    X_encoded = X.copy()
    
    for col, mapping in category_mappings.items():
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].map(mapping).fillna(-1).astype(int)
    
    return X_encoded

def decode_data_to_original(
    X_encoded: pd.DataFrame,
    category_mappings: Dict[str, Dict]
) -> pd.DataFrame:
    """Decode categorical features back to original values.

    This function reverses the categorical encoding by mapping numeric codes back
    to their original categorical values using the training mappings.

    Args:
        X_encoded (pd.DataFrame): Encoded DataFrame with numeric codes.
        category_mappings (Dict[str, Dict]): Category mappings from training in
                                             format {feature: {category: code}}.

    Returns:
        pd.DataFrame: Decoded DataFrame with original categorical values.
    """
    X_decoded = X_encoded.copy()
    
    for col, mapping in category_mappings.items():
        if col in X_decoded.columns:
            reverse_map = {code: cat for cat, code in mapping.items()}
            
            X_decoded[col] = X_decoded[col].map(reverse_map)
            
            if X_decoded[col].isna().any():
                print(f"[WARNING] {col} has {X_decoded[col].isna().sum()} unknown codes")
    
    return X_decoded

def get_category_from_code(
    feature_name: str,
    code: int,
    category_mappings: Dict[str, Dict]
) -> str:
    """Get original category value from numeric code.

    This function retrieves the original categorical value corresponding to a
    numeric code for a specified feature.

    Args:
        feature_name (str): Name of the categorical feature.
        code (int): Numeric code to decode.
        category_mappings (Dict[str, Dict]): Category mappings from training.

    Returns:
        str: Original category value, or string representation of code if not found.
    """
    if feature_name not in category_mappings:
        return str(code)
    
    reverse_map = {c: cat for cat, c in category_mappings[feature_name].items()}
    return reverse_map.get(code, f"UNKNOWN_{code}")

def decode_rule_conditions(
    rules: List[Dict],
    category_mappings: Dict[str, Dict],
    verbose: bool = False
) -> List[Dict]:
    """Decode rule conditions from numeric codes to original categorical values.

    This function converts numeric threshold conditions in rules to their original
    categorical representations. For example, conditions like "grade <= 2" are
    converted to "grade in ['A', 'B', 'C']".

    Args:
        rules (List[Dict]): List of rules with numeric conditions.
        category_mappings (Dict[str, Dict]): Category mappings in format
                                             {feature: {category: code}}.
        verbose (bool): Whether to print progress and examples.

    Returns:
        List[Dict]: List of rules with decoded categorical conditions. Each rule
                   includes both 'rule_conditions' (decoded) and
                   'rule_conditions_encoded' (original numeric).
    """
    if verbose:
        print(f"\n[INFO] Decoding {len(rules)} rules...")
    
    decoded_rules = []
    
    for rule in rules:
        decoded_rule = rule.copy()
        decoded_conditions = []
        
        for cond in rule['rule_conditions']:
            decoded_cond = cond
            
            for cat_feature in category_mappings.keys():
                if cat_feature in cond:
                    
                    if f"{cat_feature} <= " in cond:
                        threshold = float(cond.split(' <= ')[1].strip())
                        code = int(threshold)
                        
                        reverse_map = {c: cat for cat, c in category_mappings[cat_feature].items()}
                        categories = [
                            reverse_map[c] 
                            for c in range(code + 1) 
                            if c in reverse_map
                        ]
                        
                        decoded_cond = f"{cat_feature} in {categories}"
                        break
                    
                    elif f"{cat_feature} > " in cond:
                        threshold = float(cond.split(' > ')[1].strip())
                        code = int(threshold)
                        
                        reverse_map = {c: cat for cat, c in category_mappings[cat_feature].items()}
                        max_code = max(reverse_map.keys())
                        categories = [
                            reverse_map[c] 
                            for c in range(code + 1, max_code + 1) 
                            if c in reverse_map
                        ]
                        
                        decoded_cond = f"{cat_feature} in {categories}"
                        break
            
            decoded_conditions.append(decoded_cond)
        
        decoded_rule['rule_conditions'] = decoded_conditions
        decoded_rule['rule_conditions_encoded'] = rule['rule_conditions']
        decoded_rules.append(decoded_rule)
    
    if verbose:
        print(f"[INFO] Decoded {len(decoded_rules)} rules")
        
        if decoded_rules:
            print("\n[INFO] Example decoded rule:")
            example = decoded_rules[0]
            print(f"  Tree {example['tree_idx']}, Leaf {example['leaf_idx']}")
            print(f"  Original conditions:")
            for cond in example['rule_conditions_encoded']:
                print(f"    - {cond}")
            print(f"  Decoded conditions:")
            for cond in example['rule_conditions']:
                print(f"    - {cond}")
    
    return decoded_rules

def compare_models(
    rf_metrics: Dict = None,
    catboost_metrics: Dict = None,
    lightgbm_metrics: Dict = None,
    focus_metric: str = 'train_f1'
) -> pd.DataFrame:
    """Compare performance metrics across trained models.

    This function creates a summary table comparing key performance indicators for
    Random Forest, CatBoost, and LightGBM models, including their optimal thresholds
    and training metrics.

    Args:
        rf_metrics (Dict, optional): Metrics dictionary from Random Forest training.
        catboost_metrics (Dict, optional): Metrics dictionary from CatBoost training.
        lightgbm_metrics (Dict, optional): Metrics dictionary from LightGBM training.
        focus_metric (str): Primary metric to highlight for model selection.

    Returns:
        pd.DataFrame: Comparison table with models as rows and metrics as columns.
    """
    models_data = []
    
    if rf_metrics:
        models_data.append({
            'Model': 'Random Forest',
            'Threshold': rf_metrics.get('optimal_threshold', 0.5),
            'Train Recall': rf_metrics.get('train_recall', np.nan),
            'Train Precision': rf_metrics.get('train_precision', np.nan),
            'Train F1': rf_metrics.get('train_f1', np.nan),
            'Train AUC': rf_metrics.get('train_auc', np.nan),
            'Train LogLoss': rf_metrics.get('train_logloss', np.nan),
            'CV Recall': rf_metrics.get('cv_recall_mean', np.nan),
            'CV F1': rf_metrics.get('cv_f1_mean', np.nan),
            'Val Recall': np.nan,
            'Val F1': np.nan
        })
    
    if catboost_metrics:
        models_data.append({
            'Model': 'CatBoost',
            'Threshold': catboost_metrics.get('optimal_threshold', 0.5),
            'Train Recall': catboost_metrics.get('train_recall', np.nan),
            'Train Precision': catboost_metrics.get('train_precision', np.nan),
            'Train F1': catboost_metrics.get('train_f1', np.nan),
            'Train AUC': catboost_metrics.get('train_auc', np.nan),
            'Train LogLoss': catboost_metrics.get('train_logloss', np.nan),
            'CV Recall': np.nan,
            'CV F1': np.nan,
            'Val Recall': catboost_metrics.get('val_recall', np.nan),
            'Val F1': catboost_metrics.get('val_f1', np.nan)
        })
    
    if lightgbm_metrics:
        models_data.append({
            'Model': 'LightGBM',
            'Threshold': lightgbm_metrics.get('optimal_threshold', 0.5),
            'Train Recall': lightgbm_metrics.get('train_recall', np.nan),
            'Train Precision': lightgbm_metrics.get('train_precision', np.nan),
            'Train F1': lightgbm_metrics.get('train_f1', np.nan),
            'Train AUC': lightgbm_metrics.get('train_auc', np.nan),
            'Train LogLoss': lightgbm_metrics.get('train_logloss', np.nan),
            'CV Recall': np.nan,
            'CV F1': np.nan,
            'Val Recall': lightgbm_metrics.get('val_recall', np.nan),
            'Val F1': lightgbm_metrics.get('val_f1', np.nan)
        })
    
    comparison_df = pd.DataFrame(models_data)
    
    if len(comparison_df) > 0:
        print("\n[INFO] -------------------- Model Comparison --------------------")
        print(f"\n{comparison_df.to_string(index=False)}")
        print("\n[INFO] -------------------- Comparison Complete --------------------")
        
        if focus_metric.replace('train_', '').replace('Train ', '') in str(comparison_df.columns):
            col_name = [c for c in comparison_df.columns if focus_metric.lower().replace('_', ' ') in c.lower()]
            if col_name:
                best_idx = comparison_df[col_name[0]].idxmax()
                best_model = comparison_df.loc[best_idx, 'Model']
                best_score = comparison_df.loc[best_idx, col_name[0]]
                print(f"\n[INFO] Best model by {col_name[0]}: {best_model} ({best_score:.4f})")
                print("\n[INFO] ----------------------------------------------------------------")
    
    return comparison_df

def read_all_pkl_files(pkl_path: str) -> dict:
    """Read all pickle files from a directory.

    This function scans the specified directory for .pkl files and loads their
    contents into a dictionary.

    Args:
        pkl_path (str): Path to the directory containing pickle files.

    Returns:
        dict: Dictionary mapping filenames to their deserialized contents.
    """
    pkl_contents = {}
    for filename in listdir(pkl_path):
        if filename.endswith(".pkl"):
            with open(path.join(pkl_path, filename), "rb") as f:
                pkl_contents[filename] = pk.load(f)
    return pkl_contents

def write_pkl_file(obj: any, file_path: str) -> None:
    """Write an object to a pickle file.

    This function serializes a Python object and saves it to the specified path
    using the pickle protocol.

    Args:
        obj (any): The object to be pickled.
        file_path (str): The path where the pickle file will be saved.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pk.dump(obj, f)

def select_top_features_manual(
    df: pd.DataFrame, target: str = 'bad_good'
) -> tuple[pd.DataFrame, list[str]]:
    """Select top 12 features based on KS and IV analysis.

    This function filters the DataFrame to retain only the top performing features
    identified through Kolmogorov-Smirnov statistics for numerical features and
    Information Value for categorical features.

    Args:
        df (pd.DataFrame): Input DataFrame with all features.
        target (str): Name of the target variable column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Filtered DataFrame with selected features and target
            - list: List of column names that were dropped
    """
    selected_features = [
        'last_fico_range_high',
        'recoveries',  #<---- aquí está el impostor.... Así te quería agarrar puerco!
        'term_num',
        'fico_range_low',
        'total_rec_late_fee',
        'acc_open_past_24mths',
        'dti',
        'mort_acc',
        'loan_amnt',
        'debt_settlement_flag',
        'grade',
        'verification_status'
    ]
    
    available_features = [f for f in selected_features if f in df.columns]
    
    columns_to_keep = available_features + [target]
    
    all_columns = [col for col in df.columns if col != target]
    cols_to_drop = [col for col in all_columns if col not in available_features]
    
    df_filtered = df[columns_to_keep].copy()
    
    print(f"\n[INFO] Selecting top features:")
    print(f"  Original features: {len(all_columns)}")
    print(f"  Selected features: {len(available_features)}")
    print(f"  Dropped features:  {len(cols_to_drop)}")
    
    return df_filtered, cols_to_drop

if __name__ == "__main__":
    
    from config import LOAN_FILE, MODEL_DIR, PKL_DIR
    
    df = dl.load_data(LOAN_FILE)
    df = dl.inspect_and_prepare_data(df)
    dl.print_dataframe_stats(df)

    print("[INFO] Data loaded and preprocessed successfully.")

    df_processed, dropped_missing = pp.handle_missing_values(df.copy(), missing_percentage_threshold=30)
    df_processed = pp.convert_specific_categorical_to_numeric(df_processed)
    df_processed = pp.impute_missing_values(df_processed)
    df_processed, dropped_low_variance = pp.drop_low_variance_features(df_processed, variance_threshold=0.01)
    df_processed, dropped_descriptive = pp.drop_descriptive_features(df_processed)
    df_processed, dropped_correlated = pp.drop_highly_correlated_features(df_processed, correlation_threshold=0.6)
    df_processed, dropped_low_ks = pp.drop_low_ks_features(df_processed, target_col='bad_good', ks_threshold=0.03)
    df_processed, dropped_low_iv, iv_df = pp.drop_low_iv_categorical_features(df_processed, df_processed.select_dtypes(include=['object']).columns.tolist(), target='bad_good', iv_threshold=0.02)

    df_processed, dropped_low_ks_iv = select_top_features_manual(df_processed, target='bad_good')

    print("\n[INFO] Processed DataFrame head:")
    print(df_processed.head())
    print("\n[INFO] Processed DataFrame info:")    
    df_processed.info()

    print(f"\n[INFO] Processed DataFrame shape: {df_processed.shape}")

    X_train, X_test, y_train, y_test = split_data(df_processed, target_col='bad_good', test_size=0.3)
    
    write_pkl_file(X_train, path.join(PKL_DIR, "X_train.pkl"))
    write_pkl_file(y_train, path.join(PKL_DIR, "y_train.pkl"))
    write_pkl_file(X_test, path.join(PKL_DIR, "X_test.pkl"))      
    write_pkl_file(y_test, path.join(PKL_DIR, "y_test.pkl"))

    print("\n[INFO] Data split into training and testing sets and saved as pickle files.")
    
    rf_pipeline, rf_metrics, X_train_encoded, cat_mappings = train_random_forest_optimized(
        X_train, y_train,
        n_estimators=200,
        max_features='sqrt',
        max_depth=5,
        verbose=True
    )
    
    rf_paths = save_model(
        model=rf_pipeline,
        metrics=rf_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="random_forest_NEW_encoded",
        output_dir=MODEL_DIR
    )
    
    catboost_model, catboost_metrics = None, None
    if CATBOOST_AVAILABLE:
        catboost_model, catboost_metrics, X_train_encoded, cat_mappings = train_catboost_optimized(
            X_train, y_train,
            depth=5,
            iterations=200,
            verbose=True,
            learning_rate=1
        )

    cb_paths = save_model(
        model=catboost_model,
        metrics=catboost_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="catboost_NEW_encoded",
        output_dir=MODEL_DIR
    )
    
    lightgbm_model, lightgbm_metrics = None, None
  
    lightgbm_model, lightgbm_metrics, X_train_encoded, cat_mappings = train_lightgbm_optimized(
        X_train, y_train,
        max_depth=5,
        n_estimators=200,
        validation_split=0.3,
        verbose=True
    )
    
    lightgbm_paths = save_model(
        model=lightgbm_model,
        metrics=lightgbm_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="lightgbm_NEW_encoded",
        output_dir=MODEL_DIR
    )
    
    comparison = compare_models(
        rf_metrics=rf_metrics,
        catboost_metrics=catboost_metrics,
        lightgbm_metrics=lightgbm_metrics,
        focus_metric='Train F1'
    )
    
    print("\n[INFO] -------------------- Models Saved --------------------")

    print(f"\n[INFO] Random Forest:")
    print(f"  {rf_paths['model']}")
    print(f"\n[INFO] CatBoost:")
    print(f"  {cb_paths['model']}")
    print(f"\n[INFO] LightGBM:")
    print(f"  {lightgbm_paths['model']}")
    print("\n[INFO] -------------------- Save Complete --------------------")

    print("\n[INFO] CatBoost model evaluating start.")

    X_test_encoded = encode_data_like_training(X_test, cat_mappings)
    threshold = catboost_metrics['optimal_threshold']
    em.evaluate_model(catboost_model, X_test_encoded, y_test, threshold=threshold)
    print("\n[INFO] CatBoost model evaluating successfully.")
    print(cat_mappings)