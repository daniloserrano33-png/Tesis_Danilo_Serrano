# src/model/evaluate_model.py

import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, recall_score, 
    precision_score, f1_score, log_loss, classification_report, accuracy_score
)
from src.loader import data_loader as dl
from src.preprocess import preprocessing as pp
from src.model import train_model as mod

def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    category_mappings: Optional[Dict[str, Dict]] = None,
    threshold: Optional[float] = None,
    model_name: str = "Model",
    verbose: bool = True
) -> Dict:
    """Evaluate trained model performance on test set.

    This function computes comprehensive performance metrics including accuracy,
    precision, recall, F1-score, specificity, ROC-AUC, and log loss. The function
    handles categorical encoding automatically if mappings are provided and applies
    a classification threshold for binary predictions. Performance results are
    displayed in a structured format including confusion matrix and classification
    report.

    Args:
        model (object): The trained model (CatBoost, LightGBM, RandomForest, etc.).
        X_test (pd.DataFrame): The test feature matrix.
        y_test (pd.Series): The true target variable for the test set.
        category_mappings (Optional[Dict[str, Dict]]): Category mappings for encoding
                                                        categorical features in format
                                                        {feature: {category: code}}.
        threshold (Optional[float]): Classification threshold for binary predictions.
                                     Uses 0.5 if None and model supports predict_proba.
        model_name (str): Name identifier for display purposes in output.
        verbose (bool): Whether to print detailed results to console.

    Returns:
        Dict: Dictionary containing evaluation metrics:
            - accuracy: Overall accuracy score
            - recall: Recall score (sensitivity)
            - sensitivity: True positive rate
            - precision: Precision score
            - f1: F1-score
            - specificity: True negative rate
            - confusion_matrix: Confusion matrix array
            - true_negatives: Count of true negatives
            - false_positives: Count of false positives
            - false_negatives: Count of false negatives
            - true_positives: Count of true positives
            - threshold: Classification threshold used
            - n_samples: Total number of test samples
            - n_positive: Number of positive class samples
            - n_negative: Number of negative class samples
            - auc: ROC-AUC score (if probabilities available)
            - logloss: Log loss (if probabilities available)
            - predictions_proba: Predicted probabilities (if available)
    """
    if verbose:
        print("\n[INFO] -------------------- Model Evaluation --------------------")
        print(f"{model_name.upper()} - TEST SET EVALUATION")
        print("[INFO] ----------------------------------------------------------------")
    
    if category_mappings is not None and len(category_mappings) > 0:
        X_test_encoded = mod.encode_data_like_training(X_test, category_mappings)
        if verbose:
            print(f"\n[INFO] Encoded {len(category_mappings)} categorical features")
    else:
        X_test_encoded = X_test
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_encoded)[:, 1]
        
        if threshold is None:
            threshold = 0.5
        
        y_pred = (y_proba >= threshold).astype(int)
        
        if verbose:
            print(f"[INFO] Using classification threshold: {threshold:.3f}")
    else:
        y_pred = model.predict(X_test_encoded)
        y_proba = None
        threshold = None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)
    else:
        auc = None
        logloss = None
    
    if verbose:
        print(f"\n[INFO] -------------------- Performance Metrics --------------------")
        print(f"\nTest samples: {len(y_test):,}")
        print(f"Positive class (bad loans): {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
        
        print(f"\n{'Metric':<20} {'Value'}")
        print("-" * 40)
        print(f"{'Accuracy':<20} {accuracy:.4f}")
        print(f"{'Recall (Sensitivity)':<20} {recall:.4f}")
        print(f"{'Precision':<20} {precision:.4f}")
        print(f"{'F1-Score':<20} {f1:.4f}")
        print(f"{'Specificity':<20} {specificity:.4f}")
        
        if auc is not None:
            print(f"{'ROC-AUC':<20} {auc:.4f}")
        if logloss is not None:
            print(f"{'Log Loss':<20} {logloss:.4f}")
        
        print(f"\n[INFO] -------------------- Confusion Matrix --------------------")
        print(f"\n                 Predicted")
        print(f"               Good (0)  Bad (1)")
        print(f"Actual  Good   {tn:8,}  {fp:8,}")
        print(f"        Bad    {fn:8,}  {tp:8,}")
        
        print(f"\n{'Metric':<25} {'Count':<10} {'Percentage'}")
        print("-" * 50)
        print(f"{'True Negatives (TN)':<25} {tn:>8,}   {tn/(tn+fp)*100:>6.2f}% of predicted good")
        print(f"{'False Positives (FP)':<25} {fp:>8,}   {fp/(tn+fp)*100:>6.2f}% of predicted good")
        print(f"{'False Negatives (FN)':<25} {fn:>8,}   {fn/(fn+tp)*100:>6.2f}% of predicted bad")
        print(f"{'True Positives (TP)':<25} {tp:>8,}   {tp/(fn+tp)*100:>6.2f}% of predicted bad")
        
        print(f"\n[INFO] -------------------- Classification Report --------------------")
        try:
            print(classification_report(
                y_test, y_pred,
                target_names=['Good (0)', 'Bad (1)'],
                zero_division=0,
                digits=4
            ))
        except ValueError:
            print(classification_report(y_test, y_pred, zero_division=0, digits=4))
        
        print("\n[INFO] -------------------- Evaluation Complete --------------------")
    
    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'sensitivity': sensitivity,
        'precision': precision,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'threshold': threshold,
        'n_samples': len(y_test),
        'n_positive': int(y_test.sum()),
        'n_negative': int((y_test == 0).sum())
    }
    
    if auc is not None:
        metrics['auc'] = auc
    if logloss is not None:
        metrics['logloss'] = logloss
    if y_proba is not None:
        metrics['predictions_proba'] = y_proba
    
    return metrics

if __name__ == '__main__':
    print("[INFO] Running evaluate_model.py script")

    from config import LOAN_FILE
    
    df = dl.load_data(LOAN_FILE)
    df = dl.inspect_and_prepare_data(df)
    dl.print_dataframe_stats(df)

    print("[INFO] Original DataFrame head:")
    print(df.head())
    print("\n[INFO] Original DataFrame info:")
    df.info()

    df_processed, dropped_missing = pp.handle_missing_values(df.copy(), missing_percentage_threshold=30)
    df_processed = pp.convert_specific_categorical_to_numeric(df_processed)
    df_processed = pp.impute_missing_values(df_processed)
    df_processed, dropped_low_variance = pp.drop_low_variance_features(df_processed, variance_threshold=0.01)
    df_processed, dropped_descriptive = pp.drop_descriptive_features(df_processed)
    df_processed, dropped_correlated = pp.drop_highly_correlated_features(df_processed, correlation_threshold=0.6)
    df_processed, dropped_low_ks = pp.drop_low_ks_features(df_processed, target_col='bad_good', ks_threshold=0.03)
    df_processed, dropped_low_iv, iv_df = pp.drop_low_iv_categorical_features(df_processed, df_processed.select_dtypes(include=['object']).columns.tolist(), target='bad_good', iv_threshold=0.02)

    print("\n[INFO] Final Processed DataFrame head:")
    print(df_processed.head())
    print("\n[INFO] Final Processed DataFrame info:")
    df_processed.info()
    print(f"\n[INFO] Processed DataFrame shape: {df_processed.shape}")