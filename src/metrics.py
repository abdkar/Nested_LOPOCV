# mlflow_pipeline/src/metrics.py
# In src/metrics.py (or your equivalent metrics file)
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, matthews_corrcoef)
from collections import Counter
import numpy as np

def metric_row(pred, y_true, pid: str, train_acc: float, cv_acc: float) -> dict:
    """Calculates standard classification metrics (monolithic style)."""
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(pred, np.ndarray): pred = np.array(pred)

    if len(y_true) == 0: # Handle empty ground truth for a fold
        return {
            "Participant": pid, "Actual_Label": "N/A", "Predicted_Labels": [],
            "Correct_Classify": 0, "Miss_Classify": 0, "Confusion_Matrix": [],
            "Test Accuracy": np.nan, "Training Accuracy": train_acc,
            "Group CV Validation Accuracy": cv_acc,
            "Train-Test Gap": train_acc - cv_acc if not np.isnan(train_acc) and not np.isnan(cv_acc) else np.nan,
            "Test Precision": np.nan, "Test Recall": np.nan, "Test F1 Score": np.nan, "Test MCC": np.nan,
            "Overfitting Indicator": "N/A",
        }

    # Ensure all unique labels from both true and predicted are used for CM, esp. for limited fold data
    unique_labels = np.unique(np.concatenate((y_true, pred)))
    cm = confusion_matrix(y_true, pred, labels=unique_labels)
    acc = accuracy_score(y_true, pred)

    # Determine averaging method for multiclass/binary cases
    num_unique_true_labels = len(np.unique(y_true))
    avg_method = 'binary'
    if num_unique_true_labels > 2: # Multiclass
        avg_method = 'weighted'
    # If num_unique_true_labels is 1, it's tricky. 'binary' with zero_division=0 might result in 0.
    # For single-class y_true, precision/recall/F1 are often ill-defined or 0 unless predictions vary.

    prec = precision_score(y_true, pred, zero_division=0, average=avg_method, labels=unique_labels)
    rec = recall_score(y_true, pred, zero_division=0, average=avg_method, labels=unique_labels)
    f1 = f1_score(y_true, pred, zero_division=0, average=avg_method, labels=unique_labels)

    mcc = np.nan
    if len(unique_labels) > 1 : # MCC needs at least two classes represented overall
        try:
            mcc = matthews_corrcoef(y_true, pred)
        except ValueError: # Handles cases where MCC is undefined
            pass # mcc remains np.nan

    overfitting_indicator = "N/A"
    if not np.isnan(train_acc) and not np.isnan(acc): # acc is test accuracy for the current fold
        overfitting_indicator = "+" if (train_acc - acc) > 0.10 else "-"

    return {
        "Participant": pid,
        "Actual_Label": Counter(y_true).most_common(1)[0][0] if y_true.size > 0 else "N/A",
        "Predicted_Labels": pred.tolist(),
        "Correct_Classify": int(np.sum(pred == y_true)),
        "Miss_Classify": int(np.sum(pred != y_true)),
        "Confusion_Matrix": cm.tolist(),
        "Test Accuracy": acc,
        "Training Accuracy": train_acc, # Fold's training data accuracy
        "Group CV Validation Accuracy": cv_acc, # Fold's inner CV accuracy
        "Train-Test Gap": train_acc - cv_acc if not np.isnan(train_acc) and not np.isnan(cv_acc) else np.nan,
        "Test Precision": prec,
        "Test Recall": rec,
        "Test F1 Score": f1,
        "Test MCC": mcc,
        "Overfitting Indicator": overfitting_indicator,
    }