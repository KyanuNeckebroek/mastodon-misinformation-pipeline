"""
src/evaluation/metrics.py

Gedeelde evaluatiefuncties voor alle modellen.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger


def compute_metrics(y_true: list, y_pred: list, y_prob: list = None) -> dict:
    """
    Berekent alle evaluatiemetrieken.

    Args:
        y_true:  Werkelijke labels
        y_pred:  Voorspelde labels
        y_prob:  Kansen voor klasse 1 (optioneel, voor AUC-ROC)

    Returns:
        Dictionary met alle metrieken
    """
    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1_macro":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_binary":        f1_score(y_true, y_pred, average="binary", zero_division=0),
        "precision":        precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_binary": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall":           recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_binary":    recall_score(y_true, y_pred, average="binary", zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = float("nan")

    return metrics


def print_classification_report(y_true: list, y_pred: list, model_name: str = "Model") -> None:
    """Print een gedetailleerd classificatierapport."""
    logger.info(f"\n=== {model_name.upper()} - CLASSIFICATIERAPPORT ===")
    report = classification_report(
        y_true, y_pred,
        target_names=["Onwaar (0)", "Waar (1)"],
        digits=4,
    )
    for line in report.split("\n"):
        logger.info(line)

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")


def compute_per_group_metrics(
    y_true: list,
    y_pred: list,
    groups: list,
    group_name: str = "groep",
) -> dict:
    """
    Bereken metrieken per subgroep (bijv. per lengtecategorie of bronbetrouwbaarheid).

    Args:
        y_true:     Werkelijke labels
        y_pred:     Voorspelde labels
        groups:     Groepslabels per sample
        group_name: Naam van de groepsvariabele

    Returns:
        Dictionary met metrieken per groep
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    groups = np.array(groups)

    results = {}
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 5:
            continue
        results[group] = {
            group_name: group,
            "n_samples": int(mask.sum()),
            "f1_macro": f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "precision": precision_score(y_true[mask], y_pred[mask], average="macro", zero_division=0),
            "recall": recall_score(y_true[mask], y_pred[mask], average="macro", zero_division=0),
            "error_rate": 1 - accuracy_score(y_true[mask], y_pred[mask]),
        }

    return results
