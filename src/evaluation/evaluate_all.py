"""
src/evaluation/evaluate_all.py

Voert de eindevaluatie uit op de onafhankelijke testset voor alle drie de modellen.
Vergelijkt SVM, Logistic Regression en BERT op basis van F1, Precision en Recall.

Gebruik:
    python src/evaluation/evaluate_all.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs, get_device
from src.evaluation.metrics import compute_metrics, print_classification_report


def load_test_data(config: dict) -> tuple:
    """Laad de testset."""
    proc_dir = config["preprocessing"]["processed_dir"]
    test_df = pd.read_csv(f"{proc_dir}test_combined.csv").dropna(subset=["clean_text"])
    logger.info(f"Testset geladen: {len(test_df)} samples")
    return test_df


def predict_sklearn_model(model_path: str, X_test: list) -> tuple:
    """Laad een sklearn-model en maak voorspellingen."""
    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def predict_bert_model(model_path: str, X_test: list, config: dict) -> tuple:
    """Laad het BERT-model en maak voorspellingen op de testset."""
    try:
        import torch
        from transformers import BertTokenizerFast, BertForSequenceClassification
        from torch.utils.data import DataLoader
        from src.models.train_bert import MisinformatieDataset
        from tqdm import tqdm

        device = get_device()
        bert_cfg = config["bert"]

        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        # Dummy labels (worden niet gebruikt voor voorspelling)
        dummy_labels = [0] * len(X_test)
        dataset = MisinformatieDataset(X_test, dummy_labels, tokenizer, bert_cfg["max_length"])
        loader = DataLoader(dataset, batch_size=bert_cfg["batch_size"] * 2, shuffle=False)

        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="BERT testset voorspelling"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    except Exception as e:
        logger.error(f"BERT-voorspelling mislukt: {e}")
        return None, None


def plot_confusion_matrices(results: dict, y_test: list, figures_dir: str) -> None:
    """Plot en sla confusion matrices op voor alle modellen."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, data["y_pred"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_pct, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=["Onwaar", "Waar"],
            yticklabels=["Onwaar", "Waar"],
            ax=ax, cbar=False,
        )
        # Voeg absolute aantallen toe
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f"(n={cm[i,j]})",
                        ha="center", va="center", fontsize=9, color="gray")

        f1 = data["metrics"]["f1_macro"]
        ax.set_title(f"{model_name}\nF1-macro: {f1:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Voorspeld label")
        ax.set_ylabel("Werkelijk label")

    plt.tight_layout()
    path = f"{figures_dir}confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrices opgeslagen: {path}")


def plot_metrics_comparison(results: dict, figures_dir: str) -> None:
    """Staafdiagram voor metriekvergelijking."""
    metrics_to_plot = ["f1_macro", "precision", "recall", "accuracy", "auc_roc"]
    metric_labels   = ["F1 (macro)", "Precision", "Recall", "Accuracy", "AUC-ROC"]

    model_names = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for i, (model_name, data) in enumerate(results.items()):
        values = [data["metrics"].get(m, 0) for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        # Voeg waarden boven de balken toe
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Metriek", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Vergelijking van modelprestaties op testset", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{figures_dir}metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Metriekvergelijking opgeslagen: {path}")


def plot_roc_curves(results: dict, y_test: list, figures_dir: str) -> None:
    """Plot ROC-curves voor alle modellen."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for (model_name, data), color in zip(results.items(), colors):
        if data["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f"{model_name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Kansmodel (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-curves — Misinformatie Detectie", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{figures_dir}roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC-curves opgeslagen: {path}")


def evaluate_all_models(config: dict) -> dict:
    """Evalueer alle modellen op de testset en sla resultaten op."""
    test_df = load_test_data(config)
    X_test = test_df["clean_text"].tolist()
    y_test = test_df["label"].tolist()

    results = {}

    # --- SVM ---
    svm_path = config["svm"]["model_path"]
    if os.path.exists(svm_path):
        logger.info("SVM evalueren op testset...")
        y_pred, y_prob = predict_sklearn_model(svm_path, X_test)
        results["SVM"] = {
            "y_pred": y_pred, "y_prob": y_prob,
            "metrics": compute_metrics(y_test, y_pred, y_prob),
        }
        print_classification_report(y_test, y_pred, model_name="SVM")
    else:
        logger.warning(f"SVM-model niet gevonden: {svm_path}")

    # --- Logistic Regression ---
    lr_path = config["logistic_regression"]["model_path"]
    if os.path.exists(lr_path):
        logger.info("Logistic Regression evalueren op testset...")
        y_pred, y_prob = predict_sklearn_model(lr_path, X_test)
        results["Logistic Regression"] = {
            "y_pred": y_pred, "y_prob": y_prob,
            "metrics": compute_metrics(y_test, y_pred, y_prob),
        }
        print_classification_report(y_test, y_pred, model_name="Logistic Regression")
    else:
        logger.warning(f"LR-model niet gevonden: {lr_path}")

    # --- BERT ---
    bert_path = config["bert"]["model_path"]
    if os.path.exists(bert_path):
        logger.info("BERT evalueren op testset...")
        y_pred, y_prob = predict_bert_model(bert_path, X_test, config)
        if y_pred is not None:
            results["BERT"] = {
                "y_pred": y_pred, "y_prob": y_prob,
                "metrics": compute_metrics(y_test, y_pred, y_prob),
            }
            print_classification_report(y_test, y_pred, model_name="BERT")
    else:
        logger.warning(f"BERT-model niet gevonden: {bert_path}")

    return results, y_test, test_df


def save_results(results: dict, y_test: list, config: dict) -> None:
    """Sla alle resultaten op als CSV en JSON."""
    res_dir = config["evaluation"]["results_dir"]

    # Vergelijkingstabel
    rows = []
    for model_name, data in results.items():
        row = {"model": model_name, **data["metrics"]}
        rows.append(row)

    comparison_df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison_df.to_csv(f"{res_dir}model_comparison_test.csv", index=False)
    logger.info(f"\n=== VERGELIJKING OP TESTSET ===\n{comparison_df.to_string(index=False)}")

    # Individuele voorspellingen (voor foutanalyse)
    # (Dit slaan we niet op om bestandsgrootte te beperken)


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])

    figures_dir = config["evaluation"]["figures_dir"]
    ensure_dirs(figures_dir, config["evaluation"]["results_dir"])

    results, y_test, test_df = evaluate_all_models(config)

    if not results:
        logger.error("Geen modellen gevonden. Train eerst de modellen.")
        return

    # Visualisaties
    plot_confusion_matrices(results, y_test, figures_dir)
    plot_metrics_comparison(results, figures_dir)
    plot_roc_curves(results, y_test, figures_dir)

    save_results(results, y_test, config)
    logger.info("Eindevaluatie voltooid! Resultaten in results/")


if __name__ == "__main__":
    main()
