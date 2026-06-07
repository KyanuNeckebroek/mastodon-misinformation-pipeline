"""
src/models/train_logistic_regression.py

Traint een Logistic Regression classificator met TF-IDF vectorisatie.

Gebruik:
    python src/models/train_logistic_regression.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs, set_seed
from src.evaluation.metrics import compute_metrics, print_classification_report


def build_pipeline(config: dict) -> Pipeline:
    """Bouw de TF-IDF + Logistic Regression pipeline."""
    tfidf_cfg = config["tfidf"]
    lr_cfg = config["logistic_regression"]

    tfidf = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
        strip_accents="unicode",
    )

    lr = LogisticRegression(
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        solver=lr_cfg["solver"],
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("tfidf", tfidf), ("lr", lr)])


def grid_search_lr(
    pipeline: Pipeline,
    X_train: list,
    y_train: list,
    config: dict,
) -> Pipeline:
    """Hyperparameter-optimalisatie voor Logistic Regression."""
    gs_cfg = config["logistic_regression"]["grid_search"]

    param_grid = {
        "lr__C": gs_cfg["C"],
        "lr__solver": gs_cfg["solver"],
        "tfidf__max_features": [30000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    }

    logger.info("Grid search LR starten...")
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Beste parameters: {gs.best_params_}")
    logger.info(f"Beste CV F1: {gs.best_score_:.4f}")
    return gs.best_estimator_


def save_coefficients(pipeline: Pipeline, out_dir: str, top_n: int = 50) -> None:
    """Sla de belangrijkste coëfficiënten op (interpretatie)."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        lr = pipeline.named_steps["lr"]
        feature_names = tfidf.get_feature_names_out()
        coefs = lr.coef_[0]

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
        }).sort_values("coefficient", ascending=False)

        coef_df.head(top_n).to_csv(f"{out_dir}lr_top_waar_features.csv", index=False)
        coef_df.tail(top_n).to_csv(f"{out_dir}lr_top_onwaar_features.csv", index=False)
        logger.info(f"Coëfficiënten opgeslagen in {out_dir}")
    except Exception as e:
        logger.warning(f"Kon coëfficiënten niet opslaan: {e}")


def train_logistic_regression(config: dict, use_grid_search: bool = False) -> None:
    """Volledige LR-trainingspipeline."""
    set_seed(config["project"]["random_seed"])
    out_dir = config["logistic_regression"]["output_dir"]
    ensure_dirs(out_dir)

    proc_dir = config["preprocessing"]["processed_dir"]

    logger.info("Data laden voor LR-training...")
    train_df = pd.read_csv(f"{proc_dir}train_combined.csv").dropna(subset=["clean_text"])
    val_df   = pd.read_csv(f"{proc_dir}val_combined.csv").dropna(subset=["clean_text"])

    X_train, y_train = train_df["clean_text"].tolist(), train_df["label"].tolist()
    X_val, y_val     = val_df["clean_text"].tolist(), val_df["label"].tolist()

    logger.info(f"Trainingsset: {len(X_train)} | Validatieset: {len(X_val)}")

    pipeline = build_pipeline(config)

    if use_grid_search:
        pipeline = grid_search_lr(pipeline, X_train, y_train, config)
    else:
        logger.info("Logistic Regression trainen...")
        pipeline.fit(X_train, y_train)

    y_pred_val = pipeline.predict(X_val)
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]

    val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)
    logger.info("=== LOGISTIC REGRESSION VALIDATIERESULTATEN ===")
    print_classification_report(y_val, y_pred_val, model_name="Logistic Regression")

    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    model_path = config["logistic_regression"]["model_path"]
    joblib.dump(pipeline, model_path)
    logger.info(f"LR-model opgeslagen: {model_path}")

    metrics_df = pd.DataFrame([{"model": "Logistic Regression", **val_metrics}])
    metrics_df.to_csv(f"{out_dir}lr_val_metrics.csv", index=False)

    save_coefficients(pipeline, out_dir)
    return pipeline, val_metrics


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])
    train_logistic_regression(config, use_grid_search=False)


if __name__ == "__main__":
    main()
