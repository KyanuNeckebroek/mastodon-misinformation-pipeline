"""
src/models/train_svm.py

Traint een Support Vector Machine (SVM) classificator met TF-IDF vectorisatie
voor misinformatie-detectie.

Gebruik:
    python src/models/train_svm.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs, set_seed
from src.evaluation.metrics import compute_metrics, print_classification_report


def build_pipeline(config: dict) -> Pipeline:
    """Bouw de TF-IDF + SVM pipeline."""
    tfidf_cfg = config["tfidf"]
    svm_cfg = config["svm"]

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

    # LinearSVC is sneller dan SVC voor grote datasets
    # CalibratedClassifierCV voegt kansschattingen toe
    svm = CalibratedClassifierCV(
        LinearSVC(
            C=svm_cfg["C"],
            class_weight=svm_cfg["class_weight"],
            max_iter=svm_cfg["max_iter"],
            random_state=42,
        )
    )

    return Pipeline([("tfidf", tfidf), ("svm", svm)])


def grid_search_svm(
    pipeline: Pipeline,
    X_train: list,
    y_train: list,
    config: dict,
) -> Pipeline:
    """Voer hyperparameter-optimalisatie uit met grid search."""
    gs_cfg = config["svm"]["grid_search"]

    param_grid = {
        "svm__estimator__C": gs_cfg["C"],
        "tfidf__max_features": [30000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    }

    logger.info("Grid search starten (dit kan enkele minuten duren)...")
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
    logger.info(f"Beste CV F1-score: {gs.best_score_:.4f}")

    return gs.best_estimator_


def train_svm(config: dict, use_grid_search: bool = False) -> None:
    """Volledige SVM-trainingspipeline."""
    set_seed(config["project"]["random_seed"])
    out_dir = config["svm"]["output_dir"]
    ensure_dirs(out_dir)

    proc_dir = config["preprocessing"]["processed_dir"]

    logger.info("Data laden voor SVM-training...")
    train_df = pd.read_csv(f"{proc_dir}train_combined.csv")
    val_df   = pd.read_csv(f"{proc_dir}val_combined.csv")

    # Verwijder rijen met lege tekst
    train_df = train_df.dropna(subset=["clean_text"])
    val_df   = val_df.dropna(subset=["clean_text"])

    X_train = train_df["clean_text"].tolist()
    y_train = train_df["label"].tolist()
    X_val   = val_df["clean_text"].tolist()
    y_val   = val_df["label"].tolist()

    logger.info(f"Trainingsset: {len(X_train)} samples")
    logger.info(f"Validatieset: {len(X_val)} samples")

    pipeline = build_pipeline(config)

    if use_grid_search:
        logger.info("Grid search ingeschakeld.")
        pipeline = grid_search_svm(pipeline, X_train, y_train, config)
    else:
        logger.info("SVM trainen met standaard hyperparameters...")
        pipeline.fit(X_train, y_train)

    # Evalueer op validatieset
    logger.info("Evalueren op validatieset...")
    y_pred_val = pipeline.predict(X_val)
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]

    val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)
    logger.info("=== SVM VALIDATIERESULTATEN ===")
    print_classification_report(y_val, y_pred_val, model_name="SVM")

    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Sla model op
    model_path = config["svm"]["model_path"]
    joblib.dump(pipeline, model_path)
    logger.info(f"SVM-model opgeslagen: {model_path}")

    # Sla validatiemetrieken op
    metrics_df = pd.DataFrame([{"model": "SVM", **val_metrics}])
    metrics_df.to_csv(f"{out_dir}svm_val_metrics.csv", index=False)

    # Sla top TF-IDF kenmerken op (voor analyse)
    save_top_features(pipeline, out_dir)

    return pipeline, val_metrics


def save_top_features(pipeline: Pipeline, out_dir: str, top_n: int = 50) -> None:
    """Sla de meest discriminerende TF-IDF kenmerken op per klasse."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        svm_cal = pipeline.named_steps["svm"]
        svm = svm_cal.estimator

        feature_names = tfidf.get_feature_names_out()
        coefs = svm.coef_[0]

        top_false = pd.DataFrame({
            "feature": feature_names[np.argsort(coefs)[:top_n]],
            "coefficient": np.sort(coefs)[:top_n],
            "class": "onwaar (negatief)",
        })
        top_true = pd.DataFrame({
            "feature": feature_names[np.argsort(coefs)[-top_n:][::-1]],
            "coefficient": np.sort(coefs)[-top_n:][::-1],
            "class": "waar (positief)",
        })

        top_features = pd.concat([top_true, top_false])
        top_features.to_csv(f"{out_dir}svm_top_features.csv", index=False)
        logger.info(f"Top {top_n} kenmerken opgeslagen: {out_dir}svm_top_features.csv")

    except Exception as e:
        logger.warning(f"Kon kenmerken niet opslaan: {e}")


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])

    # Zet use_grid_search=True voor hyperparameter-optimalisatie
    train_svm(config, use_grid_search=False)


if __name__ == "__main__":
    main()
