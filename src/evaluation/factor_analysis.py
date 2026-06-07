"""
src/evaluation/factor_analysis.py

Analyseert de invloed van beïnvloedende factoren op classificatieprestaties:
- Berichtlengte (kort / middellang / lang / zeer lang)
- Bronbetrouwbaarheid (laag / middel / hoog)
- Taalgebruik-indicatoren (uitroeptekens, vraagtekens, URL-aanwezigheid)

Statistische toetsen: Chi-kwadraat en ANOVA (eenzijdig)

Gebruik:
    python src/evaluation/factor_analysis.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs
from src.evaluation.metrics import compute_per_group_metrics


def load_test_predictions(config: dict) -> pd.DataFrame:
    """
    Laad de testset en voeg modelvoorspellingen toe.
    Geeft een DataFrame terug met werkelijke labels, voorspellingen en kenmerken.
    """
    proc_dir = config["preprocessing"]["processed_dir"]
    test_df = pd.read_csv(f"{proc_dir}test_combined.csv").dropna(subset=["clean_text"])
    X_test = test_df["clean_text"].tolist()
    y_test = test_df["label"].tolist()

    # Laad SVM-model als referentiemodel voor factoranalyse
    # (herhaal voor alle modellen indien gewenst)
    model_predictions = {}

    for model_name, model_path_key in [
        ("SVM", "svm.model_path"),
        ("LR", "logistic_regression.model_path"),
    ]:
        # Haal geneste config-sleutel op
        keys = model_path_key.split(".")
        path = config
        for k in keys:
            path = path[k]

        if os.path.exists(path):
            pipeline = joblib.load(path)
            preds = pipeline.predict(X_test)
            model_predictions[model_name] = preds
            logger.info(f"{model_name} voorspellingen geladen.")
        else:
            logger.warning(f"{model_name} model niet gevonden: {path}")

    # Voeg voorspellingen toe aan DataFrame
    for model_name, preds in model_predictions.items():
        test_df[f"pred_{model_name}"] = preds
        test_df[f"error_{model_name}"] = (test_df["label"] != preds).astype(int)

    test_df["y_true"] = y_test
    return test_df


def analyze_length_effect(df: pd.DataFrame, figures_dir: str) -> pd.DataFrame:
    """
    Analyseer hoe berichtlengte de classificatieprestaties beïnvloedt.
    Statistische toets: Chi-kwadraat (categorische lengte vs fout)
    """
    logger.info("\n=== FACTOR: BERICHTLENGTE ===")
    results_rows = []

    for model_col in [c for c in df.columns if c.startswith("pred_")]:
        model_name = model_col.replace("pred_", "")
        error_col = f"error_{model_name}"
        if error_col not in df.columns:
            continue

        group_metrics = compute_per_group_metrics(
            df["y_true"].tolist(),
            df[model_col].tolist(),
            df["length_category"].tolist(),
            group_name="length_category",
        )

        for group, metrics in group_metrics.items():
            results_rows.append({"model": model_name, **metrics})

        # Chi-kwadraat toets: lengtecategorie vs classificatiefout
        contingency = pd.crosstab(df["length_category"], df[error_col])
        chi2, p_val, dof, _ = stats.chi_kwadraat = stats.chi2_contingency(contingency)

        logger.info(f"\n{model_name} - Lengte vs Classificatiefout:")
        logger.info(f"  Chi² = {chi2:.4f}, df = {dof}, p = {p_val:.6f}")
        if p_val < 0.05:
            logger.info(f"  → Significant effect van berichtlengte (p < 0.05)")
        else:
            logger.info(f"  → Geen significant effect (p ≥ 0.05)")

        for group, metrics in group_metrics.items():
            logger.info(f"  {group}: n={metrics['n_samples']}, "
                        f"F1={metrics['f1_macro']:.4f}, "
                        f"error_rate={metrics['error_rate']:.4f}")

    results_df = pd.DataFrame(results_rows)

    # Visualisatie
    if not results_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = ["kort", "middellang", "lang", "zeer_lang"]
        models = results_df["model"].unique()
        colors = ["#2196F3", "#4CAF50", "#FF5722"]
        x = np.arange(len(order))
        width = 0.8 / len(models)

        for i, (model, color) in enumerate(zip(models, colors)):
            subset = results_df[results_df["model"] == model]
            f1_values = [
                subset[subset["length_category"] == cat]["f1_macro"].values[0]
                if len(subset[subset["length_category"] == cat]) > 0 else 0
                for cat in order
            ]
            ax.bar(x + i * width, f1_values, width, label=model, color=color, alpha=0.8)

        ax.set_xlabel("Berichtlengte Categorie", fontsize=12)
        ax.set_ylabel("F1-score (macro)", fontsize=12)
        ax.set_title("F1-score per Berichtlengte Categorie", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(["Kort\n(<50 tekens)", "Middellang\n(50-150)", "Lang\n(150-300)", "Zeer lang\n(>300)"])
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}f1_per_lengte.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Grafiek opgeslagen: {figures_dir}f1_per_lengte.png")

    return results_df


def analyze_source_trust_effect(df: pd.DataFrame, figures_dir: str) -> pd.DataFrame:
    """
    Analyseer hoe bronbetrouwbaarheid de classificatieprestaties beïnvloedt.
    Statistische toets: ANOVA (trust score vs fout)
    """
    logger.info("\n=== FACTOR: BRONBETROUWBAARHEID ===")

    # Categoriseer bronbetrouwbaarheid
    threshold = 5.0
    df = df.copy()
    df["trust_category"] = pd.cut(
        df["source_trust_score"],
        bins=[0, 3.5, 6.5, 10],
        labels=["laag (1-3.5)", "middel (3.5-6.5)", "hoog (6.5-10)"],
    )

    results_rows = []

    for model_col in [c for c in df.columns if c.startswith("pred_")]:
        model_name = model_col.replace("pred_", "")
        error_col = f"error_{model_name}"
        if error_col not in df.columns:
            continue

        # ANOVA: is er een significant verschil in foutpercentage per trustcategorie?
        groups_data = [
            df[df["trust_category"] == cat][error_col].dropna().values
            for cat in df["trust_category"].cat.categories
            if len(df[df["trust_category"] == cat]) >= 5
        ]

        if len(groups_data) >= 2:
            f_stat, p_val = stats.f_oneway(*groups_data)
            logger.info(f"\n{model_name} - ANOVA (trust vs fout):")
            logger.info(f"  F = {f_stat:.4f}, p = {p_val:.6f}")
            if p_val < 0.05:
                logger.info("  → Significant effect van bronbetrouwbaarheid (p < 0.05)")
            else:
                logger.info("  → Geen significant effect (p ≥ 0.05)")

        group_metrics = compute_per_group_metrics(
            df["y_true"].tolist(),
            df[model_col].tolist(),
            df["trust_category"].astype(str).tolist(),
            group_name="trust_category",
        )

        for group, metrics in group_metrics.items():
            results_rows.append({"model": model_name, **metrics})
            logger.info(f"  {group}: F1={metrics['f1_macro']:.4f}, "
                        f"error_rate={metrics['error_rate']:.4f}")

    results_df = pd.DataFrame(results_rows)

    # Scatterplot: trust score vs foutpercentage
    for model_col in [c for c in df.columns if c.startswith("pred_")]:
        model_name = model_col.replace("pred_", "")
        error_col = f"error_{model_name}"
        if error_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        # Groepeer per trust score (afgerond op 1 decimaal)
        df["trust_rounded"] = df["source_trust_score"].round(1)
        trust_error = df.groupby("trust_rounded")[error_col].mean()

        ax.plot(trust_error.index, trust_error.values, "o-", color="#2196F3", linewidth=2.5, markersize=5)
        ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.7, label=f"Drempel ({threshold})")
        ax.set_xlabel("Bronbetrouwbaarheidsscore (1-10)", fontsize=12)
        ax.set_ylabel("Foutpercentage", fontsize=12)
        ax.set_title(f"{model_name} — Foutpercentage vs Bronbetrouwbaarheid", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}trust_vs_error_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return results_df


def analyze_language_features(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Analyseer de relatie tussen taalgebruik-kenmerken en classificatiefouten.
    Kenmerken: uitroeptekens, vraagtekens, URL-aanwezigheid, hoofdletterverhouding
    """
    logger.info("\n=== FACTOR: TAALGEBRUIK ===")

    feature_cols = ["exclamation_count", "question_count", "caps_ratio", "url_present"]
    feature_labels = ["Uitroeptekens", "Vraagtekens", "Hoofdletterverhouding", "URL aanwezig"]

    for model_col in [c for c in df.columns if c.startswith("pred_")]:
        model_name = model_col.replace("pred_", "")
        error_col = f"error_{model_name}"
        if error_col not in df.columns:
            continue

        logger.info(f"\n{model_name} — Correlatie taalkenmerken vs fout:")

        fig, axes = plt.subplots(1, len(feature_cols), figsize=(16, 5))
        fig.suptitle(f"{model_name} — Taalkenmerken vs Classificatiefout", fontsize=13, fontweight="bold")

        for ax, feat, label in zip(axes, feature_cols, feature_labels):
            if feat not in df.columns:
                continue

            if feat == "url_present":
                # Staafdiagram voor binaire kenmerk
                error_by_group = df.groupby(feat)[error_col].mean()
                ax.bar(["Geen URL", "URL aanwezig"], error_by_group.values, color=["#4CAF50", "#FF5722"], alpha=0.8)
                ax.set_ylabel("Foutpercentage")
            else:
                # Boxplot voor continue kenmerken
                correct = df[df[error_col] == 0][feat].dropna()
                incorrect = df[df[error_col] == 1][feat].dropna()
                ax.boxplot([correct, incorrect], labels=["Correct", "Fout"],
                           patch_artist=True,
                           boxprops=dict(facecolor="#2196F3", alpha=0.7),
                           medianprops=dict(color="red", linewidth=2))
                ax.set_ylabel(label)

                # Mann-Whitney U toets
                if len(correct) >= 5 and len(incorrect) >= 5:
                    stat, p = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")
                    logger.info(f"  {label}: Mann-Whitney U={stat:.1f}, p={p:.6f}")
                    if p < 0.05:
                        ax.set_title(f"{label}\n* p={p:.4f}", fontsize=10)
                    else:
                        ax.set_title(f"{label}\np={p:.4f}", fontsize=10)
                else:
                    ax.set_title(label, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{figures_dir}taalgebruik_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Grafiek opgeslagen: {figures_dir}taalgebruik_{model_name.lower()}.png")


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])

    figures_dir = config["evaluation"]["figures_dir"]
    res_dir = config["evaluation"]["results_dir"]
    ensure_dirs(figures_dir, res_dir)

    logger.info("Testset en voorspellingen laden...")
    test_df = load_test_predictions(config)

    if not any(c.startswith("pred_") for c in test_df.columns):
        logger.error("Geen modelvoorspellingen gevonden. Train eerst de modellen.")
        return

    logger.info("Factoranalyse uitvoeren...")

    length_results = analyze_length_effect(test_df, figures_dir)
    trust_results  = analyze_source_trust_effect(test_df, figures_dir)
    analyze_language_features(test_df, figures_dir)

    # Sla samenvattende resultaten op
    if not length_results.empty:
        length_results.to_csv(f"{res_dir}factor_length_analysis.csv", index=False)
    if not trust_results.empty:
        trust_results.to_csv(f"{res_dir}factor_trust_analysis.csv", index=False)

    logger.info("\nFactoranalyse voltooid! Grafieken en tabellen opgeslagen in results/")


if __name__ == "__main__":
    main()
