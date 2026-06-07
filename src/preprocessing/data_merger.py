"""
src/preprocessing/data_merger.py

Combineert de verwerkte LIAR-dataset met de gelabelde Mastodon-berichten
tot één gecombineerde dataset voor training, validatie en testing.

Vereisten:
- data/processed/liar_train.csv, liar_val.csv, liar_test.csv (van liar_preprocessor.py)
- data/mastodon/mastodon_posts.csv (handmatig gelabeld na mastodon_collector.py)

Gebruik:
    python src/preprocessing/data_merger.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs
from src.preprocessing.liar_preprocessor import clean_text, compute_text_features


REQUIRED_COLUMNS = ["clean_text", "label", "source_trust_score", "source",
                    "char_length", "word_count", "length_category"]


def load_mastodon_labeled(file_path: str, config: dict) -> pd.DataFrame:
    """
    Laad de handmatig gelabelde Mastodon-berichten.
    Verwijder rijen zonder label.
    """
    if not os.path.exists(file_path):
        logger.warning(
            f"Mastodon-bestand niet gevonden: {file_path}\n"
            "Sla dit over en gebruik alleen LIAR-data.\n"
            "Voer eerst mastodon_collector.py uit en label de data handmatig."
        )
        return pd.DataFrame()

    df = pd.read_csv(file_path, encoding="utf-8", sep=";")
    logger.info(f"Mastodon geladen: {len(df)} rijen")

    # Verwijder ongelabelde rijen
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Controleer op geldige labels
    df = df[df["label"].isin([0, 1])]
    logger.info(f"Mastodon na labeling-filter: {len(df)} rijen")

    if len(df) == 0:
        logger.warning("Geen gelabelde Mastodon-berichten gevonden.")
        return pd.DataFrame()

    # Reinig tekst als nog niet gedaan
    if "clean_text" not in df.columns:
        df["clean_text"] = df["content"].apply(clean_text)

    # Voeg kenmerken toe
    df = compute_text_features(df, "clean_text")

    # Bronbetrouwbaarheid (al berekend door collector, maar vul ontbrekende in)
    if "source_trust_score" not in df.columns:
        df["source_trust_score"] = 5.0  # Neutraal voor onbekende bronnen

    df["source"] = "mastodon"

    label_counts = df["label"].value_counts()
    logger.info(f"Mastodon labels: {label_counts.to_dict()}")
    return df


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Zorg dat alle vereiste kolommen aanwezig zijn."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col in ["char_length", "word_count"]:
                df[col] = 0
            elif col == "source_trust_score":
                df[col] = 5.0
            elif col == "length_category":
                df[col] = "onbekend"
            else:
                df[col] = ""
    return df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]]


def merge_and_split(
    liar_train: pd.DataFrame,
    liar_val: pd.DataFrame,
    liar_test: pd.DataFrame,
    mastodon_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strategie:
    - LIAR train/val/test behoudt zijn originele split
    - Mastodon-data wordt gesplitst (70% train, 15% val, 15% test)
    - Beide datasets worden samengevoegd per split
    """
    seed = config["project"]["random_seed"]

    if len(mastodon_df) > 0:
        # Split Mastodon-data
        masto_train_val, masto_test = train_test_split(
            mastodon_df, test_size=0.15, random_state=seed, stratify=mastodon_df["label"]
        )
        masto_train, masto_val = train_test_split(
            masto_train_val,
            test_size=0.15 / 0.85,
            random_state=seed,
            stratify=masto_train_val["label"],
        )

        train = pd.concat([liar_train, masto_train], ignore_index=True)
        val   = pd.concat([liar_val,   masto_val],   ignore_index=True)
        test  = pd.concat([liar_test,  masto_test],  ignore_index=True)

        logger.info(f"Mastodon toegevoegd: train+{len(masto_train)}, "
                    f"val+{len(masto_val)}, test+{len(masto_test)}")
    else:
        train, val, test = liar_train, liar_val, liar_test
        logger.info("Geen Mastodon-data — alleen LIAR-dataset.")

    # Schud de datasets door elkaar
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    val   = val.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train, val, test


def print_dataset_stats(df: pd.DataFrame, name: str) -> None:
    """Print uitgebreide statistieken van een dataset."""
    logger.info(f"\n=== {name.upper()} STATISTIEKEN ===")
    logger.info(f"Totaal rijen: {len(df)}")
    logger.info(f"Labels: {df['label'].value_counts().to_dict()}")
    logger.info(f"Balans: {df['label'].mean()*100:.1f}% waar")
    logger.info(f"Bronnen: {df['source'].value_counts().to_dict()}")
    logger.info(f"Gemiddelde lengte: {df['char_length'].mean():.0f} tekens")
    logger.info(f"Lengtecategorieën:\n{df['length_category'].value_counts()}")
    logger.info(f"Gem. bronbetrouwbaarheid: {df['source_trust_score'].mean():.2f}/10")


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])
    out_dir = config["preprocessing"]["processed_dir"]
    ensure_dirs(out_dir)

    # Laad LIAR splits
    logger.info("LIAR-splits laden...")
    liar_train = pd.read_csv(f"{out_dir}liar_train.csv")
    liar_val   = pd.read_csv(f"{out_dir}liar_val.csv")
    liar_test  = pd.read_csv(f"{out_dir}liar_test.csv")

    # Laad Mastodon (optioneel)
    logger.info("Mastodon-berichten laden...")
    mastodon_df = load_mastodon_labeled(
        config["mastodon"]["output_file"], config
    )

    # Zorg voor consistente kolommen
    liar_train = align_columns(liar_train)
    liar_val   = align_columns(liar_val)
    liar_test  = align_columns(liar_test)
    if len(mastodon_df) > 0:
        mastodon_df = align_columns(mastodon_df)

    # Samenvoegen
    train, val, test = merge_and_split(
        liar_train, liar_val, liar_test, mastodon_df, config
    )

    # Sla op
    train.to_csv(f"{out_dir}train_combined.csv", index=False)
    val.to_csv(f"{out_dir}val_combined.csv", index=False)
    test.to_csv(f"{out_dir}test_combined.csv", index=False)

    print_dataset_stats(train, "Train")
    print_dataset_stats(val, "Validatie")
    print_dataset_stats(test, "Test")

    logger.info("\nGecombineerde datasets opgeslagen in data/processed/")


if __name__ == "__main__":
    main()
