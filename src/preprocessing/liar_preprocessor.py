"""
src/preprocessing/liar_preprocessor.py

Verwerkt de LIAR-dataset:
- Laadt de TSV-bestanden
- Zet de 6 labels om naar binaire classificatie (waar/onwaar)
- Reinigt de tekst
- Voegt annotaties toe (berichtlengte, bronbetrouwbaarheid, taalgebruik)

Download de LIAR-dataset van: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
Pak uit naar data/raw/

Gebruik:
    python src/preprocessing/liar_preprocessor.py
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs

# Kolomnamen van de LIAR-dataset (geen header in het bestand)
LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state_info",
    "party_affiliation",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_on_fire_count",
    "context",
]


def load_liar_split(file_path: str) -> pd.DataFrame:
    """Laad een LIAR TSV-bestand."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"LIAR-bestand niet gevonden: {file_path}\n"
            "Download de dataset van: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
        )
    df = pd.read_csv(file_path, sep="\t", header=None, names=LIAR_COLUMNS)
    logger.info(f"Geladen: {file_path} ({len(df)} rijen)")
    return df


def binarize_labels(df: pd.DataFrame, true_labels: list, false_labels: list) -> pd.DataFrame:
    """
    Zet de 6-klasse LIAR-labels om naar binair (0=onwaar, 1=waar).

    Originele labels:
        true, mostly-true → 1 (waar)
        half-true         → 1 (waar, grensgevallen)
        barely-true       → 0 (onwaar)
        false             → 0 (onwaar)
        pants-fire        → 0 (onwaar)
    """
    original_counts = df["label"].value_counts()
    logger.info(f"Originele labelsverdeling:\n{original_counts}")

    df = df.copy()

    def map_label(label):
        if label in true_labels:
            return 1
        elif label in false_labels:
            return 0
        else:
            return np.nan  # Onbekend label

    df["binary_label"] = df["label"].apply(map_label)

    # Verwijder rijen met onbekende labels
    before = len(df)
    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)
    after = len(df)

    if before - after > 0:
        logger.warning(f"Verwijderd: {before - after} rijen met onbekende labels")

    binary_counts = df["binary_label"].value_counts()
    logger.info(f"Binaire labelsverdeling:\n{binary_counts}")
    logger.info(f"Balans: {binary_counts[1] / len(df) * 100:.1f}% waar")

    return df


def clean_text(text: str) -> str:
    """
    Reinig en normaliseer tekst:
    - Zet om naar kleine letters
    - Verwijder HTML-tags
    - Verwijder speciale tekens (behoud leestekens)
    - Verwijder extra witruimte
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Zet om naar kleine letters
    text = text.lower()

    # Verwijder HTML-tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Verwijder URL's
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)

    # Verwijder @mentions
    text = re.sub(r"@\w+", "[MENTION]", text)

    # Verwijder #hashtags maar bewaar de tekst
    text = re.sub(r"#(\w+)", r"\1", text)

    # Verwijder overbodige witruimte
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_text_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Bereken tekstkenmerken voor factoranalyse.
    """
    df = df.copy()

    # Berichtlengte
    df["char_length"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()

    # Taalgebruik-indicatoren
    df["exclamation_count"] = df[text_col].str.count(r"!")
    df["question_count"] = df[text_col].str.count(r"\?")
    df["caps_ratio"] = df[text_col].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df["url_present"] = df[text_col].str.contains(r"\[URL\]", na=False).astype(int)

    # Categoriseer berichtlengte
    def categorize_length(n_chars):
        if n_chars < 50:
            return "kort"
        elif n_chars < 150:
            return "middellang"
        elif n_chars < 300:
            return "lang"
        else:
            return "zeer_lang"

    df["length_category"] = df["char_length"].apply(categorize_length)

    return df


def compute_speaker_trust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schat bronbetrouwbaarheid voor LIAR op basis van historische nauwkeurigheid.

    We gebruiken de geaggregeerde beoordelingshistorie van de spreker
    als proxy voor betrouwbaarheid (beschikbaar in de LIAR-dataset).
    """
    df = df.copy()

    count_cols = [
        "barely_true_count",
        "false_count",
        "half_true_count",
        "mostly_true_count",
        "pants_on_fire_count",
    ]

    # Zet naar numeriek
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    total = df[count_cols].sum(axis=1).replace(0, 1)

    true_statements = df["mostly_true_count"] + df["half_true_count"] * 0.5
    df["source_trust_score"] = (true_statements / total * 9 + 1).clip(1, 10).round(2)

    return df


def process_liar(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Volledige LIAR-verwerkingspipeline. Geeft (train, val, test) terug."""
    true_labels = config["liar"]["true_labels"]
    false_labels = config["liar"]["false_labels"]

    splits = {}
    for split_name, file_key in [("train", "train_file"), ("val", "valid_file"), ("test", "test_file")]:
        df = load_liar_split(config["liar"][file_key])
        
        # Zet de tekstlabels om naar binaire 0 en 1 in een nieuwe kolom
        df = binarize_labels(df, true_labels, false_labels)

        # Reinig tekst
        df["clean_text"] = df["statement"].apply(clean_text)

        # Verwijder te korte teksten
        min_len = config["preprocessing"]["min_text_length"]
        df = df[df["clean_text"].str.len() >= min_len]

        # Verwijder duplicaten
        before = len(df)
        df = df.drop_duplicates(subset="clean_text")
        logger.info(f"{split_name}: {before - len(df)} duplicaten verwijderd")

        # Voeg tekstkenmerken toe
        df = compute_text_features(df, "clean_text")

        # Voeg bronbetrouwbaarheid toe
        df = compute_speaker_trust(df)

        # Print hier de statistieken met de binary_label kolom (die we zeker weten getallen bevat!)
        logger.info(f"LIAR {split_name}: {len(df)} rijen, "
                    f"{df['binary_label'].sum()} waar ({df['binary_label'].mean()*100:.1f}%)")

        # Nu we klaar zijn met de berekeningen: verwijder de oude tekst-label kolom 
        # en hernoem de binary kolom naar 'label' zodat de rest van de pipeline klopt
        df = df.drop(columns=["label"])
        df = df.rename(columns={"binary_label": "label"})
        df["source"] = "liar"
        
        splits[split_name] = df

    return splits["train"], splits["val"], splits["test"]


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])
    ensure_dirs(config["preprocessing"]["processed_dir"])

    train_df, val_df, test_df = process_liar(config)

    out_dir = config["preprocessing"]["processed_dir"]
    train_df.to_csv(f"{out_dir}liar_train.csv", index=False)
    val_df.to_csv(f"{out_dir}liar_val.csv", index=False)
    test_df.to_csv(f"{out_dir}liar_test.csv", index=False)

    logger.info("LIAR-dataset verwerkt en opgeslagen.")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
