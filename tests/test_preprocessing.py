"""
tests/test_preprocessing.py

Unittests voor de preprocessing- en evaluatiemodules.

Gebruik:
    pytest tests/ -v
    pytest tests/ -v --cov=src
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.liar_preprocessor import clean_text, binarize_labels, compute_text_features
from src.evaluation.metrics import compute_metrics, compute_per_group_metrics
from src.utils import label_to_int, int_to_label, compute_class_weights


# ── PREPROCESSING TESTS ────────────────────────────────────────────

class TestCleanText:

    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_html_removal(self):
        result = clean_text("<p>Dit is <b>tekst</b></p>")
        assert "<" not in result and ">" not in result
        assert "tekst" in result

    def test_url_replacement(self):
        result = clean_text("Kijk op https://example.com voor meer info")
        assert "[URL]" in result
        assert "https://" not in result

    def test_mention_replacement(self):
        result = clean_text("@gebruiker schreef dit")
        assert "[MENTION]" in result

    def test_hashtag_text_preserved(self):
        result = clean_text("#klimaatverandering is een feit")
        assert "klimaatverandering" in result
        assert "#" not in result

    def test_extra_whitespace_removed(self):
        result = clean_text("te   veel    spaties")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_handling(self):
        assert clean_text(None) == ""


class TestBinarizeLabels:

    def setup_method(self):
        self.df = pd.DataFrame({
            "label": ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        })
        self.true_labels  = ["true", "mostly-true", "half-true"]
        self.false_labels = ["barely-true", "false", "pants-fire"]

    def test_true_labels_become_1(self):
        result = binarize_labels(self.df, self.true_labels, self.false_labels)
        assert all(result[result["label"].isin(self.true_labels)]["binary_label"] == 1)

    def test_false_labels_become_0(self):
        result = binarize_labels(self.df, self.true_labels, self.false_labels)
        assert all(result[result["label"].isin(self.false_labels)]["binary_label"] == 0)

    def test_unknown_labels_removed(self):
        df_with_unknown = pd.DataFrame({"label": ["true", "unknown_label", "false"]})
        result = binarize_labels(df_with_unknown, self.true_labels, self.false_labels)
        assert len(result) == 2

    def test_output_dtype_is_int(self):
        result = binarize_labels(self.df, self.true_labels, self.false_labels)
        assert result["binary_label"].dtype == int


class TestTextFeatures:

    def test_char_length(self):
        df = pd.DataFrame({"clean_text": ["hallo", "een langere zin met meer woorden"]})
        result = compute_text_features(df)
        assert result.loc[0, "char_length"] == 5
        assert result.loc[1, "char_length"] > 5

    def test_word_count(self):
        df = pd.DataFrame({"clean_text": ["een twee drie"]})
        result = compute_text_features(df)
        assert result.loc[0, "word_count"] == 3

    def test_length_category_short(self):
        df = pd.DataFrame({"clean_text": ["hi"]})
        result = compute_text_features(df)
        assert result.loc[0, "length_category"] == "kort"

    def test_length_category_very_long(self):
        long_text = "woord " * 100  # 600 tekens
        df = pd.DataFrame({"clean_text": [long_text.strip()]})
        result = compute_text_features(df)
        assert result.loc[0, "length_category"] == "zeer_lang"


# ── METRICS TESTS ──────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_predictions(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_returns_required_keys(self):
        y_true = [0, 1, 0]
        y_pred = [0, 1, 1]
        metrics = compute_metrics(y_true, y_pred)
        assert "f1_macro" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "accuracy" in metrics

    def test_with_probabilities(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        y_prob = [0.1, 0.9, 0.2, 0.8]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert "auc_roc" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0


class TestPerGroupMetrics:

    def test_basic_grouping(self):
        y_true  = [0, 0, 1, 1, 0, 1]
        y_pred  = [0, 0, 1, 1, 0, 1]
        groups  = ["A", "A", "A", "B", "B", "B"]
        result  = compute_per_group_metrics(y_true, y_pred, groups, "group")
        assert "A" in result
        assert "B" in result

    def test_small_groups_excluded(self):
        y_true = [0, 1]
        y_pred = [0, 1]
        groups = ["klein", "klein"]
        result = compute_per_group_metrics(y_true, y_pred, groups)
        # Groep met < 5 samples wordt overgeslagen
        assert len(result) == 0


# ── UTILS TESTS ────────────────────────────────────────────────────

class TestUtils:

    def test_label_to_int_waar(self):
        assert label_to_int("waar") == 1
        assert label_to_int("true") == 1
        assert label_to_int("1") == 1

    def test_label_to_int_onwaar(self):
        assert label_to_int("onwaar") == 0
        assert label_to_int("false") == 0
        assert label_to_int("0") == 0

    def test_label_to_int_unknown(self):
        assert label_to_int("onbekend") == -1

    def test_int_to_label(self):
        assert int_to_label(1) == "waar"
        assert int_to_label(0) == "onwaar"

    def test_class_weights_balanced(self):
        labels = [0, 0, 0, 1, 1, 1]  # Gebalanceerd
        weights = compute_class_weights(labels)
        assert abs(weights[0] - weights[1]) < 0.01

    def test_class_weights_imbalanced(self):
        labels = [0, 0, 0, 0, 1]  # Ongebalanceerd
        weights = compute_class_weights(labels)
        assert weights[1] > weights[0]  # Minderheidsklasse krijgt meer gewicht
