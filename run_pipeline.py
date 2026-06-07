"""
run_pipeline.py

Hoofdscript dat de volledige pipeline uitvoert in de juiste volgorde.
Ideaal voor een schone, reproduceerbare run van begin tot einde.

Gebruik:
    python run_pipeline.py [--skip-mastodon] [--grid-search] [--skip-bert]
"""

import argparse
import sys
import os
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))
from src.utils import load_config, setup_logging, ensure_dirs, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Misinformatie Detectie Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voorbeeld gebruik:
  python run_pipeline.py                     # Volledige pipeline
  python run_pipeline.py --skip-mastodon     # Sla Mastodon-verzameling over
  python run_pipeline.py --grid-search       # Gebruik grid search voor SVM/LR
  python run_pipeline.py --skip-bert         # Sla BERT-training over (alleen SVM + LR)
        """
    )
    parser.add_argument("--skip-mastodon", action="store_true",
                        help="Sla Mastodon-verzameling over (gebruik alleen LIAR)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Gebruik grid search voor hyperparameter-optimalisatie")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Sla BERT-training over (sneller, maar minder volledig)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Voer alleen evaluatie uit (vereist getrainde modellen)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    setup_logging(config["logging"]["log_dir"])
    set_seed(config["project"]["random_seed"])

    logger.info("=" * 60)
    logger.info("MISINFORMATIE DETECTIE — VOLLEDIGE PIPELINE")
    logger.info("=" * 60)

    if not args.eval_only:
        # ── STAP 1: DATA VERZAMELEN ───────────────────────────────────
        logger.info("\n[STAP 1/4] DATA VERZAMELEN EN VOORBEREIDEN")

        logger.info("LIAR-dataset verwerken...")
        from src.preprocessing.liar_preprocessor import main as liar_main
        liar_main()

        if not args.skip_mastodon:
            logger.info("Mastodon-berichten verzamelen overgeslagen — bestaand gelabeld bestand wordt gebruikt.")

        logger.info("Datasets samenvoegen...")
        from src.preprocessing.data_merger import main as merger_main
        merger_main()

        # ── STAP 2: MODELLEN TRAINEN ──────────────────────────────────
        logger.info("\n[STAP 2/4] MODELLEN TRAINEN")

        logger.info("SVM trainen...")
        from src.models.train_svm import train_svm
        train_svm(config, use_grid_search=args.grid_search)

        logger.info("Logistic Regression trainen...")
        from src.models.train_logistic_regression import train_logistic_regression
        train_logistic_regression(config, use_grid_search=args.grid_search)

        if not args.skip_bert:
            logger.info("BERT fine-tunen (dit kan lang duren op CPU)...")
            from src.models.train_bert import train_bert
            train_bert(config)
        else:
            logger.info("BERT-training overgeslagen (--skip-bert)")

    # ── STAP 3: EINDEVALUATIE ────────────────────────────────────────
    logger.info("\n[STAP 3/4] EINDEVALUATIE OP TESTSET")
    from src.evaluation.evaluate_all import main as eval_main
    eval_main()

    # ── STAP 4: FACTORANALYSE ────────────────────────────────────────
    logger.info("\n[STAP 4/4] FACTORANALYSE")
    from src.evaluation.factor_analysis import main as factor_main
    factor_main()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE VOLTOOID!")
    logger.info("Resultaten beschikbaar in: results/")
    logger.info("Grafieken beschikbaar in:  results/figures/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
