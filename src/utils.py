"""
src/utils.py
Gedeelde hulpfuncties voor het hele project.
"""

import os
import yaml
import random
import logging
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> dict:
    """Laad de YAML-configuratie."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Zorg voor reproduceerbaarheid door alle random seeds te fixeren."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configureer loguru logger."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    logger.remove()
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}",
        rotation="50 MB",
        retention="30 days",
        encoding="utf-8",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    logger.info(f"Logging gestart. Logbestand: {log_file}")


def ensure_dirs(*dirs: str) -> None:
    """Maak mappen aan als ze nog niet bestaan."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_device():
    """Geef het beste beschikbare torch-apparaat terug."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU gevonden: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("Geen GPU gevonden, gebruik CPU. BERT-training zal langzaam zijn!")
        return device
    except ImportError:
        logger.error("PyTorch niet geïnstalleerd.")
        return None


def label_to_int(label: str) -> int:
    """Converteer tekstlabel naar integer (0=onwaar, 1=waar)."""
    mapping = {"waar": 1, "onwaar": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    return mapping.get(str(label).lower().strip(), -1)


def int_to_label(label_int: int) -> str:
    """Converteer integer terug naar tekstlabel."""
    return "waar" if label_int == 1 else "onwaar"


def compute_class_weights(labels: list) -> dict:
    """Bereken klassegewichten voor ongebalanceerde datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))
