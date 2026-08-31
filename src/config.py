"""Configuration centrale : URLs des sources publiques, codes indicateurs,
chemins du projet.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------- #
# Réseau Sentinelles — export CSV « all » (historique complet)
#   https://www.sentiweb.fr/datasets/all/inc-<INDICATEUR>-<GEO>.csv
# La 1re ligne du CSV est un commentaire JSON (préfixe '#') à sauter.
# La colonne `week` est une semaine ISO au format AAAASS (ex. 202634).
# --------------------------------------------------------------------------- #
SENTIWEB_CSV = "https://www.sentiweb.fr/datasets/all/inc-{indicator}-{geo}.csv"

INDICATORS = {
    3: "Syndromes grippaux",
    6: "Diarrhée aiguë",
    7: "Varicelle",
    25: "Infections respiratoires aiguës (IRA)",
}

# geo : PAY = France entière ; REG = régions ; DEP = départements
GEO_LEVELS = ("PAY", "REG", "DEP")

TARGET_INDICATOR = 3          # syndromes grippaux
TARGET_GEO = "PAY"           # série nationale pour la modélisation principale

# --------------------------------------------------------------------------- #
# Météo-France — données publiques SYNOP (features exogènes, optionnel)
#   https://donneespubliques.meteofrance.fr  (archives mensuelles synop.<AAAAMM>.csv.gz)
# --------------------------------------------------------------------------- #
METEO_SYNOP_MONTHLY = (
    "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.{yyyymm}.csv.gz"
)

# Horizons de prévision évalués (en semaines).
FORECAST_HORIZONS = (1, 2, 3, 4)
