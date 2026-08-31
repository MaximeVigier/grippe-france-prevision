"""Récupération et mise en cache des données du réseau Sentinelles.

L'export CSV « all » contient tout l'historique hebdomadaire d'un indicateur
pour un niveau géographique. On le télécharge une fois dans ``data/raw/`` puis
on le relit localement.

Format du CSV :
* 1re ligne = commentaire JSON (préfixe ``#``) — on la saute ;
* colonnes : ``week, indicator, inc, inc_low, inc_up, inc100, inc100_low,
  inc100_up, geo_insee, geo_name`` ;
* ``week`` = semaine ISO ``AAAASS`` (ex. ``202634`` = semaine 34 de 2026).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_DIR, SENTIWEB_CSV


def _local_path(indicator: int, geo: str) -> Path:
    return RAW_DIR / f"sentiweb_inc-{indicator}-{geo}.csv"


def download_sentiweb(indicator: int, geo: str = "PAY", *, force: bool = False) -> Path:
    """Télécharge l'historique complet d'un indicateur Sentinelles (cache local)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = _local_path(indicator, geo)
    if dest.exists() and not force:
        return dest
    url = SENTIWEB_CSV.format(indicator=indicator, geo=geo)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def yearweek_to_date(yearweek: int | str) -> pd.Timestamp:
    """Convertit une semaine ISO ``AAAASS`` en date du lundi correspondant.

    Ex. ``202634`` -> lundi de la semaine ISO 34 de 2026.
    """
    s = str(yearweek)
    year, week = int(s[:4]), int(s[4:])
    return pd.Timestamp.fromisocalendar(year, week, 1)


def load_incidence(indicator: int, geo: str = "PAY") -> pd.DataFrame:
    """Charge une série Sentinelles nettoyée a minima.

    Returns
    -------
    DataFrame trié par date croissante, indexé par ``date`` (lundi ISO), avec
    au moins ``inc`` (incidence estimée) et ``inc100`` (taux pour 100 000 hab.).
    """
    path = download_sentiweb(indicator, geo)
    df = pd.read_csv(path, comment="#")
    df["date"] = df["week"].apply(yearweek_to_date)
    df = df.sort_values("date").set_index("date")
    return df


if __name__ == "__main__":
    s = load_incidence(3, "PAY")
    print(f"Syndromes grippaux — France : {len(s)} semaines, "
          f"de {s.index.min():%Y-%m-%d} à {s.index.max():%Y-%m-%d}")
