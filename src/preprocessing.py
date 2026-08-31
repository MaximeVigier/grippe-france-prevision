"""Nettoyage de la série Sentinelles et mise sur un calendrier hebdomadaire régulier.

L'export national des syndromes grippaux est propre (2 182 semaines de 1984 à 2026,
pas de trou de calendrier). Il reste à :

* convertir ``inc`` / ``inc100`` en numérique (ils arrivent en texte) ;
* combler les rares semaines sans estimation par interpolation courte ;
* réindexer sur un pas hebdomadaire strict, pour que tout trou éventuel soit
  explicite ;
* étiqueter les saisons grippales (semaine 40 de l'année N à semaine 20 de N+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loader


def serie_hebdo(indicator: int = 3, geo: str = "PAY", *,
                colonne: str = "inc100", depuis: str | None = "1990-08-01") -> pd.Series:
    """Série hebdomadaire nettoyée de l'indicateur demandé.

    ``colonne`` : ``inc100`` (taux pour 100 000 hab., par défaut, comparable dans
    le temps) ou ``inc`` (nombre de cas estimé).
    ``depuis`` : on coupe le tout début si besoin ; par défaut on démarre en
    1990 (les cycles complets et la méthode d'estimation sont stables ensuite).
    """
    df = data_loader.load_incidence(indicator, geo)
    s = pd.to_numeric(df[colonne], errors="coerce")
    s.name = {"inc100": "taux_100k", "inc": "cas"}.get(colonne, colonne)

    # calendrier hebdomadaire strict (lundis ISO)
    plein = pd.date_range(s.index.min(), s.index.max(), freq="7D")
    s = s.reindex(plein)

    # trous courts -> interpolation ; trous longs -> on laisse le NaN visible
    s = s.interpolate(limit=2, limit_area="inside")

    if depuis:
        s = s.loc[depuis:]
    return s


def table_saisons(s: pd.Series) -> pd.DataFrame:
    """Ajoute la saison grippale et la position dans la saison.

    Saison ``2018-2019`` = de la semaine 40 de 2018 à la semaine 20 de 2019.
    ``sem_saison`` = 0 à la semaine 40, croît jusqu'à la fin de saison.
    """
    iso = s.index.isocalendar()
    annee, semaine = iso["year"].to_numpy(), iso["week"].to_numpy()

    saison_debut = np.where(semaine >= 40, annee, annee - 1)
    df = pd.DataFrame({"valeur": s.to_numpy(), "sem_iso": semaine}, index=s.index)
    df["saison"] = [f"{a}-{a + 1}" for a in saison_debut]
    df["sem_saison"] = np.where(semaine >= 40, semaine - 40, semaine + 12)
    # hors saison épidémique (été) : marqué à part
    df["hors_saison"] = (semaine > 20) & (semaine < 40)
    return df


def pics_par_saison(s: pd.Series) -> pd.DataFrame:
    """Date et hauteur du pic pour chaque saison grippale complète."""
    t = table_saisons(s)
    t = t[~t["hors_saison"]]
    out = (t.groupby("saison")["valeur"]
             .agg(pic="max", date_pic=lambda x: x.idxmax(), n="count"))
    return out[out["n"] >= 25]  # saisons complètes seulement
