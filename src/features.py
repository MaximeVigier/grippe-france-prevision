"""Variables explicatives pour les modèles « machine learning » et le LSTM.

Convention : une ligne indexée à la semaine ``t`` ne contient que de l'information
connue **à la fin de la semaine ``t``** (le niveau courant, les semaines passées,
le calendrier). La cible est la valeur à ``t + horizon``. Aucune fuite du futur.

Tout est vectorisé : le backtest appelle ``construire`` des centaines de fois.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 52)


def _calendrier(index: pd.DatetimeIndex) -> pd.DataFrame:
    sem = index.isocalendar()["week"].to_numpy()
    return pd.DataFrame({
        "sem_sin": np.sin(2 * np.pi * sem / 52.0),
        "sem_cos": np.cos(2 * np.pi * sem / 52.0),
        "saison_epi": ((sem >= 40) | (sem <= 20)).astype(int),
    }, index=index)


def construire(s: pd.Series, *, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Features connues à la fin de la semaine ``t`` (index = ``t``)."""
    d = s.diff()
    X = pd.DataFrame(index=s.index)
    X["niveau"] = s                              # s[t]
    for k in lags:
        X[f"lag_{k}"] = s.shift(k)               # s[t-k]
    for w in (3, 8):
        X[f"moy_{w}"] = s.rolling(w).mean()      # moyenne s[t-w+1..t]
        X[f"tendance_{w}"] = d.rolling(w).mean()  # variation moyenne récente
    X["diff_1"] = d                              # s[t] - s[t-1]
    X["accel"] = d.diff()
    return X.join(_calendrier(s.index))


def supervise(s: pd.Series, horizon: int, *, lags: tuple[int, ...] = LAGS
              ) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) pour prévoir ``s`` à ``t + horizon`` — lignes complètes seulement."""
    X = construire(s, lags=lags)
    y = s.shift(-horizon).rename("cible")
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]
