"""Backtest à origine glissante.

On avance une « date d'aujourd'hui » semaine après semaine sur la période de
test. À chaque origine, le modèle ne voit que le passé, puis prévoit 1 à 4
semaines devant. Réentraînement périodique (``refit_every``) pour garder un coût
raisonnable — entre deux réentraînements, les modèles d'état (SARIMA, Holt-Winters)
sont simplement rechargés sur l'historique courant.

Métriques : MAE, RMSE, et MASE (MAE rapportée à celle de la marche naïve
saisonnière sur la même période — sans unité, < 1 = mieux que le naïf).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_origin(make_model, y: pd.Series, *, debut_test: str,
                   horizons=(1, 2, 3, 4), pas: int = 1, refit_every: int = 13,
                   verbose: bool = False) -> pd.DataFrame:
    y = y.astype(float).dropna()
    dates = y.index
    i0 = dates.get_indexer([pd.Timestamp(debut_test)], method="bfill")[0]
    hmax = max(horizons)

    modele = None
    lignes = []
    for k, i in enumerate(range(i0, len(y) - hmax, pas)):
        hist = y.iloc[:i + 1]
        if modele is None or (k * pas) % refit_every == 0:
            modele = make_model().fit(hist)
        else:
            modele.update(hist)  # ré-ancrage rapide, sans ré-optimisation
        prev = modele.forecast(hmax)
        for h in horizons:
            d = dates[i + h]
            lignes.append({"origine": dates[i], "horizon": h, "date": d,
                           "reel": y.iloc[i + h], "prev": float(prev.iloc[h - 1])})
        if verbose and k % 20 == 0:
            print(f"  origine {dates[i].date()}  ({k})")
    return pd.DataFrame(lignes)


def _mase(res: pd.DataFrame, naive: pd.DataFrame) -> float:
    mae = (res["reel"] - res["prev"]).abs().mean()
    mae_naive = (naive["reel"] - naive["prev"]).abs().mean()
    return mae / mae_naive if mae_naive else np.nan


def scores(res: pd.DataFrame, naive: pd.DataFrame | None = None) -> pd.DataFrame:
    g = res.groupby("horizon")
    out = pd.DataFrame({
        "MAE": g.apply(lambda d: (d["reel"] - d["prev"]).abs().mean()),
        "RMSE": g.apply(lambda d: np.sqrt(((d["reel"] - d["prev"]) ** 2).mean())),
    })
    if naive is not None:
        out["MASE"] = [
            _mase(res[res.horizon == h], naive[naive.horizon == h]) for h in out.index
        ]
    return out.round(3)


def erreur_pic(res: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Pour chaque saison : écart entre le pic prévu et le pic réellement observé,
    en semaines (date) et en pourcentage (amplitude)."""
    d = res[res["horizon"] == horizon].copy()
    d["saison"] = d["date"].dt.isocalendar().apply(
        lambda r: f"{r['year'] - (0 if r['week'] >= 40 else 1)}", axis=1)
    out = []
    for sais, sub in d.groupby("saison"):
        if len(sub) < 20:
            continue
        i_reel, i_prev = sub["reel"].idxmax(), sub["prev"].idxmax()
        out.append({
            "saison": sais,
            "decalage_sem": int(round((sub.loc[i_prev, "date"] - sub.loc[i_reel, "date"]).days / 7)),
            "err_amplitude_%": round(100 * (sub.loc[i_prev, "prev"] - sub.loc[i_reel, "reel"])
                                     / sub.loc[i_reel, "reel"], 1),
        })
    return pd.DataFrame(out)
