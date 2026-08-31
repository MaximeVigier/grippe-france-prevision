"""Figures : exploration de la série et résultats de prévision.

Titres en français, palette sobre, PNG dans ``results/figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
BLEU, ROUGE, GRIS = "#2f6690", "#c1436d", "#8d99ae"

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except ImportError:
    plt.rcParams["axes.grid"] = True


def save(fig, name: str, *, dpi: int = 150) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    return p


def serie(s: pd.Series, saisons_surlignees: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(s.index, s.values, lw=0.8, color=BLEU)
    ax.set(ylabel=s.name or "incidence", xlabel="",
           title="Incidence hebdomadaire des syndromes grippaux — France")
    if saisons_surlignees:
        for sais in saisons_surlignees:
            a = int(sais.split("-")[0])
            ax.axvspan(pd.Timestamp(f"{a}-10-01"), pd.Timestamp(f"{a + 1}-05-31"),
                       color=ROUGE, alpha=0.12)
    fig.tight_layout()
    return fig


def decomposition(s: pd.Series, period: int = 52):
    from statsmodels.tsa.seasonal import STL
    res = STL(s.dropna(), period=period, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    for ax, (lab, comp) in zip(axes, [("observé", s), ("tendance", res.trend),
                                      ("saison", res.seasonal), ("résidu", res.resid)]):
        ax.plot(comp.index, comp.values, lw=0.8, color=BLEU)
        ax.set_ylabel(lab)
    axes[0].set_title("Décomposition STL")
    fig.tight_layout()
    return fig


def profil_saisonnier(table: pd.DataFrame):
    """`table` = sortie de preprocessing.table_saisons."""
    fig, ax = plt.subplots(figsize=(11, 5))
    saisons = [s for s in table["saison"].unique() if table[table.saison == s].shape[0] >= 25]
    for sais in saisons:
        sub = table[table["saison"] == sais].sort_values("sem_saison")
        recent = int(sais.split("-")[0]) >= 2018
        ax.plot(sub["sem_saison"], sub["valeur"], lw=1.6 if recent else 0.6,
                alpha=0.9 if recent else 0.35,
                color=ROUGE if sais == "2020-2021" else (BLEU if recent else GRIS),
                label=sais if recent else None)
    ax.set(xlabel="semaines depuis la semaine 40", ylabel="taux / 100 000",
           title="Profil des saisons grippales (2020-2021 en rouge)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def acf_pacf(s: pd.Series, lags: int = 105):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(s.dropna(), lags=lags, ax=a1)
    plot_pacf(s.dropna(), lags=lags, ax=a2, method="ywm")
    a1.set_title("ACF"); a2.set_title("PACF")
    fig.tight_layout()
    return fig


def prevision(hist: pd.Series, prev: pd.Series, reel: pd.Series | None = None,
              titre: str = "", fenetre_semaines: int = 80):
    fig, ax = plt.subplots(figsize=(12, 4))
    h = hist.iloc[-fenetre_semaines:]
    ax.plot(h.index, h.values, color=GRIS, lw=1, label="historique")
    ax.plot(prev.index, prev.values, color=ROUGE, lw=2, marker="o", ms=3, label="prévision")
    if reel is not None:
        r = reel.reindex(prev.index)
        ax.plot(r.index, r.values, color=BLEU, lw=2, label="réalisé")
    ax.set_title(titre or "Prévision"); ax.legend()
    fig.tight_layout()
    return fig


def scores_backtest(tables: dict[str, pd.DataFrame], metrique: str = "MASE"):
    """`tables` = {nom_modèle: DataFrame de src.backtest.scores}."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    largeur = 0.8 / len(tables)
    horizons = next(iter(tables.values())).index
    x = np.arange(len(horizons))
    for i, (nom, t) in enumerate(tables.items()):
        ax.bar(x + i * largeur, t[metrique].values, largeur, label=nom)
    ax.axhline(1.0, color=GRIS, ls="--", lw=1)
    ax.set_xticks(x + 0.4 - largeur / 2, [f"+{h} sem." for h in horizons])
    ax.set(ylabel=metrique, title=f"{metrique} par horizon de prévision (backtest)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
