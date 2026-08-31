"""Tests des features et de l'interface des modèles (pas de fuite du futur)."""
import numpy as np
import pandas as pd
import pytest

from src import features, models


@pytest.fixture
def serie():
    idx = pd.date_range("2010-01-04", periods=400, freq="7D")
    saison = 20 + 15 * np.sin(2 * np.pi * np.arange(400) / 52)
    return pd.Series(saison + np.random.default_rng(0).normal(0, 1, 400), index=idx,
                     name="taux_100k")


def test_supervise_pas_de_fuite(serie):
    X, y = features.supervise(serie, horizon=1)
    # ligne t : niveau = s[t], lag_1 = s[t-1], cible = s[t+1]
    t = X.index[100]
    assert y.loc[t] == pytest.approx(serie.loc[t + pd.Timedelta("7D")])
    assert X.loc[t, "niveau"] == pytest.approx(serie.loc[t])
    assert X.loc[t, "lag_1"] == pytest.approx(serie.loc[t - pd.Timedelta("7D")])


def test_construire_est_causal(serie):
    """Changer une valeur future ne doit pas modifier les features passées."""
    X1 = features.construire(serie)
    s2 = serie.copy(); s2.iloc[-1] = 9999
    X2 = features.construire(s2)
    pd.testing.assert_frame_equal(X1.iloc[:-60], X2.iloc[:-60])


def test_seasonal_naive_boucle_au_dela_dun_an(serie):
    m = models.SeasonalNaive().fit(serie)
    f = m.forecast(60)                     # > 52 : doit boucler sans planter
    assert len(f) == 60 and f.notna().all()


def test_forecast_renvoie_dates_futures(serie):
    m = models.SeasonalNaive().fit(serie)
    f = m.forecast(4)
    assert (f.index > serie.index[-1]).all()
    assert (f.index.to_series().diff().dropna() == pd.Timedelta("7D")).all()
