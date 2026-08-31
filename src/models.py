"""Modèles de prévision, interface commune :

    m.fit(y)            # entraînement complet sur la série y (index hebdomadaire)
    m.update(y)         # ré-ancrage rapide sur un historique allongé (sans ré-optimiser)
    m.forecast(steps)   # -> Series indexée par les dates futures

Le couple ``fit`` / ``update`` sert au backtest : on ré-optimise de temps en
temps (coûteux) et on se contente d'``update`` entre deux (quasi gratuit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WEEK = pd.Timedelta("7D")
_PERIOD = 52


def _future_index(last, steps):
    return pd.date_range(last + _WEEK, periods=steps, freq="7D")


# --------------------------------------------------------------------------- #
class SeasonalNaive:
    """Valeur observée 52 semaines plus tôt. La référence à battre (MASE)."""

    def fit(self, y):
        self.y_ = y.astype(float)
        return self

    update = fit

    def forecast(self, steps):
        # step h -> valeur à la même position dans la saison, 52 semaines plus tôt
        # (avec bouclage si l'horizon dépasse un an)
        vals = [self.y_.iloc[-_PERIOD + ((h - 1) % _PERIOD)] for h in range(1, steps + 1)]
        return pd.Series(vals, index=_future_index(self.y_.index[-1], steps), name="prev")


# --------------------------------------------------------------------------- #
class HoltWinters:
    """Lissage exponentiel, tendance amortie + saison additive (52)."""

    def fit(self, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.res_ = ExponentialSmoothing(
            y.astype(float), trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=_PERIOD,
            initialization_method="heuristic",
        ).fit()
        self._last = y.index[-1]
        return self

    def update(self, y):
        # ETS de statsmodels : pas d'append fiable ici, on ré-ajuste (rapide, ~0,5 s)
        return self.fit(y)

    def forecast(self, steps):
        f = np.clip(self.res_.forecast(steps), 0, None)
        return pd.Series(np.asarray(f), index=_future_index(self._last, steps), name="prev")


# --------------------------------------------------------------------------- #
class Sarima:
    """SARIMAX sur fenêtre glissante (par défaut 6 saisons). ``update`` prolonge
    le filtre sans ré-optimiser (``append(refit=False)``)."""

    def __init__(self, order=(2, 0, 1), seasonal_order=(1, 1, 0, _PERIOD), window=312):
        self.order, self.seasonal_order, self.window = order, seasonal_order, window

    def fit(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = y.astype(float).iloc[-self.window:]
        self.res_ = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False
                            ).fit(disp=False)
        self._last = y.index[-1]
        return self

    def update(self, y):
        y = y.astype(float)
        nouveaux = y.loc[y.index > self._last]
        if len(nouveaux):
            self.res_ = self.res_.append(nouveaux, refit=False)
            self._last = y.index[-1]
        return self

    def forecast(self, steps):
        f = np.clip(self.res_.forecast(steps), 0, None)
        return pd.Series(np.asarray(f), index=_future_index(self._last, steps), name="prev")


# --------------------------------------------------------------------------- #
class FourierArima:
    """Erreurs ARIMA + termes de Fourier pour le cycle annuel (régression
    harmonique). Beaucoup plus rapide que SARIMA, souvent aussi bon."""

    def __init__(self, order=(2, 1, 2), K=4, window=800):
        self.order, self.K, self.window = order, K, window

    def _fourier(self, index):
        t = (index - self._origin).days.to_numpy() / 7.0
        cols = [f(2 * np.pi * k * t / _PERIOD)
                for k in range(1, self.K + 1) for f in (np.sin, np.cos)]
        return np.column_stack(cols)

    def fit(self, y):
        from statsmodels.tsa.arima.model import ARIMA
        y = y.astype(float).iloc[-self.window:]
        self._origin = y.index[0]
        self.res_ = ARIMA(y, order=self.order, exog=self._fourier(y.index)).fit()
        self._last = y.index[-1]
        return self

    def update(self, y):
        y = y.astype(float)
        nouveaux = y.loc[y.index > self._last]
        if len(nouveaux):
            self.res_ = self.res_.append(nouveaux, exog=self._fourier(nouveaux.index))
            self._last = y.index[-1]
        return self

    def forecast(self, steps):
        idx = _future_index(self._last, steps)
        f = np.clip(self.res_.forecast(steps, exog=self._fourier(idx)), 0, None)
        return pd.Series(np.asarray(f), index=idx, name="prev")


# --------------------------------------------------------------------------- #
class MLForecaster:
    """Régresseur scikit-learn entraîné à prévoir t+1, appliqué en cascade."""

    def __init__(self, estimator, lags=None):
        from src.features import LAGS
        self.estimator, self.lags = estimator, lags or LAGS

    def fit(self, y):
        from src.features import supervise
        X, target = supervise(y.astype(float), horizon=1, lags=self.lags)
        self.estimator.fit(X, target)
        self.y_ = y.astype(float).copy()
        return self

    def update(self, y):
        self.y_ = y.astype(float).copy()  # features recalculées, modèle inchangé
        return self

    def forecast(self, steps):
        from src.features import construire
        hist = self.y_.copy()
        out = {}
        for _ in range(steps):
            x = construire(hist, lags=self.lags).iloc[[-1]]
            x = x.fillna(hist.iloc[-1])
            yhat = float(np.clip(self.estimator.predict(x)[0], 0, None))
            nxt = hist.index[-1] + _WEEK
            hist.loc[nxt] = yhat
            out[nxt] = yhat
        return pd.Series(out, name="prev")


def gradient_boosting(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return MLForecaster(HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=3, max_iter=400, l2_regularization=1.0,
        random_state=0, **kw))


def ridge():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return MLForecaster(make_pipeline(StandardScaler(), RidgeCV()))


# --------------------------------------------------------------------------- #
class LSTMForecaster:
    """Petit LSTM univarié sur fenêtre glissante. Simple à dessein : la question
    est de savoir s'il apporte quelque chose face à SARIMA."""

    def __init__(self, fenetre=52, units=32, epochs=60):
        self.fenetre, self.units, self.epochs = fenetre, units, epochs

    def fit(self, y):
        import keras
        v = y.astype(float).to_numpy()
        self.mu_, self.sd_ = v.mean(), v.std() + 1e-8
        vn = (v - self.mu_) / self.sd_
        X = np.stack([vn[i:i + self.fenetre] for i in range(len(vn) - self.fenetre)])[..., None]
        t = vn[self.fenetre:]
        self.model_ = keras.Sequential([
            keras.layers.Input((self.fenetre, 1)),
            keras.layers.LSTM(self.units),
            keras.layers.Dense(1),
        ])
        self.model_.compile(optimizer="adam", loss="mse")
        self.model_.fit(X, t, epochs=self.epochs, batch_size=32, verbose=0,
                        validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(patience=8,
                                                                 restore_best_weights=True)])
        self.y_ = y.astype(float).copy()
        return self

    def update(self, y):
        self.y_ = y.astype(float).copy()
        return self

    def forecast(self, steps):
        hist = ((self.y_.to_numpy() - self.mu_) / self.sd_).tolist()
        out = []
        for _ in range(steps):
            w = np.array(hist[-self.fenetre:])[None, :, None]
            yn = float(self.model_.predict(w, verbose=0)[0, 0])
            hist.append(yn)
            out.append(max(0.0, yn * self.sd_ + self.mu_))
        return pd.Series(out, index=_future_index(self.y_.index[-1], steps), name="prev")
