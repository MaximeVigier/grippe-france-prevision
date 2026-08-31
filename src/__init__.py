"""Modules réutilisables du projet Prévision grippe / réseau Sentinelles.

Ordre logique d'utilisation :
    config        -> URLs API, codes indicateurs, chemins
    data_loader   -> récupération + cache des séries Sentinelles (et Météo-France)
    preprocessing -> nettoyage, calendrier hebdomadaire régulier, gestion des révisions
    features      -> variables de décalage (lags), fenêtres glissantes, encodage saisonnier
    models        -> SARIMA, lissage exponentiel, modèles ML, LSTM
    backtest      -> backtest à origine glissante, métriques (MAE, RMSE, MASE, erreur de pic)
    plotting      -> décomposition, profils saisonniers, courbes de prévision
"""
