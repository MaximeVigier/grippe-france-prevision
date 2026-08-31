# Synthèse — Prévision des syndromes grippaux en France (réseau Sentinelles)

## Question

Prévoir l'incidence hebdomadaire nationale à 1-4 semaines, en conditions réalistes
(aucune information future), et déterminer quel modèle vaut le coup.

## Données

Réseau Sentinelles, indicateur « syndromes grippaux », série nationale (`geo=PAY`).
2 182 semaines d'octobre 1984 à août 2026 ; modélisation à partir de 1990. Taux pour
100 000 habitants (comparable dans le temps). Série complète, sans trou de calendrier.

Faits marquants : saisonnalité annuelle forte (pic ISO ~semaine 6, creux ~semaine 32),
amplitude des pics variable d'un facteur 3 à 5, et **saison 2020-2021 quasi plate** (effet
des mesures COVID : pic à 59/100 000 contre 314 l'année précédente).

## Méthode

- Nettoyage et calendrier hebdomadaire strict (`src/preprocessing.py`).
- Modèles, interface commune `fit` / `update` / `forecast` (`src/models.py`) :
  naïf saisonnier (référence), Holt-Winters, SARIMA (fenêtre glissante de 6 saisons),
  Fourier + ARIMA (régression harmonique), Ridge et gradient boosting sur variables
  décalées (prévision récursive), LSTM (Keras 3 / backend PyTorch).
- **Backtest à origine glissante** 2016-2024 (`src/backtest.py`) : à chaque semaine, le
  modèle ne voit que le passé, prévoit 1-4 semaines, ré-estimation annuelle.
- Métrique : MASE (MAE rapportée à celle du naïf saisonnier ; < 1 = mieux).

## Résultats

**MASE par horizon (backtest 2016-2024)**

| Horizon | Naïf | SARIMA | Fourier + ARIMA | Gradient boosting |
|---------|:----:|:------:|:---------------:|:-----------------:|
| +1 sem. | 1,00 | 0,305 | **0,273** | 0,278 |
| +2 sem. | 1,00 | 0,571 | **0,533** | 0,549 |
| +3 sem. | 1,00 | 0,764 | **0,749** | 0,761 |
| +4 sem. | 1,00 | 0,893 | 0,902 | 0,960 |

- **Fourier + ARIMA** est le meilleur à 1-3 semaines, pour une fraction du temps de calcul
  de SARIMA. SARIMA est juste derrière.
- Le **gradient boosting** est compétitif à 1-2 semaines puis se dégrade le plus vite
  (récursion).
- **Holt-Winters** est instable sur cette série (tendance qui s'emballe selon la fenêtre) —
  écarté.
- Le **LSTM** (notebook 05) fait moins bien que SARIMA (MAE ~19 vs ~11 à 1 semaine) pour un
  coût d'entraînement très supérieur : trop peu de données sur une seule série univariée.

**Erreur sur le pic épidémique** (SARIMA, horizon 1 semaine) : ±0-1 semaine sur la date et
±10-25 % sur l'amplitude pour les saisons normales. La saison 2020-2021 est ininterprétable
(pas de vrai pic).

**Saison 2020-2021** : en septembre 2020, tout modèle raisonnable prévoyait une saison
normale. Aucun signal dans l'historique ne pouvait annoncer l'effet des mesures COVID.

## Recommandation

- **Modèle** : Fourier + ARIMA (ou SARIMA), ré-estimé une fois par an, prévision à
  1-2 semaines. MASE ~0,3-0,5, pic calé à ±1-2 semaines.
- **Horizon utile** : 1 à 3 semaines. Au-delà, la dernière saison est une prévision aussi
  bonne.
- **Angle mort assumé** : les ruptures de régime (pandémie, changement de comportement)
  ne sont pas prévisibles à partir de la série ; un système opérationnel garde l'humain
  dans la boucle.

## Prolongements

Régresseurs exogènes (météo Météo-France SYNOP, vacances scolaires, IRA) ; modèle
hiérarchique par région (13 séries en parallèle) ; intervalles de prévision calibrés par
backtest ; petit tableau de bord Streamlit.

---

*Figures : `results/figures/`. Tables : `results/tables/`. Analyse : `notebooks/01` → `06`.*
