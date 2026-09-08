import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Prévision des syndromes grippaux en France", layout="wide")

# Titre et introduction
st.title("🤖 Prévision des syndromes grippaux en France")
st.markdown("""
**Réseau Sentinelles – indicateur « syndromes grippaux » (série nationale)**

Prévoir l'incidence hebdomadaire à 1-4 semaines, en conditions réalistes, 
et déterminer quel modèle vaut le coup. Basé sur les résultats du projet 
[grippe-france-prevision](https://github.com/MaximeVigier/grippe-france-prevision).
""")

# 1. Série temporelle complète
st.header("📅 Série temporelle complète (1990-2026)")
complete_fig_path = "results/figures/02_serie_complete.png"
if os.path.exists(complete_fig_path):
    st.image(complete_fig_path, caption="Série nationale des syndromes grippaux – 1990 à 2026", use_container_width=True)
else:
    st.warning("La figure de la série complète n'existe pas.")
    
# 2. Comparaison des modèles
st.header("📊 Performance des modèles : MASE par horizon")
mase_filepath = "results/tables/06_mase_par_horizon.csv"
if os.path.exists(mase_filepath):
    mase_data = pd.read_csv(mase_filepath, index_col=0)
    st.dataframe(mase_data)
else:
    mase_fig_path = "results/figures/06_mase_par_horizon.png"
    if os.path.exists(mase_fig_path):
        st.image(mase_fig_path, caption="MASE par horizon (backtest 2016-2024)", use_container_width=True)
    else:
        st.warning("Aucune donnée MASE ni figure disponible.")

# 3. Focus sur 2020-2021
st.header("🔍 Saison 2020-2021 : l'effet COVID")
covid_fig_path = "results/figures/06_focus_2020_2021.png"
if os.path.exists(covid_fig_path):
    st.image(covid_fig_path, caption="Saison 2020-2021 : la rupture de régime", use_container_width=True)
else:
    st.warning("La figure du focus COVID n'existe pas.")

# Conclusion
st.header("🏆 Résultats et recommandations")
st.markdown("""
- **Meilleur modèle** : Fourier + ARIMA (ou SARIMA), ré-estimé une fois par an.
- **Horizon utile** : 1 à 3 semaines. Au-delà, la prévision devient équivalente à une extrapolation simple.
- **Angle mort assumé** : les ruptures de régime (pandémie, changement de comportement) ne sont pas prévisibles à partir de la série.
""")

# Lien vers le repo
st.markdown("---")
st.markdown("[💡 Repository GitHub du projet](https://github.com/MaximeVigier/grippe-france-prevision)")