# Spectroscopy Reservoir — reproduction Figure 2 (ver. de travail)

Ce dépôt contient un code permettant de reproduire la Figure 2 du papier
"Spectroscopy on a single nonlinear mode recognizes quantum states"
(Verstraelen et al.). L'objectif: calculer les spectres S(ω) d'une cavité non-linéaire couplée
à un environnement squeezé, extraire les moments M₀..M₄ et entraîner une régression linéaire
pour prédire le paramètre de squeezing r.

## Installation

Recommandé: créer un environnement conda ou venv.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

training-thermal-states/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ __init__.py          # définit le package Python
│  ├─ params.py            # constantes et paramètres physiques
│  ├─ utils.py             # helpers numériques (FFT, moments, normalisation)
│  ├─ simulate.py          # construction Hamiltonien / collapse ops, steady-state
│  ├─ spectrum.py          # calcul du spectre S(ω) via corrélations
│  ├─ moments.py           # calcul des moments M0..M4
│  ├─ train.py             # génération du jeu de données + régression + tracés
│  └─ plot_utils.py        # fonctions de tracé (Figure 2)

