# Training-thermal-states  
**Tomographie d’états quantiques par régression (réservoir computing)**

Ce projet permet de :  
- calculer les spectres \(S(\omega)\) d’une cavité non-linéaire couplée à un environnement squeezé,  
- extraire les moments \(M_0\ldots M_4\),  
- entraîner une régression (linéaire ou autre) pour prédire le paramètre de squeezing \(r\).  

---

## 🧩 Requirements  
Avant d’installer et de faire fonctionner le projet, assurez-vous d’avoir :  
- **Python ≥ 3.7**  
- Un **environnement virtuel** (recommandé)  
- Les **dépendances** listées dans `requirements.txt`  

Pour installer les dépendances :  
```bash
pip install -r requirements.txt
```

---

## ⚙️ Installation & utilisation  

### 1. Cloner le dépôt
```bash
git clone https://github.com/Mattegad/Training-thermal-states.git
cd Training-thermal-states
```

### 2. Créer et activer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Utiliser le Makefile  
Le projet comprend un **Makefile** qui automatise certaines tâches :

| Commande | Description |
|-----------|-------------|
| `make run` | Lance le script principal (exécution de l’apprentissage et tracés) |
| `make clean` | Supprime les fichiers temporaires et les sorties générées |
| `make test` | Lance les tests (si disponibles) |
| `make docs` | Génère la documentation (optionnel) |

Exemple :
```bash
make run
```

---

## 🧱 Organisation du projet  

```
training-thermal-states/
├─ README.md              # ce fichier explicatif
├─ requirements.txt       # dépendances Python
├─ Makefile               # commandes d’automatisation
├─ prédiction1.png        # exemple de résultat graphique
├─ src/                   # code source Python
│   ├─ __init__.py        
│   ├─ params.py          # paramètres physiques et de simulation
│   ├─ utils.py           # fonctions utilitaires
│   ├─ simulate.py        # construction de l’Hamiltonien et états stationnaires
│   ├─ spectrum.py        # calcul du spectre S(ω)
│   ├─ moments.py         # extraction des moments M0–M4
│   ├─ train.py           # orchestration de la simulation + apprentissage
│   └─ plot_utils.py      # fonctions de visualisation
```

**Résumé des rôles principaux :**
- `params.py` → paramètres de simulation (fréquences, couplages, dissipation…)  
- `simulate.py` → exécution de la simulation physique  
- `spectrum.py` → calcul des spectres  
- `moments.py` → extraction des moments statistiques  
- `train.py` → génération des données, apprentissage, tracés  
- `plot_utils.py` → création des figures  

---

## 🚀 Utilisation rapide  
1. Ajuster les paramètres dans `src/params.py` selon vos besoins.  
2. Lancer la simulation et l’apprentissage :
   ```bash
   make run
   ```
3. Les figures et résultats sont générés automatiquement (voir `prédiction1.png` comme exemple).  
4. Nettoyer le projet si nécessaire :
   ```bash
   make clean
   ```

---

## 🤝 Contribution  
Les contributions sont les bienvenues !  
- Ajouter des fonctionnalités (nouveaux modèles, formats de données, etc.)  
- Mettre à jour `requirements.txt` si de nouvelles dépendances sont ajoutées  
- Respecter la structure et le style du code existant  
- Documenter les ajouts et fournir des exemples si possible  

---

## 📘 Licence & contexte  
Ce projet illustre la **tomographie d’états quantiques** (états thermiques ou squeezés) via des approches de simulation et d’apprentissage supervisé.  
Vérifiez la **licence** du dépôt pour les conditions d’utilisation et de redistribution.  
