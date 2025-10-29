# Training-thermal-states  
**Quantum state tomography (spectroscopy) with reservoir computing**

This project aims at :  
- compute the spectrum $\(S(\omega)\)$ of a non linear cavity coupled to different quantum states (for example squeezed states),  
- extract the moments $\(M_0\ldots M_4\)$,  
- train a regression (e.g. linear) to predict the parameters of the quantum states ($\(r\)$ and $\theta$ in case of squeezed states).  

---

## 🧩 Requirements  
Avant d’installer et de faire fonctionner le projet, assurez-vous d’avoir :  
- **Python ≥ 3.7**  
- Un **environnement virtuel** (recommandé)  
- Les **dépendances** listées dans `requirements.txt`  


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
| `make setup` | Créer et configure un environnement virtuel avec les requirements |
| `make run` | Lance le script principal (exécution de l’apprentissage et tracés) |
| `make run SHOW=0` | Recharge les données sauvegardées et retrace les courbes |
| `make clean` | Supprime les fichiers temporaires et les sorties générées |
| `make push` | Commit et push les changements sur GitHub |
| `make pull` | Pull les changements effectués sur GitHub |


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
