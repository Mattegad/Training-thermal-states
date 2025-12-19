# Quantum state tomography (spectroscopy) with reservoir computing  
**Training**

This project aims at :  
- compute the spectrum $\(S(\omega)\)$ (or other features of interest) of a non linear cavity coupled to different quantum states (for example squeezed states),  
- extract the moments $\(M_0\ldots M_4\)$,  
- train a regression (e.g. linear) to predict the parameters of the quantum states ($\ r\$ and $\theta$ in case of squeezed states).  

---

## 🧩 Requirements  
Before installing and running the project make sure to have :   
- **Python ≥ 3.7**  
- A **virtual environment** (recommanded)  
- The **dependencies** listed in `requirements.txt`  


---

## ⚙️ Installation & Use  

### 1. Clone the project
```bash
git clone https://github.com/Mattegad/Training-thermal-states.git
cd Training-thermal-states
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Use the Makefile  
The projects contains a **Makefile** which automates some tasks :

| Command | Description |
|-----------|-------------|
| `make setup` | Create and configure a virtual environment with the desired requirements |
| `make run` | Run the principal script (training and plots) |
| `make run SHOW=0` | Reload the saved data and replot the plots |
| `make clean` | Clean the temporary files and the saved data |
| `make push` | Commit and push to GitHub |
| `make pull` | Pull the changes from GitHub |


Example :
```bash
make run
```

---

## 🧱 Project organization 

```
training-thermal-states/
├─ README.md              # this file
├─ requirements.txt       # Python dependencies
├─ Makefile               # automation commands
├─ prédiction1.png        # example of prediction
├─ output                 # plots and computed data
└── src/
    ├── experiments/
    │   ├── opo_cavity_spectra.py
    │   ├── run_opo_cavity_spectra.py
    │   └── train.py
    │
    ├── models/
    │   ├── cavity_sde.py
    │   ├── dissipation.py
    │   ├── hamiltonian.py
    │   ├── OPO_sde.py
    │   └── steady_state.py
    │
    ├── observables/
    │   ├── moments.py
    │   ├── spectra_sde.py
    │   └── spectrum.py
    │
    ├── simulators/
    │   └── sde_solver.py
    │
    ├── params.py
    └── utils.py
```

**Principal functions :**
- `params.py` → simulation parameters (frequencies, coupling, dissipation…)  
- `simulate.py` → physical simulation of the open quantum system  
- `spectrum.py` → spectra computation  
- `moments.py` → moments computation  
- `train.py` → training and data generation  
- `plot_utils.py` → ploting parameters  

---

## 🚀 Quick use  
1. Ajust the parameters `src/params.py` based on your needs.
2. Create open quantum system (Hamiltonian + Lindblad operators) in `src/simulate.py` 
2. Run the simulation and the training :
   ```bash
   make run
   ```
3. Plots and data are automatically generated (see `prédiction1.png` as an example).  
4. Clean the project if necessary :
   ```bash
   make clean
   ```

---

## 🤝 Contribution  
Any contribution is welcome !  
- Add features (new models, new systems, etc.)  
- Upgrade `requirements.txt` if new dependencies are created 
- Respect the structure and the style of the existing project  
- Document the changes and provide examples if possible  

---

## 🙏 Credits
This project has been developed by Matteo Gadani at the Laboratoire Kastler Brossel.
We thank Wouter Verstraelen et al. for the first results. Their contributions were essential to the development of this code.

---

## 📘 Licence & context  
This project illustrates the **quantum states tomography** via simulations and reservoir computing. It complements an experiment done in the LKB where we aim at recognizing quantum states with an exciton polariton reservoir.
Verify the **licence** for use and distribution.  

---

## SDE scheme
Pour chaque G:
    Pour chaque trajectoire (10³):
        1) OPO part de zéro → évolution libre 20/γs → collecte 1000/γs toutes les 1/γs
        2) Pour chaque échantillon OPO:
            a) Cavité part du vide
            b) Relaxation 100/γc
            c) Stockage 200/γc toutes les 0.5/γc
        3) FFT → spectre S
    Moyenne S sur toutes les trajectoires → S_mean
    Moments spectraux M0..M4

