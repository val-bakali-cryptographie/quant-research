# RISQUE DE RUINE & RETRAITE — Guide d'installation et d'utilisation
## Moteur Monte Carlo VBA + Calibration Python

---

## ARCHITECTURE DU PROJET

```
projet_risque_ruine/
│
├── calibration_europeenne.py      ← Script Python (ÉTAPE 1)
├── params_calibres.xlsx           ← Généré par Python, lu par VBA
│
└── RisqueDeRuine.xlsm             ← Classeur Excel + VBA (ÉTAPE 2+)
    │
    ├── Module1_Constantes.bas     → Constantes globales et types
    ├── Module2_ChargementParams.bas → Lecture de params_calibres.xlsx
    ├── Module3_Aleatoire.bas      → LCG + Box-Muller + Bootstrap
    ├── Module4_MonteCarlo.bas     → Moteur de simulation (3 modèles)
    ├── Module5_CarteDeChалeur.bas → Heatmap probabilité ruine × taux × alloc
    ├── Module6_SequenceRisk.bas   → Analyse sequence of returns risk
    └── Module7_Dashboard.bas      → Interface utilisateur + boutons
```

---

## ÉTAPE 1 — CALIBRATION PYTHON (obligatoire, une seule fois)

### Installation des dépendances
```bash
pip install pandas numpy scipy yfinance openpyxl
```

### Lancement
```bash
python calibration_europeenne.py
```

### Ce que ça produit
Le fichier `params_calibres.xlsx` avec 3 feuilles :
- `GBM_Params` : μ et σ actions/obligations calibrés sur données européennes post-2022
- `Regimes_Params` : paramètres chaîne de Markov (expansion/récession)
- `Bootstrap_Data` : table des rendements annuels réels historiques

> **Sources** : iShares MSCI Europe ETF (EXW1.DE) + iShares € Corp Bond (IBCE.AS)
> Inflation : 5,5 % moyenne zone euro 2022-2024 (Eurostat)

---

## ÉTAPE 2 — INSTALLATION VBA

### Créer le classeur
1. Ouvrir Excel → Nouveau classeur vide
2. Sauvegarder en `.xlsm` (classeur avec macros) dans le **même dossier** que `params_calibres.xlsx`
3. Ouvrir l'éditeur VBA (`Alt + F11`)

### Importer les modules
Pour chaque fichier `.bas` :
- `Fichier` → `Importer un fichier`
- Sélectionner les 7 modules dans l'ordre

### Première initialisation
Dans la fenêtre Exécution VBA (`Ctrl + G`) :
```vba
Call CreerDashboard
```
Cela crée la feuille Dashboard avec tous les boutons et plages nommées.

---

## ÉTAPE 3 — UTILISATION QUOTIDIENNE

### Workflow normal
1. Ouvrir `RisqueDeRuine.xlsm`
2. Les paramètres se chargent automatiquement (`Workbook_Open`)
3. Saisir les paramètres dans le Dashboard (cellules jaunes)
4. Cliquer **▶ Lancer Simulation** → résultats en ~15-30 secondes
5. Cliquer **⬛ Carte de Chaleur** → ~2-5 minutes
6. Cliquer **〜 Sequence of Returns** → analyse immédiate

### Paramètres saisissables
| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| Capital initial | 500 000 € | Épargne au moment du départ en retraite |
| Taux de retrait | 4,0 % | Fraction du capital retiré chaque année |
| Allocation actions | 60 % | Part du portefeuille en actions |
| Horizon | 30 ans | Durée de retraite simulée |
| Modèle | 1 (GBM) | 1=GBM, 2=Bootstrap, 3=Régimes |
| Retrait indexé | Vrai | Si Vrai : retrait augmente avec l'inflation |

---

## LES 3 MODÈLES DE RENDEMENT

### 1. GBM Classique (Modèle = 1)
- Rendements log-normaux avec μ et σ calibrés sur données européennes post-2022
- Actions et obligations corrélés via décomposition de Cholesky
- **Usage** : scénario de base, point de comparaison

### 2. Bootstrap Historique (Modèle = 2)
- Rééchantillonnage par blocs de 3-5 ans des vraies données de marché
- Conserve les queues de distribution et les crises passées (2000, 2008)
- **Usage** : scénarios réalistes incluant les événements extrêmes

### 3. Modèle à Régimes (Modèle = 3)
- Chaîne de Markov 2 états : Expansion (μ_exp, σ_exp) et Récession (μ_rec, σ_rec)
- Probabilités de transition calibrées par algorithme EM (Baum-Welch)
- **Usage** : le plus réaliste — capture la persistance des crises

---

## FEUILLES GÉNÉRÉES

| Feuille | Contenu |
|---------|---------|
| `Dashboard` | Interface principale, inputs, résultats |
| `Resultats` | 10 000 lignes : ruine, capital final, drawdown par trajectoire |
| `Trajectoires` | 50 trajectoires complètes année par année (pour graphiques) |
| `CarteDeChалeur` | Grille probabilité de ruine 13×9 avec colorisation |
| `SequenceRisk` | Analyse comparée des scénarios de départ (2000, 2007, 2010, 2023) |

---

## PERFORMANCE

| Calcul | Temps approximatif |
|--------|-------------------|
| Charger paramètres | < 1 seconde |
| Simulation 10 000 trajectoires (GBM) | 15-25 secondes |
| Simulation 10 000 trajectoires (Bootstrap) | 20-35 secondes |
| Simulation 10 000 trajectoires (Régimes) | 25-40 secondes |
| Carte de chaleur (2 000 traj × 117 cellules) | 2-5 minutes |
| Analyse Sequence Risk | < 30 secondes |

---

## REPRODUCTIBILITÉ

- `SEED_ALEATOIRE = 42` dans Module1 → résultats identiques à chaque run
- Mettre `SEED_ALEATOIRE = 0` pour des résultats aléatoires à chaque fois

---

## INTERPRÉTATION DES RÉSULTATS

### Probabilité de ruine
- **< 5 %** : Acceptable. Correspond à une stratégie conservatrice
- **5-10 %** : Zone de vigilance. Revoir l'allocation ou baisser le taux de retrait
- **> 10 %** : Risque élevé. La règle des 4 % américaine ne s'applique pas au contexte européen à taux bas

### Règle des 4 % en Europe
La règle de Bengen (1994, étude Trinity) a été calibrée sur données **américaines**.
En Europe, les paramètres Dimson-Marsh-Staunton (Credit Suisse Yearbook) donnent
des rendements réels actions plus faibles (~4-5 % vs ~7 % US) et une inflation
post-2022 plus élevée. Le taux de retrait "sûr" en Europe est estimé à **3-3,5 %**.
