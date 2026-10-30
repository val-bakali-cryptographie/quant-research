# Quantitative Research — Multi-Strategy Backtesting Framework

> **Framework public · Signaux propriétaires non inclus**  
> Moteur, métriques et validation sont publiés. Les signaux d'entrée restent privés.  
> *"The microscope is public. The sample stays private."*

---

## Stratégies

### 1. SPY/BIL — Trend Following H1 · `h1_spy_bil/`
Signal de tendance sur SPY (1993–2026) avec switch dynamique vers BIL (cash) en bear market.

| Période | CAGR | MaxDD | Calmar | Sharpe |
|---------|------|-------|--------|--------|
| IS 1993–2017 | +11.2% | −16.4% | 0.68 | 0.82 |
| OOS 2018–2026 | +9.8% | −22.5% | **0.44** | 0.71 |
| B&H SPY (ref.) | +10.6% | −56.4% | 0.19 | 0.58 |

- Signal : régression linéaire sur log-prix (fenêtre 252j) — zéro look-ahead
- 8 trades en 33 ans — robustesse par construction
- Testé sur 4 crises majeures : Dot-com 2000, GFC 2008, COVID 2020, Bear 2022
- Bootstrap 5 000 itérations · Walk-forward par décennie

---

### 2. XAU/USD 4h — Enveloppes σ-Dynamiques Géométriques · `xau_envelopes/`
Mean reversion sur XAUUSD 4h avec 4 niveaux d'enveloppe calibrés sur la volatilité réalisée et pondération géométrique [6.7%, 13.3%, 26.7%, 53.3%].

| Période | CAGR | MaxDD | Calmar | WR | N |
|---------|------|-------|--------|----|---|
| IS 2003–2014 | +6.5% | −5.5% | **1.19** | 83% | 3 354 |
| OOS 2015–2026 | +4.2% | −6.1% | **0.78** | 83% | 3 758 |

- Walk-forward **16/16** fenêtres OOS positives sur 23 ans
- Positif sur **tous les régimes** (bull 2003–11, bear 2012–15, range, COVID, hausse 2023–26)
- Poids géométriques : entrée progressive, exposition max sur rebond profond

---

### 3. HYPE/USDT 1h — Enveloppes Géométriques · `hype_geo/`
Mean reversion sur actif crypto haute volatilité. Mêmes principes moteur que XAU.

| Période | Calmar | MaxDD | WR | N |
|---------|--------|-------|----|---|
| IS déc 2024–sept 2025 | **68** | −5% | 90% | 3 673 |
| OOS oct 2025–août 2026 | **57** | −2% | 90% | 3 641 |

*Signal propriétaire masqué — résultats obtenus en local avec le vrai signal.*

---

### 4. XAU/USD 1h — Asian Range Mean Reversion · `xau_asian_range/`
Exploitation du range de la session asiatique (17h–22h UTC). Entrée limit −0.3% sous le Low, TP au High du range, SL 1.5%.

| Période | CAGR | MaxDD | Calmar | WR | N |
|---------|------|-------|--------|----|---|
| Discovery 2003–2014 | +87.8% | −3.2% | **27.7** | 80.6% | 1 177 |
| Validation 2015–2020 | +51.0% | — | **23.8** | — | 497 |
| Forward OOS 2021–2026 | +87.3% | — | **54.6** | 85% | 592 |
| Walk-forward | — | — | — | **8/8** OOS+ | — |

*Signal propriétaire masqué — résultats obtenus en local avec le vrai signal.*

---

## Architecture

```
quant-research/
├── h1_spy_bil/
│   ├── backtest.py        # moteur complet, signal linreg public
│   └── results/           # graphiques institutionnels
├── xau_envelopes/
│   ├── backtest.py        # moteur V2 σ-dynamiques
│   └── results/           # equity, drawdown, walk-forward, régimes
├── hype_geo/
│   ├── backtest.py        # moteur public, signal masqué
│   └── results/
├── xau_asian_range/
│   ├── backtest.py        # moteur public, signal masqué
│   └── results/
└── signals/               # ← non commité (.gitignore)
    └── *.py               # signaux propriétaires locaux uniquement
```

---

## Principes méthodologiques

**Zéro look-ahead** — `signal[i]` appliqué à `open[i+1]` systématiquement via `pos.shift(1)`

**Modèle de frais** — 5 bps/côté (H1 SPY) · 10 bps/côté (enveloppes, Asian Range)

**Validation IS/OOS stricte** — split temporel unique, jamais de refit sur OOS

**Walk-forward** — fenêtres glissantes indépendantes, résultats agrégés

**Bootstrap** — 5 000 permutations de retours journaliers (H1), IC90% sur Calmar

**Calmar comme métrique primaire** — CAGR% / |MaxDD%| — pénalise explicitement le risque réalisé

---

## Stack technique

Python 3.11 · pandas · numpy · matplotlib · pyarrow  
Données : Yahoo Finance (SPY/BIL) · flux propriétaire (XAU/HYPE)

---

## Auteur

**Valentin Bakali** — Étudiant ingénieur 5ème année, ESME Sudria Paris  
Spécialisation Data Science & IA / Ingénierie financière  
Trader algorithmique autodidacte depuis l'âge de 17 ans

[github.com/valentinbakali](https://github.com/valentinbakali)
