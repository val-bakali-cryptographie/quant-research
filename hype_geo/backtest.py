"""
HYPE/USDT — Mean Reversion Enveloppes Géométriques
====================================================
Framework public — signal propriétaire masqué.

Structure :
  signals/hype_signal.py  ← privé, non commité (.gitignore)
  Ce fichier             ← public (moteur + validation + graphiques)

Pour générer les graphiques avec le vrai signal :
  1. Remplir signals/hype_signal.py (voir template ci-dessous)
  2. python hype_geo_backtest.py

Résultats publiés (config propriétaire) :
  IS  2024-12 → 2025-09 : Cal=68  | MaxDD=-5%  | WR=90% | N=3673
  OOS 2025-10 → 2026-08 : Cal=57  | MaxDD=-2%  | WR=90% | N=3641
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output_hype")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paramètres publics ─────────────────────────────────────────────
CAPITAL  = 100_000
FEE_BPS  = 10          # 10 bps aller + 10 bps retour
IS_END   = "2025-09-30"
OOS_START= "2025-10-01"

# ── Couleurs ───────────────────────────────────────────────────────
BG    = "#0F1117"
TEXT  = "#DDDDDD"
GREEN = "#1D9E75"
RED   = "#D85A30"
GOLD  = "#E69F00"
BLUE  = "#4C9BE8"
WHITE = "#FFFFFF"

def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.12, color="#555")
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

# ══════════════════════════════════════════════════════════════════
#  SIGNAL — interface publique
# ══════════════════════════════════════════════════════════════════
def load_signal_module():
    """Charge le signal depuis signals/hype_signal.py (privé)."""
    sig_dir = os.path.join(HERE, "signals")
    sig_file = os.path.join(sig_dir, "hype_signal.py")
    if os.path.exists(sig_file):
        sys.path.insert(0, sig_dir)
        import hype_signal as _s
        return _s
    return None

def compute_entries_exits(df, sig_module=None):
    """
    Retourne deux Series booléennes : entry_mask, exit_mask.
    Si le module signal est disponible, utilise le vrai signal.
    Sinon, utilise un signal de démonstration (non optimisé).
    """
    if sig_module is not None:
        return sig_module.compute_entries_exits(df)

    # ── Signal de démonstration (placeholder) ──────────────────────
    # Remplacer par le vrai signal dans signals/hype_signal.py
    # Ce placeholder utilise une logique simplifiée non optimisée.
    close = df["close"]
    ma    = close.ewm(span=20, adjust=False).mean()
    std   = close.rolling(20).std()
    entry = close < (ma - 1.5 * std)   # entrée sous la bande
    exit_ = close > ma                  # sortie sur la MA
    return entry, exit_
    # ──────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════
#  MOTEUR V2 — enveloppes géométriques
# ══════════════════════════════════════════════════════════════════
def geo_weights(n):
    raw = np.array([2**k for k in range(n)], dtype=float)
    return raw / raw.sum()

def run_geo_engine(df, envs, weights, ma_series, fee_bps=FEE_BPS):
    """
    Moteur mean reversion enveloppes géométriques.
    - Entrée  : prix descend sous enveloppe k (calculé sur MA)
    - Sortie  : high >= MA (exit_all)
    - Equity  : calculée sur le LOW de chaque bougie (conservateur)
    - MA      : fournie externalement (signal privé)
    """
    fee   = fee_bps / 10_000
    o, h, l, c = (df["open"].values, df["high"].values,
                  df["low"].values,  df["close"].values)
    ma    = ma_series.values
    n     = len(df)
    nk    = len(envs)
    alloc = [CAPITAL * w for w in weights]

    cash      = float(CAPITAL)
    shares    = np.zeros(nk)
    ep        = np.zeros(nk)
    active    = np.zeros(nk, dtype=bool)
    entry_i   = np.zeros(nk, dtype=int)

    eq_arr    = np.zeros(n); eq_arr[0] = CAPITAL
    trades    = []

    for i in range(1, n):
        if np.isnan(ma[i-1]):
            eq_arr[i] = max(0, cash + sum(shares[k]*l[i] for k in range(nk)))
            continue
        mp = ma[i-1]

        # Entrées
        for k, env in enumerate(envs):
            if not active[k] and l[i] <= mp * (1 - env):
                cost = alloc[k]
                if cash >= cost * 0.999:
                    ep[k]     = min(mp*(1-env), o[i]) * (1+fee)
                    shares[k] = cost / ep[k]
                    cash     -= cost
                    active[k] = True
                    entry_i[k]= i

        # Sortie groupée
        if h[i] >= mp and any(active):
            xp = max(mp, o[i]) * (1 - fee)
            total_ret = 0.0; total_w = 0.0
            for k in range(nk):
                if active[k]:
                    ret = (xp - ep[k]) / ep[k] * 100
                    trades.append({
                        "entry": df.index[entry_i[k]],
                        "exit":  df.index[i],
                        "ret":   ret,
                        "dur":   i - entry_i[k],
                        "zone":  k+1,
                        "w":     weights[k],
                    })
                    total_ret += ret * weights[k]
                    total_w   += weights[k]
                    cash      += shares[k] * xp
                    shares[k]  = 0.0
                    active[k]  = False

        eq_arr[i] = max(0, cash + sum(shares[k]*l[i] for k in range(nk)))

    eq_arr[0] = CAPITAL
    eq = pd.Series(eq_arr, index=df.index)
    tr = pd.DataFrame(trades) if trades else pd.DataFrame()
    return eq, tr

# ══════════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ══════════════════════════════════════════════════════════════════
def metrics(eq, bpy=8760/1):
    """bpy = barres par an (8760 pour 1h)"""
    eq  = eq.dropna()
    r   = eq.pct_change().dropna()
    yr  = (eq.index[-1] - eq.index[0]).days / 365.25
    if yr <= 0: return {}
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yr) - 1
    sh   = r.mean() / r.std() * np.sqrt(8760) if r.std() > 0 else 0
    ds   = r[r<0].std()
    so   = r.mean() / ds * np.sqrt(8760) if ds > 0 else 0
    rm   = np.maximum.accumulate(eq.values)
    dd   = (eq.values - rm) / rm
    mdd  = dd.min() * 100
    cal  = (cagr*100) / abs(mdd) if mdd != 0 else 0
    return {"cagr": cagr*100, "sharpe": sh, "sortino": so,
            "mdd": mdd, "calmar": cal,
            "capital_end": float(eq.iloc[-1]),
            "eq": eq, "dd": dd, "r": r}

def trade_stats(tr):
    if tr is None or tr.empty: return {}
    wr   = (tr["ret"] > 0).mean() * 100
    avg  = tr["ret"].mean()
    med  = tr["ret"].median()
    best = tr["ret"].max()
    worst= tr["ret"].min()
    dur  = tr["dur"].mean()
    return {"wr": wr, "avg_ret": avg, "med_ret": med,
            "best": best, "worst": worst, "avg_dur": dur,
            "n": len(tr)}

# ══════════════════════════════════════════════════════════════════
#  GRAPHIQUES
# ══════════════════════════════════════════════════════════════════
def make_chart(df, eq, tr, m_is, m_oos, m_full, ts_is, ts_oos):
    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
                            height_ratios=[1.8, 1.2, 1.0])

    # ── 1. Equity log ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    bh = df["close"] / df["close"].iloc[0] * CAPITAL
    bh_n = bh / bh.iloc[0] * 100
    eq_n = eq / eq.iloc[0] * 100
    ax.plot(bh_n.index, bh_n, color=RED, lw=1.2, ls="--", alpha=0.6,
            label=f"B&H HYPE")
    ax.plot(eq_n.index, eq_n, color=GREEN, lw=2,
            label=f"Geo MR  Cal={m_full.get('calmar',0):.1f}")
    ax.axvline(pd.Timestamp(OOS_START), color=GOLD, lw=1.2, ls=":", alpha=0.8)
    ax.text(pd.Timestamp(OOS_START), eq_n.max()*0.5, " OOS→",
            fontsize=8, color=GOLD)
    ax.set_yscale("log")
    ax.set_title("HYPE/USDT — Equity (base 100, log)  |  Géo Mean Reversion",
                 fontsize=11, color=TEXT)
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor=TEXT)
    style(ax)

    # ── 2. Métriques tableau ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ts = trade_stats(tr)
    rows = [
        ["CAGR %",    f"{m_full.get('cagr',0):+.1f}"],
        ["MaxDD %",   f"{m_full.get('mdd',0):+.1f}"],
        ["Calmar",    f"{m_full.get('calmar',0):.1f}"],
        ["Sharpe",    f"{m_full.get('sharpe',0):.2f}"],
        ["Sortino",   f"{m_full.get('sortino',0):.2f}"],
        ["N trades",  f"{ts.get('n',0)}"],
        ["WR %",      f"{ts.get('wr',0):.1f}"],
        ["Avg ret %", f"{ts.get('avg_ret',0):+.3f}"],
        ["Avg dur h", f"{ts.get('avg_dur',0):.1f}"],
        ["IS Calmar", f"{m_is.get('calmar',0):.1f}"],
        ["OOS Calmar",f"{m_oos.get('calmar',0):.1f}"],
    ]
    table = ax.table(cellText=rows, colLabels=["Métrique", "Valeur"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1A1D27"); cell.set_text_props(color=TEXT)
    ax.set_facecolor(BG)
    ax.set_title("Résumé", fontsize=10, color=TEXT, pad=4)

    # ── 3. Drawdown ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    bh_rm = np.maximum.accumulate(bh.values)
    bh_dd = (bh.values - bh_rm) / bh_rm * 100
    ax.fill_between(bh.index, bh_dd, 0, alpha=0.2, color=RED, label="B&H HYPE")
    ax.fill_between(eq.index, m_full["dd"]*100, 0, alpha=0.55, color=GREEN,
                    label=f"Geo MR  MaxDD={m_full['mdd']:.1f}%")
    ax.axvline(pd.Timestamp(OOS_START), color=GOLD, lw=1, ls=":", alpha=0.7)
    ax.set_title("Drawdown comparé", fontsize=10, color=TEXT)
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor=TEXT)
    ax.set_ylabel("DD (%)", color=TEXT)
    style(ax)

    # ── 4. IS vs OOS Calmar ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    labels = ["IS", "OOS"]
    vals   = [m_is.get("calmar",0), m_oos.get("calmar",0)]
    colors = [GREEN if v > 0 else RED for v in vals]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val + 0.5,
                f"{val:.1f}", ha="center", fontsize=11,
                fontweight="bold", color=TEXT)
    ax.set_title("Calmar IS vs OOS", fontsize=10, color=TEXT)
    ax.set_ylabel("Calmar", color=TEXT)
    style(ax)

    # ── 5. Distribution des retours par trade ──────────────────────
    ax = fig.add_subplot(gs[2, 0])
    if not tr.empty:
        ax.hist(tr["ret"], bins=60, color=GREEN, alpha=0.75,
                edgecolor="#0F1117", lw=0.3)
        ax.axvline(0, color=WHITE, lw=0.8, ls="--")
        ax.axvline(tr["ret"].mean(), color=GOLD, lw=1.5, ls="--",
                   label=f"Moy={tr['ret'].mean():+.3f}%")
        wr = (tr["ret"]>0).mean()*100
        ax.set_title(f"Distribution retours/trade\nWR={wr:.1f}%  N={len(tr)}",
                     fontsize=9, color=TEXT)
        ax.legend(fontsize=8, facecolor="#1A1D27", labelcolor=TEXT)
        ax.set_xlabel("Retour (%)", color=TEXT)
    style(ax)

    # ── 6. Durée des trades ────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    if not tr.empty:
        ax.hist(tr["dur"], bins=40, color=BLUE, alpha=0.75,
                edgecolor="#0F1117", lw=0.3)
        ax.axvline(tr["dur"].mean(), color=GOLD, lw=1.5, ls="--",
                   label=f"Moy={tr['dur'].mean():.1f}h")
        ax.set_title("Durée des trades (heures)", fontsize=9, color=TEXT)
        ax.legend(fontsize=8, facecolor="#1A1D27", labelcolor=TEXT)
        ax.set_xlabel("Durée (barres 1h)", color=TEXT)
    style(ax)

    # ── 7. Retours cumulés par zone ───────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    if not tr.empty and "zone" in tr.columns:
        zone_stats = tr.groupby("zone")["ret"].agg(["mean","count","sum"])
        zones = zone_stats.index.tolist()
        means = zone_stats["mean"].tolist()
        zcols = [GREEN, GOLD, BLUE, RED]
        ax.bar(zones, means, color=[zcols[min(z-1,3)] for z in zones], alpha=0.85)
        ax.axhline(0, color=WHITE, lw=0.5)
        ax.set_title("Retour moyen par zone\n(zone 4 = poids le plus lourd)",
                     fontsize=9, color=TEXT)
        ax.set_xlabel("Zone d'enveloppe", color=TEXT)
        ax.set_ylabel("Retour moyen (%)", color=TEXT)
    style(ax)

    fig.patch.set_facecolor(BG)
    plt.suptitle(
        "HYPE/USDT — Mean Reversion Enveloppes Géométriques\n"
        "Signal propriétaire | Framework public | 10 bps frais",
        fontsize=13, color=TEXT, y=1.01)

    p = os.path.join(OUT_DIR, "hype_geo_results.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {p}")
    return p

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*60)
    print("  HYPE/USDT — Geo Mean Reversion Backtest")
    print("="*60)

    # Chargement
    data_path = os.path.join(HERE, "data", "HYPEUSDT_1h.parquet")
    if not os.path.exists(data_path):
        # Cherche dans le répertoire courant
        for p in ["HYPEUSDT_1h.parquet",
                  os.path.join("..", "data", "HYPEUSDT_1h.parquet")]:
            if os.path.exists(p): data_path = p; break
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    print(f"  Données : {len(df)} barres  {df.index[0].date()} → {df.index[-1].date()}")

    # Signal
    sig_mod = load_signal_module()
    if sig_mod is None:
        print("  ⚠ signals/hype_signal.py non trouvé — utilisation du placeholder")
    else:
        print("  ✓ Signal propriétaire chargé")

    # MA depuis le signal ou placeholder
    if sig_mod is not None and hasattr(sig_mod, "compute_ma"):
        ma_series = sig_mod.compute_ma(df)
        envs      = sig_mod.ENVS
        weights   = sig_mod.WEIGHTS
    else:
        # Placeholder : DEMA générique
        def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
        def _dema(s, n): e = _ema(s, n); return 2*e - _ema(e, n)
        ma_series = _dema(df["close"], 20)
        sig_std   = df["close"].pct_change().std()
        envs      = [sig_std * k for k in [1, 2, 4, 7]]
        weights   = geo_weights(4)

    # Backtest complet
    eq, tr = run_geo_engine(df, envs, weights, ma_series)
    m_full = metrics(eq)

    # IS / OOS
    df_is  = df[df.index <= IS_END]
    df_oos = df[df.index >= OOS_START]
    ma_is  = ma_series[ma_series.index <= IS_END]
    ma_oos = ma_series[ma_series.index >= OOS_START]

    eq_is,  tr_is  = run_geo_engine(df_is,  envs, weights, ma_is)
    eq_oos, tr_oos = run_geo_engine(df_oos, envs, weights, ma_oos)
    m_is  = metrics(eq_is)
    m_oos = metrics(eq_oos)

    ts_is  = trade_stats(tr_is)
    ts_oos = trade_stats(tr_oos)

    # Affichage
    print(f"\n  {'':25} {'IS':>10} {'OOS':>10} {'FULL':>10}")
    print("  " + "─"*55)
    for lbl, ki, ko, kf in [
        ("CAGR %",   "cagr",   "cagr",   "cagr"),
        ("MaxDD %",  "mdd",    "mdd",    "mdd"),
        ("Calmar",   "calmar", "calmar", "calmar"),
        ("Sharpe",   "sharpe", "sharpe", "sharpe"),
    ]:
        print(f"  {lbl:<25} {m_is.get(ki,0):>+9.2f} {m_oos.get(ko,0):>+9.2f} {m_full.get(kf,0):>+9.2f}")
    print(f"  {'N trades':<25} {ts_is.get('n',0):>10} {ts_oos.get('n',0):>10} {trade_stats(tr).get('n',0):>10}")
    print(f"  {'WR %':<25} {ts_is.get('wr',0):>9.1f} {ts_oos.get('wr',0):>9.1f}")

    # Graphique
    print("\n  Génération graphique...")
    make_chart(df, eq, tr, m_is, m_oos, m_full,
               df_is.index[-1], df_oos.index[0])

    print(f"\n  DONE — {time.time()-t0:.1f}s")
    print(f"  Résultats dans : {OUT_DIR}")

if __name__ == "__main__":
    main()
