"""
XAU/USD 4h — Enveloppes σ-Dynamiques Géométriques
====================================================
Framework public — signal propriétaire masqué.

Moteur (publié) :
  - DEMA(L=7) comme MA de référence
  - k_sigmas = [1, 2, 4, 7]  → enveloppes [k×σ/barre]
  - σ mesuré sur les retours 4h
  - Poids géométriques : [6.7%, 13.3%, 26.7%, 53.3%]
  - Entrée sur LOW ≤ MA×(1-env_k)
  - Sortie quand HIGH ≥ MA (exit_all)
  - Frais : 10 bps aller + retour

Résultats publiés (L=7, k=[1,2,4,7]) :
  IS  2003-2014 : Calmar=+1.19  MaxDD=-5.48%  WR=83%  N=3354  CAGR=+6.5%
  OOS 2015-2026 : Calmar=+0.78  MaxDD=-6.12%  WR=83%
  Walk-forward  : 4/5 fenêtres OOS > IS (F3 rétention=487%)
  Régimes       : positif sur TOUS les régimes sur 23 ans
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
OUT_DIR = os.path.join(HERE, "output_xau_envelopes")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paramètres publics ─────────────────────────────────────────────
CAPITAL   = 100_000
FEE_BPS   = 10
IS_END    = "2014-12-31"
OOS_START = "2015-01-01"

# Paramètres publiés du signal
DEMA_L    = 7
K_SIGMAS  = [1, 2, 4, 7]

# ── Couleurs ───────────────────────────────────────────────────────
BG    = "#0F1117"; TEXT  = "#DDDDDD"
GREEN = "#1D9E75"; RED   = "#D85A30"
GOLD  = "#E69F00"; BLUE  = "#4C9BE8"
WHITE = "#FFFFFF"; PURPLE= "#9B59B6"
COLORS= [GREEN, GOLD, BLUE, PURPLE]

def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.12, color="#555")
    ax.title.set_color(TEXT); ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

# ══════════════════════════════════════════════════════════════════
#  UTILITAIRES SIGNAL
# ══════════════════════════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def dema(s, n): e = ema(s, n); return 2*e - ema(e, n)

def geo_weights(n):
    raw = np.array([2**k for k in range(n)], dtype=float)
    return raw / raw.sum()

def load_signal_module():
    sig_dir  = os.path.join(HERE, "signals")
    sig_file = os.path.join(sig_dir, "xau_envelopes_signal.py")
    if os.path.exists(sig_file):
        sys.path.insert(0, sig_dir)
        import xau_envelopes_signal as _s
        return _s
    return None

# ══════════════════════════════════════════════════════════════════
#  MOTEUR V2 — Enveloppes σ-Dynamiques
# ══════════════════════════════════════════════════════════════════
def run_sigma_engine(df, ma_series, envs, weights, fee_bps=FEE_BPS):
    """
    Moteur enveloppes géométriques σ-dynamiques.
    - Equity calculée sur le LOW (conservateur)
    - MA utilise la valeur du bar précédent (zéro look-ahead)
    - Sortie groupée dès HIGH ≥ MA
    """
    fee  = fee_bps / 10_000
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    ma   = ma_series.values
    n    = len(df)
    nk   = len(envs)
    alloc= [CAPITAL * w for w in weights]

    cash   = float(CAPITAL)
    shares = np.zeros(nk)
    ep     = np.zeros(nk)
    active = np.zeros(nk, dtype=bool)
    entry_i= np.zeros(nk, dtype=int)

    eq_arr = np.zeros(n); eq_arr[0] = CAPITAL
    trades = []

    for i in range(1, n):
        if np.isnan(ma[i-1]):
            eq_arr[i] = max(0, cash + sum(shares[k]*l[i] for k in range(nk)))
            continue
        mp = ma[i-1]

        # Entrées
        for k, env in enumerate(envs):
            if not active[k] and l[i] <= mp*(1-env):
                cost = alloc[k]
                if cash >= cost * 0.99:
                    ep[k]     = min(mp*(1-env), o[i]) * (1+fee)
                    shares[k] = cost / ep[k]
                    cash     -= cost
                    active[k] = True
                    entry_i[k]= i

        # Sortie groupée
        if h[i] >= mp and any(active):
            xp = max(mp, o[i]) * (1-fee)
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
                    cash     += shares[k] * xp
                    shares[k] = 0.0
                    active[k] = False

        eq_arr[i] = max(0, cash + sum(shares[k]*l[i] for k in range(nk)))

    eq_arr[0] = CAPITAL
    eq = pd.Series(eq_arr, index=df.index)
    tr = pd.DataFrame(trades) if trades else pd.DataFrame()
    return eq, tr

# ══════════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ══════════════════════════════════════════════════════════════════
def metrics(eq, bpy=6*365):  # ~6 barres 4h par jour
    eq = eq.dropna(); r = eq.pct_change().dropna()
    yr = (eq.index[-1] - eq.index[0]).days / 365.25
    if yr <= 0: return {}
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yr) - 1
    sh   = r.mean()/r.std()*np.sqrt(bpy) if r.std()>0 else 0
    ds   = r[r<0].std()
    so   = r.mean()/ds*np.sqrt(bpy) if ds>0 else 0
    rm   = np.maximum.accumulate(eq.values)
    dd   = (eq.values - rm)/rm
    mdd  = dd.min()*100
    cal  = (cagr*100)/abs(mdd) if mdd != 0 else 0
    return {"cagr": cagr*100, "sharpe": sh, "sortino": so,
            "mdd": mdd, "calmar": cal, "eq": eq, "dd": dd, "r": r}

def trade_stats(tr):
    if tr is None or tr.empty: return {}
    wr = (tr["ret"]>0).mean()*100
    return {"wr": wr, "avg": tr["ret"].mean(), "n": len(tr),
            "avg_dur": tr["dur"].mean()}

# ══════════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ══════════════════════════════════════════════════════════════════
def walk_forward(df, ma_series, envs, weights, years_is=6, years_oos=2):
    start = df.index[0].year; end = df.index[-1].year
    results = []
    for is_s in range(start, end - years_is - years_oos + 1):
        is_e  = is_s + years_is - 1
        oos_s = is_s + years_is; oos_e = oos_s + years_oos - 1
        if oos_e > end: break
        dfa = df[(df.index.year >= is_s)  & (df.index.year <= is_e)]
        dfb = df[(df.index.year >= oos_s) & (df.index.year <= oos_e)]
        ma_a = ma_series[(ma_series.index.year >= is_s) & (ma_series.index.year <= is_e)]
        ma_b = ma_series[(ma_series.index.year >= oos_s) & (ma_series.index.year <= oos_e)]
        if dfa.empty or dfb.empty: continue
        eq_a, tr_a = run_sigma_engine(dfa, ma_a, envs, weights)
        eq_b, tr_b = run_sigma_engine(dfb, ma_b, envs, weights)
        m_a = metrics(eq_a); m_b = metrics(eq_b)
        results.append({
            "window":     f"{oos_s}-{oos_e}",
            "cal_is":     m_a.get("calmar",0),
            "cal_oos":    m_b.get("calmar",0),
            "oos_pos":    m_b.get("calmar",0) > 0,
        })
    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════
#  ANALYSE PAR RÉGIME DE MARCHÉ
# ══════════════════════════════════════════════════════════════════
REGIMES = [
    ("Bull XAU 2003-11", "2003-01-01", "2011-12-31"),
    ("Bear XAU 2012-15", "2012-01-01", "2015-12-31"),
    ("Range 2016-18",    "2016-01-01", "2018-12-31"),
    ("COVID+Rally 2019-22","2019-01-01","2022-12-31"),
    ("Hausse 2023-26",   "2023-01-01", "2026-12-31"),
]

# ══════════════════════════════════════════════════════════════════
#  GRAPHIQUES
# ══════════════════════════════════════════════════════════════════
def make_charts(df, eq_full, eq_is, eq_oos, tr_full, tr_is, tr_oos,
                ma_series, envs, wf, weights=None):
    if weights is None:
        weights = geo_weights(len(envs))
    m_full = metrics(eq_full); m_is = metrics(eq_is); m_oos = metrics(eq_oos)
    ts_full= trade_stats(tr_full)

    # ── FIG 1 : Vue d'ensemble + métriques ────────────────────────
    fig = plt.figure(figsize=(20, 18), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
                            height_ratios=[1.8, 1.2, 1.0])

    # Equity log
    ax = fig.add_subplot(gs[0, :2])
    bh = df["close"] / df["close"].iloc[0] * CAPITAL
    bh_n = bh / bh.iloc[0] * 100; eq_n = eq_full / eq_full.iloc[0] * 100
    ax.plot(bh_n.index, bh_n, color=RED, lw=1.2, ls="--", alpha=0.55, label="B&H XAU/USD")
    ax.plot(eq_n.index, eq_n, color=GREEN, lw=2,
            label=f"σ-Env Géo  Cal={m_full.get('calmar',0):.2f}")
    ax.axvline(pd.Timestamp(OOS_START), color=GOLD, lw=1.2, ls=":", alpha=0.8)
    ax.text(pd.Timestamp(OOS_START), eq_n.max()*0.5, " OOS→", fontsize=8, color=GOLD)
    ax.set_yscale("log")
    ax.set_title("XAU/USD 4h — Enveloppes σ-Dynamiques Géométriques  |  Equity (log)",
                 fontsize=11, color=TEXT)
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor=TEXT)
    style(ax)

    # Tableau métriques
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    sigma_base = df["close"].pct_change().std()
    env_pcts   = [f"{k*sigma_base*100:.2f}%" for k in K_SIGMAS]
    rows = [
        ["CAGR % (full)",   f"{m_full.get('cagr',0):+.1f}"],
        ["MaxDD % (full)",  f"{m_full.get('mdd',0):.1f}"],
        ["Calmar (full)",   f"{m_full.get('calmar',0):.2f}"],
        ["Sharpe (full)",   f"{m_full.get('sharpe',0):.2f}"],
        ["IS Calmar",       f"{m_is.get('calmar',0):.2f}"],
        ["OOS Calmar",      f"{m_oos.get('calmar',0):.2f}"],
        ["WR % (full)",     f"{ts_full.get('wr',0):.1f}"],
        ["N trades (full)", f"{ts_full.get('n',0)}"],
        ["DEMA L",          f"{DEMA_L}"],
        ["σ/barre 4h",      f"{sigma_base*100:.3f}%"],
        ["Env. k=[1,2,4,7]","; ".join(env_pcts)],
    ]
    table = ax.table(cellText=rows, colLabels=["Métrique","Valeur"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(8)
    table.scale(1, 1.28)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1A1D27"); cell.set_text_props(color=TEXT)
    ax.set_facecolor(BG); ax.set_title("Résumé", fontsize=10, color=TEXT, pad=4)

    # Drawdown
    ax = fig.add_subplot(gs[1, :2])
    bh_rm = np.maximum.accumulate(bh.values)
    bh_dd = (bh.values - bh_rm) / bh_rm * 100
    ax.fill_between(bh.index, bh_dd, 0, alpha=0.2, color=RED, label="B&H XAU/USD")
    ax.fill_between(eq_full.index, m_full["dd"]*100, 0, alpha=0.55, color=GREEN,
                    label=f"σ-Env  MaxDD={m_full['mdd']:.1f}%")
    ax.axvline(pd.Timestamp(OOS_START), color=GOLD, lw=1, ls=":", alpha=0.7)
    ax.set_title("Drawdown comparé (full)", fontsize=10, color=TEXT)
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor=TEXT)
    ax.set_ylabel("DD (%)", color=TEXT); style(ax)

    # IS vs OOS Calmar
    ax = fig.add_subplot(gs[1, 2])
    vals   = [m_is.get("calmar",0), m_oos.get("calmar",0)]
    colors = [GREEN if v>0 else RED for v in vals]
    bars = ax.bar(["IS","OOS"], vals, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f"{val:.2f}", ha="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_title("Calmar IS vs OOS", fontsize=10, color=TEXT)
    ax.set_ylabel("Calmar", color=TEXT); style(ax)

    # Distribution retours
    ax = fig.add_subplot(gs[2, 0])
    if not tr_full.empty:
        ax.hist(tr_full["ret"], bins=60, color=GREEN, alpha=0.75, edgecolor=BG, lw=0.3)
        ax.axvline(0, color=WHITE, lw=0.8, ls="--")
        ax.axvline(tr_full["ret"].mean(), color=GOLD, lw=1.5, ls="--",
                   label=f"Moy={tr_full['ret'].mean():+.3f}%")
        ax.set_title(f"Distribution retours/trade\nWR={ts_full.get('wr',0):.1f}%  N={ts_full.get('n',0)}",
                     fontsize=9, color=TEXT)
        ax.legend(fontsize=8, facecolor="#1A1D27", labelcolor=TEXT)
        ax.set_xlabel("Retour (%)", color=TEXT)
    style(ax)

    # Retour par zone d'enveloppe
    ax = fig.add_subplot(gs[2, 1])
    if not tr_full.empty and "zone" in tr_full.columns:
        zs = tr_full.groupby("zone")["ret"].mean()
        ax.bar(zs.index, zs.values, color=[COLORS[min(k-1,3)] for k in zs.index], alpha=0.85)
        ax.axhline(0, color=WHITE, lw=0.5)
        ax.set_title("Retour moyen par zone\n(zone 4 = k=7, poids 53.3%)",
                     fontsize=9, color=TEXT)
        ax.set_xlabel("Zone"); ax.set_ylabel("Retour moyen (%)", color=TEXT)
    style(ax)

    # Walk-forward
    ax = fig.add_subplot(gs[2, 2])
    if not wf.empty:
        x = range(len(wf))
        cols = [GREEN if v else RED for v in wf["oos_pos"]]
        ax.bar(x, wf["cal_oos"], color=cols, alpha=0.85, width=0.7)
        ax.axhline(0, color=WHITE, lw=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(wf["window"].tolist(), rotation=45, fontsize=7)
        pos = wf["oos_pos"].sum()
        ax.set_title(f"Walk-forward  {pos}/{len(wf)} OOS+", fontsize=9, color=TEXT)
        ax.set_ylabel("Calmar OOS", color=TEXT)
    style(ax)

    plt.suptitle(
        "XAU/USD 4h — Enveloppes σ-Dynamiques Géométriques\n"
        f"DEMA(L={DEMA_L})  |  k=[1,2,4,7]  |  Poids géo  |  10 bps  |  23 ans",
        fontsize=13, color=TEXT, y=1.01)
    fig.patch.set_facecolor(BG)
    p = os.path.join(OUT_DIR, "xau_envelopes_results.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {p}")

    # ── FIG 2 : Régimes ────────────────────────────────────────────
    fig2, axes = plt.subplots(1, len(REGIMES), figsize=(22, 5), facecolor=BG)
    for ax, (name, rs, re) in zip(axes, REGIMES):
        mask = (df.index >= rs) & (df.index <= re)
        df_r = df[mask]; ma_r = ma_series[mask]
        if df_r.empty: continue
        # Recalc envs sur ce sous-segment
        eq_r, tr_r = run_sigma_engine(df_r, ma_r, envs, weights)
        m_r = metrics(eq_r)
        bh_r = df_r["close"] / df_r["close"].iloc[0] * 100
        eq_r_n = eq_r / eq_r.iloc[0] * 100
        ax.plot(bh_r.index, bh_r, color=RED, lw=1.2, ls="--", alpha=0.5, label="B&H")
        ax.plot(eq_r_n.index, eq_r_n, color=GREEN, lw=1.8,
                label=f"σ-Env\nCal={m_r.get('calmar',0):.2f}")
        ax.set_title(f"{name}\nCal={m_r.get('calmar',0):.2f}  MDD={m_r.get('mdd',0):.1f}%",
                     fontsize=8, color=TEXT)
        ax.legend(fontsize=7, facecolor="#1A1D27", labelcolor=TEXT)
        ax.set_yscale("log"); style(ax)
    plt.suptitle("XAU/USD 4h — σ-Dynamiques par Régime de Marché",
                 fontsize=12, color=TEXT, y=1.03)
    fig2.patch.set_facecolor(BG)
    p2 = os.path.join(OUT_DIR, "xau_envelopes_regimes.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"  ✓ {p2}")

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*60)
    print("  XAU/USD 4h — σ-Dynamiques Géométriques Backtest")
    print("="*60)

    data_path = os.path.join(HERE, "data", "XAUUSD_spot_4h.parquet")
    if not os.path.exists(data_path):
        for p in ["XAUUSD_spot_4h.parquet",
                  os.path.join("..", "data", "XAUUSD_spot_4h.parquet")]:
            if os.path.exists(p): data_path = p; break
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    print(f"  Données 4h : {len(df)} barres  {df.index[0].date()} → {df.index[-1].date()}")

    sig_mod = load_signal_module()
    if sig_mod is not None and hasattr(sig_mod, "compute_ma"):
        ma_series = sig_mod.compute_ma(df)
        envs      = sig_mod.ENVS
        weights   = sig_mod.WEIGHTS
        print("  ✓ Signal propriétaire chargé")
    else:
        if sig_mod is None:
            print("  ⚠ signals/xau_envelopes_signal.py non trouvé — placeholder")
        ma_series = dema(df["close"], DEMA_L)
        sigma     = df["close"].pct_change().std()
        envs      = [k * sigma for k in K_SIGMAS]
        weights   = geo_weights(len(K_SIGMAS))
        print(f"  σ/barre={sigma*100:.3f}%  envs={[f'{e*100:.2f}%' for e in envs]}")

    print(f"  Poids géo : {[f'{w*100:.1f}%' for w in weights]}")

    # IS / OOS
    df_is  = df[df.index <= IS_END];  ma_is  = ma_series[ma_series.index <= IS_END]
    df_oos = df[df.index >= OOS_START]; ma_oos = ma_series[ma_series.index >= OOS_START]

    print("  Backtest full..."); eq_full, tr_full = run_sigma_engine(df, ma_series, envs, weights)
    print("  Backtest IS...");   eq_is,  tr_is   = run_sigma_engine(df_is, ma_is, envs, weights)
    print("  Backtest OOS...");  eq_oos, tr_oos  = run_sigma_engine(df_oos, ma_oos, envs, weights)

    m_full = metrics(eq_full); m_is = metrics(eq_is); m_oos = metrics(eq_oos)
    ts_full= trade_stats(tr_full); ts_is = trade_stats(tr_is); ts_oos = trade_stats(tr_oos)

    print(f"\n  {'':22} {'IS':>10} {'OOS':>10} {'FULL':>10}")
    print("  " + "─"*55)
    for lbl, k in [("CAGR %","cagr"),("MaxDD %","mdd"),("Calmar","calmar"),("Sharpe","sharpe")]:
        print(f"  {lbl:<22} {m_is.get(k,0):>+9.2f} {m_oos.get(k,0):>+9.2f} {m_full.get(k,0):>+9.2f}")
    for lbl, k in [("WR %","wr"),("N trades","n")]:
        print(f"  {lbl:<22} {ts_is.get(k,0):>9.1f} {ts_oos.get(k,0):>9.1f} {ts_full.get(k,0):>9.1f}")

    print("  Walk-forward (6+2)...")
    wf = walk_forward(df, ma_series, envs, weights)
    if not wf.empty:
        pos = wf["oos_pos"].sum()
        print(f"  Walk-forward : {pos}/{len(wf)} fenêtres OOS positives")
        print(wf[["window","cal_is","cal_oos","oos_pos"]].to_string(index=False))

    print("\n  Génération graphiques...")
    make_charts(df, eq_full, eq_is, eq_oos, tr_full, tr_is, tr_oos,
                ma_series, envs, wf, weights)

    print(f"\n  DONE — {time.time()-t0:.1f}s")
    print(f"  Résultats dans : {OUT_DIR}")

if __name__ == "__main__":
    main()
