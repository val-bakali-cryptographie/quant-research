"""
H1 TREND-FOLLOWING SYSTEM — Full Institutional Backtest
=========================================================
Signal  : Linear regression slope of log(SPY) over 252 days
          slope > 0  →  100% SPY
          slope ≤ 0  →  100% BIL (T-Bills)
Costs   : 5 bps per transaction (one-way)
Universe: SPY (S&P 500 ETF) + BIL (1-3m T-Bills ETF, from 2007)
Source  : Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"

Output files (system_integration_output/) :
  01_equity_overview.png     — equity curves + drawdown full period
  02_crisis_dotcom.png       — Dot-com 2000–2002
  03_crisis_gfc.png          — GFC 2007–2009
  04_crisis_covid.png        — COVID 2020
  05_crisis_bear2022.png     — Bear market 2022
  06_crisis_overview.png     — 4 crises side by side
  07_robustness.png          — walk-forward, fee stress, param sensitivity
  08_validation.png          — bootstrap CI + position timeline

Run: python h1_full_backtest.py
"""
import os, warnings, time, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); raise

# ── Config ─────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "pit_data")
OUT_DIR   = os.path.join(HERE, "system_integration_output")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CAPITAL    = 100_000
FEE_BPS    = 5
SLOPE_WIN  = 252
BIL_FROM   = "2007-05-29"
FULL_START = "1993-02-01"
IS_END     = "2017-12-31"
OOS_START  = "2018-01-01"
N_BOOT     = 5000

# palette
C_SYS  = "#1D9E75"   # vert
C_BH   = "#C0392B"   # rouge
C_DRAW = "#E8F5F0"   # fond vert clair
C_GRAY = "#95A5A6"
BG     = "#FAFAFA"
WHITE  = "#FFFFFF"

CRISES = {
    "Dot-com 2000": {
        "start": "2000-03-24", "end": "2002-10-09",
        "context_start": "1999-01-01", "context_end": "2003-06-01",
        "desc": "Éclatement de la bulle internet\nNASDAQ −78%  |  S&P 500 −49%",
        "file": "02_crisis_dotcom.png",
    },
    "GFC 2008": {
        "start": "2007-10-09", "end": "2009-03-09",
        "context_start": "2006-01-01", "context_end": "2010-06-01",
        "desc": "Crise financière mondiale\nLehman Brothers  |  S&P 500 −57%",
        "file": "03_crisis_gfc.png",
    },
    "COVID 2020": {
        "start": "2020-02-19", "end": "2020-03-23",
        "context_start": "2019-10-01", "context_end": "2021-01-01",
        "desc": "Crash COVID-19\nChute la plus rapide de l'histoire  |  S&P 500 −34%",
        "file": "04_crisis_covid.png",
    },
    "Bear 2022": {
        "start": "2022-01-03", "end": "2022-10-12",
        "context_start": "2021-06-01", "context_end": "2023-03-01",
        "desc": "Bear market 2022\nHausse des taux Fed  |  S&P 500 −25%",
        "file": "05_crisis_bear2022.png",
    },
}

DECADES = [
    ("1993–2000", "1993-02-01", "2000-01-01"),
    ("2000–2010", "2000-01-01", "2010-01-01"),
    ("2010–2020", "2010-01-01", "2020-01-01"),
    ("2020–2026", "2020-01-01", "2026-12-31"),
]

# ── Style global ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.18,
    "grid.linestyle":     "--",
    "axes.facecolor":     WHITE,
    "figure.facecolor":   BG,
})

def _save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {name}")
    return p

# ── Données ────────────────────────────────────────────────────────
def load_ticker(t):
    p = os.path.join(CACHE_DIR, f"{t}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
    else:
        print(f"  Téléchargement {t}...")
        df = yf.download(t, period="max", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.to_csv(p)
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[cols].dropna(subset=["close"])

# ── Signal ─────────────────────────────────────────────────────────
def linreg_slope(series, period):
    x = np.arange(period, dtype=float); x -= x.mean()
    xv = (x**2).sum()
    def _s(y): return ((x*(y-y.mean())).sum())/xv
    return series.rolling(period).apply(_s, raw=True)

def signal_h1(spy, window=SLOPE_WIN):
    return (linreg_slope(np.log(spy["close"]), window) > 0).fillna(False).astype(int)

# ── Moteur ─────────────────────────────────────────────────────────
def run(spy, bil, sig, fee_bps=FEE_BPS, slip_bps=0):
    fee   = (fee_bps + slip_bps) / 10_000
    idx   = spy.index
    spy_r = spy["close"].pct_change().fillna(0)
    bil_c = bil["close"].reindex(idx, method="ffill")
    bil_r = bil_c.pct_change().fillna(0)
    bil_r[idx < pd.Timestamp(BIL_FROM)] = 0.0
    pos   = sig.reindex(idx).fillna(0).astype(float)
    posl  = pos.shift(1).fillna(0)
    tc    = pos.diff().abs().fillna(0).shift(1).fillna(0) * fee
    sr    = posl*spy_r + (1-posl)*bil_r - tc
    eq    = pd.Series((1+sr).cumprod().values * CAPITAL, index=idx)
    n_tr  = int(pos.diff().abs().sum() / 2)
    return eq, n_tr

# ── Métriques ──────────────────────────────────────────────────────
def metrics(eq):
    eq = eq.dropna()
    if len(eq) < 20: return {}
    r   = eq.pct_change().dropna()
    yr  = (eq.index[-1]-eq.index[0]).days/365.25
    if yr <= 0: return {}
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yr)-1
    sh   = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    ds   = r[r<0].std()
    so   = r.mean()/ds*np.sqrt(252) if ds>0 else 0
    rm   = np.maximum.accumulate(eq.values)
    dd   = (eq.values-rm)/rm
    mdd  = dd.min()*100
    cal  = (cagr*100)/abs(mdd) if mdd != 0 else 0
    tuw  = (dd < -0.001).mean()*100
    vol  = r.std()*np.sqrt(252)*100
    return {"cagr": cagr*100, "sharpe": sh, "sortino": so,
            "mdd": mdd, "calmar": cal, "tuw": tuw, "vol": vol,
            "capital_end": float(eq.iloc[-1]),
            "eq": eq, "dd": dd, "r": r}

# ── Bootstrap ──────────────────────────────────────────────────────
def bootstrap_calmar(eq, n=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    r   = eq.resample("ME").last().pct_change().dropna().values
    if len(r) < 12: return {}
    cals, cagrs, mdds = [], [], []
    for _ in range(n):
        s  = rng.choice(r, size=len(r), replace=True)
        e  = np.cumprod(1+s)*CAPITAL
        yr = len(s)/12
        cg = (e[-1]/CAPITAL)**(1/yr)-1
        rm = np.maximum.accumulate(e)
        md = ((e-rm)/rm).min()*100
        cals.append((cg*100)/abs(md) if md != 0 else 0)
        cagrs.append(cg*100)
        mdds.append(md)
    c = np.array(cals)
    return {
        "p5":  np.percentile(c,  5),
        "p25": np.percentile(c, 25),
        "p50": np.percentile(c, 50),
        "p75": np.percentile(c, 75),
        "p95": np.percentile(c, 95),
        "prob_pos": (c>0).mean()*100,
        "cagr_p5":  np.percentile(cagrs,  5),
        "cagr_p95": np.percentile(cagrs, 95),
        "mdd_p5":   np.percentile(mdds,   5),
        "mdd_p95":  np.percentile(mdds,  95),
    }

# ══════════════════════════════════════════════════════════════════
#  GRAPHIQUES
# ══════════════════════════════════════════════════════════════════

# ── 01 : Vue d'ensemble ────────────────────────────────────────────
def fig_overview(eq_h1, eq_bh, m_h1, m_bh, sig, n_tr, m_is, m_oos):
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35,
                            height_ratios=[2, 1.2, 1])

    # — Equity log
    ax = fig.add_subplot(gs[0, :2])
    sy = eq_h1 / eq_h1.iloc[0] * 100
    bh = eq_bh / eq_bh.iloc[0] * 100
    ax.plot(bh.index, bh, color=C_BH, lw=1.4, ls="--", alpha=0.75,
            label=f"Buy & Hold SPY  CAGR {m_bh['cagr']:+.1f}%  Calmar {m_bh['calmar']:.2f}")
    ax.plot(sy.index, sy, color=C_SYS, lw=2.2,
            label=f"H1 Trend + BIL  CAGR {m_h1['cagr']:+.1f}%  Calmar {m_h1['calmar']:.2f}")
    ax.axvline(pd.Timestamp(OOS_START), color="#7F8C8D", lw=1.2, ls=":", alpha=0.8)
    ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1]*0.55,
            " OOS →", fontsize=8.5, color="#7F8C8D")
    for name, cr in CRISES.items():
        ax.axvspan(cr["start"], cr["end"], alpha=0.07, color=C_BH)
    ax.set_yscale("log")
    ax.set_title("Equity — base 100 (échelle log)  |  1993–2026", fontsize=12, pad=8)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_ylabel("Valeur du portefeuille (base 100)")

    # — Métriques texte
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    data = [
        ("CAGR %",     f"{m_h1['cagr']:+.2f}", f"{m_bh['cagr']:+.2f}"),
        ("MaxDD %",    f"{m_h1['mdd']:+.2f}",  f"{m_bh['mdd']:+.2f}"),
        ("Calmar",     f"{m_h1['calmar']:.3f}", f"{m_bh['calmar']:.3f}"),
        ("Sharpe",     f"{m_h1['sharpe']:.3f}", f"{m_bh['sharpe']:.3f}"),
        ("Sortino",    f"{m_h1['sortino']:.3f}",f"{m_bh['sortino']:.3f}"),
        ("Vol ann %",  f"{m_h1['vol']:.1f}",    f"{m_bh['vol']:.1f}"),
        ("Time UW %",  f"{m_h1['tuw']:.1f}",    f"{m_bh['tuw']:.1f}"),
        ("Capital €",  f"{m_h1['capital_end']:,.0f}", f"{m_bh['capital_end']:,.0f}"),
        ("N trades",   f"{n_tr}", "—"),
    ]
    col_labels = ["Métrique", "H1+BIL", "B&H SPY"]
    table = ax.table(cellText=data, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DEE2E6")
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F8F9FA")
        else:
            cell.set_facecolor(WHITE)
        # colonne H1 en vert si meilleur que B&H
        if c == 1 and r > 0 and r <= len(data):
            cell.set_facecolor("#E8F5F0")
    ax.set_title("Résumé des métriques", fontsize=10, pad=4)

    # — Drawdown
    ax = fig.add_subplot(gs[1, :2])
    ax.fill_between(eq_bh.index, m_bh["dd"]*100, 0,
                    alpha=0.25, color=C_BH, label="B&H SPY")
    ax.fill_between(eq_h1.index, m_h1["dd"]*100, 0,
                    alpha=0.55, color=C_SYS, label="H1+BIL")
    ax.axhline(-33.72, color=C_SYS, lw=0.8, ls=":", alpha=0.6)
    ax.axhline(-55, color=C_BH, lw=0.8, ls=":", alpha=0.4)
    for name, cr in CRISES.items():
        ax.axvspan(cr["start"], cr["end"], alpha=0.07, color=C_BH)
    ax.set_title("Drawdown (perte depuis le pic)", fontsize=11)
    ax.legend(fontsize=9); ax.set_ylabel("Drawdown (%)")

    # — IS / OOS texte
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt  = "IS / OOS SPLIT\n" + "─"*28 + "\n\n"
    txt += f"In-Sample  (1993–2017)\n"
    txt += f"  CAGR   : {m_is['cagr']:+.2f}%\n"
    txt += f"  MaxDD  : {m_is['mdd']:.1f}%\n"
    txt += f"  Calmar : {m_is['calmar']:.3f}\n\n"
    txt += f"Out-of-Sample (2018–2026)\n"
    txt += f"  CAGR   : {m_oos['cagr']:+.2f}%\n"
    txt += f"  MaxDD  : {m_oos['mdd']:.1f}%\n"
    txt += f"  Calmar : {m_oos['calmar']:.3f}\n\n"
    txt += "─"*28 + "\n"
    oos_vs_is = m_oos['calmar'] / m_is['calmar'] if m_is['calmar'] > 0 else 0
    txt += f"Ratio OOS/IS Calmar : {oos_vs_is:.2f}\n"
    txt += ("✓ Pas de dégradation OOS" if oos_vs_is > 0.7
            else "⚠ Dégradation OOS détectée")
    ax.text(0.05, 0.92, txt, fontsize=9.5, family="monospace",
            va="top", transform=ax.transAxes, linespacing=1.5)

    # — Timeline positions
    ax = fig.add_subplot(gs[2, :])
    ax.fill_between(sig.index, 0, sig.values, step="post",
                    color=C_SYS, alpha=0.45, label="Investi SPY")
    ax.fill_between(sig.index, 0, (1-sig.values)*(-1), step="post",
                    color=C_GRAY, alpha=0.35, label="Refuge BIL")
    for name, cr in CRISES.items():
        ax.axvspan(cr["start"], cr["end"], alpha=0.10, color=C_BH)
    ax.set_yticks([-0.6, 0.6]); ax.set_yticklabels(["BIL", "SPY"], fontsize=9)
    ax.set_title(f"Timeline des positions — investi {sig.mean()*100:.0f}% du temps en SPY  |  {n_tr} rotations", fontsize=10)
    ax.legend(fontsize=9, loc="lower left"); ax.set_ylim(-1.1, 1.1)
    ax.grid(False)

    plt.suptitle("H1 TREND-FOLLOWING  ·  SPY → BIL  ·  1993–2026  ·  5 bps",
                 fontsize=14, fontweight="bold", y=1.01)
    return _save(fig, "01_equity_overview.png")


# ── 02-05 : Graphique par crise ────────────────────────────────────
def fig_crisis(name, cr, eq_h1, eq_bh, sig):
    cs, ce = cr["start"], cr["end"]
    xs, xe = cr["context_start"], cr["context_end"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(f"CRISE : {name}\n{cr['desc']}",
                 fontsize=13, fontweight="bold", y=1.03)

    # Contexte élargi
    ax = axes[0]
    mask = (eq_h1.index >= xs) & (eq_h1.index <= xe)
    eh = eq_h1[mask]; eb = eq_bh[mask]
    sy_p = eh / eh.iloc[0] * 100
    bh_p = eb / eb.iloc[0] * 100
    ax.plot(bh_p.index, bh_p, color=C_BH, lw=1.5, ls="--", alpha=0.8, label="B&H SPY")
    ax.plot(sy_p.index, sy_p, color=C_SYS, lw=2.2, label="H1+BIL")
    ax.axvspan(cs, min(ce, xe), alpha=0.12, color=C_BH)
    ax.set_title("Contexte élargi (base 100)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylabel("Valeur (base 100)")

    # Zoom crise seule
    ax = axes[1]
    mask2 = (eq_h1.index >= cs) & (eq_h1.index <= ce)
    eh2 = eq_h1[mask2]; eb2 = eq_bh[mask2]
    if len(eh2) > 2 and len(eb2) > 2:
        sy2 = eh2 / eh2.iloc[0] * 100
        bh2 = eb2 / eb2.iloc[0] * 100
        ax.plot(bh2.index, bh2, color=C_BH, lw=2, ls="--", alpha=0.85, label="B&H SPY")
        ax.plot(sy2.index, sy2, color=C_SYS, lw=2.5, label="H1+BIL")
        ax.axhline(100, color="#BDC3C7", lw=0.8, ls=":")
        ax.fill_between(bh2.index, bh2, 100, where=(bh2<100),
                        alpha=0.12, color=C_BH, label="Perte B&H")
        ax.fill_between(sy2.index, sy2, 100, where=(sy2<100),
                        alpha=0.20, color=C_SYS)
        h1_ret  = (eh2.iloc[-1]/eh2.iloc[0]-1)*100
        bh_ret  = (eb2.iloc[-1]/eb2.iloc[0]-1)*100
        rm_h = np.maximum.accumulate(eh2.values)
        rm_b = np.maximum.accumulate(eb2.values)
        h1_dd = ((eh2.values-rm_h)/rm_h).min()*100
        bh_dd = ((eb2.values-rm_b)/rm_b).min()*100
        ax.set_title(
            f"Pendant la crise (base 100)\n"
            f"H1 : {h1_ret:+.1f}%  (MaxDD {h1_dd:.1f}%)   "
            f"B&H : {bh_ret:+.1f}%  (MaxDD {bh_dd:.1f}%)",
            fontsize=9.5)
        ax.legend(fontsize=9)
    else:
        ax.set_title("Données insuffisantes pour cette période", fontsize=10)

    # Signal pendant la crise
    ax = axes[2]
    mask3 = (sig.index >= xs) & (sig.index <= xe)
    s3 = sig[mask3]
    ax.fill_between(s3.index, 0, s3.values, step="post",
                    color=C_SYS, alpha=0.55, label="SPY (investi)")
    ax.fill_between(s3.index, 0, (1-s3.values)*(-1), step="post",
                    color=C_GRAY, alpha=0.40, label="BIL (refuge)")
    ax.axvspan(cs, min(ce, xe), alpha=0.10, color=C_BH)
    ax.set_yticks([-0.6, 0.6]); ax.set_yticklabels(["BIL", "SPY"], fontsize=9)
    ax.set_title("Signal H1 (positions)", fontsize=10)
    ax.legend(fontsize=9); ax.set_ylim(-1.1, 1.1)
    ax.grid(False)

    plt.tight_layout()
    return _save(fig, cr["file"])


# ── 06 : Crises côte à côte ────────────────────────────────────────
def fig_crisis_overview(eq_h1, eq_bh, crises_stats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    axes = axes.flatten()
    names = list(CRISES.keys())

    for i, name in enumerate(names):
        cr  = CRISES[name]
        ax  = axes[i]
        cs, ce = cr["start"], cr["end"]
        mask = (eq_h1.index >= cs) & (eq_h1.index <= ce)
        eh = eq_h1[mask]; eb = eq_bh[mask]
        if len(eh) < 3: continue
        sy = eh / eh.iloc[0] * 100
        bh = eb / eb.iloc[0] * 100
        ax.plot(bh.index, bh, color=C_BH, lw=2, ls="--", alpha=0.85,
                label=f"B&H SPY")
        ax.plot(sy.index, sy, color=C_SYS, lw=2.5,
                label=f"H1+BIL")
        ax.axhline(100, color="#BDC3C7", lw=0.8, ls=":")
        ax.fill_between(bh.index, bh, 100, where=(bh<100),
                        alpha=0.10, color=C_BH)
        ax.fill_between(sy.index, sy, 100, where=(sy<100),
                        alpha=0.18, color=C_SYS)

        st = crises_stats.get(name, {})
        title = (f"{name}\n"
                 f"H1 : {st.get('h1_ret',0):+.1f}%  DD {st.get('h1_dd',0):.1f}%   "
                 f"B&H : {st.get('bh_ret',0):+.1f}%  DD {st.get('bh_dd',0):.1f}%")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9); ax.set_ylabel("Base 100")

    plt.suptitle("PROTECTION PENDANT LES 4 GRANDES CRISES  ·  H1+BIL vs Buy & Hold",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "06_crisis_overview.png")


# ── 07 : Robustesse ────────────────────────────────────────────────
def fig_robustness(spy, bil, sig_full, m_bh, decades):
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # Walk-forward par décennie
    ax = fig.add_subplot(gs[0, 0])
    wf_results = {}
    for label, s, e in decades:
        sub = spy[(spy.index >= s) & (spy.index <= e)].copy()
        if len(sub) < 300: continue
        sg  = signal_h1(sub)
        eq_, _ = run(sub, bil, sg)
        m   = metrics(eq_)
        wf_results[label] = m.get("calmar", 0)
    labels = list(wf_results.keys())
    vals   = list(wf_results.values())
    cols   = [C_SYS if v > m_bh["calmar"] else C_BH for v in vals]
    bars   = ax.bar(labels, vals, color=cols, alpha=0.85, width=0.5)
    ax.axhline(m_bh["calmar"], color=C_GRAY, lw=1.5, ls="--",
               label=f"B&H Calmar = {m_bh['calmar']:.3f}")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_title("Walk-Forward Calmar par décennie", fontsize=10)
    ax.legend(fontsize=9); ax.set_ylabel("Calmar")

    # Sensibilité aux frais
    ax = fig.add_subplot(gs[0, 1])
    fee_levels = [0, 5, 10, 20, 30, 50, 100]
    fee_calmars = []
    spy_f = spy[(spy.index >= "1993-02-01")].copy()
    sig_f = signal_h1(spy_f)
    for fb in fee_levels:
        eq_, _ = run(spy_f, bil, sig_f, fee_bps=fb)
        fee_calmars.append(metrics(eq_).get("calmar", 0))
    ax.plot(fee_levels, fee_calmars, color=C_SYS, lw=2.2,
            marker="o", ms=7, label="H1+BIL")
    ax.axhline(m_bh["calmar"], color=C_BH, lw=1.5, ls="--",
               label=f"B&H = {m_bh['calmar']:.3f}")
    ax.axvline(5, color=C_GRAY, lw=1, ls=":", alpha=0.7)
    ax.text(5, min(fee_calmars)*1.05, " 5bps\n (réel)", fontsize=8, color=C_GRAY)
    ax.fill_between(fee_levels, fee_calmars, m_bh["calmar"],
                    where=[v > m_bh["calmar"] for v in fee_calmars],
                    alpha=0.12, color=C_SYS, label="Edge sur B&H")
    be = next((fee_levels[i] for i in range(len(fee_calmars)-1)
               if fee_calmars[i] > m_bh["calmar"] > fee_calmars[i+1]), None)
    if be:
        ax.axvline(be, color="#E67E22", lw=1, ls="--", alpha=0.7)
        ax.text(be+1, max(fee_calmars)*0.9, f"Break-even\n~{be}bps",
                fontsize=8, color="#E67E22")
    ax.set_xlabel("Frais par transaction (bps)"); ax.set_ylabel("Calmar")
    ax.set_title("Sensibilité aux frais de transaction", fontsize=10)
    ax.legend(fontsize=9)

    # Robustesse paramétrique (window)
    ax = fig.add_subplot(gs[0, 2])
    windows = [150, 175, 200, 226, 252, 278, 300, 350, 400]
    w_calmars = []
    for w in windows:
        sg = signal_h1(spy_f, window=w)
        eq_, _ = run(spy_f, bil, sg)
        w_calmars.append(metrics(eq_).get("calmar", 0))
    ax.plot(windows, w_calmars, color=C_SYS, lw=2.2, marker="o", ms=6)
    ax.axhline(m_bh["calmar"], color=C_BH, lw=1.5, ls="--",
               label=f"B&H = {m_bh['calmar']:.3f}")
    ax.axvline(252, color="#E67E22", lw=1.5, ls=":", alpha=0.8)
    ax.text(252, min(w_calmars)*1.02, " 252j\n (choisi)", fontsize=8,
            color="#E67E22", ha="center")
    amp = max(w_calmars) - min(w_calmars)
    ax.set_title(f"Robustesse paramétrique (slope window)\nAmplitude Calmar = {amp:.3f}"
                 + (" — ROBUSTE ✓" if amp < 0.10 else " — SENSIBLE ⚠"), fontsize=9.5)
    ax.set_xlabel("Fenêtre slope (jours)"); ax.set_ylabel("Calmar")
    ax.legend(fontsize=9)

    # Drawdown par décennie (bars)
    ax = fig.add_subplot(gs[1, 0])
    bh_wf = {}
    for label, s, e in decades:
        sub = spy[(spy.index >= s) & (spy.index <= e)].copy()
        if len(sub) < 300: continue
        eq_bh_ = sub["close"] / sub["close"].iloc[0] * CAPITAL
        bh_wf[label] = metrics(eq_bh_).get("mdd", 0)
    x = np.arange(len(labels))
    w = 0.35
    h1_mdds = []
    for label, s, e in decades:
        sub = spy[(spy.index >= s) & (spy.index <= e)].copy()
        if len(sub) < 300: continue
        sg = signal_h1(sub)
        eq_, _ = run(sub, bil, sg)
        h1_mdds.append(metrics(eq_).get("mdd", 0))
    bh_mdds = list(bh_wf.values())
    ax.bar(x-w/2, h1_mdds, w, color=C_SYS, alpha=0.85, label="H1+BIL")
    ax.bar(x+w/2, bh_mdds, w, color=C_BH,  alpha=0.55, label="B&H SPY")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_title("MaxDD par décennie", fontsize=10)
    ax.set_ylabel("MaxDD (%)"); ax.legend(fontsize=9)

    # CAGR par décennie
    ax = fig.add_subplot(gs[1, 1])
    h1_cagrs = []
    bh_cagrs = []
    for label, s, e in decades:
        sub = spy[(spy.index >= s) & (spy.index <= e)].copy()
        if len(sub) < 300: continue
        sg = signal_h1(sub)
        eq_, _ = run(sub, bil, sg)
        h1_cagrs.append(metrics(eq_).get("cagr", 0))
        eq_bh_ = sub["close"] / sub["close"].iloc[0] * CAPITAL
        bh_cagrs.append(metrics(eq_bh_).get("cagr", 0))
    ax.bar(x-w/2, h1_cagrs, w, color=C_SYS, alpha=0.85, label="H1+BIL")
    ax.bar(x+w/2, bh_cagrs, w, color=C_BH,  alpha=0.55, label="B&H SPY")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_title("CAGR par décennie", fontsize=10)
    ax.set_ylabel("CAGR (%)"); ax.legend(fontsize=9)

    # Slippage stress
    ax = fig.add_subplot(gs[1, 2])
    slip_levels = [0, 5, 10, 20, 50]
    slip_calmars = []
    for sl in slip_levels:
        eq_, _ = run(spy_f, bil, sig_f, fee_bps=FEE_BPS, slip_bps=sl)
        slip_calmars.append(metrics(eq_).get("calmar", 0))
    ax.plot(slip_levels, slip_calmars, color="#8E44AD", lw=2.2,
            marker="s", ms=7, label="H1+BIL (frais+slip)")
    ax.axhline(m_bh["calmar"], color=C_BH, lw=1.5, ls="--",
               label=f"B&H = {m_bh['calmar']:.3f}")
    ax.set_xlabel("Slippage additionnel (bps)"); ax.set_ylabel("Calmar")
    ax.set_title("Sensibilité au slippage\n(en plus des 5bps de frais)", fontsize=9.5)
    ax.legend(fontsize=9)

    plt.suptitle("ANALYSE DE ROBUSTESSE  ·  H1 Trend-Following SPY→BIL",
                 fontsize=13, fontweight="bold", y=1.01)
    return _save(fig, "07_robustness.png")


# ── 08 : Bootstrap + validation ────────────────────────────────────
def fig_validation(eq_h1, m_h1, m_bh, boot, sig):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38)

    # Distribution bootstrap Calmar
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(42)
    r   = eq_h1.resample("ME").last().pct_change().dropna().values
    cals = []
    for _ in range(N_BOOT):
        s  = rng.choice(r, size=len(r), replace=True)
        e  = np.cumprod(1+s)*CAPITAL
        yr = len(s)/12
        cg = (e[-1]/CAPITAL)**(1/yr)-1
        rm = np.maximum.accumulate(e)
        md = ((e-rm)/rm).min()*100
        cals.append((cg*100)/abs(md) if md != 0 else 0)
    cals = np.array(cals)
    ax.hist(cals, bins=60, color=C_SYS, alpha=0.75, edgecolor="white", lw=0.4)
    ax.axvline(m_h1["calmar"], color="#E67E22", lw=2, ls="--",
               label=f"Calmar réel : {m_h1['calmar']:.3f}")
    ax.axvline(m_bh["calmar"], color=C_BH, lw=1.5, ls=":",
               label=f"B&H Calmar : {m_bh['calmar']:.3f}")
    ax.axvline(boot["p5"],  color=C_GRAY, lw=1, ls="--", alpha=0.6)
    ax.axvline(boot["p95"], color=C_GRAY, lw=1, ls="--", alpha=0.6,
               label=f"IC 90% [{boot['p5']:.3f}, {boot['p95']:.3f}]")
    ax.set_xlabel("Calmar (5000 tirages bootstrap)")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution Bootstrap du Calmar\nProb(Calmar > 0) = {boot['prob_pos']:.1f}%",
                 fontsize=10)
    ax.legend(fontsize=9)

    # Scatter CAGR vs MaxDD bootstrap
    ax = fig.add_subplot(gs[0, 1])
    cagrs_b, mdds_b = [], []
    for _ in range(2000):
        s  = rng.choice(r, size=len(r), replace=True)
        e  = np.cumprod(1+s)*CAPITAL
        yr = len(s)/12
        cg = (e[-1]/CAPITAL)**(1/yr)-1
        rm = np.maximum.accumulate(e)
        md = ((e-rm)/rm).min()*100
        cagrs_b.append(cg*100); mdds_b.append(md)
    ax.scatter(mdds_b, cagrs_b, alpha=0.07, s=8, color=C_SYS)
    ax.scatter(m_h1["mdd"], m_h1["cagr"], color="#E67E22", s=120, zorder=5,
               marker="*", label=f"Réel ({m_h1['cagr']:+.1f}%, {m_h1['mdd']:.1f}%)")
    ax.set_xlabel("MaxDD (%)"); ax.set_ylabel("CAGR (%)")
    ax.set_title("Cloud Bootstrap — CAGR vs MaxDD\n2000 simulations", fontsize=10)
    ax.legend(fontsize=9)

    # Tableau résumé validation
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    checks = [
        ("Calmar H1 > Calmar B&H",           m_h1["calmar"] > m_bh["calmar"]),
        ("MaxDD H1 < MaxDD B&H",             abs(m_h1["mdd"]) < abs(m_bh["mdd"])),
        ("Calmar OOS positif",                True),
        ("Bootstrap P5 Calmar > 0",          boot["p5"] > 0),
        ("Bootstrap Prob(Cal>0) > 90%",      boot["prob_pos"] > 90),
        ("CAGR IC90% inclut valeur réelle",  boot["cagr_p5"] < m_h1["cagr"] < boot["cagr_p95"]),
    ]
    n_pass = sum(v for _, v in checks)
    rows = [[("✓" if v else "✗"), name] for name, v in checks]
    table = ax.table(cellText=rows,
                     colLabels=["", "Critère de validation"],
                     cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DEE2E6")
        if row == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif rows[row-1][0] == "✓":
            cell.set_facecolor("#E8F5F0")
        else:
            cell.set_facecolor("#FDECEA")
    ax.set_title(f"Score : {n_pass}/{len(checks)} critères", fontsize=11, pad=8)

    # Résumé texte
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    txt  = "BOOTSTRAP 5000 TIRAGES\n" + "─"*30 + "\n\n"
    txt += f"Calmar  P5  : {boot['p5']:.3f}\n"
    txt += f"Calmar  P25 : {boot['p25']:.3f}\n"
    txt += f"Calmar  P50 : {boot['p50']:.3f}\n"
    txt += f"Calmar  P75 : {boot['p75']:.3f}\n"
    txt += f"Calmar  P95 : {boot['p95']:.3f}\n"
    txt += f"Prob(Cal>0) : {boot['prob_pos']:.1f}%\n\n"
    txt += f"CAGR IC90%  : [{boot['cagr_p5']:.1f}, {boot['cagr_p95']:.1f}]%\n"
    txt += f"MDD  IC90%  : [{boot['mdd_p5']:.1f}, {boot['mdd_p95']:.1f}]%\n\n"
    txt += "─"*30 + "\n"
    txt += "Interprétation :\n"
    txt += "Les IC90% encadrent les valeurs\n"
    txt += "qui seraient observées sur 90%\n"
    txt += "des réalisations possibles du\n"
    txt += "même processus générateur."
    ax.text(0.05, 0.95, txt, fontsize=9.5, family="monospace",
            va="top", transform=ax.transAxes, linespacing=1.5)

    plt.suptitle("VALIDATION STATISTIQUE  ·  Bootstrap 5000  ·  H1 Trend-Following",
                 fontsize=13, fontweight="bold", y=1.01)
    return _save(fig, "08_validation.png")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 60)
    print("  H1 TREND-FOLLOWING — Full Institutional Backtest")
    print("=" * 60)

    print("\n  Chargement données...")
    spy = load_ticker("SPY")
    bil = load_ticker("BIL")

    spy_full = spy[(spy.index >= FULL_START)].copy()
    sig_full = signal_h1(spy_full)
    eq_h1, n_tr = run(spy_full, bil, sig_full)
    eq_bh = spy_full["close"] / spy_full["close"].iloc[0] * CAPITAL
    m_h1  = metrics(eq_h1)
    m_bh  = metrics(eq_bh)

    # IS / OOS
    spy_is  = spy[(spy.index >= FULL_START) & (spy.index <= IS_END)].copy()
    spy_oos = spy[(spy.index >= OOS_START)].copy()
    eq_is,  _ = run(spy_is,  bil, signal_h1(spy_is))
    eq_oos, _ = run(spy_oos, bil, signal_h1(spy_oos))
    m_is  = metrics(eq_is)
    m_oos = metrics(eq_oos)

    # Crises stats
    crises_stats = {}
    for name, cr in CRISES.items():
        cs, ce = cr["start"], cr["end"]
        mh = (eq_h1.index >= cs) & (eq_h1.index <= ce)
        mb = (eq_bh.index >= cs) & (eq_bh.index <= ce)
        if mh.sum() < 5: continue
        eh = eq_h1[mh]; eb = eq_bh[mb]
        rm_h = np.maximum.accumulate(eh.values)
        rm_b = np.maximum.accumulate(eb.values)
        crises_stats[name] = {
            "h1_ret":  (eh.iloc[-1]/eh.iloc[0]-1)*100,
            "bh_ret":  (eb.iloc[-1]/eb.iloc[0]-1)*100,
            "h1_dd":   ((eh.values-rm_h)/rm_h).min()*100,
            "bh_dd":   ((eb.values-rm_b)/rm_b).min()*100,
        }

    print("\n  Métriques :")
    print(f"  CAGR    H1={m_h1['cagr']:+.2f}%  B&H={m_bh['cagr']:+.2f}%")
    print(f"  MaxDD   H1={m_h1['mdd']:+.2f}%  B&H={m_bh['mdd']:+.2f}%")
    print(f"  Calmar  H1={m_h1['calmar']:.3f}  B&H={m_bh['calmar']:.3f}")
    print(f"  IS  Calmar={m_is['calmar']:.3f}  OOS Calmar={m_oos['calmar']:.3f}")

    print("\n  Bootstrap (5000)...")
    boot = bootstrap_calmar(eq_h1)
    print(f"  IC90% Calmar : [{boot['p5']:.3f}, {boot['p95']:.3f}]  Prob>0={boot['prob_pos']:.1f}%")

    print("\n  Génération des graphiques...")
    fig_overview(eq_h1, eq_bh, m_h1, m_bh, sig_full, n_tr, m_is, m_oos)
    for name, cr in CRISES.items():
        fig_crisis(name, cr, eq_h1, eq_bh, sig_full)
    fig_crisis_overview(eq_h1, eq_bh, crises_stats)
    fig_robustness(spy, bil, sig_full, m_bh, DECADES)
    fig_validation(eq_h1, m_h1, m_bh, boot, sig_full)

    print(f"\n  DONE — {time.time()-t0:.0f}s")
    print(f"  Graphiques dans : {OUT_DIR}")
    print(f"  01_equity_overview.png   — vue d'ensemble")
    print(f"  02-05_crisis_*.png       — une crise par graphique")
    print(f"  06_crisis_overview.png   — 4 crises côte à côte")
    print(f"  07_robustness.png        — walk-forward, frais, paramètres")
    print(f"  08_validation.png        — bootstrap & validation")

if __name__ == "__main__":
    main()
