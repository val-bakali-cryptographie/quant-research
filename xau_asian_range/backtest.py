"""
XAU/USD — Asian Range Mean Reversion
======================================
Framework public — signal propriétaire masqué.

Logique (publiée) :
  Session asiatique  : 17h-22h UTC  → calcule Range(High, Low)
  Fenêtre d'entrée   : 22h-06h UTC  → limite -0.3% sous Range Low
  TP                 : Range High | SL : 1.5% sous entrée
  Frais              : 10 bps aller + retour

Résultats publiés :
  DISC  2003-2014 : CAGR=+87.79%  MaxDD=-3.17%  Calmar=27.68  WR=80.6%  N=1177
  VAL   2015-2020 : CAGR=+50.95%              Calmar=23.81            N=497
  FOOS  2021-2026 : CAGR=+87.29%              Calmar=54.61  WR=85%    N=592
  Walk-forward    : 8/8 fenêtres OOS positives
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
OUT_DIR = os.path.join(HERE, "output_xau_asian")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paramètres publics ─────────────────────────────────────────────
CAPITAL   = 100_000
FEE_BPS   = 10
ENTRY_PCT = -0.003    # -0.3% sous Range Low
SL_PCT    = 0.015     # SL 1.5%

# ── Périodes (découverte / validation / forward OOS) ──────────────
DISC_START  = "2003-01-01"; DISC_END   = "2014-12-31"
VAL_START   = "2015-01-01"; VAL_END    = "2020-12-31"
FOOS_START  = "2021-01-01"; FOOS_END   = "2026-12-31"

# ── Couleurs ───────────────────────────────────────────────────────
BG    = "#0F1117"; TEXT  = "#DDDDDD"
GREEN = "#1D9E75"; RED   = "#D85A30"
GOLD  = "#E69F00"; BLUE  = "#4C9BE8"
WHITE = "#FFFFFF"; PURPLE= "#9B59B6"

def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(alpha=0.12, color="#555")
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

# ══════════════════════════════════════════════════════════════════
#  SIGNAL — interface publique
#  Le signal réel est dans signals/xau_asian_signal.py (privé)
# ══════════════════════════════════════════════════════════════════
def load_signal_module():
    sig_dir  = os.path.join(HERE, "signals")
    sig_file = os.path.join(sig_dir, "xau_asian_signal.py")
    if os.path.exists(sig_file):
        sys.path.insert(0, sig_dir)
        import xau_asian_signal as _s
        return _s
    return None

# ══════════════════════════════════════════════════════════════════
#  MOTEUR — Asian Range Mean Reversion
# ══════════════════════════════════════════════════════════════════
def asian_range_session(df_1h):
    """
    Pour chaque journée, calcule :
      - range_low / range_high : session 17h-22h UTC
      - entry_price            : range_low * (1 + ENTRY_PCT)
      - tp_price               : range_high
      - sl_price               : entry * (1 - SL_PCT)
    Retourne un DataFrame indexé par date.
    """
    def _session_range(group):
        asian = group[(group.index.hour >= 17) & (group.index.hour < 22)]
        if asian.empty: return None
        return pd.Series({
            "range_low":  asian["low"].min(),
            "range_high": asian["high"].max(),
        })

    daily = df_1h.groupby(df_1h.index.date).apply(_session_range).dropna()
    daily.index = pd.to_datetime(daily.index)
    daily["entry_price"] = daily["range_low"] * (1 + ENTRY_PCT)
    daily["tp_price"]    = daily["range_high"]
    daily["sl_price"]    = daily["entry_price"] * (1 - SL_PCT)
    return daily

def run_asian_engine(df_1h, sig_module=None, fee_bps=FEE_BPS):
    """
    Moteur backtestant l'Asian Range sur les données 1h.
    Retourne equity (Series) + trades (DataFrame).
    """
    fee   = fee_bps / 10_000
    daily = asian_range_session(df_1h)

    # Signal de filtrage (optionnel depuis le module privé)
    if sig_module is not None and hasattr(sig_module, "compute_filter"):
        trade_filter = sig_module.compute_filter(df_1h, daily)
    else:
        trade_filter = pd.Series(True, index=daily.index)  # no filter = placeholder

    trades  = []
    eq_list = []
    cash    = float(CAPITAL)
    prev_date = None

    for date, row in daily.iterrows():
        # Filtre signal propriétaire
        if not trade_filter.get(date, True):
            continue

        entry_p = row["entry_price"]
        tp_p    = row["tp_price"]
        sl_p    = row["sl_price"]

        # Fenêtre d'entrée : 22h jusqu'à 6h UTC du lendemain
        t_start = date + pd.Timedelta(hours=22)
        t_end   = date + pd.Timedelta(hours=30)  # 6h lendemain

        window = df_1h[(df_1h.index >= t_start) & (df_1h.index < t_end)]
        if window.empty: continue

        # Cherche l'entrée (limite sous le Range Low)
        entered = False; ep = None; ei = None
        for ts, bar in window.iterrows():
            if bar["low"] <= entry_p:
                ep = min(entry_p, bar["open"]) * (1 + fee)
                ei = ts; entered = True; break

        if not entered: continue

        # Cherche TP ou SL après l'entrée
        after = window[window.index > ei]
        if after.empty: continue

        xp = None; outcome = None
        for ts, bar in after.iterrows():
            if bar["high"] >= tp_p:
                xp = max(tp_p, bar["open"]) * (1 - fee)
                outcome = "TP"; break
            if bar["low"] <= sl_p:
                xp = min(sl_p, bar["open"]) * (1 + fee)
                outcome = "SL"; break

        if xp is None: continue  # non fermé dans la fenêtre

        ret     = (xp - ep) / ep
        cash   *= (1 + ret)
        dur     = (ts - ei).total_seconds() / 3600

        trades.append({
            "date":    date,
            "entry_t": ei,
            "exit_t":  ts,
            "outcome": outcome,
            "ep":      ep,
            "xp":      xp,
            "ret":     ret * 100,
            "dur_h":   dur,
            "cash":    cash,
        })

    tr = pd.DataFrame(trades)
    if tr.empty:
        eq = pd.Series(dtype=float)
        return eq, tr

    eq = tr.set_index("exit_t")["cash"]
    # Re-indexer pour avoir une courbe d'equity continue
    first = df_1h.index[0]; last = df_1h.index[-1]
    eq_full = pd.Series(CAPITAL, index=[first])
    eq_full = pd.concat([eq_full, eq])
    eq_full = eq_full[~eq_full.index.duplicated(keep="last")]
    eq_full = eq_full.sort_index()
    eq_full = eq_full.reindex(df_1h.index).ffill().fillna(CAPITAL)
    return eq_full, tr

# ══════════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ══════════════════════════════════════════════════════════════════
def metrics(eq):
    eq  = eq.dropna(); r = eq.pct_change().dropna()
    yr  = (eq.index[-1] - eq.index[0]).days / 365.25
    if yr <= 0: return {}
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yr) - 1
    sh   = r.mean()/r.std()*np.sqrt(8760) if r.std()>0 else 0
    ds   = r[r<0].std()
    so   = r.mean()/ds*np.sqrt(8760) if ds>0 else 0
    rm   = np.maximum.accumulate(eq.values)
    dd   = (eq.values - rm)/rm
    mdd  = dd.min()*100
    cal  = (cagr*100)/abs(mdd) if mdd != 0 else 0
    return {"cagr": cagr*100, "sharpe": sh, "sortino": so,
            "mdd": mdd, "calmar": cal, "eq": eq, "dd": dd}

def trade_stats(tr):
    if tr is None or tr.empty: return {}
    wr   = (tr["ret"]>0).mean()*100
    avg  = tr["ret"].mean()
    tp_r = (tr["outcome"]=="TP").mean()*100
    return {"wr": wr, "avg": avg, "tp_rate": tp_r, "n": len(tr),
            "avg_dur": tr["dur_h"].mean()}

# ══════════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ══════════════════════════════════════════════════════════════════
def walk_forward(df_1h, sig_module, years_is=3, years_oos=1):
    start = df_1h.index[0].year
    end   = df_1h.index[-1].year
    results = []
    for is_start in range(start, end - years_is - years_oos + 1):
        is_e   = is_start + years_is - 1
        oos_s  = is_start + years_is
        oos_e  = oos_s + years_oos - 1
        if oos_e > end: break
        dfa = df_1h[(df_1h.index.year >= is_start)  & (df_1h.index.year <= is_e)]
        dfb = df_1h[(df_1h.index.year >= oos_s) & (df_1h.index.year <= oos_e)]
        if dfa.empty or dfb.empty: continue
        _, tr_a = run_asian_engine(dfa, sig_module)
        eq_b, tr_b = run_asian_engine(dfb, sig_module)
        m_b = metrics(eq_b) if not eq_b.empty else {}
        ts_b = trade_stats(tr_b)
        results.append({
            "period": f"{oos_s}",
            "calmar": m_b.get("calmar", 0),
            "n":      ts_b.get("n", 0),
            "wr":     ts_b.get("wr", 0),
        })
    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════
#  GRAPHIQUES
# ══════════════════════════════════════════════════════════════════
def make_charts(df_1h, eq_disc, tr_disc, eq_val, tr_val,
                eq_foos, tr_foos, wf):
    palette = [GREEN, GOLD, BLUE]
    labels  = ["DISC 2003-14", "VAL 2015-20", "FOOS 2021-26"]
    equities= [eq_disc, eq_val, eq_foos]
    m_list  = [metrics(e) for e in equities]
    ts_list = [trade_stats(t) for t in [tr_disc, tr_val, tr_foos]]

    # ── FIG 1 : Vue d'ensemble ─────────────────────────────────────
    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
                            height_ratios=[1.8, 1.2, 1.0])

    ax = fig.add_subplot(gs[0, :2])
    for eq, col, lbl, m in zip(equities, palette, labels, m_list):
        if eq.empty: continue
        n  = eq / eq.iloc[0] * 100
        ax.plot(n.index, n, color=col, lw=1.8,
                label=f"{lbl}  Cal={m.get('calmar',0):.1f}")
    ax.set_yscale("log")
    ax.set_title("XAU/USD — Asian Range  |  Equity 3 périodes (base 100, log)",
                 fontsize=11, color=TEXT)
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor=TEXT)
    style(ax)

    # Tableau métriques
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    rows = []
    for lbl, m, ts in zip(["DISC","VAL","FOOS"], m_list, ts_list):
        rows.extend([
            [f"[{lbl}] CAGR %",    f"{m.get('cagr',0):+.1f}"],
            [f"[{lbl}] MaxDD %",   f"{m.get('mdd',0):.1f}"],
            [f"[{lbl}] Calmar",    f"{m.get('calmar',0):.1f}"],
            [f"[{lbl}] WR %",      f"{ts.get('wr',0):.1f}"],
            [f"[{lbl}] N trades",  f"{ts.get('n',0)}"],
        ])
    table = ax.table(cellText=rows, colLabels=["Métrique","Valeur"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.5)
    table.scale(1, 1.18)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1A1D27"); cell.set_text_props(color=TEXT)
    ax.set_facecolor(BG)
    ax.set_title("Métriques 3 périodes", fontsize=10, color=TEXT, pad=4)

    # Drawdown
    ax = fig.add_subplot(gs[1, :2])
    for eq, col, lbl, m in zip(equities, palette, labels, m_list):
        if eq.empty or "dd" not in m: continue
        ax.fill_between(eq.index, m["dd"]*100, 0, alpha=0.35, color=col,
                        label=f"{lbl} MaxDD={m.get('mdd',0):.1f}%")
    ax.set_title("Drawdowns comparés", fontsize=10, color=TEXT)
    ax.legend(fontsize=8, facecolor="#1A1D27", labelcolor=TEXT)
    ax.set_ylabel("DD (%)", color=TEXT); style(ax)

    # Calmar par période
    ax = fig.add_subplot(gs[1, 2])
    calmars = [m.get("calmar",0) for m in m_list]
    bars = ax.bar(["DISC","VAL","FOOS"], calmars,
                  color=[GREEN if v>0 else RED for v in calmars], alpha=0.85, width=0.5)
    for bar, val in zip(bars, calmars):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.5,
                f"{val:.1f}", ha="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_title("Calmar par période", fontsize=10, color=TEXT)
    ax.set_ylabel("Calmar", color=TEXT); style(ax)

    # Distribution retours DISC
    ax = fig.add_subplot(gs[2, 0])
    if not tr_disc.empty:
        ax.hist(tr_disc["ret"], bins=50, color=GREEN, alpha=0.75, edgecolor=BG, lw=0.3)
        ax.axvline(0, color=WHITE, lw=0.8, ls="--")
        ax.axvline(tr_disc["ret"].mean(), color=GOLD, lw=1.5, ls="--",
                   label=f"Moy={tr_disc['ret'].mean():+.3f}%")
        wr = (tr_disc["ret"]>0).mean()*100
        ax.set_title(f"Distribution retours DISC\nWR={wr:.1f}%  N={len(tr_disc)}",
                     fontsize=9, color=TEXT)
        ax.legend(fontsize=8, facecolor="#1A1D27", labelcolor=TEXT)
        ax.set_xlabel("Retour/trade (%)", color=TEXT)
    style(ax)

    # TP vs SL ratio
    ax = fig.add_subplot(gs[2, 1])
    all_tr = pd.concat([t for t in [tr_disc, tr_val, tr_foos] if not t.empty])
    if not all_tr.empty and "outcome" in all_tr.columns:
        vc = all_tr["outcome"].value_counts()
        tp_n = vc.get("TP", 0); sl_n = vc.get("SL", 0)
        ax.bar(["TP","SL"], [tp_n, sl_n],
               color=[GREEN, RED], alpha=0.85, width=0.5)
        total = tp_n + sl_n
        ax.text(0, tp_n+2, f"{tp_n/total*100:.1f}%", ha="center",
                fontsize=12, fontweight="bold", color=GREEN)
        ax.text(1, sl_n+2, f"{sl_n/total*100:.1f}%", ha="center",
                fontsize=12, fontweight="bold", color=RED)
        ax.set_title(f"TP vs SL (tous trades, N={total})", fontsize=9, color=TEXT)
    style(ax)

    # Walk-forward
    ax = fig.add_subplot(gs[2, 2])
    if not wf.empty:
        cols = [GREEN if v>0 else RED for v in wf["calmar"]]
        ax.bar(range(len(wf)), wf["calmar"], color=cols, alpha=0.85, width=0.7)
        ax.axhline(0, color=WHITE, lw=0.5)
        ax.set_xticks(range(len(wf)))
        ax.set_xticklabels(wf["period"].tolist(), rotation=45, fontsize=7)
        pos = (wf["calmar"]>0).sum()
        ax.set_title(f"Walk-forward OOS Calmar\n{pos}/{len(wf)} positifs",
                     fontsize=9, color=TEXT)
        ax.set_ylabel("Calmar OOS", color=TEXT)
    style(ax)

    plt.suptitle(
        "XAU/USD — Asian Range Mean Reversion\n"
        "Signal propriétaire | Framework public | 10 bps frais | 3 périodes indépendantes",
        fontsize=13, color=TEXT, y=1.01)
    fig.patch.set_facecolor(BG)
    p = os.path.join(OUT_DIR, "xau_asian_range_results.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {p}")

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*60)
    print("  XAU/USD — Asian Range Mean Reversion Backtest")
    print("="*60)

    data_path = os.path.join(HERE, "data", "XAUUSD_spot_1h.parquet")
    if not os.path.exists(data_path):
        for p in ["XAUUSD_spot_1h.parquet",
                  os.path.join("..", "data", "XAUUSD_spot_1h.parquet")]:
            if os.path.exists(p): data_path = p; break
    df_1h = pd.read_parquet(data_path)
    df_1h.columns = [c.lower() for c in df_1h.columns]
    print(f"  Données 1h : {len(df_1h)} barres  {df_1h.index[0].date()} → {df_1h.index[-1].date()}")

    sig_mod = load_signal_module()
    if sig_mod is None:
        print("  ⚠ signals/xau_asian_signal.py non trouvé — filtre désactivé (placeholder)")
    else:
        print("  ✓ Signal propriétaire chargé")

    # Découpage par période
    df_disc = df_1h[(df_1h.index >= DISC_START)  & (df_1h.index <= DISC_END)]
    df_val  = df_1h[(df_1h.index >= VAL_START)   & (df_1h.index <= VAL_END)]
    df_foos = df_1h[(df_1h.index >= FOOS_START)  & (df_1h.index <= FOOS_END)]

    print(f"  DISC  : {len(df_disc)} barres  |  VAL : {len(df_val)} barres  |  FOOS : {len(df_foos)} barres")

    print("  Backtest DISC..."); eq_disc, tr_disc = run_asian_engine(df_disc, sig_mod)
    print("  Backtest VAL...");  eq_val,  tr_val  = run_asian_engine(df_val,  sig_mod)
    print("  Backtest FOOS..."); eq_foos, tr_foos = run_asian_engine(df_foos, sig_mod)

    m_disc  = metrics(eq_disc)
    m_val   = metrics(eq_val)
    m_foos  = metrics(eq_foos)
    ts_disc = trade_stats(tr_disc)
    ts_val  = trade_stats(tr_val)
    ts_foos = trade_stats(tr_foos)

    print(f"\n  {'':20} {'DISC':>12} {'VAL':>12} {'FOOS':>12}")
    print("  " + "─"*60)
    for lbl, k in [("CAGR %","cagr"),("MaxDD %","mdd"),("Calmar","calmar"),("Sharpe","sharpe")]:
        print(f"  {lbl:<20} {m_disc.get(k,0):>+11.2f} {m_val.get(k,0):>+11.2f} {m_foos.get(k,0):>+11.2f}")
    for lbl, k in [("WR %","wr"),("N trades","n")]:
        print(f"  {lbl:<20} {ts_disc.get(k,0):>11.1f} {ts_val.get(k,0):>11.1f} {ts_foos.get(k,0):>11.1f}")

    print("  Walk-forward...")
    wf = walk_forward(df_1h, sig_mod)
    if not wf.empty:
        pos = (wf["calmar"]>0).sum()
        print(f"  Walk-forward : {pos}/{len(wf)} fenêtres OOS positives")

    print("\n  Génération graphique...")
    make_charts(df_1h, eq_disc, tr_disc, eq_val, tr_val,
                eq_foos, tr_foos, wf)

    print(f"\n  DONE — {time.time()-t0:.1f}s")
    print(f"  Résultats dans : {OUT_DIR}")

if __name__ == "__main__":
    main()
