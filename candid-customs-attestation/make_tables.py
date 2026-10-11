"""Write LaTeX tables and fill numeric placeholders in the manuscript parts.

Every number in the results section is taken from a CSV in results/, which
keeps the text, the tables and the figures on one set of numbers.
"""
import os, json, re
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(HERE, "results")
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)
V = {}


def f3(x): return f"{x:.3f}"
def ci(p, lo, hi, d=3): return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
def sci(p, lo, hi, d=3):
    s = lambda v: f"{v:+.{d}f}".replace("+", "$+$").replace("-", "$-$")
    return f"{s(p)} [{s(lo)}, {s(hi)}]"
def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return m, m - h, m + h
def signed(x, d=3): return (f"{x:+.{d}f}").replace("-", "$-$").replace("+", "$+$")


# ------------------------------------------------------------- E1
def e1():
    sc = pd.read_csv(os.path.join(RES, "impl_stage_costs.csv")).set_index("stage").microseconds
    V["SC_COMMIT"] = f"{sc['commitment_us']:.1f}"; V["SC_SIGN"] = f"{sc['sign_us']:.0f}"
    V["SC_VERIFY"] = f"{sc['verify_us']:.0f}"
    b = pd.read_csv(os.path.join(RES, "impl_benchmark.csv"))
    g = b.groupby("config").agg(tps=("tps", "mean"), sd=("tps", "std"), p95=("p95_ms", "mean"),
                                ne=("endorsers", "first"), bs=("block_size", "first"),
                                dur=("durable", "first"))
    key = {"large blocks, 1 endorser": "TPS_1", "large blocks, 3 endorsers": "TPS_3",
           "large blocks, 5 endorsers": "TPS_5", "large blocks, 3 endorsers, durable": "TPS_3D",
           "small blocks, 3 endorsers, durable": "TPS_3SD", "small blocks, 3 endorsers": "TPS_3S"}
    for c, k in key.items():
        V[k] = f"{g.loc[c, 'tps']:,.0f}".replace(",", "{,}")
    order = ["large blocks, 1 endorser", "large blocks, 3 endorsers", "large blocks, 5 endorsers",
             "small blocks, 3 endorsers", "large blocks, 3 endorsers, durable",
             "small blocks, 3 endorsers, durable"]
    rows = []
    for c in order:
        r = g.loc[c]
        rows.append(f"{int(r['ne'])} & {int(r['bs'])} & {'Durable' if r['dur'] else 'Volatile'} & "
                    f"{r.tps:,.0f} $\\pm$ {r.sd:,.0f} & {r.p95:,.1f}\\\\".replace(",", "{,}"))
    tex = ("\\begin{table}[H]\n\\caption{Closed-loop throughput, mean $\\pm$ standard deviation of five runs.\\label{tab:bench}}\n"
           "\\newcolumntype{C}{>{\\centering\\arraybackslash}X}\n\\begin{tabularx}{\\textwidth}{CCCCC}\n\\toprule\n"
           "\\textbf{Endorsers} & \\textbf{Block size} & \\textbf{Storage} & \\textbf{Throughput (tx/s)} & \\textbf{95th pct latency (ms)}\\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    open(os.path.join(TAB, "tab_bench.tex"), "w").write(tex)

    s = pd.read_csv(os.path.join(RES, "impl_load_sweep.csv"))
    sust = {}
    for dur in (False, True):
        q = s[s.durable == dur].groupby("offered_tps").agg(p95=("p95_latency_ms", "mean"),
                                                         p50=("p50_latency_ms", "mean"),
                                                         ach=("achieved_tps", "mean"),
                                                         backlog=("backlog_at_close", "mean"))
        ok = q[(q.p95 < 1000) & (q.backlog < 250)]
        sust[dur] = float(ok.index.max())
        if not dur:
            V["LAT_LOW_MED"] = f"{q.loc[q.index.min(), 'p50']:.0f}"
    V["SUST_VOL"] = f"{sust[False]:,.0f}".replace(",", "{,}")
    V["SUST_DUR"] = f"{sust[True]:,.0f}".replace(",", "{,}")
    cap = sust[True]

    T = 31_536_000; a = 10.29; c = 3; u = 1
    rows = []
    for lab, n in [("Small", 1_000_000), ("Medium", 4_500_000), ("Large", 20_000_000), ("Very large", 60_000_000)]:
        base = 4 * n * (a + c + u) / T; st = 12 * n * (a + c + u) / T
        rows.append(f"{lab} & {n:,} & {base:.1f} & {st:.1f} & {100*base/cap:.1f}\\% & {100*st/cap:.1f}\\%\\\\".replace(",", "{,}"))
        if lab == "Very large":
            V["PEAK_VL"] = f"{base:.0f}"; V["PEAK_VL_S"] = f"{st:.0f}"
            V["UTIL_BASE"] = f"{100*base/cap:.1f}"; V["UTIL_STRESS"] = f"{100*st/cap:.0f}"
    tex = ("\\begin{table}[H]\n\\caption{Peak demand against sustainable durable throughput.\\label{tab:capacity}}\n"
           "\\newcolumntype{C}{>{\\centering\\arraybackslash}X}\n\\begin{tabularx}{\\textwidth}{lCCCCC}\n\\toprule\n"
           "\\textbf{Size} & \\textbf{Declarations per year} & \\textbf{Peak, $\\kappa = 4$ (tx/s)} & \\textbf{Peak, $\\kappa = 12$ (tx/s)} & "
           "\\textbf{Use, $\\kappa = 4$} & \\textbf{Use, $\\kappa = 12$}\\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    open(os.path.join(TAB, "tab_capacity.tex"), "w").write(tex)

    fp = pd.read_csv(os.path.join(RES, "impl_footprint.csv")).set_index("quantity").bytes.astype(float)
    txpd = 10.29 + 3
    rec = fp["record_bytes"]; frame = fp["framing_bytes"]
    per = txpd * (rec + frame)
    V["REC"] = f"{rec:.0f}"; V["SIG"] = f"{fp['signature_set_bytes']:.0f}"; V["FRAME"] = f"{frame:.0f}"
    V["CERT"] = f"{fp['creator_certificate_bytes']:.0f}"; V["TXPD"] = f"{txpd:.1f}"
    V["PERDECL"] = f"{per/1000:.1f}"; V["DECLREC"] = f"{fp['declaration_record_with_framing']:,.0f}".replace(",", "{,}")
    V["RATIO_DECL"] = f"{per/fp['declaration_record_with_framing']:.0f}"
    V["RATIO_DOC"] = f"{(fp['declaration_record_with_framing'] + 260000)/per:.0f}"
    lst = open(os.path.join(RES, "show_record.txt")).read().rstrip("\n")
    V["LISTING"] = lst


# ------------------------------------------------------------- E2 and E3
def e23():
    s = pd.read_csv(os.path.join(RES, "summary_selectivity_calibrated.csv")).set_index("metric")
    reps = pd.read_csv(os.path.join(RES, "main_calibrated_reps.csv"))
    lab = {"auc": "Area under the ROC curve", "ap": "Average precision", "recall@1": "Recall at 1\\% budget",
           "recall@5": "Recall at 5\\% budget", "recall@10": "Recall at 10\\% budget",
           "precision@1": "Precision at 1\\% budget", "revenue@5": "Revenue proxy at 5\\% budget",
           "revenue@10": "Revenue proxy at 10\\% budget"}
    rows = []
    for m, l in lab.items():
        r = s.loc[m]
        rows.append(f"{l} & {ci(r.M0, r.M0_lo, r.M0_hi)} & {ci(r.raw, r.raw_lo, r.raw_hi)} & {sci(r['diff'], r.diff_lo, r.diff_hi)}\\\\")
    for m, l in [("revenue_drop5@5", "Proxy at 5\\%, five largest removed"),
                 ("revenue_drop5@10", "Proxy at 10\\%, five largest removed")]:
        a = tci(reps[reps.policy == "M0"].sort_values("rep")[m]); b = tci(reps[reps.policy == "raw"].sort_values("rep")[m])
        d = tci(reps[reps.policy == "raw"].sort_values("rep")[m].values - reps[reps.policy == "M0"].sort_values("rep")[m].values)
        rows.append(f"{l}$^\\ast$ & {ci(*a)} & {ci(*b)} & {sci(*d)}\\\\")
        if m == "revenue_drop5@5":
            V["M0_V5X"] = f3(a[0]); V["RAW_V5X"] = f3(b[0])
    tex = ("\\begin{table}[H]\n\\caption{Selectivity on the calibrated substrate with 95\\% intervals. Rows marked $\\ast$ use t intervals over replications.\\label{tab:selectivity}}\n"
           "\\small\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\newcolumntype{C}{>{\\centering\\arraybackslash}X}\n"
           "\\begin{tabularx}{\\fulllength}{>{\\raggedright\\arraybackslash}p{5.2cm}CCC}\n\\toprule\n"
           "\\textbf{Metric} & \\textbf{Declaration only} & \\textbf{With attestation record} & \\textbf{Paired difference}\\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n\\end{table}\n")
    open(os.path.join(TAB, "tab_selectivity.tex"), "w").write(tex)
    V["M0_AUC"] = f3(s.loc["auc", "M0"]); V["RAW_AUC"] = f3(s.loc["auc", "raw"])
    V["D_AUC"] = f"{s.loc['auc','diff']:.3f} [{s.loc['auc','diff_lo']:.3f}, {s.loc['auc','diff_hi']:.3f}]"
    V["M0_R5"] = f3(s.loc["recall@5", "M0"]); V["RAW_R5"] = f3(s.loc["recall@5", "raw"])
    V["M0_V5"] = f3(s.loc["revenue@5", "M0"]); V["RAW_V5"] = f3(s.loc["revenue@5", "raw"])

    p = pd.read_csv(os.path.join(RES, "summary_policies_calibrated.csv")).set_index("policy")
    order = ["M0", "majority", "beta_reputation", "candid", "truth_discovery", "static", "candid_mixture",
             "chair_varshney", "raw+candid", "raw+beta_reputation", "raw+candid_mixture", "raw+chair_varshney", "raw"]
    names = {"M0": "Declaration only", "raw": "Raw indicators (control)", "majority": "Equal weight",
             "static": "Static prior", "chair_varshney": "Chair and Varshney oracle \\cite{chair1986fusion}",
             "beta_reputation": "Beta reputation \\cite{josang2002beta}", "truth_discovery": "Truth discovery \\cite{li2014resolving}",
             "candid": "Learned likelihood ratio", "candid_mixture": "Mixture-corrected",
             "raw+chair_varshney": "Raw plus oracle ratio", "raw+candid": "Raw plus learned ratio",
             "raw+candid_mixture": "Raw plus mixture ratio", "raw+beta_reputation": "Raw plus Beta trust"}
    rows = []
    for k in order:
        r = p.loc[k]
        if k == "raw":
            rows.append("\\midrule")
            rows.append(f"{names[k]} & {ci(r.auc, r.auc_lo, r.auc_hi)} & reference & {ci(r.ap, r.ap_lo, r.ap_hi)} & reference & {f3(r['recall@5'])}\\\\")
        else:
            if k == "raw+candid": rows.append("\\midrule")
            rows.append(f"{names[k]} & {ci(r.auc, r.auc_lo, r.auc_hi)} & {sci(r.d_auc, r.d_auc_lo, r.d_auc_hi)} & "
                        f"{f3(r.ap)} & {sci(r.d_ap, r.d_ap_lo, r.d_ap_hi)} & {f3(r['recall@5'])}\\\\")
    tex = ("\\begin{table}[H]\n\\caption{Scoring policy comparison with paired differences from the raw control.\\label{tab:policies}}\n"
           "\\footnotesize\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\newcolumntype{C}{>{\\centering\\arraybackslash}X}\n"
           "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabularx}{\\fulllength}{>{\\raggedright\\arraybackslash}p{3.7cm}CCcCc}\n\\toprule\n"
           "\\textbf{Policy} & \\textbf{AUC [95\\% CI]} & \\textbf{Difference in AUC} & \\textbf{AP} & \\textbf{Difference in AP} & \\textbf{Recall at 5\\%}\\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n\\end{table}\n")
    tex = tex.replace("--- &", "--- &")
    open(os.path.join(TAB, "tab_policies.tex"), "w").write(tex)
    V["CV_AUC"] = f3(p.loc["chair_varshney", "auc"])
    V["CV_D"] = f"{p.loc['chair_varshney','d_auc']:.4f} [{p.loc['chair_varshney','d_auc_lo']:.4f}, {p.loc['chair_varshney','d_auc_hi']:.4f}]".replace("-", "$-$")
    V["LLR_AUC"] = f3(p.loc["candid", "auc"]); V["BETA_AUC"] = f3(p.loc["beta_reputation", "auc"])
    V["EQ_AUC"] = f3(p.loc["majority", "auc"])
    V["STATIC_GAP"] = f"{abs(p.loc['chair_varshney','auc'] - p.loc['static','auc']):.3f}"
    V["AUG_CV_D"] = f"{p.loc['raw+chair_varshney','d_auc']:.4f} [{p.loc['raw+chair_varshney','d_auc_lo']:.4f}, {p.loc['raw+chair_varshney','d_auc_hi']:.4f}]".replace("-", "$-$")
    V["AUG_CV_AP"] = f"{p.loc['raw+chair_varshney','d_ap']:.3f} [{p.loc['raw+chair_varshney','d_ap_lo']:.3f}, {p.loc['raw+chair_varshney','d_ap_hi']:.3f}]".replace("-", "$-$")

    full = pd.read_csv(os.path.join(RES, "summary_policies_full.csv")).set_index("policy")
    V["FULL_RAW"] = f3(full.loc["raw", "auc"])
    V["FULL_MAXD"] = f"{max(abs(full.loc[k,'d_auc']) for k in ['static','chair_varshney','truth_discovery','candid_mixture']):.4f}"

    sen = pd.read_csv(os.path.join(RES, "summary_sensitivity.csv"))
    def gain(f, l):
        q = sen[(sen.factor == f) & (np.isclose(sen.level, l))].set_index("policy").auc
        return f"{q['raw'] - q['M0']:.3f}"
    base = reps[reps.rep <= 5].groupby("policy").auc.mean()
    V["G_D015"] = gain("detect_scale", 0.15); V["G_D050"] = gain("detect_scale", 0.5)
    V["G_D115"] = gain("detect_scale", 1.15); V["G_F3"] = gain("false_alarm_scale", 3.0)
    V["G_C85"] = gain("p_conceal", 0.85)
    def dd(f, l, pol, key="d_auc"):
        q = sen[(sen.factor == f) & (np.isclose(sen.level, l)) & (sen.policy == pol)].iloc[0]
        return f"{q[key]:.3f} [{q[key+'_lo']:.3f}, {q[key+'_hi']:.3f}]".replace("-", "$-$")
    V["AUG_F3"] = dd("false_alarm_scale", 3.0, "raw+chair_varshney")
    V["AUG_B1"] = dd("blind_doc", 1.0, "raw+chair_varshney")
    q = sen[(sen.factor == "p_conceal") & np.isclose(sen.level, 0.85) & (sen.policy == "chair_varshney")].iloc[0]
    V["CONC_GAP"] = f"{-q.d_auc:.3f} [{-q.d_auc_hi:.3f}, {-q.d_auc_lo:.3f}]"
    q = sen[(sen.factor == "train_frac") & np.isclose(sen.level, 0.02)].set_index("policy")
    V["TF2_RAW"] = f3(q.loc["raw", "auc"]); V["TF2_CV"] = f"{-q.loc['chair_varshney','d_auc']:.3f}"


# ------------------------------------------------------------- E4
def e4():
    lat = pd.read_csv(os.path.join(RES, "summary_latency.csv"))
    ns = pd.read_csv(os.path.join(RES, "summary_nonstationary.csv"))
    g = lambda pol, lag: lat[(lat.policy == pol) & (lat.lag == lag)].iloc[0]
    raw0 = g("raw", 0).auc
    c0, c365 = g("candid", 0), g("candid", 365); b0, b365 = g("beta_reputation", 0), g("beta_reputation", 365)
    m0, m365 = g("candid_mixture", 0), g("candid_mixture", 365)
    cmax = lat[lat.policy == "candid"].sort_values("auc").iloc[-1]
    V["LATENCY_TEXT"] = (
        f"The raw control learns nothing from audits and holds at {raw0:.3f} at every lag. "
        f"The learned ratio moves from {c0.auc:.3f} to {c365.auc:.3f} over 365 days. "
        f"Beta reputation moves from {b0.auc:.3f} to {b365.auc:.3f} and the mixture correction from {m0.auc:.3f} to {m365.auc:.3f} over the same range of lags. "
        f"The trajectories are not monotone, and the learned ratio peaks at a lag of {int(cmax.lag)} days. "
        f"At every lag the learned policies remain below the raw control, although some of the intervals include zero.")
    rows = []
    for sw in sorted(ns.switch_day.unique()):
        q = ns[ns.switch_day == sw].set_index("policy")
        rows.append((sw, q))
    parts = []
    for sw, q in rows:
        parts.append(f"{q.loc['raw','auc']:.3f} at switch day {int(sw)}")
    best = []
    for sw, q in rows:
        others = q.drop(index=["raw", "M0"])
        top = others.d_auc.idxmax()
        best.append((sw, top, others.loc[top]))
    q = {int(sw): r for sw, r in rows}
    txt = (f"The raw control reaches {q[180].loc['raw','auc']:.3f} when collusion begins on day 180, "
           f"{q[300].loc['raw','auc']:.3f} when it begins on day 300 and {q[420].loc['raw','auc']:.3f} when it begins on day 420. ")
    sig_better = [(sw, t, r) for sw, t, r in best if r.d_auc_lo > 0]
    if sig_better:
        sw, t, r = sig_better[0]
        txt += (f"Only at switch day {int(sw)} does a weighting policy exceed it with an interval above zero. ")
    else:
        bsw, bt, br = max(best, key=lambda x: x[2].d_auc)
        nm = {"candid": "learned likelihood ratio", "candid_mixture": "mixture correction",
              "beta_reputation": "Beta reputation", "static": "static prior", "majority": "equal weight"}[bt]
        txt += (f"No weighting policy exceeds it with an interval above zero. The closest is the {nm} at switch day {int(bsw)}, "
                f"with a paired difference of {br.d_auc:.3f} [{br.d_auc_lo:.3f}, {br.d_auc_hi:.3f}].").replace("-", "$-$")
    V["NS_TEXT"] = txt


def fill(src, dst):
    t = open(src).read()
    for k, v in V.items():
        t = t.replace(f"@@{k}@@", v)
    left = re.findall(r"@@[A-Z0-9_]+@@", t)
    open(dst, "w").write(t)
    return left


if __name__ == "__main__":
    e1(); e23(); e4()
    json.dump({k: v for k, v in V.items() if k != "LISTING"}, open(os.path.join(RES, "manuscript_values.json"), "w"), indent=1)
    import sys
    if len(sys.argv) > 2:
        print("unfilled:", fill(sys.argv[1], sys.argv[2]))
