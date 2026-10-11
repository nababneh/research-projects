"""Summaries, intervals and table values for the revised manuscript.

Reads the per-replication CSVs written by run_main.py, run_sensitivity.py and
run_robustness.py and writes summary CSVs into results/.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
import evaluation as V

RES = V.RES
NAMES = {"M0": "Declaration only", "raw": "Raw indicators", "majority": "Equal weight",
         "static": "Static prior", "chair_varshney": "Chair and Varshney oracle",
         "beta_reputation": "Beta reputation", "truth_discovery": "Truth discovery",
         "candid": "Learned likelihood ratio", "candid_mixture": "Mixture-corrected",
         "raw+chair_varshney": "Raw plus oracle ratio", "raw+candid": "Raw plus learned ratio",
         "raw+candid_mixture": "Raw plus mixture ratio", "raw+beta_reputation": "Raw plus Beta trust"}


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def main_tables(B=500):
    df = pd.read_csv(os.path.join(RES, "main_calibrated_reps.csv"))
    z = np.load(os.path.join(RES, "main_calibrated_preds.npz"))
    yt, rt = z["yt"], z["rt"]
    preds = {}
    for k in z.files:
        if "__" in k:
            r, p = k.split("__"); preds.setdefault(int(r[1:]), {})[p] = z[k]
    rows = []
    metrics = ["auc", "ap", "recall@5", "revenue@5"]
    for p in df.policy.unique():
        row = {"policy": p, "name": NAMES[p]}
        for m in metrics:
            pt, lo, hi = V.two_stage_bootstrap(preds, yt, rt, p, None, m, B=B, seed=1)
            row[f"{m}"] = pt; row[f"{m}_lo"] = lo; row[f"{m}_hi"] = hi
            if p != "raw":
                d, dlo, dhi = V.two_stage_bootstrap(preds, yt, rt, p, "raw", m, B=B, seed=2)
                row[f"d_{m}"] = d; row[f"d_{m}_lo"] = dlo; row[f"d_{m}_hi"] = dhi
                a = df[df.policy == p].set_index("rep")[m]; b = df[df.policy == "raw"].set_index("rep")[m]
                row[f"wins_{m}"] = int(((a - b) > 0).sum())
        rows.append(row)
        print(p, {k: round(v, 4) for k, v in row.items() if isinstance(v, float) and not k.endswith(("lo", "hi"))}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "summary_policies_calibrated.csv"), index=False)
    return out


def selectivity_table(B=500):
    """M0 against raw, every budget metric, with bootstrap intervals."""
    df = pd.read_csv(os.path.join(RES, "main_calibrated_reps.csv"))
    z = np.load(os.path.join(RES, "main_calibrated_preds.npz"))
    yt, rt, rtr = z["yt"], z["rt"], z["rt_raw"]
    preds = {}
    for k in z.files:
        if "__" in k:
            r, p = k.split("__"); preds.setdefault(int(r[1:]), {})[p] = z[k]
    rows = []
    for m in ["auc", "ap", "recall@1", "recall@2", "recall@5", "recall@10", "precision@1",
              "precision@5", "revenue@1", "revenue@2", "revenue@5", "revenue@10"]:
        r = {"metric": m}
        for p in ["M0", "raw"]:
            pt, lo, hi = V.two_stage_bootstrap(preds, yt, rt, p, None, m, B=B, seed=3)
            r[p] = pt; r[p + "_lo"] = lo; r[p + "_hi"] = hi
        d, lo, hi = V.two_stage_bootstrap(preds, yt, rt, "raw", "M0", m, B=B, seed=4)
        r["diff"] = d; r["diff_lo"] = lo; r["diff_hi"] = hi
        rows.append(r); print(r, flush=True)
    # descriptive revenue sensitivity, means over replications
    extra = df[df.policy.isin(["M0", "raw"])].groupby("policy")[
        [c for c in df.columns if c.startswith("revenue_")]].mean()
    extra.to_csv(os.path.join(RES, "summary_revenue_sensitivity.csv"))
    srt = np.sort(rt)[::-1]
    conc = dict(test_n=int(len(yt)), test_fraud=int(yt.sum()),
                top1_share=float(srt[0] / rt.sum()), top2_share=float(srt[:2].sum() / rt.sum()),
                top5_share=float(srt[:5].sum() / rt.sum()),
                n_capped_test=int((rt >= rt.max() - 1e-6).sum()))
    json.dump(conc, open(os.path.join(RES, "summary_revenue_concentration.json"), "w"), indent=1)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "summary_selectivity_calibrated.csv"), index=False)
    print(extra.round(4).T); print(conc)
    return out


def budget_curve():
    """Mean and t interval across replications at each budget, M0 and raw."""
    z = np.load(os.path.join(RES, "main_calibrated_preds.npz"))
    yt, rt = z["yt"], z["rt"]
    reps = sorted({int(k.split("__")[0][1:]) for k in z.files if "__" in k})
    rows = []
    for b in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]:
        for p in ["M0", "raw"]:
            rec, rev = [], []
            for r in reps:
                pr = z[f"r{r}__{p}"]; o = np.argsort(-pr, kind="stable")[:max(int(len(yt) * b), 1)]
                rec.append(yt[o].sum() / yt.sum()); rev.append(rt[o].sum() / rt.sum())
            a = tci(rec); c = tci(rev)
            rows.append(dict(budget=b, policy=p, recall=a[0], recall_lo=a[1], recall_hi=a[2],
                             revenue=c[0], revenue_lo=c[1], revenue_hi=c[2]))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "summary_budget_curve.csv"), index=False)
    return out


def full_substrate():
    df = pd.read_csv(os.path.join(RES, "main_full_reps.csv"))
    rows = []
    for p in df.policy.unique():
        a = tci(df[df.policy == p].auc); ap = tci(df[df.policy == p].ap)
        r = dict(policy=p, name=NAMES[p], auc=a[0], auc_lo=a[1], auc_hi=a[2], ap=ap[0])
        if p != "raw":
            d = (df[df.policy == p].set_index("rep").auc - df[df.policy == "raw"].set_index("rep").auc)
            m = tci(d); r.update(d_auc=m[0], d_auc_lo=m[1], d_auc_hi=m[2], wins=int((d > 0).sum()))
        rows.append(r)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "summary_policies_full.csv"), index=False)
    print(out.round(4).to_string())
    return out


def sensitivity():
    df = pd.read_csv(os.path.join(RES, "sensitivity_reps.csv"))
    base = pd.read_csv(os.path.join(RES, "main_calibrated_reps.csv"))
    base = base[base.rep <= 5].copy()
    rows = []
    def summarise(sub, factor, level):
        g = sub.groupby("policy")
        for p, s in g:
            r = dict(factor=factor, level=level, policy=p, auc=s.auc.mean(), ap=s.ap.mean(),
                     recall5=s["recall@5"].mean())
            if p != "raw" and "raw" in sub.policy.values:
                d = (s.set_index("rep").auc - sub[sub.policy == "raw"].set_index("rep").auc).dropna()
                m = tci(d); r.update(d_auc=m[0], d_auc_lo=m[1], d_auc_hi=m[2])
                dap = (s.set_index("rep").ap - sub[sub.policy == "raw"].set_index("rep").ap).dropna()
                m2 = tci(dap); r.update(d_ap=m2[0], d_ap_lo=m2[1], d_ap_hi=m2[2])
            rows.append(r)
    for (f, l), sub in df.groupby(["factor", "level"]):
        summarise(sub, f, l)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "summary_sensitivity.csv"), index=False)
    return out


def robustness():
    out = {}
    lat = pd.read_csv(os.path.join(RES, "latency_reps.csv"))
    rows = []
    for (lag, p), s in lat.groupby(["lag", "policy"]):
        a = tci(s.auc); r = dict(lag=lag, policy=p, auc=a[0], auc_lo=a[1], auc_hi=a[2])
        if p != "raw":
            d = s.set_index("rep").auc - lat[(lat.lag == lag) & (lat.policy == "raw")].set_index("rep").auc
            m = tci(d); r.update(d_auc=m[0], d_auc_lo=m[1], d_auc_hi=m[2])
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "summary_latency.csv"), index=False)
    ns = pd.read_csv(os.path.join(RES, "nonstationary_reps.csv"))
    rows = []
    for (sw, p), s in ns.groupby(["switch_day", "policy"]):
        a = tci(s.auc); r = dict(switch_day=sw, policy=p, auc=a[0], auc_lo=a[1], auc_hi=a[2])
        if p != "raw":
            d = s.set_index("rep").auc - ns[(ns.switch_day == sw) & (ns.policy == "raw")].set_index("rep").auc
            m = tci(d); r.update(d_auc=m[0], d_auc_lo=m[1], d_auc_hi=m[2])
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "summary_nonstationary.csv"), index=False)


if __name__ == "__main__":
    what = sys.argv[1:] or ["main", "selectivity", "curve", "full"]
    if "main" in what: main_tables()
    if "selectivity" in what: selectivity_table()
    if "curve" in what: budget_curve()
    if "full" in what: full_substrate()
    if "sensitivity" in what: sensitivity()
    if "robustness" in what: robustness()
