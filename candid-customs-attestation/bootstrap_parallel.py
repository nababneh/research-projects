"""Two-stage bootstrap summaries computed in parallel (same estimator as analyze.py)."""
import os, json, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
import evaluation as V
from analyze import NAMES, budget_curve, full_substrate

B = int(sys.argv[1]) if len(sys.argv) > 1 else 400
z = np.load(os.path.join(V.RES, "main_calibrated_preds.npz"))
YT, RT = z["yt"], z["rt"]
PRED = {}
for k in z.files:
    if "__" in k:
        r, p = k.split("__"); PRED.setdefault(int(r[1:]), {})[p] = z[k]


def pol_task(p):
    row = {"policy": p, "name": NAMES[p]}
    for m in ["auc", "ap", "recall@5", "revenue@5"]:
        pt, lo, hi = V.two_stage_bootstrap(PRED, YT, RT, p, None, m, B=B, seed=1)
        row[m] = pt; row[m + "_lo"] = lo; row[m + "_hi"] = hi
        if p != "raw":
            d, dlo, dhi = V.two_stage_bootstrap(PRED, YT, RT, p, "raw", m, B=B, seed=2)
            row["d_" + m] = d; row["d_" + m + "_lo"] = dlo; row["d_" + m + "_hi"] = dhi
    print("policy", p, flush=True)
    return row


def sel_task(m):
    r = {"metric": m}
    for p in ["M0", "raw"]:
        pt, lo, hi = V.two_stage_bootstrap(PRED, YT, RT, p, None, m, B=B, seed=3)
        r[p] = pt; r[p + "_lo"] = lo; r[p + "_hi"] = hi
    d, lo, hi = V.two_stage_bootstrap(PRED, YT, RT, "raw", "M0", m, B=B, seed=4)
    r["diff"] = d; r["diff_lo"] = lo; r["diff_hi"] = hi
    print("metric", m, flush=True)
    return r


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(V.RES, "main_calibrated_reps.csv"))
    pols = list(df.policy.unique())
    mets = ["auc", "ap", "recall@1", "recall@5", "recall@10", "precision@1", "revenue@5", "revenue@10"]
    with Pool(2) as pool:
        sel = pool.map(sel_task, mets)
        pd.DataFrame(sel).to_csv(os.path.join(V.RES, "summary_selectivity_calibrated.csv"), index=False)
        rows = pool.map(pol_task, pols)
    out = pd.DataFrame(rows)
    for p in pols:
        if p == "raw":
            continue
        for m in ["auc", "ap", "recall@5", "revenue@5"]:
            a = df[df.policy == p].set_index("rep")[m]; b = df[df.policy == "raw"].set_index("rep")[m]
            out.loc[out.policy == p, "wins_" + m] = int(((a - b) > 0).sum())
    out.to_csv(os.path.join(V.RES, "summary_policies_calibrated.csv"), index=False)
    extra = df[df.policy.isin(["M0", "raw"])].groupby("policy")[
        [c for c in df.columns if c.startswith("revenue_")]].mean()
    extra.to_csv(os.path.join(V.RES, "summary_revenue_sensitivity.csv"))
    srt = np.sort(RT)[::-1]
    json.dump(dict(test_n=int(len(YT)), test_fraud=int(YT.sum()), top1_share=float(srt[0] / RT.sum()),
                   top2_share=float(srt[:2].sum() / RT.sum()), top5_share=float(srt[:5].sum() / RT.sum())),
              open(os.path.join(V.RES, "summary_revenue_concentration.json"), "w"), indent=1)
    budget_curve(); full_substrate()
    print("done", flush=True)
