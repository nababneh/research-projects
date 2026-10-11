"""Sensitivity of E2 and E3 to the attestation generator and to label budgets.

One factor is varied at a time around the baseline configuration. Each cell
uses five paired replications on the calibrated substrate.

Writes results/sensitivity_reps.csv
"""
import os, sys, time
import numpy as np, pandas as pd
import corpus as C
import evaluation as V

POL = ["M0", "raw", "chair_varshney", "candid", "candid_mixture",
       "beta_reputation", "raw+chair_varshney"]

GRID = (
    [("detect_scale", m) for m in (0.15, 0.3, 0.5, 0.75, 1.0, 1.15)] +
    [("false_alarm_scale", m) for m in (0.5, 2.0, 3.0)] +
    [("p_conceal", p) for p in (0.0, 0.25, 0.65, 0.85)] +
    [("blind_doc", b) for b in (0.5, 1.0)] +
    [("blind_ind", b) for b in (0.0, 0.5)] +
    [("train_frac", f) for f in (0.02, 0.05, 0.10, 0.25)] +
    [("audit_frac", a) for a in (0.01, 0.10)]
)


def cell(factor, level):
    gen, params, kw = {}, {}, {}
    if factor == "detect_scale":
        d = np.clip(C.DETECT * level, 0.01, 0.99)
        gen["detect"] = d; params["detect"] = d
    elif factor == "false_alarm_scale":
        q = np.clip(C.FALSE_ALARM * level, 0.01, 0.99)
        gen["false_alarm"] = q; params["false_alarm"] = q
    elif factor == "p_conceal":
        gen["p_conceal"] = level
    elif factor == "blind_doc":
        gen["blind_doc"] = level
    elif factor == "blind_ind":
        gen["blind_ind"] = level
    elif factor == "train_frac":
        kw["train_frac"] = level
    elif factor == "audit_frac":
        kw["audit_frac"] = level
    return gen, params, kw


def main(nrep=5):
    S = V.Substrate(); view = S.view(True)
    out = os.path.join(V.RES, "sensitivity_reps.csv")
    rows = []
    t0 = time.time()
    for factor, level in GRID:
        gen, params, kw = cell(factor, level)
        pol = POL if factor != "audit_frac" else ["raw", "candid", "candid_mixture", "beta_reputation"]
        for r in range(1, nrep + 1):
            rr, _ = V.run_replication(S, view, r, pol, gen=gen, params=params, **kw)
            for x in rr:
                x.update(factor=factor, level=level)
            rows += rr
        df = pd.DataFrame(rows); df.to_csv(out, index=False)
        cur = df[(df.factor == factor) & (df.level == level)].groupby("policy").auc.mean()
        print(f"{factor:18s} {level:5}  {time.time()-t0:6.0f}s  " +
              "  ".join(f"{k} {v:.3f}" for k, v in cur.items()), flush=True)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
