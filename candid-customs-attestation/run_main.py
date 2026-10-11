"""E2 and E3 under the single protocol, on both substrates.

Writes
  results/main_calibrated_reps.csv   one row per replication and policy
  results/main_full_reps.csv         same on the uncalibrated substrate
  results/main_calibrated_preds.npz  test-quarter predictions for bootstrap
"""
import os, sys, time
import numpy as np, pandas as pd
import evaluation as V

POLICIES = ["M0", "raw", "majority", "static", "chair_varshney",
            "beta_reputation", "truth_discovery", "candid", "candid_mixture",
            "raw+chair_varshney", "raw+candid", "raw+candid_mixture",
            "raw+beta_reputation"]


def main(n_cal=20, n_full=10):
    S = V.Substrate()
    for tag, cal, nrep in [("calibrated", True, n_cal), ("full", False, n_full)]:
        view = S.view(cal)
        rows, store = [], {}
        t0 = time.time()
        for r in range(1, nrep + 1):
            rr, pr = V.run_replication(S, view, r, POLICIES, keep_pred=cal)
            rows += rr
            if cal:
                store[r] = pr
            print(f"[{tag}] rep {r}/{nrep}  {time.time()-t0:6.0f}s  "
                  f"raw {rr[1]['auc']:.4f}  M0 {rr[0]['auc']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(V.RES, f"main_{tag}_reps.csv"), index=False)
        if cal:
            te = view["split"] == "test"
            np.savez_compressed(
                os.path.join(V.RES, "main_calibrated_preds.npz"),
                yt=view["y"][te], rt=view["rar"][te], rt_raw=view["rar_raw"][te],
                **{f"r{r}__{k}": v for r, d in store.items() for k, v in d.items()})


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
