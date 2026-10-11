"""E4: audit latency and non-stationary compromise, ten paired replications.

Writes results/latency_reps.csv and results/nonstationary_reps.csv
"""
import os, sys, time
import pandas as pd
import evaluation as V

LAGS = (0, 30, 90, 180, 365)
SWITCH = (180, 300, 420)
COMPROMISED = [0, 5]      # partner export record and port community system


def main(nrep=10):
    S = V.Substrate(); view = S.view(True)
    t0 = time.time()
    rows = []
    for lag in LAGS:
        for r in range(1, nrep + 1):
            rr, _ = V.run_replication(S, view, r,
                                      ["raw", "candid", "candid_mixture", "beta_reputation"],
                                      audit_lag_days=lag)
            for x in rr: x.update(lag=lag)
            rows += rr
        pd.DataFrame(rows).to_csv(os.path.join(V.RES, "latency_reps.csv"), index=False)
        print(f"lag {lag}  {time.time()-t0:5.0f}s", flush=True)
    rows = []
    for sw in SWITCH:
        gen = dict(compromised=COMPROMISED, switch_day=sw)
        for r in range(1, nrep + 1):
            rr, _ = V.run_replication(S, view, r,
                                      ["M0", "raw", "majority", "static", "beta_reputation",
                                       "candid", "candid_mixture"], gen=gen)
            for x in rr: x.update(switch_day=sw)
            rows += rr
        pd.DataFrame(rows).to_csv(os.path.join(V.RES, "nonstationary_reps.csv"), index=False)
        print(f"switch {sw}  {time.time()-t0:5.0f}s", flush=True)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
