"""
Single evaluation protocol used for every selectivity number in the revised
manuscript.

Protocol
  substrate      Customs Import Declaration Dataset (Jeong et al., 2022),
                 published temporal split (12 months train, 3 months
                 validation, 3 months test). Train and validation are pooled
                 for fitting and the test quarter is held out.
  calibration    all non-fraud records are kept and fraud records are
                 subsampled once (seed 11) to a 3 per cent base rate.
  replication    replication r draws (i) the synthetic attestation layer with
                 seed r, (ii) the audited subset used by the learned policies
                 with seed r, and (iii) the classifier hold-out with seed r.
                 Every policy sees the same draw within a replication, which
                 makes the comparison paired.
  estimator      the metric is computed for each replication and averaged.
                 No ensembling of predictions is used anywhere.
  uncertainty    95 per cent intervals come from a two-stage bootstrap that
                 resamples replications and then test records, stratified by
                 label. Intervals for a policy difference are paired.
  revenue        a duty-at-risk proxy. The dataset records the assessed item
                 price (KRW) and the tax rate but not the duty actually
                 recovered, so for a fraudulent record r_i = price_i x
                 rate_i / 100 and r_i = 0 otherwise. r_i is winsorised at the
                 99th percentile of its positive values in the calibrated
                 sample. Revenue at budget B is the share of the test-quarter
                 total captured by the top B fraction of the ranking.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import corpus as C
import estimators as E
from experiments import declaration_features

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

BUDGETS = (0.01, 0.02, 0.05, 0.10)
TARGET_RATE = 0.03
GB = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=48,
          l2_regularization=1.0)


def calibrate(d, rate=TARGET_RATE, seed=11):
    rng = np.random.default_rng(seed)
    y = d["Fraud"].values.astype(bool)
    nf = np.flatnonzero(~y); fr = np.flatnonzero(y)
    keep = int(round(len(nf) * rate / (1 - rate)))
    return np.sort(np.concatenate([nf, rng.choice(fr, min(keep, len(fr)), replace=False)]))


class Substrate:
    """Loads the dataset once and exposes the full and calibrated views."""

    def __init__(self):
        self.d_full = C.load_substrate()
        self.X0_full = declaration_features(self.d_full)
        self.sel = calibrate(self.d_full)

    def view(self, calibrated=True):
        if calibrated:
            idx = self.sel
        else:
            idx = np.arange(len(self.d_full))
        d = self.d_full.iloc[idx].reset_index(drop=True)
        y = d["Fraud"].values.astype(int)
        price = d["Item Price"].values.astype(float)
        rate = d["Tax Rate"].values.astype(float)
        rar_raw = price * rate / 100.0 * y
        pos = rar_raw[rar_raw > 0]
        cap = np.percentile(pos, 99) if len(pos) else 1.0
        return dict(idx=idx, d=d, X0=self.X0_full[idx], y=y,
                    split=d["split"].values, day=d["day"].values,
                    rar=np.minimum(rar_raw, cap), rar_raw=rar_raw, cap=cap)


def fit_predict(X, y, split, seed, train_frac=1.0):
    tr = np.flatnonzero(split != "test"); te = split == "test"
    if train_frac < 1.0:
        rng = np.random.default_rng(1000 + seed)
        tr = np.sort(rng.choice(tr, int(round(len(tr) * train_frac)), replace=False))
    clf = HistGradientBoostingClassifier(random_state=seed, **GB)
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X[te])[:, 1]


def metrics(p, yt, rt, rt_raw=None, budgets=BUDGETS, drop_top=(1, 5)):
    o = np.argsort(-p, kind="stable")
    out = {"auc": roc_auc_score(yt, p), "ap": average_precision_score(yt, p)}
    n = len(yt)
    for b in budgets:
        m = max(int(n * b), 1); s = o[:m]; t = f"{b*100:g}"
        out[f"precision@{t}"] = float(yt[s].mean())
        out[f"recall@{t}"] = float(yt[s].sum() / max(yt.sum(), 1))
        out[f"revenue@{t}"] = float(rt[s].sum() / max(rt.sum(), 1e-12))
        if rt_raw is not None:
            out[f"revenue_unw@{t}"] = float(rt_raw[s].sum() / max(rt_raw.sum(), 1e-12))
    # sensitivity of the revenue proxy to its largest contributions: the k
    # largest test consignments are removed from the test quarter entirely
    for k in drop_top:
        big = np.argsort(-rt, kind="stable")[:k]
        keep = np.ones(n, bool); keep[big] = False
        pk, yk, rk = p[keep], yt[keep], rt[keep]
        ok = np.argsort(-pk, kind="stable")
        for b in (0.05, 0.10):
            m = max(int(len(yk) * b), 1)
            out[f"revenue_drop{k}@{b*100:g}"] = float(rk[ok[:m]].sum() / max(rk.sum(), 1e-12))
    return out


def policy_features(name, agree, present, day, ff, params=None, audit_seed=3,
                    audit_lag_days=0, audit_frac=0.03):
    params = params or {}
    if name == "raw":
        return E.features_raw(agree, present, day)
    if name.startswith("raw+"):
        base = E.features_raw(agree, present, day)
        extra = policy_features(name[4:], agree, present, day, ff, params,
                                audit_seed, audit_lag_days, audit_frac)
        return np.column_stack([base, extra])
    fn = E.POLICIES[name]
    return fn(agree, present, day, false_field=ff, seed=audit_seed,
              audit_lag_days=audit_lag_days, audit_frac=audit_frac,
              detect=params.get("detect"), false_alarm=params.get("false_alarm"))


def run_replication(S, view, r, policies, gen=None, params=None, train_frac=1.0,
                    audit_lag_days=0, audit_frac=0.03, keep_pred=False):
    """One replication: draw the attestation layer and score every policy."""
    gen = gen or {}
    agree_f, present_f, ff_f, conc_f = C.build_attestations(S.d_full, seed=r, **gen)
    idx = view["idx"]
    agree, present, ff = agree_f[idx], present_f[idx], ff_f[idx]
    y, split, day = view["y"], view["split"], view["day"]
    te = split == "test"
    yt, rt, rtr = y[te], view["rar"][te], view["rar_raw"][te]
    rows, preds = [], {}
    for name in policies:
        if name == "M0":
            X = view["X0"]; nx = 0
        else:
            F = policy_features(name, agree, present, day, ff, params,
                                audit_seed=r, audit_lag_days=audit_lag_days,
                                audit_frac=audit_frac)
            X = np.column_stack([view["X0"], F]); nx = F.shape[1]
        p = fit_predict(X, y, split, seed=r, train_frac=train_frac)
        m = metrics(p, yt, rt, rtr)
        m.update(policy=name, rep=r, n_extra=nx)
        rows.append(m)
        if keep_pred:
            preds[name] = p.astype(np.float32)
    return rows, preds


# ---------------------------------------------------------------- bootstrap

def _auc_fast(y, p):
    from scipy.stats import rankdata
    ranks = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _metric_fn(name):
    if name == "auc":
        return lambda y, p, r: _auc_fast(y, p)
    if name == "ap":
        def ap(y, p, r):
            o = np.argsort(-p, kind="stable"); yy = y[o]
            tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
            return float((tp / k)[yy == 1].sum() / max(yy.sum(), 1))
        return ap
    kind, b = name.split("@"); b = float(b) / 100.0
    def f(y, p, r):
        m = max(int(len(y) * b), 1); s = np.argsort(-p, kind="stable")[:m]
        if kind == "recall":
            return y[s].sum() / max(y.sum(), 1)
        if kind == "precision":
            return y[s].mean()
        if kind == "revenue":
            return r[s].sum() / max(r.sum(), 1e-12)
        raise ValueError(name)
    return f


def two_stage_bootstrap(preds, yt, rt, a, b=None, metric="auc", B=1000, seed=0):
    """Mean over replications of metric(a) or of metric(a) minus metric(b).

    preds maps replication -> {policy: test predictions}. Replications are
    resampled with replacement, then test records stratified by label.
    Returns (point, lo, hi).
    """
    rng = np.random.default_rng(seed)
    f = _metric_fn(metric)
    reps = sorted(preds)
    def stat(rep_idx, rec):
        vals = []
        for rr in rep_idx:
            pa = preds[rr][a][rec]
            v = f(yt[rec], pa, rt[rec])
            if b is not None:
                v -= f(yt[rec], preds[rr][b][rec], rt[rec])
            vals.append(v)
        return float(np.mean(vals))
    allrec = np.arange(len(yt))
    point = stat(reps, allrec)
    pos = np.flatnonzero(yt == 1); neg = np.flatnonzero(yt == 0)
    out = []
    for _ in range(B):
        rep_idx = rng.choice(reps, len(reps), replace=True)
        rec = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append(stat(rep_idx, rec))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_t(df, policy, base="raw", metric="auc"):
    a = df[df.policy == policy].set_index("rep")[metric]
    b = df[df.policy == base].set_index("rep")[metric]
    dlt = (a - b).dropna()
    from scipy import stats
    n = len(dlt); m = dlt.mean(); s = dlt.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h, int((dlt > 0).sum()), n
