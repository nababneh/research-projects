"""Declaration features (M0), computed causally from the substrate."""
import os, csv, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import corpus as C
import estimators as E



def declaration_features(d):
    """M0: substrate attributes only, all history computed causally."""
    n = len(d)
    price = d["Item Price"].values.astype(float)
    mass = d["Net Mass"].values.astype(float)
    rate = d["Tax Rate"].values.astype(float)
    hs = d["HS6 Code"].values
    lp = np.log1p(np.clip(price, 0, None))
    lm = np.log1p(np.clip(mass, 0, None))
    uv = np.log1p(np.clip(price, 0, None) / np.maximum(mass, 1e-3))

    # causal unit-value z score within HS6
    order = np.argsort(d["day"].values, kind="stable")
    z = np.zeros(n)
    s1, s2, cnt = {}, {}, {}
    hist = {}
    hcnt = {}
    imp_rate = np.zeros(n); imp_n = np.zeros(n)
    sel_rate = np.zeros(n); sel_n = np.zeros(n)
    imp = d["Importer ID"].values; sell = d["Seller ID"].values
    y = d["Fraud"].values
    ih, ic, sh, sc = {}, {}, {}, {}
    for j in order:
        h = hs[j]
        if cnt.get(h, 0) > 20:
            m = s1[h] / cnt[h]; v = max(s2[h] / cnt[h] - m * m, 1e-9)
            z[j] = (uv[j] - m) / np.sqrt(v)
        s1[h] = s1.get(h, 0) + uv[j]; s2[h] = s2.get(h, 0) + uv[j] ** 2
        cnt[h] = cnt.get(h, 0) + 1
        a = imp[j]
        imp_n[j] = ic.get(a, 0)
        imp_rate[j] = ih.get(a, 0) / ic[a] if ic.get(a, 0) > 5 else -1.0
        ic[a] = ic.get(a, 0) + 1; ih[a] = ih.get(a, 0) + y[j]
        b = sell[j]
        sel_n[j] = sc.get(b, 0)
        sel_rate[j] = sh.get(b, 0) / sc[b] if sc.get(b, 0) > 5 else -1.0
        sc[b] = sc.get(b, 0) + 1; sh[b] = sh.get(b, 0) + y[j]

    cat = lambda col: pd.factorize(d[col].astype(str))[0]
    X = np.column_stack([
        lp, lm, uv, z, rate, lp * rate / 100.0,
        imp_rate, np.log1p(imp_n), sel_rate, np.log1p(sel_n),
        cat("HS6 Code"), cat("Country of Origin"), cat("Country of Departure"),
        (d["Country of Origin"] != d["Country of Departure"]).astype(int).values,
        cat("Office ID"), cat("Import Type"), cat("Import Use"),
        cat("Payment Type"), cat("Mode of Transport"), cat("Tax Type"),
        cat("Country of Origin Indicator"), cat("Process Type"),
        cat("Declarant ID"), cat("Courier ID"),
        d["day"].values / 365.0,
    ])
    return X
