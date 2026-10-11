"""
Semi-synthetic CANDID corpus.

Substrate: the Customs Import Declaration Dataset (Jeong et al., 2022), 54,000
import declarations with fraud labels, synthesised with CTGAN from 24.7 million
real Korean Customs Service declarations filed between January 2020 and June
2021. The declaration attributes and the labels are taken unchanged. We add
only an attestation layer, and we are explicit that this layer is ours.

Attestation sources are bound to entities that exist in the substrate, so their
availability follows the real data rather than an assumption:

  partner_export_record   the counterparty's export filing; exists when the
                          departure country is a data-exchange partner
  carrier_manifest        bound to Courier ID
  broker_filing           bound to Declarant ID
  payment_record          exists when Payment Type is a documentary credit
  origin_certificate      exists when a preferential Tax Type is claimed
  port_community_system   bound to Office ID

Attestation content is synthetic. Each source has a detection rate on false
fields and a false-alarm rate on true ones, and upstream concealment blinds the
documentary sources on a share of fraudulent consignments. Concealment is a
single latent variable shared across sources, which is what makes their errors
correlated and what breaks the independence assumption of the naive rule.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("KCS_DATA", os.path.join(HERE, "kcs_data"))

FIELDS = ["value", "classification", "origin"]
W_FIELD = np.array([0.55, 0.25, 0.20])

# name, kind, observes (value, class, origin), detection, false alarm
SOURCES = [
    ("partner_export_record", "independent", (1, 1, 1), 0.86, 0.08),
    ("carrier_manifest",      "independent", (0, 1, 1), 0.72, 0.10),
    ("broker_filing",         "documentary", (1, 1, 0), 0.68, 0.14),
    ("payment_record",        "documentary", (1, 0, 0), 0.78, 0.15),
    ("origin_certificate",    "documentary", (0, 1, 1), 0.80, 0.06),
    ("port_community_system", "documentary", (1, 1, 1), 0.70, 0.13),
]
N_SRC = len(SOURCES)
OBS = np.array([s[2] for s in SOURCES], dtype=bool)
DETECT = np.array([s[3] for s in SOURCES])
FALSE_ALARM = np.array([s[4] for s in SOURCES])
IS_DOC = np.array([s[1] == "documentary" for s in SOURCES])
PRIOR_REL = np.array([0.85, 0.78, 0.75, 0.80, 0.84, 0.76])

BLIND_DOC = 0.85
BLIND_IND = 0.25
P_CONCEAL = 0.45


def load_substrate():
    tr = pd.read_csv(os.path.join(DATA, "df_syn_train_eng.csv"))
    va = pd.read_csv(os.path.join(DATA, "df_syn_valid_eng.csv"))
    te = pd.read_csv(os.path.join(DATA, "df_syn_test_eng.csv"))
    tr["split"], va["split"], te["split"] = "train", "valid", "test"
    d = pd.concat([tr, va, te], ignore_index=True)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date", kind="stable").reset_index(drop=True)
    d["day"] = (d["Date"] - d["Date"].min()).dt.days
    return d


def build_attestations(d, seed=0, compromised=None, switch_day=None,
                       p_conceal=P_CONCEAL, detect=None, false_alarm=None,
                       blind_doc=BLIND_DOC, blind_ind=BLIND_IND):
    """Return agree (n,S,3) in {-1,0,1} and present (n,S,3) bool.

    agree = 1 means the source corroborates the declared value, 0 means it
    contradicts it, -1 means no attestation for that field group.
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    fraud = d["Fraud"].values.astype(bool)

    # --- which fields a fraudulent declaration actually falsifies ---
    # value is always implicated (fraud here is duty reduction); classification
    # and origin follow observable proxies in the substrate
    false_field = np.zeros((n, 3), dtype=bool)
    false_field[:, 0] = fraud
    hs_odd = (d["HS6 Code"].values % 7) < 3
    false_field[:, 1] = fraud & hs_odd
    orig_mismatch = (d["Country of Origin"].values != d["Country of Departure"].values)
    false_field[:, 2] = fraud & (orig_mismatch | (rng.random(n) < 0.25))

    # --- availability, driven by the substrate ---
    dep = d["Country of Departure"].astype(str).values
    top_partners = set(pd.Series(dep).value_counts().head(20).index)
    has_partner = np.array([c in top_partners for c in dep]) & (rng.random(n) < 0.80)
    has_carrier = d["Courier ID"].notna().values & (rng.random(n) < 0.90)
    has_broker = d["Declarant ID"].notna().values
    pay = d["Payment Type"].values
    has_payment = np.isin(pay, [11, 12, 13, 21, 22]) | (rng.random(n) < 0.25)
    tt = d["Tax Type"].astype(str).values
    has_coo = np.array([t.startswith(("F", "A", "C")) for t in tt])
    has_pcs = np.ones(n, dtype=bool)
    avail = np.column_stack([has_partner, has_carrier, has_broker,
                             has_payment, has_coo, has_pcs])

    present = avail[:, :, None] & OBS[None, :, :]

    # --- concealment: one latent per declaration, shared across sources ---
    concealed = fraud & (rng.random(n) < p_conceal)
    blind_p = np.where(IS_DOC, blind_doc, blind_ind)[None, :, None]
    blinded = concealed[:, None, None] & (rng.random((n, N_SRC, 3)) < blind_p)

    det = DETECT.copy() if detect is None else np.asarray(detect, float)
    fa = FALSE_ALARM.copy() if false_alarm is None else np.asarray(false_alarm, float)
    ff = false_field[:, None, :]
    detected = rng.random((n, N_SRC, 3)) < det[None, :, None]
    alarmed = rng.random((n, N_SRC, 3)) < fa[None, :, None]
    supports = np.where(ff, np.where(blinded | ~detected, 1, 0),
                        np.where(alarmed, 0, 1))

    # --- collusion ---
    if compromised is not None and len(compromised):
        mask = np.zeros(N_SRC, dtype=bool); mask[list(compromised)] = True
        if switch_day is None:
            active = np.ones(n, dtype=bool)
        else:
            active = d["day"].values >= switch_day
        coll = (mask[None, :, None] & active[:, None, None]
                & (rng.random((n, N_SRC, 3)) < 0.85))
        supports = np.where(coll, 1, supports)

    agree = np.where(present, supports, -1).astype(np.int8)
    return agree, present, false_field, concealed


if __name__ == "__main__":
    d = load_substrate()
    print(f"declarations {len(d)}  fraud {d['Fraud'].mean():.4f}")
    print(d.groupby("split").agg(n=("Fraud", "size"), fraud=("Fraud", "mean")).round(4).to_string())
    a, p, ff, c = build_attestations(d, seed=1)
    print(f"\nattestations present per declaration: {p.any(2).sum(1).mean():.2f} of {N_SRC}")
    for i, s in enumerate(SOURCES):
        print(f"  {s[0]:24s} available {p[:, i].any(1).mean():6.1%}   "
              f"contradiction rate {(a[:, i] == 0).any(1).mean():6.1%}")
    print(f"\nconcealed share of fraud: {c.sum() / d['Fraud'].sum():.1%}")
    print(f"false fields per fraud: value {ff[:,0].mean():.3f} class {ff[:,1].mean():.3f} origin {ff[:,2].mean():.3f}")
