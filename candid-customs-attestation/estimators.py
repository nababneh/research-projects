"""
Attestation-scoring policies.

Each policy turns the attestation matrix into per-declaration features. They
differ only in how much weight a corroboration or a contradiction carries.

  raw               no weighting at all; the per-source indicator vector is
                    handed to the classifier, which may learn its own weights.
                    This is the control the paper must beat to claim anything.
  majority          equal weight to every source
  static            fixed prior reliability, never updated
  chair_varshney    log likelihood ratio with the true detection and
                    false-alarm rates supplied by an oracle. Upper bound on
                    any policy of this family, and not attainable in practice.
  beta_reputation   Josang and Ismail Beta reputation with an exponential
                    forgetting factor, used as a scalar trust weight
  truth_discovery   CRH-style unsupervised iteration between source weights
                    and inferred truth
  candid            log likelihood ratio with detection and false-alarm rates
                    learned from the audited subset under discounted updating
  candid_mixture    candid, with the conditional independence assumption
                    replaced by a two-component mixture over the shared latent
                    concealment state

All learned policies observe audited labels only after `audit_lag_days`.
"""
import numpy as np
from corpus import N_SRC, OBS, DETECT, FALSE_ALARM, PRIOR_REL, IS_DOC, W_FIELD, BLIND_DOC, BLIND_IND

EPS = 1e-3
LAMBDA = 0.35          # forgetting rate per 30 days


def _blocks(day):
    days = np.unique(day)
    return days, {d: np.flatnonzero(day == d) for d in days}


def _stack(lam, extra=None):
    """lam is (n,3) evidence that each field group is false."""
    delta = lam @ W_FIELD
    out = [ -delta, delta, lam ]
    if extra is not None:
        out.append(extra)
    return np.column_stack(out)


def features_raw(agree, present, day, y_audit=None, **kw):
    """Per-source, per-field indicators. No weighting."""
    att = (agree >= 0).astype(np.int8)
    con = (agree == 0).astype(np.int8)
    n = agree.shape[0]
    return np.column_stack([
        att.reshape(n, -1), con.reshape(n, -1),
        att.sum((1, 2)), con.sum((1, 2)),
        (agree == 0).any(1).astype(np.int8),
    ])


def _llr_features(agree, present, d_s, q_s):
    d_s = np.clip(d_s, EPS, 1 - EPS); q_s = np.clip(q_s, EPS, 1 - EPS)
    w_flag = np.log(d_s / q_s)
    w_supp = np.log((1 - d_s) / (1 - q_s))
    contrib = np.where(present, np.where(agree == 0, w_flag[None, :, None],
                                         w_supp[None, :, None]), 0.0)
    return contrib.sum(axis=1)


def features_majority(agree, present, day, **kw):
    lam = _llr_features(agree, present, np.full(N_SRC, .75), np.full(N_SRC, .25))
    return _stack(lam)


def features_static(agree, present, day, **kw):
    lam = _llr_features(agree, present, PRIOR_REL, 1 - PRIOR_REL)
    return _stack(lam)


def features_chair_varshney(agree, present, day, detect=None, false_alarm=None, **kw):
    """Oracle: the true generative detection and false-alarm rates."""
    lam = _llr_features(agree, present,
                        DETECT if detect is None else detect,
                        FALSE_ALARM if false_alarm is None else false_alarm)
    return _stack(lam)


def _audited_mask(n, rng, audit_frac):
    return rng.random(n) < audit_frac


def features_beta_reputation(agree, present, day, false_field=None,
                             audit_frac=0.03, audit_lag_days=0, seed=3, **kw):
    """Josang and Ismail: one Beta per source over correct/incorrect calls."""
    rng = np.random.default_rng(seed)
    n = agree.shape[0]
    audited = _audited_mask(n, rng, audit_frac)
    a = np.ones(N_SRC); b = np.ones(N_SRC)
    days, idx_by_day = _blocks(day)
    trust_used = np.zeros((n, N_SRC))
    prev = days[0]
    for dd in days:
        g = np.exp(-LAMBDA * (dd - prev) / 30.0)
        a = 1.0 + (a - 1.0) * g; b = 1.0 + (b - 1.0) * g
        trust = a / (a + b)
        idx = idx_by_day[dd]
        trust_used[idx] = trust[None, :]
        # audited evidence that has matured
        rel = idx_by_day.get(dd - audit_lag_days)
        if rel is not None and false_field is not None:
            au = rel[audited[rel]]
            if len(au):
                aa = agree[au]; pp = present[au]
                ffl = np.repeat(false_field[au][:, None, :], N_SRC, axis=1)
                correct = np.where(ffl, aa == 0, aa == 1) & pp
                wrong = (~np.where(ffl, aa == 0, aa == 1)) & pp
                a += correct.sum((0, 2)); b += wrong.sum((0, 2))
        prev = dd
    # scalar trust weighting of contradictions
    con = (agree == 0).astype(float) * trust_used[:, :, None]
    sup = (agree == 1).astype(float) * trust_used[:, :, None]
    lam = (con - sup).sum(axis=1)
    return _stack(lam, trust_used)


def features_truth_discovery(agree, present, day, n_iter=8, **kw):
    """CRH-style: alternate between source weights and inferred truth.

    Unsupervised. Weight is the negative log of a source's normalised
    disagreement with the current consensus.
    """
    n = agree.shape[0]
    obs = present
    a = np.where(obs, agree, np.nan).astype(float)
    w = np.ones(N_SRC)
    truth = np.nanmean(a, axis=1)               # (n,3) in [0,1]
    truth = np.nan_to_num(truth, nan=1.0)
    for _ in range(n_iter):
        err = np.nansum(np.abs(a - truth[:, None, :]) * obs, axis=(0, 2))
        cnt = obs.sum((0, 2)).astype(float)
        rate = np.where(cnt > 0, err / np.maximum(cnt, 1), 0.5)
        w = -np.log(np.clip(rate, EPS, 1 - EPS))
        w = np.clip(w, 0.0, None)
        num = np.nansum(np.where(obs, a, 0.0) * w[None, :, None], axis=1)
        den = np.nansum(obs * w[None, :, None], axis=1)
        truth = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0)
    con = (agree == 0).astype(float) * w[None, :, None]
    sup = (agree == 1).astype(float) * w[None, :, None]
    lam = (con - sup).sum(axis=1)
    return _stack(lam, np.repeat(truth, 1, axis=1))


def _learn_dq(agree, present, day, false_field, audited, audit_lag_days,
              days, idx_by_day, discount=True):
    """Yield per-day (d_s, q_s) learned from matured audited evidence."""
    a_d = np.ones(N_SRC); b_d = np.ones(N_SRC)
    a_q = np.ones(N_SRC); b_q = np.ones(N_SRC)
    prev = days[0]
    out = {}
    for dd in days:
        if discount:
            g = np.exp(-LAMBDA * (dd - prev) / 30.0)
            a_d = 1 + (a_d - 1) * g; b_d = 1 + (b_d - 1) * g
            a_q = 1 + (a_q - 1) * g; b_q = 1 + (b_q - 1) * g
        out[dd] = (a_d / (a_d + b_d), a_q / (a_q + b_q))
        rel = idx_by_day.get(dd - audit_lag_days)
        if rel is not None:
            au = rel[audited[rel]]
            if len(au):
                aa = agree[au]; pp = present[au]
                ffl = np.repeat(false_field[au][:, None, :], N_SRC, axis=1)
                flag = (aa == 0) & pp
                of = ffl & pp; ot = (~ffl) & pp
                a_d += (flag & of).sum((0, 2)); b_d += ((~flag) & of).sum((0, 2))
                a_q += (flag & ot).sum((0, 2)); b_q += ((~flag) & ot).sum((0, 2))
        prev = dd
    return out


def features_candid(agree, present, day, false_field=None, audit_frac=0.03,
                    audit_lag_days=0, seed=3, track=False, **kw):
    rng = np.random.default_rng(seed)
    n = agree.shape[0]
    audited = _audited_mask(n, rng, audit_frac)
    days, idx_by_day = _blocks(day)
    dq = _learn_dq(agree, present, day, false_field, audited, audit_lag_days,
                   days, idx_by_day)
    lam = np.zeros((n, 3)); wsup = np.zeros((n, N_SRC))
    hist = []
    for dd in days:
        idx = idx_by_day[dd]
        d_s, q_s = dq[dd]
        lam[idx] = _llr_features(agree[idx], present[idx], d_s, q_s)
        ds = np.clip(d_s, EPS, 1 - EPS); qs = np.clip(q_s, EPS, 1 - EPS)
        wsup[idx] = np.log((1 - ds) / (1 - qs))[None, :]
        if track:
            hist.append([int(dd)] + list(np.round(d_s, 4)) + list(np.round(q_s, 4))
                        + list(np.round(np.log((1 - ds) / (1 - qs)), 4)))
    f = _stack(lam, wsup)
    return (f, hist) if track else f


def features_candid_mixture(agree, present, day, false_field=None,
                            audit_frac=0.03, audit_lag_days=0, seed=3, **kw):
    """CANDID with the independence assumption replaced by a mixture.

    Concealment is one latent binary per declaration, shared across sources.
    Conditional on the field being false we therefore have a two-component
    mixture, and the likelihood ratio is taken over the mixture rather than
    over a product of independent terms.
    """
    rng = np.random.default_rng(seed)
    n = agree.shape[0]
    audited = _audited_mask(n, rng, audit_frac)
    days, idx_by_day = _blocks(day)
    dq = _learn_dq(agree, present, day, false_field, audited, audit_lag_days,
                   days, idx_by_day)

    # blinding probabilities and the concealment prior are estimated once from
    # the matured audited subset by a two-component Bernoulli EM
    au = np.flatnonzero(audited)
    if false_field is not None and len(au):
        pi, b_s = _em_conceal(agree[au], present[au], false_field[au])
    else:
        pi, b_s = 0.4, np.where(IS_DOC, BLIND_DOC, BLIND_IND)

    lam = np.zeros((n, 3))
    for dd in days:
        idx = idx_by_day[dd]
        d_s, q_s = dq[dd]
        d_s = np.clip(d_s, EPS, 1 - EPS); q_s = np.clip(q_s, EPS, 1 - EPS)
        a = agree[idx]; p = present[idx]
        # P(a | true)
        lt = np.where(p, np.where(a == 0, q_s[None, :, None], (1 - q_s)[None, :, None]), 1.0)
        log_true = np.log(np.clip(lt, EPS, None)).sum(axis=1)
        comp = []
        for c, bb in [(0, np.zeros(N_SRC)), (1, b_s)]:
            eff = (1 - bb) * d_s
            lf = np.where(p, np.where(a == 0, eff[None, :, None],
                                      (1 - eff)[None, :, None]), 1.0)
            comp.append(np.log(np.clip(lf, EPS, None)).sum(axis=1))
        m = np.maximum(comp[0] + np.log(1 - pi), comp[1] + np.log(pi))
        log_false = m + np.log(np.exp(comp[0] + np.log(1 - pi) - m)
                               + np.exp(comp[1] + np.log(pi) - m))
        lam[idx] = log_false - log_true
    return _stack(lam)


def _em_conceal(agree, present, false_field, n_iter=40):
    """Two-component Bernoulli EM over audited fraudulent declarations."""
    sel = false_field.any(1)
    a = agree[sel]; p = present[sel]; ff = false_field[sel]
    if len(a) < 30:
        return 0.4, np.where(IS_DOC, BLIND_DOC, BLIND_IND)
    obs = p & ff[:, None, :]
    flag = (a == 0) & obs
    pi = 0.4
    d0 = DETECT.copy()
    b = np.where(IS_DOC, 0.7, 0.2)
    for _ in range(n_iter):
        ll = []
        for c, bb in [(0, np.zeros(N_SRC)), (1, b)]:
            eff = np.clip((1 - bb) * d0, EPS, 1 - EPS)
            l = np.where(obs, np.where(flag, eff[None, :, None],
                                       (1 - eff)[None, :, None]), 1.0)
            ll.append(np.log(l).sum((1, 2)))
        m = np.maximum(ll[0] + np.log(1 - pi), ll[1] + np.log(pi))
        p1 = np.exp(ll[1] + np.log(pi) - m)
        p0 = np.exp(ll[0] + np.log(1 - pi) - m)
        r = p1 / np.maximum(p0 + p1, 1e-12)
        pi = float(np.clip(r.mean(), 0.02, 0.95))
        num0 = (flag * (1 - r)[:, None, None]).sum((0, 2))
        den0 = (obs * (1 - r)[:, None, None]).sum((0, 2))
        num1 = (flag * r[:, None, None]).sum((0, 2))
        den1 = (obs * r[:, None, None]).sum((0, 2))
        d0 = np.clip(np.where(den0 > 0, num0 / np.maximum(den0, 1), DETECT), EPS, 1 - EPS)
        d1 = np.clip(np.where(den1 > 0, num1 / np.maximum(den1, 1), 0.1), EPS, 1 - EPS)
        b = np.clip(1 - d1 / np.maximum(d0, EPS), 0.0, 0.98)
    return pi, b


POLICIES = {
    "raw":              features_raw,
    "majority":         features_majority,
    "static":           features_static,
    "chair_varshney":   features_chair_varshney,
    "beta_reputation":  features_beta_reputation,
    "truth_discovery":  features_truth_discovery,
    "candid":           features_candid,
    "candid_mixture":   features_candid_mixture,
}
