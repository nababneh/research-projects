"""Result figures for the revised manuscript (vector PDF plus PNG previews).

Every interval drawn is a 95 per cent interval. Budget curves show the mean
over 20 replications with a t interval across replications. Policy contrasts
show the paired two-stage bootstrap interval of the difference from the raw
control. Sensitivity panels show paired t intervals over five replications.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "figures", "R2")
BLUE, VERM, GREEN, PURPLE, GREY = "#0072B2", "#D55E00", "#009E73", "#8C5AA8", "#7a7a7a"
INK, INK2, GRID = "#1a1a1a", "#454545", "#dedede"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.4, "legend.fontsize": 7.0,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300})
CM = 1 / 2.54
W = 17.2 * CM
NAMES = {"M0": "declaration only", "raw": "raw indicators", "majority": "equal weight",
         "static": "static prior", "chair_varshney": "Chair-Varshney oracle",
         "beta_reputation": "Beta reputation", "truth_discovery": "truth discovery",
         "candid": "learned likelihood ratio", "candid_mixture": "mixture-corrected",
         "raw+chair_varshney": "raw + oracle ratio", "raw+candid": "raw + learned ratio",
         "raw+candid_mixture": "raw + mixture ratio", "raw+beta_reputation": "raw + Beta trust"}


def tidy(ax, title=None):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=5)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def implementation():
    b = pd.read_csv(os.path.join(RES, "impl_benchmark.csv"))
    s = pd.read_csv(os.path.join(RES, "impl_load_sweep.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 6.4 * CM), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    ax = axes[0]
    order = ["large blocks, 1 endorser", "large blocks, 3 endorsers", "large blocks, 5 endorsers",
             "small blocks, 3 endorsers", "large blocks, 3 endorsers, durable",
             "small blocks, 3 endorsers, durable"]
    labels = ["1 endorser, b = 250", "3 endorsers, b = 250", "5 endorsers, b = 250",
              "3 endorsers, b = 10", "3 endorsers, b = 250,\ndurable", "3 endorsers, b = 10,\ndurable"]
    g = b.groupby("config").tps.agg(["mean", "std"]).reindex(order)
    y = np.arange(len(order))[::-1]
    cols = [BLUE, BLUE, BLUE, BLUE, VERM, VERM]
    ax.barh(y, g["mean"], xerr=1.96 * g["std"] / np.sqrt(5), color=[c for c in cols], alpha=0.85,
            height=0.6, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    for yi, m, sd in zip(y, g["mean"], g["std"]):
        ax.text(m + 1.96 * sd / np.sqrt(5) + 220, yi, f"{m:,.0f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel("closed-loop throughput (tx/s)")
    ax.set_xlim(0, g["mean"].max() * 1.22)
    tidy(ax, "(a) Drain rate by configuration")
    ax = axes[1]
    for dur, col, lab in [(False, BLUE, "volatile"), (True, VERM, "durable")]:
        q = s[s.durable == dur].groupby("offered_tps").agg(p95=("p95_latency_ms", "mean"),
                                                         ach=("achieved_tps", "mean")).reset_index()
        ax.plot(q.offered_tps, q.p95 / 1000, "-o", color=col, ms=3, lw=1.2, label=f"95th pct latency, {lab}")
    ax.axhline(1.0, color=GREY, lw=0.7, ls="--")
    ax.text(60, 1.12, "one-second objective", fontsize=6.6, color=GREY)
    ax.set_yscale("log"); ax.set_xlabel("offered load (tx/s, open loop)"); ax.set_ylabel("latency (s)")
    ax.legend(loc="upper left")
    tidy(ax, "(b) Latency under offered load")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig4_implementation")


def selectivity():
    c = pd.read_csv(os.path.join(RES, "summary_budget_curve.csv"))
    p = pd.read_csv(os.path.join(RES, "summary_policies_calibrated.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 7.4 * CM), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    ax = axes[0]
    for pol, col, lab in [("M0", VERM, "declaration only"), ("raw", BLUE, "with attestation record")]:
        q = c[c.policy == pol]
        x = q.budget * 100
        ax.plot(x, q.recall, "-o", color=col, ms=2.8, lw=1.2, label=f"recall, {lab}")
        ax.fill_between(x, q.recall_lo, q.recall_hi, color=col, alpha=0.15, lw=0)
        ax.plot(x, q.revenue, "--s", color=col, ms=2.6, lw=1.0, label=f"revenue proxy, {lab}")
    ax.axvline(5, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("inspection budget (%)"); ax.set_ylabel("share recovered")
    ax.set_ylim(0, 1.03); hl = ax.get_legend_handles_labels()
    tidy(ax, "(a) Recall and revenue proxy")
    ax = axes[1]
    pp = p[p.policy != "raw"].copy()
    pp = pp[pp.policy != "M0"]
    pp = pp.sort_values("d_auc")
    y = np.arange(len(pp))
    col = [GREEN if s.startswith("raw+") else (PURPLE if s == "chair_varshney" else INK2) for s in pp.policy]
    for yi, (_, r), c in zip(y, pp.iterrows(), col):
        ax.errorbar([r.d_auc], [yi], xerr=[[r.d_auc - r.d_auc_lo], [r.d_auc_hi - r.d_auc]], fmt="none",
                    ecolor=c, elinewidth=0.9, capsize=2)
    ax.scatter(pp.d_auc, y, c=col, s=16, zorder=3)
    ax.axvline(0, color=BLUE, lw=0.9, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([NAMES[s] for s in pp.policy], fontsize=6.8)
    ax.set_xlabel("difference in AUC from raw indicators")
    ax.text(0.0005, -0.9, "raw control", color=BLUE, fontsize=6.4, va="center")
    tidy(ax, "(b) Paired contrasts with the raw control")
    fig.legend(*hl, loc="lower center", ncol=4, fontsize=6.2, columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.6, rect=[0, 0.07, 1, 1])
    save(fig, "fig5_selectivity")


def sensitivity():
    s = pd.read_csv(os.path.join(RES, "summary_sensitivity.csv"))
    base = pd.read_csv(os.path.join(RES, "main_calibrated_reps.csv"))
    base5 = base[base.rep <= 5]
    def with_base(factor, level0):
        sub = s[s.factor == factor].copy()
        # baseline cell from the main run, first five replications
        rows = []
        for pol in ["M0", "raw", "chair_varshney", "candid", "candid_mixture", "beta_reputation",
                    "raw+chair_varshney"]:
            a = base5[base5.policy == pol].set_index("rep")
            r = dict(factor=factor, level=level0, policy=pol, auc=a.auc.mean(), ap=a.ap.mean())
            if pol != "raw":
                d = a.auc - base5[base5.policy == "raw"].set_index("rep").auc
                from scipy import stats
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                r.update(d_auc=d.mean(), d_auc_lo=d.mean() - h, d_auc_hi=d.mean() + h)
            rows.append(r)
        if not ((sub.level == level0).any()):
            sub = pd.concat([sub, pd.DataFrame(rows)], ignore_index=True)
        return sub.sort_values("level")

    fig, axes = plt.subplots(2, 3, figsize=(W, 11.0 * 0.5 * 2 * CM))
    panels = [("detect_scale", 1.0, "detection rate (times baseline)", "(a) Source detection"),
              ("false_alarm_scale", 1.0, "false-alarm rate (times baseline)", "(b) Source false alarms"),
              ("p_conceal", 0.45, "share of frauds concealed", "(c) Concealment")]
    for k, (f, l0, xl, tt) in enumerate(panels):
        sub = with_base(f, l0)
        ax = axes[0, k]
        for pol, col, lab in [("M0", VERM, "declaration only"), ("raw", BLUE, "raw indicators")]:
            q = sub[sub.policy == pol]
            ax.plot(q.level, q.auc, "-o", color=col, ms=2.6, lw=1.1, label=lab)
        ax.set_xlabel(xl); ax.set_ylabel("AUC" if k == 0 else None)
        if k == 0: ax.legend(loc="lower right", fontsize=6.3)
        tidy(ax, tt)
        ax = axes[1, k]
        for pol, col, lab, dx in [("chair_varshney", PURPLE, "oracle ratio", -0.012),
                                  ("candid", INK2, "learned ratio", 0.0),
                                  ("raw+chair_varshney", GREEN, "raw + oracle ratio", 0.012)]:
            q = sub[sub.policy == pol]
            span = sub.level.max() - sub.level.min()
            xx = q.level + dx * span
            ax.errorbar(xx, q.d_auc, yerr=[q.d_auc - q.d_auc_lo, q.d_auc_hi - q.d_auc], fmt="-o",
                        color=col, ms=2.4, lw=0.9, elinewidth=0.7, capsize=1.5, label=lab)
        ax.axhline(0, color=BLUE, lw=0.8, ls="--")
        ax.set_xlabel(xl); ax.set_ylabel("AUC difference from raw" if k == 0 else None)
        if k == 0: ax.legend(loc="lower right", fontsize=6.2)
        tidy(ax)
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save(fig, "fig6_sensitivity")


def labels_panel():
    """Training share and audited share."""
    s = pd.read_csv(os.path.join(RES, "summary_sensitivity.csv"))
    base = pd.read_csv(os.path.join(RES, "main_calibrated_reps.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 6.0 * CM))
    sub = s[s.factor == "train_frac"].copy()
    b5 = base[base.rep <= 5]
    extra = []
    for pol in ["raw", "chair_varshney", "candid", "candid_mixture", "raw+chair_varshney"]:
        a = b5[b5.policy == pol].set_index("rep")
        r = dict(factor="train_frac", level=1.0, policy=pol, auc=a.auc.mean())
        if pol != "raw":
            from scipy import stats
            d = a.auc - b5[b5.policy == "raw"].set_index("rep").auc
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            r.update(d_auc=d.mean(), d_auc_lo=d.mean() - h, d_auc_hi=d.mean() + h)
        extra.append(r)
    sub = pd.concat([sub, pd.DataFrame(extra)], ignore_index=True)
    ticks = [2, 5, 10, 25, 100]
    ax = axes[0]
    for pol, col, lab in [("chair_varshney", PURPLE, "oracle ratio"), ("candid", INK2, "learned ratio"),
                          ("candid_mixture", GREY, "mixture ratio"),
                          ("raw+chair_varshney", GREEN, "raw + oracle ratio")]:
        q = sub[sub.policy == pol].sort_values("level")
        ax.errorbar(q.level * 100, q.d_auc, yerr=[q.d_auc - q.d_auc_lo, q.d_auc_hi - q.d_auc], fmt="-o",
                    color=col, ms=2.4, lw=0.9, elinewidth=0.7, capsize=1.5, label=lab)
    ax.axhline(0, color=BLUE, lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("share of training records used (%)"); ax.set_ylabel("AUC difference from raw")
    ax.legend(fontsize=6.2); tidy(ax, "(a) Fewer labels for the classifier")
    ax = axes[1]
    for pol, col, lab in [("raw", BLUE, "raw indicators"), ("chair_varshney", PURPLE, "oracle ratio"),
                          ("candid", INK2, "learned ratio")]:
        q = sub[sub.policy == pol].sort_values("level")
        ax.plot(q.level * 100, q.auc, "-o", color=col, ms=2.6, lw=1.1, label=lab)
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("share of training records used (%)"); ax.set_ylabel("AUC")
    ax.legend(fontsize=6.2); tidy(ax, "(b) Absolute AUC")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig7_labels")


def robustness():
    lat = pd.read_csv(os.path.join(RES, "summary_latency.csv"))
    ns = pd.read_csv(os.path.join(RES, "summary_nonstationary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 6.2 * CM))
    ax = axes[0]
    for pol, col, lab, dx in [("raw", BLUE, "raw indicators", -6), ("candid", INK2, "learned ratio", -2),
                              ("candid_mixture", GREEN, "mixture ratio", 2),
                              ("beta_reputation", VERM, "Beta reputation", 6)]:
        q = lat[lat.policy == pol].sort_values("lag")
        ax.errorbar(q.lag + dx, q.auc, yerr=[q.auc - q.auc_lo, q.auc_hi - q.auc], fmt="-o", color=col,
                    ms=2.6, lw=1.0, elinewidth=0.7, capsize=1.5, label=lab)
    ax.set_xlabel("audit lag (days)"); ax.set_ylabel("AUC"); ax.legend(fontsize=6.0, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.5), handletextpad=0.2, columnspacing=0.8)
    tidy(ax, "(a) Audit latency")
    ax = axes[1]
    pols = [("raw", BLUE, "raw"), ("majority", GREY, "equal"), ("static", "#b59b00", "static"),
            ("beta_reputation", VERM, "Beta"), ("candid", INK2, "learned"), ("candid_mixture", GREEN, "mixture")]
    for j, (pol, col, lab) in enumerate(pols):
        q = ns[ns.policy == pol].sort_values("switch_day")
        x = np.arange(len(q)) + (j - 2.5) * 0.12
        ax.errorbar(x, q.auc, yerr=[q.auc - q.auc_lo, q.auc_hi - q.auc], fmt="o", color=col, ms=2.8,
                    elinewidth=0.8, capsize=1.5, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["day 180", "day 300", "day 420"])
    ax.set_xlabel("collusion begins"); ax.set_ylabel("AUC"); ax.legend(fontsize=6.0, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.5), handletextpad=0.2, columnspacing=0.8)
    tidy(ax, "(b) Two sources turn")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig8_robustness")


if __name__ == "__main__":
    import sys
    for w in (sys.argv[1:] or ["implementation", "selectivity", "sensitivity", "labels", "robustness"]):
        {"implementation": implementation, "selectivity": selectivity, "sensitivity": sensitivity,
         "labels": labels_panel, "robustness": robustness}[w]()
