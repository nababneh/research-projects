"""Architecture, lifecycle and evaluation-workflow diagrams (vector PDF).

Notation used in every diagram
  rounded box      software component
  cylinder         data store
  dashed box       external actor or component evaluated but not adopted
  solid arrow      data flow, labelled with the artefact that flows
  dashed arrow     feedback that arrives later (audited outcomes)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures", "R2")
os.makedirs(FIG, exist_ok=True)

BLUE, VERM, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#8C5AA8"
INK, INK2, MUTED, GREY = "#1a1a1a", "#454545", "#6f6f6f", "#9a9a9a"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
CM = 1 / 2.54


def tint(hexc, a=0.10):
    h = hexc.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (1 - a + a * r / 255, 1 - a + a * g / 255, 1 - a + a * b / 255)


def comp(ax, x, y, w, h, text, color, dashed=False, fs=6.6, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc="white", ec=color, lw=0.9,
                                ls=(0, (3, 2)) if dashed else "-", zorder=3))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h / 2 + (1.15 if len(lines) > 1 else 0), lines[0], ha="center",
                va="center", fontsize=fs, color=INK, weight="bold", zorder=4)
        if len(lines) > 1:
            ax.text(x + w / 2, y + h / 2 - 1.35, "\n".join(lines[1:]), ha="center", va="center",
                    fontsize=fs - 0.5, color=INK2, zorder=4, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, zorder=4, linespacing=1.2)


def store(ax, x, y, w, h, text, color, fs=6.6):
    e = 1.1
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc="white", ec="none", zorder=3))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=color, lw=0.9, zorder=4)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=color, lw=0.9, zorder=4)
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc="white", ec=color, lw=0.9, zorder=4))
    ax.add_patch(matplotlib.patches.Arc((x + w / 2, y + e / 2), w, e, theta1=180, theta2=360,
                                        ec=color, lw=0.9, zorder=4))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h / 2 - 0.1 + (1.6 if len(lines) > 2 else 0.8 if len(lines) > 1 else 0), lines[0], ha="center",
            va="center", fontsize=fs, weight="bold", color=INK, zorder=5)
    if len(lines) > 1:
        ax.text(x + w / 2, y + h / 2 - (1.6 if len(lines) > 2 else 1.3), "\n".join(lines[1:]), ha="center", va="center",
                fontsize=fs - 0.5, color=INK2, zorder=5, linespacing=1.15)


def actor(ax, x, y, w, h, text, fs=6.6):
    ax.add_patch(Rectangle((x, y), w, h, fc="#f6f6f6", ec=GREY, lw=0.8, ls=(0, (3, 2)), zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=4, linespacing=1.15, style="italic")


def arrow(ax, pts, label=None, lpos=None, color=INK2, dashed=False, fs=5.3, ha="center",
          va="center", lw=0.8, rot=0):
    for (a, b) in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                ls=(0, (3, 2)) if dashed else "-", zorder=2, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=3.2,head_width=1.8",
                                 color=color, lw=lw, ls=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=2))
    if label:
        x, y = lpos
        ax.text(x, y, label, fontsize=fs, color=INK2, ha=ha, va=va, zorder=6, rotation=rot,
                bbox=dict(fc="white", ec="none", pad=0.4), linespacing=1.1)


def band(ax, y0, y1, color, title, verdict, x0=14.5, x1=96.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=tint(color, 0.07), ec=tint(color, 0.35), lw=0.8, zorder=1))
    ax.text(x0 + 1.2, y1 - 1.7, title, fontsize=7.4, weight="bold", color=color, va="center", zorder=2)
    ax.text(x1 - 1.2, y1 - 1.7, verdict, fontsize=6.0, color=color, va="center", ha="right",
            style="italic", zorder=2)


# ------------------------------------------------------------------ figure 1
def architecture():
    fig, ax = plt.subplots(figsize=(17.6 * CM, 15.6 * CM))
    ax.set_xlim(0, 112); ax.set_ylim(-0.5, 99); ax.axis("off")
    W, H = 18.0, 6.6
    c = [16.6, 37.0, 57.4, 77.8]
    FS = 5.9

    band(ax, 4, 30, VERM, "Attestation plane", "a signed log suffices inside one administration")
    band(ax, 33, 53, GREEN, "Provenance plane", "no ledger required")
    band(ax, 56, 77, PURPLE, "Control plane", "no ledger required")
    band(ax, 80, 98, BLUE, "Interoperability plane", "a distributed ledger is required")

    # attestation plane
    yA, yS = 17.5, 5.8
    comp(ax, c[0], yA, W, H, "Commitment service\n$c_f = H(r_f \\Vert x_f)$", VERM, fs=FS)
    comp(ax, c[1], yA, W, H, "Endorsing peers\nECDSA P-256", VERM, fs=FS)
    comp(ax, c[2], yA, W, H, "Ordering service\nblocks, MVCC check", VERM, fs=FS)
    store(ax, c[3], yA - 0.4, W, H + 0.8, "Attestation ledger\ncommitments, signatures", VERM, fs=FS)
    store(ax, c[0], yS, W, H + 0.6, "System of record\npayloads, salt key", VERM, fs=FS)
    actor(ax, 0.5, yA, 12.2, H, "Off-chain\nsources", fs=FS)
    actor(ax, 0.5, yS + 0.3, 12.2, H, "Trader or\nbroker", fs=FS)

    mA, mS = yA + H / 2, yS + (H + 0.6) / 2
    arrow(ax, [(12.7, mA), (c[0], mA)], None)
    ax.text(6.6, yA - 1.9, "attestation $(s, f, a_{s,f})$", fontsize=5.3, color=INK2, ha="center", va="center")
    arrow(ax, [(12.7, mS), (c[0], mS)], None)
    ax.text(6.6, yS - 1.6, "lodges declaration $d$", fontsize=5.3, color=INK2, ha="center", va="center")
    arrow(ax, [(c[0] + W / 2, yS + H + 0.6), (c[0] + W / 2, yA)], "$x_f$, $r_f$", (c[0] + W / 2 + 3.0, 15.0))
    arrow(ax, [(c[0] + W, mA + 1.2), (c[1], mA + 1.2)], "proposal", (35.8, mA + 4.4))
    arrow(ax, [(c[1] + W, mA), (c[2], mA)], "endorsed\ntx", (56.2, mA + 5.0))
    arrow(ax, [(c[2] + W, mA), (c[3], mA)], "block", (76.6, mA + 4.4))

    # provenance plane
    yP = 38.5
    comp(ax, c[0], yP, W, H, "Credibility weighting\nevaluated, not adopted", GREY, dashed=True, fs=FS)
    store(ax, c[1], yP - 0.4, W, H + 0.8, "Attestation matrix\n$a_{s,f}(d)\\in\\{0,1,\\bot\\}$", GREEN, fs=FS)
    store(ax, c[2], yP - 0.4, W, H + 0.8, "Source register\nidentity, competence", GREEN, fs=FS)
    comp(ax, c[3], yP, W, H, "Provenance builder\nevents to matrix rows", GREEN, fs=FS)
    mP = yP + H / 2
    arrow(ax, [(c[3] + W / 2, yA + H + 0.4), (c[3] + W / 2, yP)], "committed\nevents", (c[3] + W / 2, 31.5))
    arrow(ax, [(c[2] + W, mP + 1.0), (c[3], mP + 1.0)], "source\nprofile", (76.6, mP + 5.6))
    arrow(ax, [(c[3] + 3.0, yP), (c[3] + 3.0, 35.4), (c[1] + W / 2, 35.4), (c[1] + W / 2, yP - 0.4)],
          "matrix rows", (58.0 + W / 2, 35.4))
    arrow(ax, [(c[1], mP), (c[0] + W, mP)], None, color=GREY, dashed=True)

    # control plane
    yC = 61.0
    comp(ax, c[0], yC, W, H, "Post-clearance audit\nlabels released goods", PURPLE, fs=FS)
    comp(ax, c[1], yC, W, H, "Risk model\nexisting classifier", PURPLE, fs=FS)
    comp(ax, c[2], yC, W, H, "Rule engine\ndeclarative rules", PURPLE, fs=FS)
    actor(ax, 99.0, yC, 12.5, H, "Customs\nofficer", fs=FS)
    mC = yC + H / 2
    arrow(ax, [(c[1] + W / 2, yP + H + 0.4), (c[1] + W / 2, yC)], "$a_{s,f}(d)$", (c[1] + W / 2 + 3.4, 51.0))
    arrow(ax, [(c[0], mS - 1.2), (15.3, mS - 1.2), (15.3, 70.6), (c[1] + 4.0, 70.6), (c[1] + 4.0, yC + H)],
          "declaration features", (26.0, 70.6))
    arrow(ax, [(c[1] + W, mC), (c[2], mC)], "risk\nscore", (56.2, mC + 5.0))
    arrow(ax, [(c[2] + W, mC), (99.0, mC)], "recommendation\nand reasons", (87.0, mC + 3.4))
    arrow(ax, [(105.25, yC + H), (105.25, 74.0), (c[0] + W / 2, 74.0), (c[0] + W / 2, yC + H)],
          "release or inspection decision", (66.0, 74.0))
    arrow(ax, [(c[0] + W, mC + 1.4), (c[1], mC + 1.4)], "labels", (35.8, mC + 4.4), dashed=True)
    arrow(ax, [(c[0] + W / 2, yC), (c[0] + W / 2, yP + H)], "audited\noutcomes", (c[0] + W / 2, 54.6),
          dashed=True, color=GREY)

    # interoperability plane
    yI = 85.0
    store(ax, c[2], yI - 0.4, W, H + 0.8, "Key register\nendorser public keys", BLUE, fs=FS)
    comp(ax, c[3], yI, W, H, "Corridor gateway\nrelay and verify", BLUE, fs=FS)
    actor(ax, 99.0, yI, 12.5, H, "Partner\nadministration", fs=FS)
    mI = yI + H / 2
    arrow(ax, [(c[3] + W, mA - 1.4), (97.8, mA - 1.4), (97.8, yI + 1.2), (c[3] + W, yI + 1.2)],
          "commitments and signatures", (97.8, 52.0), rot=90, fs=5.6)
    arrow(ax, [(c[2] + W, mI), (c[3], mI)], "public\nkeys", (76.6, mI + 5.0))
    arrow(ax, [(c[3] + W, mI + 1.6), (99.0, mI + 1.6)], None)
    arrow(ax, [(99.0, mI - 0.4), (c[3] + W, mI - 0.4)], None)
    ax.text(105.25, yI - 2.6, "attestation out,\nverification in", fontsize=5.3, color=INK2,
            ha="center", va="center")

    # legend
    lx, ly = 16.5, 0.0
    comp(ax, lx, ly, 8.0, 3.8, "component", INK2, fs=5.6, bold_first=False)
    store(ax, lx + 9.5, ly - 0.2, 7.0, 4.2, "store", INK2, fs=5.6)
    actor(ax, lx + 18.0, ly, 10.5, 3.8, "external actor", fs=5.4)
    ax.plot([lx + 30.0, lx + 33.5], [ly + 3.0, ly + 3.0], color=INK2, lw=0.8)
    ax.text(lx + 34.2, ly + 3.0, "data flow", fontsize=5.6, va="center", color=INK2)
    ax.plot([lx + 30.0, lx + 33.5], [ly + 0.8, ly + 0.8], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(lx + 34.2, ly + 0.8, "delayed or offline", fontsize=5.6, va="center", color=INK2)
    fig.savefig(os.path.join(FIG, "fig1_architecture.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, "fig1_architecture.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def lifecycle():
    fig, ax = plt.subplots(figsize=(17.6 * CM, 11.6 * CM))
    ax.set_xlim(0, 112); ax.set_ylim(0, 74); ax.axis("off")
    lanes = [("Trader or\nbroker", GREY), ("Off-chain\nsource", GREY), ("Attestation\nplane", VERM),
             ("Provenance\nplane", GREEN), ("Control\nplane", PURPLE), ("Customs\nofficer", GREY),
             ("Interoperability\nplane", BLUE), ("Partner\nadministration", GREY)]
    x0, dx = 18.0, 13.2
    X = [x0 + i * dx for i in range(len(lanes))]
    top, bot = 69.5, 2.0
    for (name, col), x in zip(lanes, X):
        ext = col == GREY
        ax.add_patch(FancyBboxPatch((x - 6.25, top - 0.6), 12.5, 5.0,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="#f6f6f6" if ext else tint(col, 0.12), ec=col, lw=0.85,
                                    ls=(0, (3, 2)) if ext else "-", zorder=3))
        ax.text(x, top + 1.9, name, ha="center", va="center", fontsize=5.4, color=INK,
                weight="normal" if ext else "bold", style="italic" if ext else "normal",
                linespacing=1.1, zorder=4)
        ax.plot([x, x], [bot, top - 0.6], color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)

    phases = [("Pre-arrival", 66.5, 43.0), ("At the border", 43.0, 25.5),
              ("After release", 25.5, 17.0), ("Cross-border\nexchange", 17.0, 2.0)]
    for i, (lab, y1, y0) in enumerate(phases):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.5, y0), 111, y1 - y0, fc="#f7f7f5", ec="none", zorder=0))
        ax.text(1.5, (y0 + y1) / 2, lab, fontsize=6.3, color=INK2, va="center", ha="left",
                style="italic", linespacing=1.1)

    def act(lane, y0, y1, col):
        ax.add_patch(Rectangle((X[lane] - 0.8, y0), 1.6, y1 - y0, fc=tint(col, 0.35), ec=col,
                               lw=0.6, zorder=3))

    def msg(n, a, b, y, text, dashed=False, col=INK2, lx=None, ha="left"):
        xa, xb = X[a], X[b]
        sgn = 1 if xb > xa else -1
        arrow(ax, [(xa + 0.8 * sgn, y), (xb - 0.8 * sgn, y)], None, color=col, dashed=dashed)
        cx = xa + 2.6 * sgn
        ax.add_patch(matplotlib.patches.Circle((cx, y), 1.2, fc=INK, ec="none", zorder=7))
        ax.text(cx, y, str(n), ha="center", va="center", fontsize=5.0, color="white",
                weight="bold", zorder=8)
        if lx is None:
            lx = min(xa, xb) + 1.6
        ax.text(lx, y + 1.8, text, ha=ha, va="center", fontsize=5.6, color=INK, zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.25))

    def selfmsg(n, a, y, text, col):
        x = X[a]
        ax.plot([x + 0.8, x + 3.6, x + 3.6], [y + 1.2, y + 1.2, y - 1.2], color=INK2, lw=0.8, zorder=2)
        ax.add_patch(FancyArrowPatch((x + 3.6, y - 1.2), (x + 0.8, y - 1.2),
                                     arrowstyle="-|>,head_length=3.0,head_width=1.7", color=INK2,
                                     lw=0.8, zorder=2))
        ax.text(x + 4.4, y, text, ha="left", va="center", fontsize=5.6, color=INK, zorder=6,
                linespacing=1.1, bbox=dict(fc="white", ec="none", pad=0.3))
        ax.add_patch(matplotlib.patches.Circle((x - 2.4, y), 1.2, fc=INK, ec="none", zorder=7))
        ax.text(x - 2.4, y, str(n), ha="center", va="center", fontsize=5.0, color="white",
                weight="bold", zorder=8)

    act(2, 47.6, 63.0, VERM)
    msg(1, 0, 2, 61.0, "lodge declaration $d$ in the system of record")
    msg(2, 1, 2, 56.0, "attest field group $f$: corroborate or contradict")
    selfmsg(3, 2, 50.8, "commit $c_f$, endorse,\norder and validate", VERM)
    msg(4, 2, 3, 45.4, "committed event $(s, f, a_{s,f})$")
    act(4, 28.8, 40.6, PURPLE)
    msg(5, 3, 4, 39.0, "matrix $a_{s,f}(d)$")
    selfmsg(6, 4, 35.4, "score $d$ and\napply rules", PURPLE)
    msg(7, 4, 5, 30.2, "recommendation and reasons")
    msg(8, 5, 0, 27.0, "release or inspection decision")
    msg(9, 5, 4, 20.0, "audited outcome, used as training labels", dashed=True)
    act(6, 4.0, 13.5, BLUE)
    msg(10, 7, 6, 12.0, "request attestation for $d$", lx=X[7] - 1.6, ha="right")
    msg(11, 6, 2, 8.5, "fetch commitment and endorsement signatures")
    msg(12, 6, 7, 4.5, "commitment, signatures, opening on consent", lx=X[7] - 1.6, ha="right")
    fig.savefig(os.path.join(FIG, "fig2_lifecycle.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, "fig2_lifecycle.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ workflow
def workflow():
    fig, ax = plt.subplots(figsize=(17.6 * CM, 10.4 * CM))
    ax.set_xlim(0, 112); ax.set_ylim(0, 66); ax.axis("off")
    FS = 5.8
    W, H = 16.6, 8.2
    xs = [1.0, 19.6, 38.2, 56.8, 75.4, 94.0]

    def lane(y0, y1, col, title):
        ax.add_patch(FancyBboxPatch((0.3, y0), 111.4, y1 - y0, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=tint(col, 0.06), ec=tint(col, 0.30), lw=0.7, zorder=0))
        ax.text(1.2, y1 - 1.6, title, fontsize=6.6, weight="bold", color=col, va="center")

    lane(42.0, 65.5, GREEN, "Data preparation (shared by E2, E3, E4 and the sensitivity analysis)")
    lane(17.5, 40.5, PURPLE, "Selectivity experiments, repeated for replications r = 1 to 20")
    lane(0.3, 16.0, VERM, "E1: reference implementation")

    y1 = 47.5
    comp(ax, xs[0], y1, W, H, "Dataset\n54,000 CTGAN rows\n22 attributes", GREEN, fs=FS)
    comp(ax, xs[1], y1, W, H, "Preprocessing\nsorted by date\n25 causal features", GREEN, fs=FS)
    comp(ax, xs[2], y1, W, H, "Calibration\nfraud cut to 3%\nn = 43,657", GREEN, fs=FS)
    comp(ax, xs[3], y1, W, H, "Temporal split\ntest 6,845 rows\n199 fraud", GREEN, fs=FS)
    store(ax, xs[4], y1 - 0.3, W, H + 0.6, "Revenue proxy\nprice x tax rate\ncapped at p99", GREEN, fs=FS)
    comp(ax, xs[5], y1, W, H, "Settings\nTable 4 or a\nsensitivity cell", GREY, fs=FS)
    for i in range(3):
        arrow(ax, [(xs[i] + W, y1 + H / 2), (xs[i + 1], y1 + H / 2)], None)

    y2 = 24.0
    comp(ax, xs[0], y2, W, H, "Attestation layer\ndrawn with\nseed r", PURPLE, fs=FS)
    store(ax, xs[1], y2 - 0.3, W, H + 0.6, "Matrix\n$a_{s,f}(d)$", PURPLE, fs=FS)
    comp(ax, xs[2], y2, W, H, "Scoring policy\nraw, weighted,\nraw plus weights", PURPLE, fs=FS)
    comp(ax, xs[3], y2, W, H, "Classifier\nboosted trees\nseed r", PURPLE, fs=FS)
    comp(ax, xs[4], y2, W, H, "Metrics\nAUC, AP, recall,\nrevenue proxy", PURPLE, fs=FS)
    comp(ax, xs[5], y2, W, H, "Inference\npaired contrasts\nbootstrap CI", PURPLE, fs=FS)
    for i in range(5):
        arrow(ax, [(xs[i] + W, y2 + H / 2), (xs[i + 1], y2 + H / 2)], None)
    # data preparation into the experiment lane
    arrow(ax, [(xs[3] + W / 2, y1), (xs[3] + W / 2, y2 + H)], "features, labels", (xs[3] + W / 2 + 5.6, 41.2))
    arrow(ax, [(xs[4] + W / 2, y1 - 0.3), (xs[4] + W / 2, y2 + H)], None)
    arrow(ax, [(xs[5] + W / 2, y1), (xs[5] + W / 2, 43.6), (xs[0] + W / 2, 43.6), (xs[0] + W / 2, y2 + H)],
          "parameters", (24.0, 43.6))
    # replication loop
    arrow(ax, [(xs[4] + W / 2, y2), (xs[4] + W / 2, 20.0), (xs[0] + W / 2, 20.0), (xs[0] + W / 2, y2)],
          "next replication r + 1, same draw for every policy", (47.0, 20.0), dashed=True)
    ax.text(xs[2] + W / 2, y2 + H + 2.0, "E3 compares policies, E4 adds audit lag or collusion",
            fontsize=5.4, color=INK2, ha="center", va="center", style="italic")

    y3 = 2.6
    comp(ax, xs[0], y3, W, H, "Reference node\nSHA-256, P-256,\nMVCC, SQLite", VERM, fs=FS)
    comp(ax, xs[1], y3, W, H, "Stage costs\ncommit, sign,\nverify", VERM, fs=FS)
    comp(ax, xs[2], y3, W, H, "Closed loop\n1, 3, 5 endorsers\nvolatile, durable", VERM, fs=FS)
    comp(ax, xs[3], y3, W, H, "Open loop\nPoisson arrivals\nbacklog kept", VERM, fs=FS)
    comp(ax, xs[4], y3, W, H, "Footprint\nrecord bytes\nand framing", VERM, fs=FS)
    comp(ax, xs[5], y3, W, H, "Capacity check\npeak demand\nvs throughput", VERM, fs=FS)
    for i in range(5):
        arrow(ax, [(xs[i] + W, y3 + H / 2), (xs[i + 1], y3 + H / 2)], None)
    fig.savefig(os.path.join(FIG, "fig3_workflow.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, "fig3_workflow.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    lifecycle()
    workflow()
