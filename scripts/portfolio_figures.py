"""Portfolio summary figure (Direction 2 + E1). Reads raw result JSON, writes SVG+PNG.

Run from the repo root:  python3 scripts/portfolio_figures.py
E1 v1 pilot numbers are not in a JSON; they are copied from docs/direction2/results/e1-pilot.md
(pooled_adapt0 row and the honest-cost paragraph in section 4).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "portfolio"
OUT.mkdir(parents=True, exist_ok=True)

main = json.loads((ROOT / "results/direction2/main.json").read_text())
hold = json.loads((ROOT / "results/direction2/holdout/result.json").read_text())
e1 = json.loads((ROOT / "results/direction2/e1v2/result.json").read_text())

cell = next(c for c in main["cells"] if c["r"] == 0.5)
main_runs = [(r["D"]["mean"], r["D"]["lo"], r["D"]["hi"]) for r in cell["runs"]]
hold_runs = [(r["D"]["mean"], r["D"]["lo"], r["D"]["hi"]) for r in hold["runs"]]
delta = hold["delta"]  # 0.005

# colours chosen to read on white (figure carries its own white background)
BLUE, ORANGE, GREY, RED = "#1f5fbf", "#c25a00", "#555555", "#b3261e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1.1]})
fig.patch.set_facecolor("white")

# --- Panel A: Direction 2, advantage D of learned liar over best handwritten liar
ax = axs[0]
labels, rows, cols = [], [], []
for i, t in enumerate(main_runs):
    labels.append(f"main test\nrun {i}"); rows.append(t); cols.append(BLUE)
for i, t in enumerate(hold_runs):
    labels.append(f"holdout\nrun {i}"); rows.append(t); cols.append(ORANGE)
ys = [5, 4, 3, 1.8, 0.8, -0.2]
for y, (m, lo, hi), c in zip(ys, rows, cols):
    ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="butt")
    ax.plot(m, y, "o", color=c, ms=6)
ax.plot([delta, delta], [-0.7, 2.4], color=RED, ls="--", lw=1.4)  # holdout rows only (ys 1.8, 0.8, -0.2)
ax.axvline(0, color=GREY, lw=0.8)
ax.text(delta + 0.0003, 2.35, "holdout threshold 0.005\n(registered after main test)", color=RED, fontsize=7.5, va="top")
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_ylim(-0.9, 6.2); ax.set_xlim(-0.002, 0.021)
ax.set_xlabel("D = learned liar gain minus best handwritten gain")
ax.set_title("A. Learned liar beats the best handwritten one\n(95% two-sided; 32 seeds main, 64 holdout)", fontsize=10, loc="left")

# --- Panel B: E1 reduction of attacker gain
ax = axs[1]
v1 = (0.0086, 0.0047, 0.0125)
v2 = (e1["mean_D"], e1["LB"], e1["UB_D"])
for y, (m, lo, hi), c, lab in [(1, v1, GREY, "v1 pilot\nexploratory,\n95% interval"),
                               (0, v2, BLUE, "v2 preregistered\ntest, 1024 seeds,\n90% interval")]:
    ax.plot([lo, hi], [y, y], color=c, lw=2.6, solid_capstyle="butt")
    ax.plot(m, y, "o", color=c, ms=7)
ax.axvline(e1["delta"], color=RED, ls="--", lw=1.4)
ax.text(e1["delta"] + 0.0002, 1.6, "needed:\n0.005", color=RED, fontsize=8, va="top")
ax.set_yticks([1, 0]); ax.set_yticklabels(["v1 pilot\n(exploratory)", "v2 preregistered\n(final)"], fontsize=8.5)
ax.set_ylim(-0.7, 1.7); ax.set_xlim(0, 0.014)
ax.set_xlabel("reduction of the attacker's gain")
ax.set_title("B. Retuned audit: reduction falls short\n(v2 interval excludes 0.005)", fontsize=10, loc="left")
ax.annotate("v2: 0.00424\n[0.00388, 0.00459]", xy=(v2[0], 0), xytext=(0.0062, 0.12), fontsize=8, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE))
ax.text(0.0001, -0.55, "v1 selection and evaluation shared the same 32 seeds;\nv2 interval = two one-sided 95% bounds", fontsize=7, color=GREY, va="bottom")

# --- Panel C: honest cost, upper bound vs limit
ax = axs[2]
h = e1["honest"]
items = [("honest regret\n(limit 0.002)", 0.002, h["mean_regret"], h["ub_regret"], 0.00179, 0.0033),
         ("ineligible share\n(limit 0.01)", 0.01, h["mean_alpha"], h["ub_alpha"], 0.00653, 0.01179)]
# normalise to the limit so both items share one axis
for k, (lab, lim, m2, ub2, m1, ub1) in enumerate(items):
    y = 1.0 - k * 1.6
    ax.barh(y + 0.28, ub1 / lim, height=0.4, color="#ececec", edgecolor=GREY, linestyle="--")
    ax.plot(m1 / lim, y + 0.28, "o", color=GREY, ms=5)
    ax.barh(y - 0.2, ub2 / lim, height=0.4, color="#a9c4ee", edgecolor=BLUE)
    ax.plot(m2 / lim, y - 0.2, "o", color=BLUE, ms=5)
ax.axvline(1.0, color=RED, ls="--", lw=1.4)
ax.text(1.0, 1.75, "limit", color=RED, fontsize=8, ha="center")
ax.set_yticks([1.0, -0.6]); ax.set_yticklabels([items[0][0], items[1][0]], fontsize=8.5)
ax.set_ylim(-1.3, 1.9); ax.set_xlim(0, 1.9)
ax.set_xlabel("upper bound / limit (dot = estimate)\nv1: 95% CI end; v2: 1-sided 95% bound")
ax.set_title("C. Honest cost: within limit in v2, not free\n(grey = v1 exploratory, blue = v2)", fontsize=10, loc="left")

axs[2].text(0.97, 0.02, "different interval types:\nnot directly comparable", transform=axs[2].transAxes, ha="right", va="bottom", fontsize=7.5, color=RED)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(OUT / f"direction2-e1-summary.{ext}", dpi=170, facecolor="white")
print("wrote", OUT)
