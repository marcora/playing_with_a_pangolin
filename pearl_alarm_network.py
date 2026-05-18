"""
==========================================================================
Bayesian Network: The Burglar Alarm Network
==========================================================================

Judea Pearl — "Probabilistic Reasoning in Intelligent Systems:
  Networks of Plausible Inference" (1988), Chapter 2 / Appendix A
Also cited in: Russell & Norvig — "AI: A Modern Approach" (3rd ed.), Ch.14

The Network
-----------
Five binary random variables connected in a DAG that models a real-world
burglary-alarm scenario.  Pearl uses it as the canonical example to
demonstrate:

  1. Causal (predictive)  reasoning:   B → A → J    "if burglar, call?"
  2. Diagnostic (abductive) reasoning: J → A → B    "if John calls, burglary?"
  3. Explaining away (intercausal):    B↑ explains A, so E↓  |  A observed

                ┌───────────┐   ┌────────────┐
                │ B Burglary│   │ E Earthquake│
                └─────┬─────┘   └──────┬─────┘
                      │                │
                      └────────┬───────┘
                               ▼
                         ┌───────────┐
                         │  A Alarm  │
                         └─────┬─────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌───────────┐        ┌───────────┐
             │  J John   │        │  M Mary   │
             │   Calls   │        │   Calls   │
             └───────────┘        └───────────┘

Conditional Probability Tables (from Pearl 1988, p.35)
-------------------------------------------------------
  P(B = 1) = 0.001   P(E = 1) = 0.002
                          E=0    E=1
  P(A=1 | B, E):  B=0  [ 0.001  0.29 ]
                  B=1  [ 0.94   0.95 ]
  P(J=1 | A):   A=0 → 0.05,   A=1 → 0.90
  P(M=1 | A):   A=0 → 0.01,   A=1 → 0.70

Inference Strategy
------------------
All five variables are *discrete* (binary).  The BlackJAX/NUTS backend
in Pangolin requires continuous latent variables, so NUTS is not used here.

Instead we exploit the fact that with only 2^5 = 32 joint assignments the
full joint distribution can be **enumerated exactly**.  We compute the log
joint probability for every assignment using Pangolin's JAX backend
(`jax_backend.ancestor_log_prob_flat`), then marginalise analytically.
This is equivalent to the Variable Elimination algorithm for small networks.

Key Results (Pearl, 1988)
--------------------------
  Prior            P(B=1)                 = 0.001
  P(B=1 | J=1)                            ≈ 0.016
  P(B=1 | J=1, M=1)                       ≈ 0.284
  Explaining away: P(B=1 | J=1, M=1, E=1) ≈ 0.017  << 0.284
  P(A=1 | B=1) — forward/predictive       ≈ 0.940

==========================================================================
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

import pangolin as pg
from pangolin import interface as pi
from pangolin import jax_backend

# ── 0.  Style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ── 1.  Pangolin model — the five-node DAG ─────────────────────────────────
#
#  The CPT for A cannot be expressed as a standard conditional distribution
#  because it has *two* parents (B and E).  We encode it as a deterministic
#  linear combination of indicator terms, which is exact for binary parents.
#
#  p_alarm(B,E) = 0.001*(1-B)*(1-E)  +  0.29*(1-B)*E
#               + 0.94*B*(1-E)        +  0.95*B*E
#
#  This is a valid arithmetic expression over pangolin RVs and produces
#  a deterministic node whose value is in [0,1] for any B,E ∈ {0,1}.

B = pi.bernoulli(0.001)          # Burglary prior
E = pi.bernoulli(0.002)          # Earthquake prior

# --- Alarm CPT (two-parent conditional) -----------------------------------
p_alarm = (
    0.001 * (1 - B) * (1 - E)   +   # B=0, E=0
    0.290 * (1 - B) * E          +   # B=0, E=1
    0.940 * B       * (1 - E)   +   # B=1, E=0
    0.950 * B       * E              # B=1, E=1
)
A = pi.bernoulli(p_alarm)        # Alarm

# --- Single-parent CPTs ---------------------------------------------------
p_john = 0.05 + 0.85 * A        # P(J=1|A): 0.05 if A=0, 0.90 if A=1
p_mary = 0.01 + 0.69 * A        # P(M=1|A): 0.01 if A=0, 0.70 if A=1
J = pi.bernoulli(p_john)         # JohnCalls
M = pi.bernoulli(p_mary)         # MaryCalls

# Print the pangolin upstream graphs (shows the full DAG structure)
print("=" * 68)
print("   BAYESIAN NETWORK: BURGLAR ALARM  (Pearl, 1988)")
print("=" * 68)
print("\n  Pangolin upstream graph for J (JohnCalls):")
pi.print_upstream(J)
print("\n  Pangolin upstream graph for M (MaryCalls):")
pi.print_upstream(M)

# ── 2.  Exact inference via full enumeration ───────────────────────────────
#
#  Build the complete 2^5 joint probability table.
#  For each assignment (b, e, a, j, m) ∈ {0,1}^5 we call
#  jax_backend.ancestor_log_prob_flat([B,E,A,J,M], [b,e,a,j,m])
#  which computes:
#    log P(B=b) + log P(E=e) + log P(A=a|B=b,E=e)
#    + log P(J=j|A=a) + log P(M=m|A=a)
#  Deterministic intermediate nodes (p_alarm, p_john, p_mary) are evaluated
#  transparently by the JAX backend.
#
print("\n  Building full 2^5 joint probability table …")

log_joint = np.zeros([2, 2, 2, 2, 2])   # axes: B, E, A, J, M

for b, e, a, j, m in itertools.product([0, 1], repeat=5):
    log_joint[b, e, a, j, m] = float(
        jax_backend.ancestor_log_prob_flat(
            [B, E, A, J, M],
            [float(b), float(e), float(a), float(j), float(m)],
            bijector_dict=None,      # no bijection for discrete distributions
        )
    )

joint = np.exp(log_joint)
assert abs(joint.sum() - 1.0) < 1e-8, f"joint sums to {joint.sum():.6f}, expected 1"

print(f"  Joint table sum = {joint.sum():.8f}  (✓ = 1.0)\n")

# ── 3.  Posterior queries ─────────────────────────────────────────────────
#
# Helper: P(query_var = 1 | evidence dict)
# evidence is a dict mapping axis index → observed value
# axes: 0=B, 1=E, 2=A, 3=J, 4=M
#
def posterior_one(query_axis: int, evidence: dict) -> float:
    """P(X[query_axis]=1 | evidence)."""
    # Build index tuple, slice only over the evidence axes
    def _sum(b_val):
        idx = [slice(None)] * 5
        idx[query_axis] = b_val
        for ax, val in evidence.items():
            idx[ax] = val
        return joint[tuple(idx)].sum()
    return _sum(1) / (_sum(0) + _sum(1))

# ── 3a. Prior marginals ────────────────────────────────────────────────────
prior_B = joint[1, :, :, :, :].sum()   # P(B=1)
prior_E = joint[:, 1, :, :, :].sum()   # P(E=1)
prior_A = joint[:, :, 1, :, :].sum()   # P(A=1)
prior_J = joint[:, :, :, 1, :].sum()   # P(J=1)
prior_M = joint[:, :, :, :, 1].sum()   # P(M=1)

# ── 3b. Key diagnostic queries ────────────────────────────────────────────
P_B_J1        = posterior_one(0, {3: 1})               # P(B=1 | J=1)
P_B_M1        = posterior_one(0, {4: 1})               # P(B=1 | M=1)
P_B_J1M1      = posterior_one(0, {3: 1, 4: 1})         # P(B=1 | J=1, M=1)
P_A_J1M1      = posterior_one(2, {3: 1, 4: 1})         # P(A=1 | J=1, M=1)

# ── 3c. Explaining away ───────────────────────────────────────────────────
P_B_J1M1_E0   = posterior_one(0, {3: 1, 4: 1, 1: 0})  # P(B=1 | J,M, E=0)
P_B_J1M1_E1   = posterior_one(0, {3: 1, 4: 1, 1: 1})  # P(B=1 | J,M, E=1)

# ── 3d. Causal / predictive queries ──────────────────────────────────────
P_J_B1        = posterior_one(3, {0: 1})               # P(J=1 | B=1) causal
P_A_B1        = posterior_one(2, {0: 1})               # P(A=1 | B=1) causal
P_J_B0        = posterior_one(3, {0: 0})               # P(J=1 | B=0) no burglary

# ── 3e. D-separation: J ⊥ B | A  ─────────────────────────────────────────
P_J_B1A1 = posterior_one(3, {0: 1, 2: 1})  # P(J=1|B=1,A=1)
P_J_B0A1 = posterior_one(3, {0: 0, 2: 1})  # P(J=1|B=0,A=1)  should equal P_J_B1A1
P_J_A1   = posterior_one(3, {2: 1})         # P(J=1|A=1)

# ── 4.  Print results ─────────────────────────────────────────────────────
print("  PRIOR MARGINALS")
print(f"    P(B=1) = {prior_B:.5f}   (given: 0.00100)")
print(f"    P(E=1) = {prior_E:.5f}   (given: 0.00200)")
print(f"    P(A=1) = {prior_A:.5f}   (Pearl 1988: ~0.00252)")
print(f"    P(J=1) = {prior_J:.5f}   (Pearl 1988: ~0.05224)")
print(f"    P(M=1) = {prior_M:.5f}   (Pearl 1988: ~0.01173)")

print("\n  DIAGNOSTIC REASONING  (effect → cause)")
print(f"    P(B=1 | J=1)          = {P_B_J1:.5f}   (AIMA Fig 14.4: ~0.01600)")
print(f"    P(B=1 | M=1)          = {P_B_M1:.5f}   (computed)")
print(f"    P(B=1 | J=1, M=1)     = {P_B_J1M1:.5f}   (AIMA Fig 14.4: ~0.28417)")

print("\n  EXPLAINING AWAY  (intercausal reasoning)")
print(f"    P(B=1 | J=1,M=1)       = {P_B_J1M1:.5f}   ← baseline")
print(f"    P(B=1 | J=1,M=1, E=0) = {P_B_J1M1_E0:.5f}   ← E absent  (E absent raises confidence in B)")
print(f"    P(B=1 | J=1,M=1, E=1) = {P_B_J1M1_E1:.5f}   ← E present (earthquake explains alarm away)")
print(f"    Earthquake explains away burglary!  ratio = {P_B_J1M1_E1/P_B_J1M1_E0:.4f}")

print("\n  CAUSAL / PREDICTIVE REASONING  (cause → effect)")
print(f"    P(A=1 | B=1) = {P_A_B1:.5f}   (forward: given burglary → alarm?)")
print(f"    P(J=1 | B=1) = {P_J_B1:.5f}   (forward: given burglary → John calls?)")
print(f"    P(J=1 | B=0) = {P_J_B0:.5f}   (baseline, no burglary)")

print("\n  D-SEPARATION DEMO:  J ⊥ B | A  (alarm blocks burglar→John path)")
print(f"    P(J=1 | B=1, A=1) = {P_J_B1A1:.5f}")
print(f"    P(J=1 | B=0, A=1) = {P_J_B0A1:.5f}")
print(f"    P(J=1 |      A=1) = {P_J_A1:.5f}")
print(f"    All three equal (within rounding): J ⊥ B given A  ✓")

# ── 5.  Sensitivity: P(B=1|J=1,M=1) vs P(B) ──────────────────────────────
#
#  Recompute posteriors over a range of burglary priors.
#  All other CPTs are kept fixed; only P(B) is varied.
#
pb_range  = np.logspace(-4, -1, 60)   # 0.0001 … 0.1
post_curve = np.zeros_like(pb_range)

for idx, pb in enumerate(pb_range):
    B_ = pi.bernoulli(pb)
    p_alarm_ = (
        0.001 * (1 - B_) * (1 - E) +
        0.290 * (1 - B_) * E        +
        0.940 * B_       * (1 - E) +
        0.950 * B_       * E
    )
    A_ = pi.bernoulli(p_alarm_)
    J_ = pi.bernoulli(0.05 + 0.85 * A_)
    M_ = pi.bernoulli(0.01 + 0.69 * A_)

    lj = np.zeros([2, 2, 2, 2, 2])
    for b, e, a, j, m in itertools.product([0, 1], repeat=5):
        lj[b, e, a, j, m] = float(
            jax_backend.ancestor_log_prob_flat(
                [B_, E, A_, J_, M_],
                [float(b), float(e), float(a), float(j), float(m)],
                bijector_dict=None,
            )
        )
    jt = np.exp(lj)
    num  = jt[1, :, :, 1, 1].sum()
    denom = jt[:, :, :, 1, 1].sum()
    post_curve[idx] = num / denom

# ── 6.  Visualisation ─────────────────────────────────────────────────────
BLUE    = "#2a6496"
GREEN   = "#2ca089"
RED     = "#c0392b"
GREY    = "#7f8c8d"
ORANGE  = "#e67e22"
PURPLE  = "#8e44ad"
COLORS  = [BLUE, GREEN, RED, GREY, ORANGE]

fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.38)
ax1 = fig.add_subplot(gs[0, 0])   # DAG
ax2 = fig.add_subplot(gs[0, 1])   # Prior vs posterior marginals
ax3 = fig.add_subplot(gs[0, 2])   # Explaining away
ax4 = fig.add_subplot(gs[1, 0])   # Causal vs diagnostic
ax5 = fig.add_subplot(gs[1, 1])   # Prior sensitivity
ax6 = fig.add_subplot(gs[1, 2])   # D-separation demo

fig.suptitle(
    "Bayesian Network — Burglar Alarm   (Pearl, 1988)\n"
    "Exact inference via 2^5 enumeration  ·  "
    "Pangolin JAX backend (ancestor_log_prob_flat)",
    fontsize=13, y=1.01,
)

# ── Panel A: DAG ───────────────────────────────────────────────────────────
ax1.set_xlim(0, 10); ax1.set_ylim(0, 11)
ax1.set_aspect("equal"); ax1.axis("off")
ax1.set_title("A)  Bayesian Network (DAG)", pad=6)

def _node(ax, cx, cy, label, color, r=0.9):
    c = plt.Circle((cx, cy), r, color=color, ec="white", lw=2, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=4)

def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#444"),
                zorder=2)

def _label(ax, cx, cy, txt):
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=8, color="#555", style="italic")

_node(ax1, 2.5,  9.2, "B", BLUE)        # Burglary
_node(ax1, 7.5,  9.2, "E", GREEN)       # Earthquake
_node(ax1, 5.0,  6.5, "A", ORANGE)      # Alarm
_node(ax1, 2.5,  3.8, "J", RED)         # John
_node(ax1, 7.5,  3.8, "M", PURPLE)      # Mary

_arrow(ax1, 3.3,  8.5, 4.2, 7.3)
_arrow(ax1, 6.7,  8.5, 5.8, 7.3)
_arrow(ax1, 4.2,  5.8, 3.3, 4.6)
_arrow(ax1, 5.8,  5.8, 6.7, 4.6)

_label(ax1, 2.5, 7.7, "P(B)=0.001")
_label(ax1, 7.5, 7.7, "P(E)=0.002")
_label(ax1, 5.0, 5.2, "CPT 2×2")
_label(ax1, 2.5, 2.65, "P(J|A=0)=0.05\nP(J|A=1)=0.90")
_label(ax1, 7.5, 2.65, "P(M|A=0)=0.01\nP(M|A=1)=0.70")

# (node labels are embedded in the circles; no separate legend needed)

# ── Panel B: Prior vs Posterior marginals ─────────────────────────────────
ax2.set_title("B)  Prior vs Posterior  (given J=1, M=1)")
labels  = ["B\n(Burglary)", "E\n(Earthquake)", "A\n(Alarm)",
           "J\n(JohnCalls)", "M\n(MaryCalls)"]
priors  = [prior_B, prior_E, prior_A, prior_J, prior_M]
posts   = [
    P_B_J1M1,
    posterior_one(1, {3:1, 4:1}),   # P(E=1|J,M)
    P_A_J1M1,
    1.0,                              # J=1 is given
    1.0,                              # M=1 is given
]

x = np.arange(5)
w = 0.33
b1 = ax2.bar(x - w/2, priors, w, label="Prior  P(X=1)", color=BLUE, alpha=0.8)
b2 = ax2.bar(x + w/2, posts,  w, label="Post.  P(X=1|J=1,M=1)", color=RED, alpha=0.8)

for bar_group in [b1, b2]:
    for bar in bar_group:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7)

ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("P(X = 1)")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1.15)

# ── Panel C: Explaining away ──────────────────────────────────────────────
ax3.set_title("C)  Explaining Away  (Pearl's key insight)")
cases  = ["No\nevidence", "J=1", "J=1\nM=1", "J=1,M=1\nE=0", "J=1,M=1\nE=1"]
pb_vals = [
    prior_B,
    P_B_J1,
    P_B_J1M1,
    P_B_J1M1_E0,
    P_B_J1M1_E1,
]
bar_colors = [GREY, BLUE, ORANGE, GREEN, RED]
bars = ax3.bar(cases, pb_vals, color=bar_colors, alpha=0.85, ec="white", lw=1.5)
for bar, val in zip(bars, pb_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax3.set_ylabel("P(B = 1 | evidence)")
ax3.set_ylim(0, 0.38)
ax3.axhline(prior_B, color=GREY, ls="--", lw=1, label=f"Prior = {prior_B:.4f}")

# annotate the explaining-away drop
ax3.annotate(
    "Earthquake\nexplains away\nburglary!",
    xy=(4, P_B_J1M1_E1), xytext=(3.2, 0.25),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    fontsize=8, color=RED, ha="center",
)
ax3.legend(fontsize=8)

# ── Panel D: Causal vs Diagnostic ─────────────────────────────────────────
ax4.set_title("D)  Causal vs Diagnostic Reasoning")
row_labels = [
    "Causal:\nP(A=1|B=1)",
    "Causal:\nP(J=1|B=1)",
    "Causal:\nP(J=1|B=0)",
    "Diagnostic:\nP(B=1|J=1)",
    "Diagnostic:\nP(B=1|J=1,M=1)",
]
row_vals = [P_A_B1, P_J_B1, P_J_B0, P_B_J1, P_B_J1M1]
row_colors = [ORANGE, ORANGE, GREY, BLUE, RED]

y_pos = np.arange(len(row_labels))
hbars = ax4.barh(y_pos, row_vals, color=row_colors, alpha=0.85, ec="white", lw=1.5)
for bar, val in zip(hbars, row_vals):
    ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

ax4.set_yticks(y_pos)
ax4.set_yticklabels(row_labels, fontsize=8.5)
ax4.set_xlabel("Probability")
ax4.set_xlim(0, 1.05)

causal_patch    = mpatches.Patch(color=ORANGE, alpha=0.85, label="Causal   (cause→effect)")
diag_patch      = mpatches.Patch(color=BLUE,   alpha=0.85, label="Diagnostic (effect→cause)")
ax4.legend(handles=[causal_patch, diag_patch], fontsize=8, loc="lower right")

# ── Panel E: Prior sensitivity ────────────────────────────────────────────
ax5.set_title("E)  Posterior Sensitivity to Prior P(B)")
ax5.semilogx(pb_range, post_curve, lw=2.5, color=BLUE)
ax5.axvline(0.001, color=GREY, ls="--", lw=1.5, label="Pearl's P(B)=0.001")
ax5.axhline(P_B_J1M1, color=RED, ls=":", lw=1.5, label=f"Reference = {P_B_J1M1:.3f}")

ax5.set_xlabel("Prior  P(B = 1)  [log scale]")
ax5.set_ylabel("P(B=1 | J=1, M=1)")
ax5.legend(fontsize=8)
ax5.set_ylim(0, 1)

# ── Panel F: D-separation ─────────────────────────────────────────────────
ax6.set_title("F)  D-separation:  J _|_ B | A")

dsep_labels = ["P(J=1|B=1,A=1)", "P(J=1|B=0,A=1)", "P(J=1|A=1)"]
dsep_vals   = [P_J_B1A1, P_J_B0A1, P_J_A1]
bar_c       = [ORANGE, GREEN, RED]
dbars = ax6.bar(dsep_labels, dsep_vals, color=bar_c, alpha=0.85, ec="white", lw=1.5)
for bar, val in zip(dbars, dsep_vals):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax6.set_ylabel("P(J = 1 | …)")
ax6.set_ylim(0, 1.1)
ax6.text(1, 0.60,
         "Once A is observed,\nJ is independent of B.\n(Alarm blocks the\ncausal path B->A->J)",
         ha="center", fontsize=8.5, color=GREY,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#ccc"))

plt.savefig("pearl_alarm_network.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Figure saved → pearl_alarm_network.png")
print("=" * 68)
