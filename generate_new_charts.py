"""
Supplementary charts for the expanded coordination experiment (P1-P5).

Reuses the seed-averaged summaries produced by process.py
(`data_summary_mean` / `data_summary_std`) so it does NOT re-run any
algorithm nor re-parse the raw CSVs. Same visual language as process.py:
viridis palette, usetex, seaborn, constrained layout, PDF export.

Outputs land in charts/experiment_export/custom/ (next to the existing
paper charts). Run:  uv run python generate_new_charts.py
"""
import os
import pickle
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Setup: match process.py styling
# ----------------------------------------------------------------------------
EXPERIMENT = "experiment_export"
OUT_DIR = f"charts/{EXPERIMENT}/custom"
FIG_DIR = "figures/charts"
os.makedirs(OUT_DIR, exist_ok=True)

matplotlib.rcParams.update({"axes.titlesize": 12, "axes.labelsize": 10})
try:
    plt.rc("text.latex", preamble=r"\usepackage{amsmath,amssymb,amsfonts,graphicx}")
    plt.rcParams.update({"text.usetex": True})
    _ = matplotlib.figure.Figure()  # cheap; real check happens at first savefig
except Exception:
    plt.rcParams.update({"text.usetex": False})

means = pickle.load(open("data_summary_mean", "rb"))[EXPERIMENT]
stdevs = pickle.load(open("data_summary_std", "rb"))[EXPERIMENT]

viridis = plt.colormaps["viridis"]
palette = {
    "bc_re": viridis(0.0),
    "bc_re_c": viridis(0.1),
    "ff_linpro": viridis(0.2),
    "ff_linpro_c": viridis(0.3),
    "ff_linpro_ac": viridis(0.4),
    "sm_av": viridis(0.8),
    "sm_av_c": viridis(0.7),
    "sm_av_ac": viridis(0.9),
}

# Ablation ordering: each base immediately followed by its clustered variants,
# grouped by base family so the base -> clustered lift reads left to right.
FAMILIES = [
    (r"\texttt{bc\_re}", ["bc_re", "bc_re_c"]),
    (r"\texttt{sm\_av}", ["sm_av", "sm_av_c"]),
    (r"\texttt{ff\_linpro}", ["ff_linpro", "ff_linpro_c", "ff_linpro_ac"]),
]
ORDER = [a for _, fam in FAMILIES for a in fam]

minTime, maxTime = 300, 1500
time = np.linspace(minTime, maxTime, means.sizes["time"])

DIAGONAL = [(1.0, 2.0), (2.0, 4.0), (3.0, 8.0)]  # (nu, zeta) as in selected_global


def scenario_title(nu, zeta):
    return r"$\nu$=" + str(int(nu)) + r" - $\zeta$=" + str(int(zeta))


def global_metric(ds):
    return ds["BodyCoverage[mean]"] * ds["FovDistance[mean]"] * (
        1 - ds["NoisePerceivedNormalized[mean]"]
    )


def shade_families(ax):
    """Light background bands separating the base-algorithm families.

    The x tick labels already carry the family names (e.g. bc_re / bc_re_c),
    so the shaded band alone is enough to read the grouping without adding
    text that would collide with the panel title.
    """
    pos = 0
    ymin, ymax = ax.get_ylim()
    for i, (_fname, fam) in enumerate(FAMILIES):
        width = len(fam)
        if i % 2 == 1:
            ax.axvspan(pos - 0.5, pos + width - 0.5, color="0.9", zorder=0)
        pos += width
    ax.set_ylim(ymin, ymax)


# ----------------------------------------------------------------------------
# P1 - Clustering ablation grouped by base algorithm
# ----------------------------------------------------------------------------
def chart_p1():
    means_ = means.assign(GlobalMetric=global_metric)
    fig, axes = plt.subplots(1, len(DIAGONAL), figsize=(18, 4.5), layout="constrained")
    for ax, (nu, zeta) in zip(axes, DIAGONAL):
        df = (
            means_.sel(CamHerdRatio=nu, NumberOfHerds=zeta)["GlobalMetric"]
            .to_dataframe()
            .reset_index()
        )
        sns.boxplot(df, ax=ax, x="Algorithm", y="GlobalMetric", order=ORDER,
                    hue="Algorithm", hue_order=ORDER, palette=palette, legend=False)
        ax.set_title(scenario_title(nu, zeta), fontsize=16)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.tick_params(labelsize=13, axis="x", rotation=45)
        ax.set(ylabel=r"$G$", xlabel="")
        ax.yaxis.get_label().set_fontsize(17)
        shade_families(ax)
    fig.suptitle(r"Clustering ablation: base vs.\ \texttt{\_c} (static) vs.\ "
                 r"\texttt{\_ac} (adaptive)", fontsize=18)
    fig.savefig(f"{OUT_DIR}/ablation_global_metric_by_family.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# P2 - Adaptive vs static clustering threshold
# ----------------------------------------------------------------------------
def chart_p2():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")

    # (a) threshold over time at the largest scenario
    ax = axes[0]
    for algo, style in [("ff_linpro_c", "--"), ("ff_linpro_ac", "-")]:
        ts = means["ClusteringDistance[mean]"].sel(
            Algorithm=algo, CamHerdRatio=3.0, NumberOfHerds=8.0
        ).values
        ax.plot(time, ts, style, color=palette[algo], linewidth=2.2,
                label=algo.replace("_", r"\_"))
    ax.set_title(r"Cut threshold over time (" + scenario_title(3.0, 8.0) + ")",
                 fontsize=14)
    ax.set_xlabel(r"time (s)", fontsize=14)
    ax.set_ylabel(r"clustering cut distance (m)", fontsize=13)
    ax.set_xlim(minTime, maxTime)
    ax.set_ylim(52, 63)
    ax.grid(True)
    ax.legend(fontsize=12)

    # (b) per-scenario distribution of the adaptive threshold (over time),
    #     one box per scenario, vs the static _c reference pinned at 60 m.
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ax = axes[1]
    order = []
    rows = []
    for nu in [1.0, 2.0, 3.0]:
        for zeta in [2.0, 4.0, 8.0]:
            label = f"({int(nu)},{int(zeta)})"
            order.append(label)
            for v in means["ClusteringDistance[mean]"].sel(
                Algorithm="ff_linpro_ac", CamHerdRatio=nu, NumberOfHerds=zeta
            ).values:
                rows.append({"scenario": label, "cut": float(v)})
    df = pd.DataFrame(rows)
    sns.boxplot(df, ax=ax, x="scenario", y="cut", order=order,
                color=palette["ff_linpro_ac"])
    ax.axhline(60, ls="--", color=palette["ff_linpro_c"], linewidth=2)
    ax.set_xlabel(r"scenario $(\nu,\zeta)$", fontsize=13)
    ax.set_ylabel(r"clustering cut distance (m)", fontsize=13)
    ax.set_ylim(52, 63)
    ax.tick_params(axis="x", rotation=45, labelsize=11)
    ax.grid(True, axis="y")
    ax.legend(handles=[
        Patch(facecolor=palette["ff_linpro_ac"], edgecolor="black",
              label=r"\texttt{ff\_linpro\_ac} (adaptive)"),
        Line2D([0], [0], ls="--", color=palette["ff_linpro_c"], linewidth=2,
               label=r"\texttt{ff\_linpro\_c} (static, 60\,m)"),
    ], fontsize=11, loc="lower right")
    fig.savefig(f"{OUT_DIR}/adaptive_vs_static_clustering.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# P3 - Worst-case (tail) perceived noise vs disturbance ceiling
# ----------------------------------------------------------------------------
def chart_p3():
    BACKGROUND_DB, CEILING_DB = 40, 80
    var = "NoisePerceived[max]"
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    df = means[var].to_dataframe().reset_index()
    sns.boxplot(df, ax=ax, x="Algorithm", y=var, order=ORDER,
                hue="Algorithm", hue_order=ORDER, palette=palette, legend=False)
    ax.axhline(BACKGROUND_DB, ls="--", color="0.4", linewidth=1.5,
               label=r"background ($\sim$40\,dB)")
    ax.axhline(CEILING_DB, ls="-", color="crimson", linewidth=2,
               label=r"disturbance ceiling (80\,dB)")
    ax.set_title("Maximum perceived noise", fontsize=15)
    ax.set_ylim(0, 85)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.tick_params(labelsize=12, axis="x", rotation=45)
    ax.set(ylabel=r"$L_{P_T}$ (dB)", xlabel="")
    ax.yaxis.get_label().set_fontsize(15)
    ax.legend(fontsize=11, loc="upper left")
    fig.savefig(f"{OUT_DIR}/tail_noise_vs_ceiling.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# P4 - Coverage vs noise trade-off (per-scenario cloud + average marker)
# ----------------------------------------------------------------------------
def chart_p4():
    """2D coverage-noise trade-off. Each algorithm shows all 9 scenario means as
    a faded cloud plus a large marker for its overall average, so both the
    frontier and the spread across scenarios are visible."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    # per-point label offsets to avoid overlapping annotations
    label_offset = {
        "ff_linpro_c": (10, 9),
        "ff_linpro_ac": (12, -17),
        "sm_av": (9, -3),
    }
    for algo in ORDER:
        xs, ys = [], []
        for nu in [1.0, 2.0, 3.0]:
            for zeta in [2.0, 4.0, 8.0]:
                sel = dict(Algorithm=algo, CamHerdRatio=nu, NumberOfHerds=zeta)
                xs.append(float(means["NoisePerceivedNormalized[mean]"].sel(sel).mean(dim="time")))
                ys.append(float(means["BodyCoverage[mean]"].sel(sel).mean(dim="time")))
        ax.scatter(xs, ys, color=palette[algo], s=45, alpha=0.55, edgecolors="none")
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], color=palette[algo], s=170, edgecolors="black",
                   linewidths=1.2, zorder=3)
        ax.annotate(algo.replace("_", r"\_"), (mx, my),
                    textcoords="offset points",
                    xytext=label_offset.get(algo, (9, 6)), fontsize=11)
    ax.set_xlabel(r"perceived noise $\rho$ (normalized)", fontsize=15)
    ax.set_ylabel(r"body coverage $\Diamond$", fontsize=15)
    ax.set_title(r"Coverage--noise trade-off (per-scenario means; "
                 r"large marker = average)", fontsize=14)
    ax.grid(True)
    fig.savefig(f"{OUT_DIR}/coverage_noise_tradeoff.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# P5 - Scaling of the global metric with herd count and fleet ratio
# ----------------------------------------------------------------------------
def chart_p5():
    means_ = means.assign(GlobalMetric=global_metric)
    zetas = [2.0, 4.0, 8.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=False,
                             layout="constrained")
    for ax, nu in zip(axes, [1.0, 2.0, 3.0]):
        for algo in ORDER:
            ys = [float(means_["GlobalMetric"].sel(
                Algorithm=algo, CamHerdRatio=nu, NumberOfHerds=z).mean(dim="time"))
                for z in zetas]
            ax.plot(zetas, ys, "-o", color=palette[algo], linewidth=2, ms=7,
                    label=algo.replace("_", r"\_"))
        ax.set_title(scenario_title(nu, 0).split(" - ")[0] + r"  (cameras/herd)",
                     fontsize=14)
        ax.set_xlabel(r"number of herds $\zeta$", fontsize=14)
        ax.set_ylabel(r"$G$", fontsize=16)
        ax.set_xticks(zetas)
        ax.grid(True)
    axes[-1].legend(fontsize=10, loc="upper left", ncol=2)
    fig.suptitle(r"Scaling of the global metric $G$ with herd count", fontsize=17)
    fig.savefig(f"{OUT_DIR}/scaling_global_metric.pdf")
    plt.close(fig)


NEW_CHARTS = [
    "ablation_global_metric_by_family.pdf",
    "adaptive_vs_static_clustering.pdf",
    "tail_noise_vs_ceiling.pdf",
    "coverage_noise_tradeoff.pdf",
    "scaling_global_metric.pdf",
]

if __name__ == "__main__":
    chart_p1()
    chart_p2()
    chart_p3()
    chart_p4()
    chart_p5()
    os.makedirs(FIG_DIR, exist_ok=True)
    for name in NEW_CHARTS:
        shutil.copy(f"{OUT_DIR}/{name}", f"{FIG_DIR}/{name}")
    print("Generated and copied to figures/charts/:")
    for name in NEW_CHARTS:
        print("  ", name)
