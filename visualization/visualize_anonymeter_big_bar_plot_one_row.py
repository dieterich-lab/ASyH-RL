"""3 attack experiments x 2 fields (attack/baseline success rate) in ONE row of 6 subplots."""
import io
import os
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove for an interactive window
import matplotlib.pyplot as plt

MODELS = ["CTGAN", "TVAE", "Gaussian Copula", "Gaussian Mixture"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# field in the table -> label shown above the subplot
FIELDS = {
    "attack_rate success_rate": "Attack success rate",
    "baseline_rate success_rate": "Baseline success rate",
}

# Each experiment: 4 tables (CSV text OR file path), in the order of MODELS.
EXPERIMENTS = {
    "Privacy risk for singling out attack": [
        "risks/singling_out/ctgan.csv", "risks/singling_out/tvae.csv",
        "risks/singling_out/copula.csv", "risks/singling_out/mixture.csv"],
    "Privacy risk for linkability attack": [
        "risks/linkability/ctgan.csv", "risks/linkability/tvae.csv",
        "risks/linkability/copula.csv", "risks/linkability/mixture.csv"],
    "Privacy risk for inference attack": [
        "risks/inference/ctgan.csv", "risks/inference/tvae.csv",
        "risks/inference/copula.csv", "risks/inference/mixture.csv"],
}

# ---- font sizes ----
PANEL_TITLE_SIZE = 20
SUBPLOT_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 16

SHARE_Y_ACROSS_ALL = True   # same y-scale in all 6 subplots (easier to compare)


def parse_table(source):
    is_path = "\n" not in source and os.path.exists(source)
    df = pd.read_csv(source if is_path else io.StringIO(source), index_col=0)
    return df.iloc[:, 0]


def main(out="combined_plot_row.png"):
    data = {name: [parse_table(s) for s in srcs] for name, srcs in EXPERIMENTS.items()}
    ymax = max(float(t[f]) for ts in data.values() for t in ts for f in FIELDS) * 1.1

    fig = plt.figure(figsize=(24, 7))
    header, body = fig.subfigures(2, 1, height_ratios=[0.14, 1], hspace=0.02)

    # legend in the top right corner (own header strip => no overlap)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    header.legend(handles, MODELS, loc="upper right", ncol=len(MODELS),
                  fontsize=LEGEND_SIZE)

    # one column-group per experiment, each holding its 2 subplots
    groups = body.subfigures(1, len(EXPERIMENTS), wspace=0.04)
    for group, (name, tables) in zip(groups, data.items()):
        group.suptitle(textwrap.fill(name, 22), fontsize=PANEL_TITLE_SIZE,
                       fontweight="bold", y=0.99)
        axes = group.subplots(1, len(FIELDS), sharey=True)
        group.subplots_adjust(top=0.76, bottom=0.06, left=0.12, right=0.98, wspace=0.12)
        for ax, (field, label) in zip(axes, FIELDS.items()):
            ax.bar(range(len(MODELS)), [float(t[field]) for t in tables], color=COLORS)
            ax.set_title(textwrap.fill(label, 12), fontsize=SUBPLOT_TITLE_SIZE, pad=12)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=TICK_SIZE)
            if SHARE_Y_ACROSS_ALL:
                ax.set_ylim(0, ymax)
        axes[0].set_ylabel("success rate", fontsize=AXIS_LABEL_SIZE)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()