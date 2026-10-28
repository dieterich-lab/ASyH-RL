"""Combine 3 attack experiments (4 model tables each) into one 3x3 figure."""
import io
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove for an interactive window
import matplotlib.pyplot as plt

MODELS = ["CTGAN", "TVAE", "Gaussian Copula", "Gaussian Mixture"]

FIELDS = ["attack_rate success_rate",
          "baseline_rate success_rate",
          "control_rate success_rate"]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

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

SHARE_Y_ACROSS_ALL = False   # True: one common y-scale for all 9 subplots


def parse_table(source):
    """CSV text or path to a CSV file -> Series indexed by field name."""
    is_path = "\n" not in source and os.path.exists(source)
    df = pd.read_csv(source if is_path else io.StringIO(source), index_col=0)
    return df.iloc[:, 0]


# def main(out="combined_plot.png"):
#     data = {name: [parse_table(s) for s in srcs] for name, srcs in EXPERIMENTS.items()}

#     fig = plt.figure(figsize=(15, 13))
#     subfigs = fig.subfigures(len(EXPERIMENTS), 1, hspace=0.05)
#     ymax = max(float(t[f]) for ts in data.values() for t in ts for f in FIELDS) * 1.1

#     for subfig, (name, tables) in zip(subfigs, data.items()):
#         subfig.suptitle(name, fontsize=14, fontweight="bold")
#         axes = subfig.subplots(1, len(FIELDS), sharey=True)
#         for ax, field in zip(axes, FIELDS):
#             ax.bar(range(len(MODELS)), [float(t[field]) for t in tables], color=COLORS)
#             ax.set_title(field, fontsize=10)
#             ax.set_xticks([])
#             if SHARE_Y_ACROSS_ALL:
#                 ax.set_ylim(0, ymax)
#         axes[0].set_ylabel("success rate")

#     handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
#     fig.legend(handles, MODELS, loc="upper right")   # one legend, top right
#     fig.savefig(out, dpi=150, bbox_inches="tight")
#     print("Saved", out)
# ---- font sizes: change these to taste ----
PANEL_TITLE_SIZE = 24     # "Privacy risk for ... attack"
SUBPLOT_TITLE_SIZE = 18   # "attack_rate success_rate", ...
AXIS_LABEL_SIZE = 16      # y-axis label
TICK_SIZE = 14            # y tick numbers
LEGEND_SIZE = 16
WRAP_SUBPLOT_TITLES = True   # put the title on 2 lines ("attack_rate\nsuccess_rate")

def main(out="combined_plot.png"):
    data = {name: [parse_table(s) for s in srcs] for name, srcs in EXPERIMENTS.items()}
    ymax = max(float(t[f]) for ts in data.values() for t in ts for f in FIELDS) * 1.2

    # fig = plt.figure(figsize=(18, 17))
    # # thin header row for the legend + one row per experiment
    # subfigs = fig.subfigures(len(EXPERIMENTS) + 1, 1, hspace=0.03,
    #                          height_ratios=[0.2] + [1] * len(EXPERIMENTS))
    fig = plt.figure(figsize=(18, 20))                       # was (18, 17)
    subfigs = fig.subfigures(len(EXPERIMENTS) + 1, 1, hspace=0.07,   # was 0.03
                            height_ratios=[0.12] + [1] * len(EXPERIMENTS))


    # legend in the top right corner (own header row => never overlaps titles)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    subfigs[0].legend(handles, MODELS, loc="upper right", ncol=len(MODELS),
                      fontsize=LEGEND_SIZE, frameon=True)

    for subfig, (name, tables) in zip(subfigs[1:], data.items()):
        # subfig.suptitle(name, fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        # axes = subfig.subplots(1, len(FIELDS), sharey=True)
        subfig.suptitle(name, fontsize=PANEL_TITLE_SIZE, fontweight="bold", y=0.99)
        axes = subfig.subplots(1, len(FIELDS), sharey=True)
        subfig.subplots_adjust(top=0.78, bottom=0.06, wspace=0.12)   # top=0.68 leaves ~32% of the row for titles
        for ax, field in zip(axes, FIELDS):
            ax.bar(range(len(MODELS)), [float(t[field]) for t in tables], color=COLORS)
            title = field.replace(" ", "\n") if WRAP_SUBPLOT_TITLES else field
            # ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE)
            ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, pad=14)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=TICK_SIZE)
            if SHARE_Y_ACROSS_ALL:
                ax.set_ylim(0, ymax)
        axes[0].set_ylabel("success rate", fontsize=AXIS_LABEL_SIZE)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main(out="combined_plot_adjusted.png")