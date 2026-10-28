"""Bar plot of success rates from model-evaluation CSV tables.

Usage:
    python plot_tables.py                           # uses the tables embedded below
    python plot_tables.py a.csv b.csv c.csv d.csv   # uses CSV files, in model order
"""
import io
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line if you want an interactive window
import matplotlib.pyplot as plt

MODELS = ["CTGAN", "TVAE", "Gaussian Copula", "Gaussian Mixture"]
FIELDS = ["attack_rate success_rate",
          "baseline_rate success_rate",
          "control_rate success_rate"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Full tables exactly as provided, in the same order as MODELS.
RAW_TABLES = [
""",0
attack_rate success_rate,0.01849674910349284
attack_rate error,0.01849674910349284
baseline_rate success_rate,0.01849674910349284
baseline_rate error,0.01849674910349284
control_rate success_rate,0.01970094022309537
control_rate error,0.019662687134248402
n_attacks,100
n_baseline,0
n_control,0.12504496255845135
n_success,0
priv_risk,0.0
priv_risk_ci,0.0
can_be_used,False
""",
""",0
attack_rate success_rate,0.01849674910349284
attack_rate error,0.01849674910349284
baseline_rate success_rate,0.01849674910349284
baseline_rate error,0.01849674910349284
control_rate success_rate,0.01849674910349284
control_rate error,0.01849674910349284
n_attacks,100
n_baseline,0
n_control,0.0
n_success,0
priv_risk,0.0
priv_risk_ci,0.0
can_be_used,False
""",
""",0
attack_rate success_rate,0.028126814121422983
attack_rate error,0.026359382057282336
baseline_rate success_rate,0.01849674910349284
baseline_rate error,0.01849674910349284
control_rate success_rate,0.01970094022309537
control_rate error,0.019662687134248402
n_attacks,100
n_baseline,0
n_control,0.12504496255845135
n_success,1
priv_risk,0.008595207568846557
priv_risk_ci,0.0
can_be_used,True
""",
""",0
attack_rate success_rate,0.01849674910349284
attack_rate error,0.01849674910349284
baseline_rate success_rate,0.028126814121422983
baseline_rate error,0.026359382057282336
control_rate success_rate,0.01849674910349284
control_rate error,0.01849674910349284
n_attacks,100
n_baseline,1
n_control,0.0
n_success,0
priv_risk,0.0
priv_risk_ci,0.0
can_be_used,False
""",
]


def parse_table(source):
    """Parse a full table (CSV text or file path) into a Series indexed by field name."""
    buf = source if "\n" not in source and source.endswith(".csv") else io.StringIO(source)
    df = pd.read_csv(buf, index_col=0)
    return df.iloc[:, 0]          # the single value column (named "0")


def main(sources):
    if len(sources) != len(MODELS):
        sys.exit(f"Expected {len(MODELS)} tables (order: {', '.join(MODELS)}), got {len(sources)}")
    series = {m: parse_table(s) for m, s in zip(MODELS, sources)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, field in zip(axes, FIELDS):
        values = [float(series[m][field]) for m in MODELS]
        ax.bar(range(len(MODELS)), values, color=COLORS)
        ax.set_title(field)
        ax.set_xticks([])
    axes[0].set_ylabel("success rate")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, MODELS, loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig("plot_anonymeter_bars.png", dpi=150)
    print("Saved the plot")
    for f in FIELDS:
        print(f, [round(float(series[m][f]), 5) for m in MODELS])


if __name__ == "__main__":
    main(sys.argv[1:] or RAW_TABLES)