import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, avoids hanging on a stale/unreachable X server
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')


# import sys

# def trace_lines(frame, event, arg):
#     if event == 'line':
#         print(f"{frame.f_code.co_filename}:{frame.f_lineno}")
#     return trace_lines

# sys.settrace(trace_lines)
THRESH_CAT_NUNIQ = 20

def convert_comma_decimal(val):
        if isinstance(val, str):
            try:
                return float(val.replace(',', '.'))
            except:
                return val
        return val


def plot_pathological_distributions_fixed(real_path='train.csv', syn_path='synthetic_data_gmm_optimized.csv'):
    print("=" * 60)
    print("Pathological distribution plot (fixed version)")
    print("Goal: avoid KeyError from missing columns")
    print("=" * 60)

    # --- 1. Robust CSV loading ---
    def robust_load(path, sep=','):
        # Try utf-8-sig first, then fallback encodings
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Failed to read file: {path}")

    # try:
    print(f"Loading real data: {real_path}")
    real = robust_load(real_path, sep=';')
    print(f"Loading synthetic data: {syn_path}")
    syn = robust_load(syn_path, sep=',')

    real = real.applymap(convert_comma_decimal)
    syn = syn.applymap(convert_comma_decimal)

    # --- 2. Column intersection filter ---
    # Use only common columns to avoid KeyError
    common_cols = set(real.columns) & set(syn.columns)
    # remove r'Pat_ID*' and 'gender' from `common_cols`
    cols_to_remove = ['gender', r'Pat_ID*']
    # use regex to find and remove these cols
    for col in cols_to_remove:
        common_cols = {c for c in common_cols if not pd.Series(c).str.contains(col, regex=True).any()}

    print(f"Number of common columns: {len(common_cols)}")

    cat_vars = ['date_infection_exact', 'gender', 'ethnicity', 'gecco_537', 
                                    'gecco_552', 'gecco_550', 'gecco_551', 
                                    'gecco_538', 'gecco_553', 'gecco_665', 'gecco_658', 
                                    'gecco_655', 'gecco_522', 'gecco_524', 'gecco_656', 
                                    'gecco_533', 'gecco_534', 'gecco_535', 'gecco_536', 
                                    'gecco_528', 'gecco_529_a', 'gecco_529_b', 'gecco_530', 
                                    'gecco_664', 'gecco_539', 'gecco_540', 
                                    'gecco_541', 'gecco_542', 'gecco_543', 'gecco_544', 
                                    'gecco_545', 'gecco_546', 'gecco_547', 
                                    'gecco_548', 'score-eq5d-5l', 'Cohort.numeric']

    # leave only cat_vars which are present in common_cols
    cat_vars = [col for col in cat_vars if col in common_cols]

    # check number of unique values in each of `cat_vars`, if the number of unique values exceeds `THRESH_CAT_NUNIQ`
    # then delete it from cat_vars
    cat_vars = [col for col in cat_vars if real[col].nunique() <= THRESH_CAT_NUNIQ and syn[col].nunique() <= THRESH_CAT_NUNIQ]

    selected_cat_vars = cat_vars
    # randomly choose 3 inside `selected_cat_vars`
    selected_cat_vars = np.random.choice(selected_cat_vars, size=min(3, len(selected_cat_vars)), replace=False).tolist()

    # get other columns `selected_cont_vars` which are the result of exclusion of `selected_cat_vars` from `common_cols`
    cont_vars = list(common_cols - set(cat_vars))

    # choose randomly 3 continuous vars
    selected_cont_vars = np.random.choice(cont_vars, size=min(3, len(cont_vars)), replace=False).tolist()

    print(f"Selected categorical variables: {selected_cat_vars}")
    print(f"Selected continuous variables: {selected_cont_vars}")

    # --- 5. Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    color_real = '#1f77b4'
    color_syn = '#d62728'

    # First row: continuous
    for i, col in enumerate(selected_cont_vars):
        if i >= 3:
            break
        ax = axes[0, i]
        sns.kdeplot(data=real, x=col, color=color_real, label='Real', fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(data=syn, x=col, color=color_syn, label='Synthetic', linestyle='--', linewidth=2, ax=ax)
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density' if i == 0 else '')
        if i == 0:
            ax.legend()

    # Second row: categorical
    for i, col in enumerate(selected_cat_vars):
        if i >= 3:
            break
        ax = axes[1, i]

        # Cap the plotted categories to real data's domain; some synthetic columns
        # (e.g. gecco_550/551/552) come out with tens of thousands of noisy unique
        # values instead of a handful of codes, which makes barplot pathologically slow.
        max_categories = 20
        top_categories = real[col].value_counts().nlargest(max_categories).index
        real_capped = real[col].where(real[col].isin(top_categories), 'Other')
        syn_capped = syn[col].where(syn[col].isin(top_categories), 'Other')

        r_counts = real_capped.value_counts(normalize=True).reset_index()
        r_counts.columns = [col, 'Proportion']
        r_counts['Source'] = 'Real'

        s_counts = syn_capped.value_counts(normalize=True).reset_index()
        s_counts.columns = [col, 'Proportion']
        s_counts['Source'] = 'Synthetic'

        combined = pd.concat([r_counts, s_counts]) 

        sns.barplot(data=combined, x=col, y='Proportion', hue='Source',
                    palette={'Real': color_real, 'Synthetic': color_syn}, errorbar=None, ax=ax)
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion' if i == 0 else '')
        ax.get_legend().remove()

    # Figure legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)

    plt.tight_layout()
    output_file = 'Figure_Pathology_Distributions_tight_df_full_CTGAN_rl_40ep.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[Done] Figure saved as: {output_file}")

    # except Exception as e:
    #     print(f"[Error] {e}")
    #     import traceback
    #     traceback.print_exc()


if __name__ == "__main__":
    # plot_pathological_distributions_fixed(real_path='puf_narrow/train.csv', syn_path='tables4comparison/synthetic_data_gmm_optimized_fixed.csv')
    plot_pathological_distributions_fixed(real_path='puf_wide/uf_wide_plus_target_numerics.csv',
                                          syn_path='tables4comparison/synthetic_data_CTGANModel_wide_full.csv')
                                          # 'tables4comparison/synthetic_data_TVAEModel_wide_rl_40ep.csv') # synthetic_data_GaussianCopulaModel_wide_rl_40ep.csv')
                                          # syn_path='tables4comparison/synthetic_data_gmm_wide_full.csv')