# -*- coding: gbk -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    csv_path = "results/ablation_summary_main.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Found no file at {csv_path}. Please run ablation script first.")
        return

    # 1. Load and Parse Data
    # Ensure read with utf-8 or specified encoding
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    def parse_stat(text):
        """Parse 'mean ± std' string into (mean, std) tuple"""
        try:
            parts = text.split(' ± ')
            return float(parts[0]), float(parts[1])
        except:
            return float(text), 0.0

    variants = df['Variant'].tolist()
    metrics = ['Combined', 'Utility', 'Fidelity']
    
    data = {}
    errors = {}
    
    for metric in metrics:
        parsed = df[metric].apply(parse_stat)
        data[metric] = [x[0] for x in parsed]
        errors[metric] = [x[1] for x in parsed]

    # 2. Plotting Configuration
    x = np.arange(len(variants))
    width = 0.25
    
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign display
    fig, ax = plt.subplots(figsize=(10, 6))

    # Using high-contrast colors for academic publication
    rects1 = ax.bar(x - width, data['Combined'], width, yerr=errors['Combined'], 
                    label='Combined Score', color='#1f77b4', capsize=5, alpha=0.9)
    rects2 = ax.bar(x, data['Utility'], width, yerr=errors['Utility'], 
                    label='Utility (TSTR)', color='#ff7f0e', capsize=5, alpha=0.9)
    rects3 = ax.bar(x + width, data['Fidelity'], width, yerr=errors['Fidelity'], 
                    label='Statistical Fidelity', color='#2ca02c', capsize=5, alpha=0.9)

    # 3. Refine Aesthetics
    ax.set_ylabel('Performance Metric Value', fontsize=12)
    ax.set_title('Ablation Study: Component Performance Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(0, 1.15) 
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()

    # 4. Save Image
    output_image = "results/ablation_performance_comparison.png"
    plt.savefig(output_image, dpi=300)
    print(f"Visualization saved to: {output_image}")

if __name__ == "__main__":
    main()