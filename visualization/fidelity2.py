# -*- coding: gbk -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')


def convert_comma_decimal(val):
        if isinstance(val, str):                                                                                                           
            try:                     
                return float(val.replace(',', '.')) 
            except:                                                    
                return val                                             
        return val


def plot_pathological_distributions_fixed(real_path = 'train.csv', syn_path = 'synthetic_data_gmm_optimized.csv'):
    print("="*60)
    print("正在绘制单变量分布图 (修复版)")
    print("目标: 解决中文列名 '类型' 导致的 KeyError")
    print("="*60)
    
    # --- 1. 鲁棒的数据加载函数 ---
    def robust_load(path, sep=','):
        # 优先尝试 utf-8-sig (合成数据通常用这个)
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法读取文件 {path}，请检查编码")

    try:
        print(f"读取真实数据: {real_path}")
        real = robust_load(real_path)
        print(f"读取合成数据: {syn_path}")
        syn = robust_load(syn_path)
        if len(syn.columns) != len(real.columns):
            print(f"[警告] 合成数据列数 ({len(syn.columns)}) 与真实数据列数 ({len(real.columns)}) 不一致。请检查数据。")
            # 尝试使用分隔符 ';'
            try:
                syn = robust_load(syn_path, sep=';')
            except:
                print("[错误] 使用分隔符 ';' 仍无法读取合成数据，请检查文件格式。")
                return

        real = real.applymap(convert_comma_decimal)
        syn = syn.applymap(convert_comma_decimal)
        
        # --- 2. 智能筛选列 (关键修复) ---
        # 确保只画两个文件里都有的列
        common_cols = set(real.columns) & set(syn.columns)
        print(f"公共列数量: {len(common_cols)}")
        
        if '类型' not in common_cols:
            print("[警告] '类型' 列未在合成数据中找到。可能是编码问题导致列名乱码。")
            print("合成数据列名示例:", list(syn.columns)[:5])
            # 尝试自动修复：如果列数一致，强行对齐列名（可选）
            # syn.columns = real.columns 
        
        # --- 3. 选择病理指标 (Continuous) ---
        priority_keywords = [
            'QAWSS', 'WSS', 'OSI', 'RRT', 'Pressure', 'G', 'GON'
        ]
        
        available_nums = [c for c in real.select_dtypes(include=[np.number]).columns if c in common_cols]
        
        selected_cont_vars = []
        for keyword in priority_keywords:
            matches = [c for c in available_nums if keyword in c]
            for match in matches:
                if match not in selected_cont_vars:
                    selected_cont_vars.append(match)
                    if len(selected_cont_vars) >= 3: break
            if len(selected_cont_vars) >= 3: break
            
        # 补位
        if len(selected_cont_vars) < 3:
            remaining = [c for c in available_nums if c not in selected_cont_vars and c not in ['序号', 'ID']]
            selected_cont_vars.extend(remaining[:3-len(selected_cont_vars)])
            
        print(f"选定连续指标: {selected_cont_vars}")

        # --- 4. 选择离散指标 (Categorical) ---
        # 只从 common_cols 里选，防止 KeyError
        target_cat_vars = [ 'status','sex', 'status', 'location', 'HBP']
        selected_cat_vars = [c for c in target_cat_vars if c in common_cols][:3]
        
        if len(selected_cat_vars) < 3:
             all_cats = [c for c in real.select_dtypes(exclude=[np.number]).columns if c in common_cols]
             selected_cat_vars.extend([c for c in all_cats if c not in selected_cat_vars][:3-len(selected_cat_vars)])

        print(f"选定离散指标: {selected_cat_vars}")

        # --- 5. 绘图 ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        color_real = '#1f77b4'
        color_syn = '#d62728'
        
        # 第一行：连续
        for i, col in enumerate(selected_cont_vars):
            if i >= 3: break
            ax = axes[0, i]
            sns.kdeplot(data=real, x=col, color=color_real, label='Real', fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(data=syn, x=col, color=color_syn, label='Synthetic', linestyle='--', linewidth=2, ax=ax)
            ax.set_title(col, fontsize=12, fontweight='bold')
            ax.set_ylabel('Density' if i==0 else '')
            if i==0: ax.legend()

        # 第二行：离散
        for i, col in enumerate(selected_cat_vars):
            if i >= 3: break
            ax = axes[1, i]
            
            r_counts = real[col].value_counts(normalize=True).reset_index()
            r_counts.columns = [col, 'Proportion']
            r_counts['Source'] = 'Real'
            
            s_counts = syn[col].value_counts(normalize=True).reset_index()
            s_counts.columns = [col, 'Proportion']
            s_counts['Source'] = 'Synthetic'
            
            combined = pd.concat([r_counts, s_counts])
            
            sns.barplot(data=combined, x=col, y='Proportion', hue='Source', 
                        palette={'Real': color_real, 'Synthetic': color_syn}, ax=ax)
            ax.set_title(col, fontsize=12, fontweight='bold')
            ax.set_ylabel('Proportion' if i==0 else '')
            ax.get_legend().remove()

        # 总图例
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)
        
        plt.tight_layout()
        output_file = 'Figure_Pathology_Distributions_Fixed.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[成功] 图片已保存为: {output_file}")

    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_pathological_distributions_fixed(real_path='puf_narrow/train.csv', syn_path='tables4comparison/synthetic_data_gmm_optimized_fixed.csv')