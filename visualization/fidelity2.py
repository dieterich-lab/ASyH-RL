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
    print("���ڻ��Ƶ������ֲ�ͼ (�޸���)")
    print("Ŀ��: �����������? '����' ���µ� KeyError")
    print("="*60)
    
    # --- 1. ³�������ݼ��غ��� ---
    def robust_load(path, sep=','):
        # ���ȳ��� utf-8-sig (�ϳ�����ͨ�������?)
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"�޷���ȡ�ļ� {path}���������?")

    try:
        print(f"��ȡ��ʵ����: {real_path}")
        real = robust_load(real_path)
        print(f"��ȡ�ϳ�����: {syn_path}")
        syn = robust_load(syn_path)
        if len(syn.columns) != len(real.columns):
            print(f"[����] �ϳ��������� ({len(syn.columns)}) ����ʵ�������� ({len(real.columns)}) ��һ�¡��������ݡ�")
            # ����ʹ�÷ָ��� ';'
            try:
                syn = robust_load(syn_path, sep=';')
            except:
                print("[����] ʹ�÷ָ��� ';' ���޷���ȡ�ϳ����ݣ������ļ���ʽ��")
                return

        real = real.applymap(convert_comma_decimal)
        syn = syn.applymap(convert_comma_decimal)
        
        # --- 2. ����ɸѡ�� (�ؼ��޸�) ---
        # ȷ��ֻ�������ļ��ﶼ�е���
        common_cols = set(real.columns) & set(syn.columns)
        print(f"����������: {len(common_cols)}")
        
        if '����' not in common_cols:
            print("[����] '����' ��δ�ںϳ��������ҵ��������Ǳ������⵼���������롣")
            print("�ϳ���������ʾ��:", list(syn.columns)[:5])
            # �����Զ��޸����������һ�£�ǿ�ж�����������ѡ��?
            # syn.columns = real.columns 
        
        # --- 3. ѡ����ָ�� (Continuous) ---
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
            
        # ��λ
        if len(selected_cont_vars) < 3:
            remaining = [c for c in available_nums if c not in selected_cont_vars and c not in ['���?', 'ID']]
            selected_cont_vars.extend(remaining[:3-len(selected_cont_vars)])
            
        print(f"ѡ������ָ��: {selected_cont_vars}")

        # --- 4. ѡ����ɢָ�� (Categorical) ---
        # ֻ�� common_cols ��ѡ����ֹ KeyError
        target_cat_vars = [ 'status','sex', 'status', 'location', 'HBP']
        selected_cat_vars = [c for c in target_cat_vars if c in common_cols][:3]
        
        if len(selected_cat_vars) < 3:
             all_cats = [c for c in real.select_dtypes(exclude=[np.number]).columns if c in common_cols]
             selected_cat_vars.extend([c for c in all_cats if c not in selected_cat_vars][:3-len(selected_cat_vars)])

        print(f"ѡ����ɢָ��: {selected_cat_vars}")

        # --- 5. ��ͼ ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        color_real = '#1f77b4'
        color_syn = '#d62728'
        
        # ��һ�У�����
        for i, col in enumerate(selected_cont_vars):
            if i >= 3: break
            ax = axes[0, i]
            sns.kdeplot(data=real, x=col, color=color_real, label='Real', fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(data=syn, x=col, color=color_syn, label='Synthetic', linestyle='--', linewidth=2, ax=ax)
            ax.set_title(col, fontsize=12, fontweight='bold')
            ax.set_ylabel('Density' if i==0 else '')
            if i==0: ax.legend()

        # �ڶ��У���ɢ
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

        # ��ͼ��
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)
        
        plt.tight_layout()
        output_file = 'Figure_Pathology_Distributions_Fixed.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[�ɹ�] ͼƬ�ѱ���Ϊ: {output_file}")

    except Exception as e:
        print(f"[����] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # plot_pathological_distributions_fixed(real_path='puf_narrow/train.csv', syn_path='tables4comparison/synthetic_data_gmm_optimized_fixed.csv')
    plot_pathological_distributions_fixed(real_path='puf_wide/uf_wide_plus_target_numerics.csv', 
                                          syn_path='tables4comparison/synthetic_data_gmm_wide_full.csv')