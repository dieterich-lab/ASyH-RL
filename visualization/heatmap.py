# -*- coding: gbk -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def plot_correlation_difference_labeled(real_path = 'train.csv',
                                        syn_path = 'synthetic_data_gmm_optimized.csv'):
    print("="*60)
    print("正在绘制【带特征标签】的相关性矩阵差值热力图")
    print("="*60)

    def load_data(path, sep=','):
        try: 
            return pd.read_csv(path, sep=sep, encoding='gbk')
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=sep, encoding='utf-8')
        except Exception as e:
            raise e

    try:
        df_real = load_data(real_path)
        df_syn = load_data(syn_path)
        if len(df_syn.columns) != len(df_real.columns):
            print(f"[警告] 合成数据列数 ({len(df_syn.columns)}) 与真实数据列数 ({len(df_real.columns)}) 不一致。请检查数据。")
            # 尝试使用分隔符 ';'
            try:
                df_syn = load_data(syn_path, sep=';')
            except:
                print("[错误] 使用分隔符 ';' 仍无法读取合成数据，请检查文件格式。")
                return

        # 2. 自动对齐列
        common_cols = [c for c in df_real.columns if c in df_syn.columns]
        ignore_cols = ['序号', 'ID', 'Unnamed: 0', 'id']
        numeric_cols = df_real.select_dtypes(include=[np.number]).columns
        final_cols = [c for c in common_cols if c in numeric_cols and c not in ignore_cols]
        
        print(f"对齐特征数量: {len(final_cols)} 个")
        
        df_real = df_real[final_cols]
        df_syn = df_syn[final_cols]

        # 3. 计算差值
        corr_real = df_real.corr()
        corr_syn = df_syn.corr()
        diff_matrix = corr_real - corr_syn

        # 4. 绘图
        # 【关键修改】画布调大，适应密集标签
        plt.figure(figsize=(20, 18)) 
        
        # 设置中文字体支持 (如果您的列名是中文)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        mask = np.triu(np.ones_like(diff_matrix, dtype=bool))

        sns.heatmap(diff_matrix, 
                    mask=mask,
                    cmap='RdBu_r',
                    center=0,
                    vmax=0.25, vmin=-0.25, 
                    square=True, 
                    annot=False, 
                    cbar_kws={"shrink": .8, "label": "Correlation Difference"},
                    # 【关键修改】开启标签
                    xticklabels=True, 
                    yticklabels=True)

        # 【关键修改】精细调整标签样式
        # 如果特征非常多(>50)，建议字体设为 5-6；如果较少(<20)，可以设为 10
        font_size = 6 if len(final_cols) > 50 else 10
        
        plt.xticks(rotation=90, fontsize=font_size, ha='right') # X轴垂直显示
        plt.yticks(rotation=0, fontsize=font_size)              # Y轴水平显示

        plt.title('Correlation Structure Preservation (Labeled)', 
                 fontsize=20, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = 'Figure_Correlation_Diff_Labeled.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[完成] 图片已保存为: {output_file}")
        print("提示：如果标签依然太密看不清，请手动在代码中减小 fontsize 或增大 figsize。")

    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    plot_correlation_difference_labeled(real_path='puf_narrow/train.csv',
                                        syn_path='tables4comparison/out_uf_all_tvae_util_v3.csv')