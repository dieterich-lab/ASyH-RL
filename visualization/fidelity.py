# -*- coding: gbk -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

def convert_comma_decimal(val):
        if isinstance(val, str):                                                                                                           
            try:                     
                return float(val.replace(',', '.')) 
            except:                                                    
                return val                                             
        return val


def plot_tsne_distribution(real_path = 'train.csv', syn_path = 'synthetic_data_gmm_optimized.csv'):
    print("="*60)
    print("正在绘制 t-SNE 全局分布图 (修复版)")
    print("目标: 证明合成数据与真实数据在流形空间上重合")
    print("="*60)
    
    if not os.path.exists(syn_path):
        print(f"[错误] 找不到合成数据: {syn_path}")
        return

    try:
        # 读取数据
        try: real = pd.read_csv(real_path, encoding='gbk')
        except: real = pd.read_csv(real_path, encoding='utf-8')
        
        try: 
            syn = pd.read_csv(syn_path, encoding='gbk')
        except: 
            syn = pd.read_csv(syn_path, encoding='utf-8')
        finally:
            if len(syn.columns) != len(real.columns):
                print(f"[警告] 合成数据列数 ({len(syn.columns)}) 与真实数据列数 ({len(real.columns)}) 不一致。请检查数据。")
                # open with the separator ';'
                try:
                    syn = pd.read_csv(syn_path, sep=';', encoding='utf-8', engine='python')
                except:
                    syn = pd.read_csv(syn_path, sep=';', encoding='gbk', engine='python')

        real = real.applymap(convert_comma_decimal)
        syn = syn.applymap(convert_comma_decimal)

        # 【关键修复 1】: 立即删除含有缺失值的行
        print(f"原始形状 - Real: {real.shape}, Syn: {syn.shape}")
        # real = real.dropna()
        # syn = syn.dropna()
        # print(f"清洗后形状 - Real: {real.shape}, Syn: {syn.shape}")

        # 【关键修复 2】: 确保列完全一致
        common_cols = [c for c in real.columns if c in syn.columns]
        # 排除 ID 列
        ignore_cols = ['序号', 'ID', 'id', 'Unnamed: 0']
        final_cols = [c for c in common_cols if c not in ignore_cols]
        
        real = real[final_cols]
        syn = syn[final_cols]

        # 采样以加快速度 (如果数据量太大，限制在 1000 条)
        n_sample = min(len(real), len(syn), 1000)
        real_sample = real.sample(n=n_sample, random_state=42)
        syn_sample = syn.sample(n=n_sample, random_state=42)
        
        # 添加标签
        real_sample['Label'] = 'Real'
        syn_sample['Label'] = 'Synthetic'
        
        # 合并
        df_all = pd.concat([real_sample, syn_sample], axis=0).reset_index(drop=True)
        labels = df_all['Label']
        df_model = df_all.drop(columns=['Label'])
        
        # 2. 预处理流水线
        # 自动识别列类型
        num_cols = df_model.select_dtypes(include=[np.number, 'number']).columns
        cat_cols = df_model.select_dtypes(exclude=[np.number], include=['object', 'category', 'bool']).columns
        
        print(f"数值列: {len(num_cols)}, 类别列: {len(cat_cols)}")

        # 【关键修复 3】: 加入 Imputer 防止标准化时报错
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), # 填充缺失值
            ('scaler', StandardScaler())                 # 标准化
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')), # 填充缺失值
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # 编码
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols)
            ])
        
        print("正在进行数据转换...")

        import ipdb; ipdb.set_trace()  # Debugging point to inspect

        data_scaled = preprocessor.fit_transform(df_model)
        
        # 检查是否有 NaN (双重保险)
        if np.isnan(data_scaled).any():
            print("[警告] 数据转换后仍含有 NaN，正在填充 0...")
            data_scaled = np.nan_to_num(data_scaled)
        
        # 3. 先用 PCA 降噪 (加速 t-SNE)
        print("正在执行 PCA 降维...")
        # 如果特征数很少，n_components 不能太大
        n_pca = min(30, data_scaled.shape[1])
        pca = PCA(n_components=n_pca)
        data_pca = pca.fit_transform(data_scaled)
        
        # 4. t-SNE 降维
        print("正在执行 t-SNE (这可能需要几秒钟)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(data_pca)
        
        # 5. 绘图
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")
        
        # 使用学术配色: 蓝色(Real) vs 红色(Synthetic)
        sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=labels,
            palette={'Real': '#1f77b4', 'Synthetic': '#d62728'},
            alpha=0.6, # 透明度，方便看重叠
            s=40,      # 点的大小
            edgecolor='w'
        )
        
        plt.title('t-SNE Visualization: Real vs. Synthetic Data Distribution', fontsize=15, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Data Source', loc='upper right')
        
        output_file = 'Figure_tSNE.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[成功] 绘图完成: {output_file}")

    except Exception as e:
        print(f"[严重错误] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_tsne_distribution(real_path='puf_narrow/train.csv',
                           syn_path='tables4comparison/out_uf_all_tvae_util_v3.csv')