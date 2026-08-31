# -*- coding: gbk -*-
"""
Cardio-Privacy-GAN: Privacy Evaluation Suite
This script implements DCR and NNAA metrics to provide empirical evidence 
of privacy preservation.

Implementation Details for Reviewers:
- Metric: Euclidean Distance (L2 Norm)
- Scaling: Z-score Normalization (StandardScaler)
- Encoding: One-Hot Encoding for categorical variables
- Attacker Model: Linkage Attack (Identity Disclosure)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class PrivacyEvaluator:
    def __init__(self, real_path, syn_path):
        """
        Assumed Attacker Model: 
        The adversary possesses the synthetic dataset and seeks to re-identify 
        individuals by linking synthetic records to a known subset of real data.
        """
        self.real_df = self._load(real_path)
        self.syn_df = self._load(syn_path)

        if len(self.syn_df.columns) != len(self.real_df.columns):
            try:
                self.syn_df = self._load(syn_path, sep=';')
            except:
                raise ValueError("Value error happened")
            
        self._preprocess()
        
    def _load(self, path, sep=','):
        # Handles potential encoding issues during file read
        try:
            return pd.read_csv(path, encoding='utf-8', sep=sep) # .dropna()
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='gbk', sep=sep) # .dropna()

    def _convert_comma_decimal(self, val):
        if isinstance(val, str):                                                                                                           
            try:                     
                return float(val.replace(',', '.')) 
            except:                                                    
                return val                                             
        return val

    def _knn_impute_onehot(self, df, n_neighbors=5, cat_cols=None, num_cols=None):
            df = df.copy()
            if cat_cols is None:
                cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            if num_cols is None:
                num_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(cat_cols) == 0 and len(num_cols) == 0:
                return df
            
            # One-hot encode (missing stays as all-NaN row, then imputed)
            if len(cat_cols) > 0:
                df_cat = pd.get_dummies(df[cat_cols], dummy_na=True) # dummy_na=False)
                mask_nan = df[cat_cols].isna()
            else:
                df_cat = pd.DataFrame(index=df.index)
                mask_nan = pd.DataFrame(index=df.index)
            
            # Reintroduce NaNs for rows where original was missing
            for col in cat_cols:
                dummy_cols = [c for c in df_cat.columns if c.startswith(col + "_")]
                df_cat.loc[mask_nan[col], dummy_cols] = np.nan
            
            combined = pd.concat([df[num_cols], df_cat], axis=1)
            
            imputer = KNNImputer(n_neighbors=n_neighbors) # , keep_empty_features=True)
            imputed_arr = imputer.fit_transform(combined)

            # Defensive alignment in case backend behavior changes
            if imputed_arr.shape[1] != combined.shape[1]:
                imputed = pd.DataFrame(imputed_arr, index=df.index)
                imputed = imputed.reindex(columns=range(combined.shape[1]))
                imputed.columns = combined.columns
            else:
                imputed = pd.DataFrame(imputed_arr, columns=combined.columns, index=df.index)
            
            # Convert one-hot back to categorical via argmax
            for col in cat_cols:
                # try:
                dummy_cols = [c for c in imputed.columns if c.startswith(col + "_")]
                # imputed[col] = imputed[dummy_cols].idxmax(axis=1).str.replace(col + "_", "")
                # imputed.drop(columns=dummy_cols, inplace=True)
    
                # Prevent ValueError: idxmax on empty frame
                if not dummy_cols:
                    # keep column with missing values when no dummy feature exists
                    imputed[col] = pd.Series(pd.NA, index=imputed.index, dtype="object")
                    continue
    
                winners = imputed[dummy_cols].idxmax(axis=1)
                imputed[col] = winners.str.replace(col + "_", "", regex=False)
                imputed.drop(columns=dummy_cols, inplace=True)
    
                # except ValueError:
                    # print("Attempted to get argmax on an empty sequence! Entering debug mode ..")
                    # import ipdb; ipdb.set_trace()
            
            return imputed

    def _preprocess(self):
        # 1. Feature Selection: Align columns and drop identifiers
        common = [c for c in self.real_df.columns if c in self.syn_df.columns]
        # ignore = ['

        ignore = ['Unnamed: 0']
        cols = [c for c in common if c not in ignore]
        
        self.real = self.real_df[cols]
        self.syn = self.syn_df[cols]

        # apply the internal method `_convert_comma_decimal`
        self.real = self.real.applymap(self._convert_comma_decimal)
        self.syn = self.syn.applymap(self._convert_comma_decimal)

        # impute missing values in dataframes with kNN algorithm
        self.real = self._knn_impute_onehot(self.real, n_neighbors=5)
        self.syn = self._knn_impute_onehot(self.syn, n_neighbors=5)
        
        # 2. Pipeline for Scaling and Encoding
        # num_cols = self.real.select_dtypes(include=[np.number]).columns
        # cat_cols = self.real.select_dtypes(exclude=[np.number]).columns

        num_cols = []

        for col in self.real_df.columns:
            if self.real_df[col].nunique() > 20:
                num_cols.append(col)
        
        cat_cols = list([col for col in self.real_df.columns if col not in num_cols])
        
        # Explicit Scaling (StandardScaler) and Encoding (OneHotEncoder)
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])

        import ipdb; ipdb.set_trace()  # Debugging point to inspect
        # Fit on real distribution to establish the baseline feature space
        self.real_enc = preprocessor.fit_transform(self.real)
        self.syn_enc = preprocessor.transform(self.syn)
        
        print(f"Pre-processing Complete. Feature dimensions: {self.real_enc.shape[1]}")

    def evaluate_dcr(self):
        """
        Distance to Closest Record (DCR)
        Measures the L2 distance from each synthetic record to its nearest real neighbor.
        """
        print("\nCalculating DCR (Distance to Closest Record)...")
        
        # Explicitly using Euclidean Metric as requested by reviewers
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(self.real_enc)
        dists, _ = nn.kneighbors(self.syn_enc)
        
        min_dists = dists.flatten()
        exact_matches = np.sum(min_dists < 1e-5) # Potential Identity Leakage
        mean_dist = np.mean(min_dists)
        
        print(f"  -> Mean DCR: {mean_dist:.4f}")
        print(f"  -> Exact Matches (Privacy Leakage): {exact_matches} / {len(self.syn_enc)}")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.histplot(min_dists, kde=True, color='teal', bins=30)
        plt.title('DCR Distribution: Empirical Privacy Evidence', fontsize=14)
        plt.xlabel('Euclidean Distance (Normalized Space)')
        plt.ylabel('Frequency')
        plt.axvline(x=0.05, color='red', linestyle='--', label='Leakage Threshold')
        plt.legend()
        plt.savefig('Privacy_DCR_Analysis.png', dpi=300)
        print("  -> Plot saved: Privacy_DCR_Analysis.png")
        
        return mean_dist, exact_matches

    def evaluate_nnaa(self):
        """
        Nearest Neighbor Adversarial Accuracy (NNAA)
        ï¿½ï¿½ï¿½ï¿½Öµ: 0.5 (Ideal privacy where real and synthetic are indistinguishable).
        """
        print("\nCalculating NNAA (Nearest Neighbor Adversarial Accuracy)...")
        
        # Combine and label
        X = np.vstack([self.real_enc, self.syn_enc])
        y = np.array([0] * len(self.real_enc) + [1] * len(self.syn_enc))
        
        # Leave-one-out 1-NN check using Euclidean metric
        nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X)
        _, indices = nn.kneighbors(X)
        
        # Predict label based on the nearest neighbor (excluding itself)
        y_pred = y[indices[:, 1]]
        accuracy = np.mean(y == y_pred)
        
        print(f"  -> NNAA Score: {accuracy:.4f}")
        return accuracy

if __name__ == "__main__":
    print("-" * 50)
    print("EMPIRICAL PRIVACY AUDIT")
    print("-" * 50)
    
    # Update these paths to your actual file names
    evaluator = PrivacyEvaluator('puf_narrow/train.csv', 'tables4comparison/out_uf_all_tvae_util_v3.csv')
    
    dcr_val, leaks = evaluator.evaluate_dcr()
    nnaa_val = evaluator.evaluate_nnaa()
    
    print("\nSummary Report:")
    print(f"1. Identity Protection: {leaks} records leaked.")
    print(f"2. Membership Defense (NNAA): {nnaa_val:.4f} (Goal: 0.5)")
    print("\nNote: These results provide empirical evidence and not formal DP guarantees.")