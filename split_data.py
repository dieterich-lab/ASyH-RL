# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

# The path to original table which is to be split
DATA_PATH = "puf_narrow/PUF_descriptions_util_score_v4.csv"

# Paths to train, validation and test subsets
TRAIN_PATH = "puf_narrow/train.csv"
VAL_PATH   = "puf_narrow/val.csv"
TEST_PATH  = "puf_narrow/test.csv"

# Set here the target feature (used later to assess utility score)
TARGET_COL = "score-eq5d-5l_start"
COLS_TO_DROP = ["export_psn", "Unnamed: 0"] # 预防数据泄露：移除唯一标识符


RANDOM_STATE = 42

def main():

    df = pd.read_csv(DATA_PATH, sep=';', encoding="utf-8", on_bad_lines='skip')
    
    # get rid of unnecessary columns
    df.columns = df.columns.str.strip()
    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)
    # df = df.dropna()
    print(f"Total samples after cleaning: {len(df)}")


    # 1b. If target values are continuous, make them discrete, class-like
    bin_size = 0.05
    df[TARGET_COL] = (df[TARGET_COL] // bin_size) * bin_size
    # # 1c. Filter out rare classes (in target col)
    df = df.groupby(TARGET_COL).filter(lambda x: len(x) >= 2)
    
    import ipdb; ipdb.set_trace()


    # split into train and val-test parts 80/20
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=RANDOM_STATE, 
        stratify=df[TARGET_COL] # 分层抽样保持类别比例
    )

    # split the remaining part into val. and test subsets 50/50
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=RANDOM_STATE, 
        stratify=temp_df[TARGET_COL]
    )

    sets = [
        ("Training Set", train_df),
        ("Validation Set", val_df),
        ("Testing Set", test_df)
    ]
    
    print("\n" + "="*60)
    print(f"{'Data Partition':<20} | {'Samples':<10} | {'Class 0':<10} | {'Class 1':<10}")
    print("-" * 60)
    for name, data in sets:
        counts = data[TARGET_COL].value_counts().to_dict()
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        print(f"{name:<20} | {len(data):<10} | {c0:<10} | {c1:<10}")
    print("="*60)

    # Save resulting dataframes on disk as csv files
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")
    print(f"\nFiles saved: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")

if __name__ == "__main__":
    main()
