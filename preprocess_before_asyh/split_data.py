# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

# The path to original table which is to be split
DATA_PATH = "puf_wide/uf_wide_plus_target_numerics.csv"

# Paths to train, validation and test subsets
TRAIN_PATH = "puf_wide/train.csv"
VAL_PATH   = "puf_wide/val.csv"
TEST_PATH  = "puf_wide/test.csv"

# Set here the target feature (used later to assess utility score)
TARGET_COL = "score-eq5d-5l"

# columns to be removed before the split
COLS_TO_DROP = ["export_psn", "Unnamed: 0",
                'date_infection', 'date_exam',
                'Pat_ID', 'Visite', 'Cohort']


RANDOM_STATE = 42


def print_dict(dct):
    print("Items held:")
    for item, amount in dct.items():  # dct.iteritems() in Python 2
        print("{} ({})".format(item, amount))


def main():

    df = pd.read_csv(DATA_PATH, sep=';', encoding="utf-8", on_bad_lines='skip')
    
    # get rid of unnecessary columns
    df.columns = df.columns.str.strip()
    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)
    # df = df.dropna()

    # check if all columns are numeric, if not - raise an error
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric. Please check the data.")

    print(f"Total samples after cleaning: {len(df)}")

    # 1b. If target values are continuous, make them discrete, class-like
    bin_size = 0.1
    df[TARGET_COL] = (df[TARGET_COL] // bin_size) * bin_size
    # # 1c. Filter out rare classes (in target col)
    df = df.groupby(TARGET_COL).filter(lambda x: len(x) >= 2)

    # split into train and val-test parts 80/20
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=RANDOM_STATE, 
        stratify=df[TARGET_COL]
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

    # import ipdb; ipdb.set_trace()

    for name, data in sets:
        counts = data[TARGET_COL].value_counts().to_dict()
        print(f"\n{name} ({len(data)} samples):")
        print_dict(counts)

    # Save resulting dataframes on disk as csv files
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")
    print(f"\nFiles saved: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")

if __name__ == "__main__":
    main()
