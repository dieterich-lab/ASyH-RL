# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

# The path to original table which is to be split
# DATA_PATH = "puf_wide/uf_wide_plus_target_numerics.csv"
DATA_PATH = "puf_wide/uf_wide_y26m10_target_feat.csv"

# Paths to train, validation and test subsets
TRAIN_PATH = "puf_wide/train.csv"
VAL_PATH   = "puf_wide/val.csv"
TEST_PATH  = "puf_wide/test.csv"

# Set here the target feature (used later to assess utility score)
TARGET_COL = "score-eq5d-5l"

# columns to be removed before the split
COLS_TO_DROP = ["export_psn", "Unnamed: 0", 'Unnamed: 0.1']
                # 'date_infection', 'date_exam']
                # 'Visite', 'Cohort', 'Pat_ID']

COLS_TO_REMAIN = ['Pat_ID', 'Visite', 'Cohort']

COHORTS_UQ = ['hap', 'pop', 'suep']

RANDOM_STATE = 42


def print_dict(dct):
    print("Items held:")
    for item, amount in dct.items():  # dct.iteritems() in Python 2
        print("{} ({})".format(item, amount))


def main():

    df = pd.read_csv(DATA_PATH, sep=';', encoding="utf-8", on_bad_lines='skip')
    if not len(df.columns) > 1:
        df = pd.read_csv(DATA_PATH, sep=',', encoding="utf-8", on_bad_lines='skip')
    assert len(df.columns) > 1, "Dataframe has only one column. Please check the separator in the CSV file." 
    
    # get rid of unnecessary columns
    df.columns = df.columns.str.strip()
    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)
    # df = df.dropna()

    # introduce a new column "Cohort", on each row its value is the same 
    # as the first part of string in column "Pat_ID" before the undescore symbol '_'
    df['Cohort'] = df['Pat_ID'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else None)
    # replace the values in Cohort with indexes of respective string in the list `COHORTS_UQ`
    df['Cohort'] = df['Cohort'].apply(lambda x: COHORTS_UQ.index(x) if x in COHORTS_UQ else None)

    # encode all unique values in each of the columns inside COLS_TO_REMAIN with ordinal integers
    for col in COLS_TO_REMAIN:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
            # rename the column with new name col+'.numeric'
            df.rename(columns={col: col+'.numeric'}, inplace=True)

    # check if all columns are numeric or should remain, if not - raise an error
    # for col in df.columns:
    #     if not pd.api.types.is_numeric_dtype(df[col]) and not (col in COLS_TO_REMAIN):
    #         if col.endswith('.factor'):
    #             # delete the column if it is a factor column
    #             df.drop(columns=[col], inplace=True)
    #         else:
    #             raise ValueError(f"Column '{col}' is not numeric. Please check the data.")

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
