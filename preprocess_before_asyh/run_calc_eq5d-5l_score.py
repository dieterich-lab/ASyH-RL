import pandas as pd
from calc_eq5d_score import calc_eq5d_score
import argparse
# def calc_eq5d_score(idx_mo, idx_sc, idx_ua, idx_pd, idx_ad):


COLS_TYPE: int = 1  # 0 - descriptions,  1 - codebook codes, 2 - gecco codes,

EQ5D_COLS_DESC = ['Beweglichkeit_/_Mobilitaet', 'Fuer_sich_selbst_sorgen', 'Alltaegliche_Aktivitaeten',
             'Schmerzen_/_Koerperliche_Beschwerden', 'Angst_/_Niedergeschlagenheit']

EQ5D_COLS_GECCO = ['gecco_552', 'gecco_550', 'gecco_551', 'gecco_538', 'gecco_553']

EQ5D_COLS_CODEBOOK = ['covid19f-dataelement-552', 'covid19f-dataelement-550', 'covid19f-dataelement-551',
                      'covid19f-dataelement-538', 'covid19f-dataelement-553']

SUPSET_EQ5D_COLS = [EQ5D_COLS_DESC, EQ5D_COLS_GECCO]


def parse_args():
    parser = argparse.ArgumentParser(prog="Calculate utility metric - EQ-5D-5L", 
                                     description="utility metric calculation", 
                                     epilog='.')
    parser.add_argument("--in", dest="path_src_data", type=str, default='')
    parser.add_argument("--out", dest="path_out_data", type=str, default='')
    parser.add_argument("--cols-type", dest="cols_type", type=int, default=COLS_TYPE)
    parser.add_argument("--short-format", dest="short", type=bool, default=False)
    args = parser.parse_args()
    return args

args = parse_args()

PATH_UF_ALL = args.path_src_data
# PATH_UF_ALL = '/beegfs/homes/gsergei/napcode/data_proc/UF_ALL_columns_descriptions.csv'
if args.short:
    POSTFIX = '_start'
else:
    POSTFIX = ''

df_uf = pd.read_csv(PATH_UF_ALL, sep=';')
if not (len(df_uf.columns) > 1):
    df_uf = pd.read_csv(PATH_UF_ALL, sep=',')


PATH_OUT = args.path_out_data

EQ5D_COLS = SUPSET_EQ5D_COLS[args.cols_type]

# EQ5D_COLS = {}
# POP EQ5D columns
# Mobility, self-care, daily activity, pain / somatic symptoms, anxiety / depressed
# EQ5D_COLS['pop'] = ['eq5d5l1', 'eq5d5l2', 'eq5d5l3', 'eq5d5l4', 'eq5d5l5']

# for K,L in EQ5D_COLS.items():
#     for idx,_ in enumerate(L):
#         # element += '_start'
#         EQ5D_COLS[K][idx] += POSTFIX

for idx, col in enumerate(EQ5D_COLS):
    EQ5D_COLS[idx] += POSTFIX

# print(EQ5D_COLS['pop'])

def apply_calc_eq5d_score(df, col_names):
    '''
    Args:
        subframe: pd.DataFrame
        col_names: list[str]

    return : pd.Series
    '''
    # subframe has all required 5 columns for the score to be computed
    # (mobility, self-care, daily activity, pain / somatic symptoms, anxiety / depressed)
    # args = list([df[col_name] for col_name in col_names])
    print(df[col_names].head())
    # score_series = df[col_names].apply(calc_eq5d_score)
    # score_series = df.apply(lambda x: calc_eq5d_score(*list([x[col] for col in col_names])))
    score_series = df[col_names].apply(lambda row: calc_eq5d_score(*row), axis=1)
    print(pd.unique(score_series))
    return score_series


COL_SCORE = 'score-eq5d-5l' + POSTFIX

df_uf[COL_SCORE] = pd.Series()
df_uf[COL_SCORE] = apply_calc_eq5d_score(df_uf, EQ5D_COLS)

# First, let's compute the score for the pop cohort batch
# df_uf[COL_SCORE][df_uf['cohort'] == 'pop'] = apply_calc_eq5d_score(df_uf, EQ5D_COLS['pop'])
# df_uf[COL_SCORE][df_uf['cohort'] == 'hap'] = apply_calc_eq5d_score(df_uf, EQ5D_COLS['hap'])
# df_uf[COL_SCORE][df_uf['cohort'] == 'suep'] = apply_calc_eq5d_score(df_uf, EQ5D_COLS['suep'])

# print(df_uf.head())
# PATH_UF_ALL = '/beegfs/homes/gsergei/napcode/data_proc/ALL_UF_semicol.csv'
df_uf.to_csv(PATH_OUT, sep=';')
