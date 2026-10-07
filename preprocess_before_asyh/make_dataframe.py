import pandas as pd
import json

path_varset = '../../ap-4/NAPCODE/07_Variablenset/datasets/data.xlsx'

data_varset = pd.read_excel(path_varset)

cols_remain = ['Pat_ID', 'Pat_ID.numeric', 'Visite.numeric', 'Visite', 'date_infection_exact.factor', 'date_exam', 
               'datediff_to_inf', 'date_infection', 'date_infection_exact', 'age_infection']

for col in data_varset.columns:
    if col not in cols_remain and 'factor' not in col:
        cols_remain.append(col)

data_varset_sub =  data_varset[cols_remain]


data_varset_sub.to_csv('../puf_wide/uf_wide_y26m10.csv',sep=',')
# data_varset[cols_remain].shape
data_varset.to_csv('../puf_wide/uf_wide_factors_y26m10.csv',sep=',')


# P.S. Few lines for using a book of codes (it can be used on the core feature set)

# data_varset_sub.to_csv('puf_varset_wide_y26m10d06.csv', sep=',')

# path_pcs_varsatz = 'ap-4/NAPCODE/07_Variablenset/2026_05_12_NAPKON Post-COVID Variablensatz NAPKON PCS v1.8_BITTE PRÜFEN(1).xlsx'
# excel_pcs_varsatz = pd.ExcelFile(path_pcs_varsatz)
# df_pcs_varsatz = excel_pcs_varsatz.parse(excel_pcs_varsatz.sheet_names[0])

# cols4mapping = ['Category', 'Description', 'GECCO Kennung', 'Cohort', 'Codebook Kennung']
# df_pcs_4mapping = df_pcs_varsatz[cols4mapping]
# df_pcs_4mapping[['GECCO Kennung', 'Codebook Kennung']]
# df_pcs_4mapping[['Description', 'GECCO Kennung', 'Codebook Kennung']]

# cols4mapping = ['Category', 'Description', 'GECCO Kennung', 'Cohort', 'Codebook Kennung', 'Item_Kennung']
# cols4mapping = ['Category', 'Description', 'GECCO Kennung', 'Cohort', 'Codebook Kennung', 'Item Kennung']
# codebook2gecco = {x:y for x,y in zip(df_pcs_4mapping['Codebook Kennung'],df_pcs_4mapping['GECCO Kennung'])}
# gecco2codebook = {y:x for x,y in zip(df_pcs_4mapping['Codebook Kennung'],df_pcs_4mapping['GECCO Kennung'])}
# gecco2codebook = {y.strip():x for x,y in zip(df_pcs_4mapping['Codebook Kennung'],df_pcs_4mapping['GECCO Kennung'])}
# gecco2codebook = {str(y).strip():x for x,y in zip(df_pcs_4mapping['Codebook Kennung'],df_pcs_4mapping['GECCO Kennung'])}

# with open('mapping_gecco_to_codebook_codes.json', 'w') as flmap:
#     json.dump(gecco2codebook, flmap)
    
# mapp_preprocessing = pd.read_csv('Mapping_for_preprocessing_ALL.csv', sep=';')
# mapp_preprocessing['Codebook_Kennung'] = mapp_preprocessing['GECCO_Kennung'].apply(lambda x: gecco2codebook[x])
# mapp_preprocessing.to_csv('Mapping_for_preprocessing_codebook.csv')
# mapp_preprocessing.to_csv('Mapping_for_preprocessing_codebook.csv', sep=';')