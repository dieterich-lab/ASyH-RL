import pandas as pd
from typing import Union, Optional, Dict, Tuple
from anonymeter_evaluation.privacy_evaluation_script import run_anonymeter_attribute_inference                                                               
from anonymeter_evaluation.privacy_evaluation_script import run_anonymeter_singlingout                                                                       
from anonymeter_evaluation.privacy_evaluation_script import run_anonymeter_linkability


def run_anonymeter(data_main: pd.DataFrame, data_synth: pd.DataFrame, 
                   data_control: pd.DataFrame, mode: str = 'univariate', 
                   aux_cols: list = [], n_attacks: int = 400) -> Dict:
    '''
    It launches anonymity risk evaluations, on 3 main attack types: linkage,  singling out, and attribute inference

    Args:
        data_main: pd.DataFrame
        data_synth: pd.DataFrame
        data_control: pd.DataFrame
        mode: str

    Return:
        Dict
    where the returned dict has key-value pairs as <attack_type>:<risk report>
    '''                                                                                                                           
    results_df_link = run_anonymeter_linkability(data_main, data_synth, data_control, aux_cols, n_attacks=n_attacks)                              
    results_df_so = run_anonymeter_singlingout(data_main, data_synth,                                                                             
                                              data_control, mode=mode, n_attacks=n_attacks)                                                       
    results_df_ai = run_anonymeter_attribute_inference(data_main, data_synth,                                                                     
                                                       data_control, n_attacks=n_attacks)                                                         
    return {'singlingout':results_df_so,                                                                                                          
            'inference':results_df_ai,                                                                                                      
            'linkability':results_df_link}


def get_aux_columns(data_origin,                                                                                              
                    data_processed,                                                                                           
                    data_control,                                                                                             
                    unavailable_columns=['export_psn',                                                                        
                                         'mnpdocid',                                                                          
                                         'mnpvisid',                                                                          
                                         'mnpcvpid',                                                                          
                                         'mnpvislabel',                                                                       
                                         'mnpvisno'],                                                                         
                    available_columns=['Pat_ID.numeric',
                                       'ethnicity',
                                       'gender']):                                                                                    
                                                                                                                              
    shared_columns = list(set.intersection(*[set(x) for x in [data_origin.columns, data_processed.columns,                    
                                                              data_control.columns]]))                                        
    print("=== Column sets intersection: {} ===".format(shared_columns))                                                      
                                                                                                                              
    unavailable_columns = [x for x in unavailable_columns if x in shared_columns]                                             
                                                                                                                              
    if len(available_columns) > 0:                                                                                            
        available_columns = [x for x in available_columns if x in shared_columns]                                             
    else:                                                                                                                     
        available_columns = [x for x in shared_columns if x not in unavailable_columns]                                       
                                                                                                                              
    aux_columns = [                                                                                                           
        available_columns,  # available/leaked columns (laboratory results                                                    
        unavailable_columns # private/interesting columns (demographic data)                                                  
    ]                                                                                                                         
    return aux_columns 


def open_tables(path_main: str, path_synth: str, path_control: str) -> Tuple:
    df_main = pd.read_csv(path_main)
    df_synth = pd.read_csv(path_synth)
    df_control = pd.read_csv(path_control)
    return df_main, df_synth, df_control


def main():
    PATHS = ['puf_wide/train.csv',
            #  'tables4comparison/synthetic_data_GaussianCopulaModel_wide_rl_40ep.csv', # synthetic_data_gmm_wide_full.csv',
            # './synthetic_data_GaussianMixtureModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv',
            # './synthetic_data_GaussianCopulaModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv',
            # './synthetic_data_TVAEModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv',
            './synthetic_data_CTGANModel_wide_rl_50ep_pcs_score_2026-10-19-1830-long.csv',
             'puf_wide/test.csv']
    
    model_name = 'ctgan'
    
    data = open_tables(*PATHS)

    del data[1]['row_id']

    print(f"\n[Info] Running Anonymeter evaluation on the following datasets:\n"
          f"Main dataset: {PATHS[0]}\n"
          f"Synthetic dataset: {PATHS[1]}\n"
          f"Control dataset: {PATHS[2]}\n")

    # import ipdb; ipdb.set_trace()  # breakpoint to inspect variables

    aux_cols = get_aux_columns(*data)[0]

    results_anonym = run_anonymeter(*data, mode='univariate', aux_cols=aux_cols, n_attacks=100)

    # save it on the disk
    for attack_type, results_df in results_anonym.items():
        results_df.to_csv(f'anonymeter_results_{attack_type}_{model_name}.csv', index=True)
        print(f"[Done] Anonymeter results saved as: anonymeter_results_{attack_type}_{model_name}.csv")


if __name__ == "__main__":
    main()