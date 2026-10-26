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


def open_tables(path_main: str, path_synth: str, path_control: str) -> Tuple:
    df_main = pd.read_csv(path_main)
    df_synth = pd.read_csv(path_synth)
    df_control = pd.read_csv(path_control)
    return df_main, df_synth, df_control


def main():
    PATHS = ['puf_wide/train.csv',
             'tables4comparison/synthetic_data_gmm_wide_full.csv',
             'puf_wide/test.csv']
    
    data = open_tables(*PATHS)

    print(f"\n[Info] Running Anonymeter evaluation on the following datasets:\n"
          f"Main dataset: {PATHS[0]}\n"
          f"Synthetic dataset: {PATHS[1]}\n"
          f"Control dataset: {PATHS[2]}\n")

    results_anonym = run_anonymeter(*data, mode='univariate', aux_cols=[], n_attacks=400)

    # save it on the disk
    for attack_type, results_df in results_anonym.items():
        results_df.to_csv(f'anonymeter_results_{attack_type}.csv', index=False)
        print(f"[Done] Anonymeter results saved as: anonymeter_results_{attack_type}.csv")


if __name__ == "__main__":
    main()