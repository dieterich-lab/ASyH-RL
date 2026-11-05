import pandas as pd
from typing import Union, Optional, Dict, Tuple, Any
from anonymeter_evaluation.privacy_evaluation_script import run_anonymeter_attribute_inference                                                               
from anonymeter_evaluation.privacy_evaluation_script import run_anonymeter_singlingout                                                                       
from anonymeter_evaluation.privacy_evaluation_script import run_anonymeter_linkability


class PrivacyEvaluator:
    
    def __init__(self, real_train: pd.DataFrame, real_test: pd.DataFrame, 
                    synthetic_data: pd.DataFrame, 
                    discrete_columns: list = None,
                    aux_cols: list = None):
        """
        Initializes the evaluator with pre-split training and testing sets.
        A numeric target with more than `n_target_bins` unique values is quantile-binned into classes.
        """
        self.data_main = real_train
        self.data_control = real_test
        self.data_synth = synthetic_data
        self.aux_cols = aux_cols
        if self.aux_cols is None:
            self.aux_cols = self.get_aux_columns(real_train, real_test, synthetic_data)


    def privacy_risk_evaluation(self, 
                                n_attacks: int = 100, 
                                mode: str = 'univariate') -> Tuple:
        data_main = self.data_main
        data_synth = self.data_synth
        data_control = self.data_control
        aux_cols = self.aux_cols

        print("aux_cols:", aux_cols)
        print(data_main.shape, data_control.shape, data_synth.shape)

        if not all(len(g) for g in aux_cols):
            raise ValueError(f"Empty aux column group: {aux_cols}")

        shared = [c for c in data_main.columns
                  if c in data_control.columns and c in data_synth.columns and c not in self._DROPPED]
        data_main, data_control, data_synth = self._align_dtypes(data_main, data_control, data_synth, shared)

        for c in shared:
            print(c, data_main[c].dtype, data_control[c].dtype, data_synth[c].dtype)

        results_df_link = run_anonymeter_linkability(data_main, data_synth, data_control, aux_cols, n_attacks=n_attacks)                              
        results_df_so = run_anonymeter_singlingout(data_main, data_synth,                                                                             
                                                    data_control, mode=mode, n_attacks=n_attacks)                                                       
        results_df_ai = run_anonymeter_attribute_inference(data_main, data_synth,                                                                     
                                                            data_control, n_attacks=n_attacks)

        risks = []
        # append risks from results to the list `risks`
        # append the value in the field of `results_df_link`, it has two columns
        # the first one contains names of rows and the second - values
        # need to append a value from the cell in 2nd column on the row where it's written `priv_risk` in 1st column

        # import ipdb; ipdb.set_trace()  # TODO: delete this line later

        risks.append(results_df_link.loc[results_df_link.index == 'priv_risk'].iloc[0, 0])
        # do the same for results_df_so
        risks.append(results_df_so.loc[results_df_so.index == 'priv_risk'].iloc[0, 0])
        # inside the dataframe `results_df_ai` there're numerous columns but the row with the label `priv_risk` in the 1st column exists too
        # take an average of values in all columns on this row and append to the list `risks`
        risks.append(results_df_ai.loc[results_df_ai.index == 'priv_risk'].iloc[0, 0:].mean())

        avg_risk = sum(risks) / len(risks)

        return avg_risk, {'singlingout': results_df_so, 'inference': results_df_ai, 'linkability': results_df_link}
    

    def get_detailed_report(self,) -> Dict:
        """
        Returns a detailed report of the privacy evaluation results.
        """
        avg_risk, report = self.privacy_risk_evaluation()
        return {
            'average_privacy_risk': avg_risk,
            'detailed_report': report
        }
    

    _DROPPED = ('Pat_ID.numeric',)

    @staticmethod
    def _align_dtypes(data_main, data_control, data_synth, cols):
        """Keep only `cols` and make each numeric/non-numeric consistently across frames, as anonymeter requires."""
        data_main, data_control, data_synth = (d[cols].copy() for d in (data_main, data_control, data_synth))
        for col in cols:
            if pd.api.types.is_numeric_dtype(data_main[col]):
                for df in (data_control, data_synth):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                for df in (data_main, data_control, data_synth):
                    df[col] = df[col].astype(object)
        return data_main, data_control, data_synth


    @staticmethod
    def get_aux_columns(data_origin,                                                                                              
                        data_processed,                                                                                           
                        data_control,                                                                                             
                        unavailable_columns=['export_psn',                                                                        
                                            'mnpdocid',                                                                          
                                            'mnpvisid',                                                                          
                                            'mnpcvpid',                                                                          
                                            'mnpvislabel',                                                                       
                                            'mnpvisno',
                                            'ethnicity',
                                            'gender'
                                            ],                                                                         
                        available_columns=(),
                        dropped_columns=_DROPPED):                                                                                    
                                                                                                                                
        shared_columns = [c for c in data_origin.columns
                          if c in data_processed.columns and c in data_control.columns
                          and c not in dropped_columns]
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