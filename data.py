# ASyH simple CSV and Excel file Input/Output, as well as model i/o using pkl
#
# ToDos:
#   A real exception thrown when the file type cannot be determined by read().
#
#   Dataset naming?  This could help with consistent output file naming even in
#   downstream modules like *Model.
#
#   Module docstring
from typing import Optional
import re
import magic

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator
# import Pipeline
from sklearn.pipeline import Pipeline
# import SimpleImputer 
from sklearn.impute import SimpleImputer
# import methods for handling dates from utils
from utils import Utils

from sdv.metadata import SingleTableMetadata
from ASyH.metadata import Metadata
from ASyH.dataerror import DataError


# ASyH reads data into a pandas dataframe, so, the return type of the read-in
# methods are pandas dataframes.
class Data:
    # a pandas data frame holding all the actual data
    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data_content: pd.DataFrame):
        self._data = data_content

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata_content: Metadata):
        self._metadata = metadata_content

    @property
    def sdv_metadata(self) -> SingleTableMetadata:
        try:
            return SingleTableMetadata.load_from_dict(self._metadata.metadata)
        except AttributeError:
            if self._metadata is not None:
                if hasattr(self._metadata, 'metadata'):
                    return self._metadata.metadata
                else:
                    raise DataError("Metadata does not contain SDV metadata.")
            else:
                raise DataError("Metadata is not set or does not contain SDV metadata.")

    def __init__(self, data: Optional[DataFrame] = None, metadata: Metadata = None):
        # should we not .copy() the dataframe?  atm we do not manipulate ._data.
        self._data = data
        # if metadata is None -> then use the method of SingleTableMetadata to detect if from data
        if metadata is None and data is not None:
            STM = SingleTableMetadata()
            # self._metadata = Metadata(metadata=STM.detect_from_dataframe(data))
            STM.detect_from_dataframe(data)
            dict_stm = STM.to_dict()
            # the key 'ethinicity' should have the value 'categorical' if it is present in dict_stm keys
            # if 'ethnicity' in dict_stm['columns']:
            #     # dict_stm['columns']['ethnicity']['sdtype'] = 'categorical'
            #     dict_stm['columns']['ethnicity']['sdtype'] = 'numerical'
            self._metadata = Metadata(metadata=dict_stm)
        else:
            self._metadata = metadata

        col_types_dict = self.detect_column_types(data) # , num_cols=num_cols, cat_cols=cat_cols, date_cols=date_cols)
        self.cat_cols, self.num_cols, self.date_cols = col_types_dict['categorical'], col_types_dict['numerical'], col_types_dict['date']


    def read(self, input_file):
        """Generic read method inferring file format from \'magic\'.  Excel and
        CSV are supported.
        """
        filetype = magic.from_file(input_file)
        x = re.compile(".*Excel.*")
        c = re.compile(".*CSV.*")
        if x.match(filetype):
            data = pd.read_excel(input_file, index_col=False)
        elif c.match(filetype):
            data = pd.read_csv(input_file, index_col=False)
        else:
            raise DataError("Cannot determine input file type: ")
        self._data = data

    # def set_metadata(self, metadata: Optional[Metadata] = None):
    #     self._metadata = metadata

    def _convert_comma_decimal(self, val):
            if isinstance(val, str):                                                                                                           
                try:                     
                    return float(val.replace(',', '.')) 
                except:                                                    
                    return val                                             
            return val


    @staticmethod
    def detect_discrete_columns(df, threshold_share: float = 0.1):
        """ 
        Recognize categorical / discrete columns 
        Args:
            threshold_share: the percentage of rows - maximal number of categories
        """
        threshold =  max(1, int(df.shape[0] * threshold_share))
        # return [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() <= threshold]
        cols = []
        cols_nuniques = df.nunique()
        for col, nunique in zip(df.columns, cols_nuniques):
            # is_numeric = pd.api.types.is_numeric_dtype(df[col])
            # if not is_numeric and df[col].nunique() <= threshold:
            if nunique <= threshold:
                cols.append(col)
        return cols


    # the method that detects date columns
    @staticmethod
    def detect_date_columns(df):
        """ 
        Recognize date columns 
        """
        cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                cols.append(col)
            elif pd.api.types.is_object_dtype(df[col]):
                try:
                    pd.to_datetime(df[col])
                    cols.append(col)
                except (ValueError, TypeError):
                    pass
        return cols


    @staticmethod
    def detect_column_types(df: pd.DataFrame, num_cols=None, 
                            cat_cols=None, date_cols=None) -> dict:
        # TODO: remake the instructions for recognizing categorical and numerical columns
        if num_cols is None:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()

        if date_cols is None:
            # date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            date_cols = []
            # also include in date_cols the columns whose name includes a substring 'date'
            for col in df.columns:
                if 'date' in col.lower() and col not in date_cols and col not in num_cols:
                    # check if elements of the column are date strings of the forman YYYY-MM-DD
                    # pattern = r'^\d{4}-\d{2}-\d{2}$'
                    # # if df[col].dropna().apply(lambda x: isinstance(x, str) and re.match(pattern, x)).all():
                    # # like in the condition above but split into several lines for better readability
                    # # check if all elements are string (except nans)
                    # is_string_dtype = df[col].dropna().apply(lambda x: isinstance(x, str)).all()
                    # # apply reges if is string
                    # if is_string_dtype:
                    #     is_date_format = df[col].dropna().apply(lambda x: re.match(pattern, x)).all()
                    # else:
                    #     is_date_format = False

                    # if is_date_format:
                    #     date_cols.append(col)

                    non_null = df[col].dropna()
                    is_date_format = (
                        not non_null.empty
                        and non_null.map(lambda x: isinstance(x, str)).all()
                        and non_null.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all()
                    )
                    if is_date_format:
                        date_cols.append(col)

        if cat_cols is None:
            cat_cols = set(df.select_dtypes(include=['object', 'category', 'bool']).columns)
            discrete_cols = set(Data.detect_discrete_columns(df))
            cat_cols = list(cat_cols.union(discrete_cols))
            # remove from cat_cols  elements that are present in either num_cols or date_cols
            for col in cat_cols.copy():
                if col in num_cols or col in date_cols:
                    cat_cols.remove(col)

        # make sure that cat_cols, num_cols and date_cols do not overlap with each other
        # use set operations for that
        overlap = set(cat_cols) & set(num_cols) | set(cat_cols) & set(date_cols) | set(num_cols) & set(date_cols)
        assert len(overlap) == 0, f"cat_cols, num_cols and date_cols should not overlap. Overlapping columns: {overlap}"

        return {"categorical": cat_cols, "numerical": num_cols, "date": date_cols}


    # def knn_impute_onehot(self, df, n_neighbors=5, cat_cols=None, num_cols=None):
    def preprocess_data(self, 
                        n_neighbors=5, 
                        cat_cols=None, num_cols=None,
                        date_cols=None,
                        keep_nans=False,
                        inplace=False,
                        subsample=False,
                        min_n_rows: int = 1000,
                        reuse_preprocessor=None):
        
        # subsample 100 rows from `train_data_transformed` to execute `fit()` faster
        # self.train_data_transformed = self.train_data_transformed.sample(n=min(100, len(self.train_data_transformed)), random_state=42)
        if subsample:
            # df_proc = df_proc.sample(n=min(100, len(df_proc)), random_state=42)
            df = self.data.sample(n=min(min_n_rows, len(self.data)), random_state=42)
        else:
            df = self.data.copy()

        if cat_cols is None or num_cols is None or date_cols is None:
            cat_cols = self.cat_cols
            num_cols = self.num_cols
            date_cols = self.date_cols

        mask_nan = df.isna()

        self.numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median",
                                      add_indicator=False)),
            # ("scaler", StandardScaler()),
        ])

        self.categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", 
                                      add_indicator=False)),
            # ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        self.preprocessor = ColumnTransformer([
            ("num", self.numeric_transformer, num_cols),
            ("cat", self.categorical_transformer, cat_cols),
        ], remainder="passthrough")

        self.original_columns_ = list(df.columns)

        if reuse_preprocessor is not None:
            # ordinal codes/medians must come from the train fit, otherwise features mean different things across splits
            self.preprocessor = reuse_preprocessor
            df_proc = self.preprocessor.transform(df)
        else:
            df_proc = self.preprocessor.fit_transform(df)

        # TODO: FIX IN THE CODE BELOW HOW COLUMN NAMES ARE ASSIGNED !!
        column_names = self.preprocessor.get_feature_names_out()
        column_names = [name.split("__")[-1] for name in column_names]
        print(f"Column names after preprocessing: {column_names}")
        df_proc = pd.DataFrame(df_proc, columns=column_names)

        # import ipdb; ipdb.set_trace()
        # go through date_cols and convert dates
        # first check the intersection between date_cols and df_proc.columns
        cols_intersection = [col for col in date_cols if col in df_proc.columns]
        if set(cols_intersection) == set(date_cols):
            df_proc = Utils.convert_dates(df_proc, date_cols)
        else:
            print(f"Warning: Not all date columns are present in df_proc. Missing columns: {set(date_cols) - set(cols_intersection)}")
            # update date_cols class attribute
            self.date_cols = cols_intersection
            df_proc = Utils.convert_dates(df_proc, self.date_cols)

        if keep_nans:
            # df_proc = pd.DataFrame(df_proc, columns=self.preprocessor.get_feature_names_out())
            # ake sure that column names do not contain prefixes `cat_` or `num_`
            for col in self.original_columns_:
                if col in df_proc.columns:
                    df_proc.loc[mask_nan[col], col] = np.nan

        if inplace:
            # self._data = pd.DataFrame(df_proc, columns=self.preprocessor.get_feature_names_out())
            self._data = pd.DataFrame(df_proc, columns=column_names)
        else:
            return df_proc


    def inverse_transform_data(self, df_proc: pd.DataFrame, convert_dates=True) -> pd.DataFrame:
        """Inverse-transform data using the fitted sub-transformers but it 
        does it much more comprehensively by sticking to dataframe format and using column names"""

        # get the ordered list of column_names to make reconstruction easier later
        column_names = df_proc.columns.tolist()

        if not hasattr(self, "preprocessor") or self.preprocessor is None:
            raise ValueError("No fitted preprocessor found. Call _preprocess_data() first.")

        if not hasattr(self, "original_columns_"):
            raise ValueError("Missing original_columns_. Store them before fitting.")

        cat_transformer = self.preprocessor.named_transformers_.get("cat")
        num_transformer = self.preprocessor.named_transformers_.get("num")

        if cat_transformer is None or num_transformer is None:
            raise ValueError("Expected both 'cat' and 'num' transformers in the preprocessor.")

        fitted_cols = {name: cols for name, _, cols in self.preprocessor.transformers_}
        cat_cols = list(fitted_cols["cat"])
        num_cols = list(fitted_cols["num"])

        cat_feature_count = len(cat_cols)
        num_feature_count = len(num_cols)
        date_feature_count = len(self.date_cols)

        # ensure that summed feature counts is equal to the length of column_names
        assert len(column_names) == cat_feature_count + num_feature_count + date_feature_count, \
            f"Sum of feature counts ({cat_feature_count + num_feature_count + date_feature_count}) does not match number of columns ({len(column_names)})."
        
        # get indices of numerical features
        num_indices = [column_names.index(col) for col in num_cols]
        # get indices of categorical features
        cat_indices = [column_names.index(col) for col in cat_cols]
        # get indices of date columns
        date_indices = [column_names.index(col) for col in self.date_cols]

        X = np.asarray(df_proc)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        num_block = X[:, num_indices]
        cat_block = X[:, cat_indices]
        date_block = X[:, date_indices]

        # SimpleImputer has no inverse with add_indicator=False, so only invert the encoder
        ordinal = cat_transformer.named_steps["ordinal"]
        # Generative models can emit non-integer or out-of-range codes; snap them to valid category indices
        max_codes = np.array([len(c) - 1 for c in ordinal.categories_])
        cat_codes = np.rint(np.nan_to_num(cat_block.astype(float)))
        cat_codes = np.clip(cat_codes, 0, max_codes)

        cat_inv = pd.DataFrame(ordinal.inverse_transform(cat_codes), columns=cat_cols)
        num_inv = pd.DataFrame(num_block, columns=num_cols)
        date_inv = pd.DataFrame(date_block, columns=self.date_cols)

        df_inv = pd.concat([cat_inv, num_inv, date_inv], axis=1)
        df_inv = df_inv.reindex(columns=self.original_columns_)

        if convert_dates:
            df_inv = Utils.convert_back_dates(df_inv, self.date_cols)

        return df_inv


    def inverse_transform_data_obsolete(self, df_proc: pd.DataFrame) -> pd.DataFrame:
        """Inverse-transform data using the fitted sub-transformers."""
        if not hasattr(self, "preprocessor") or self.preprocessor is None:
            raise ValueError("No fitted preprocessor found. Call _preprocess_data() first.")

        if not hasattr(self, "original_columns_"):
            raise ValueError("Missing original_columns_. Store them before fitting.")

        cat_transformer = self.preprocessor.named_transformers_.get("cat")
        num_transformer = self.preprocessor.named_transformers_.get("num")

        if cat_transformer is None or num_transformer is None:
            raise ValueError("Expected both 'cat' and 'num' transformers in the preprocessor.")

        fitted_cols = {name: cols for name, _, cols in self.preprocessor.transformers_}
        cat_cols = list(fitted_cols["cat"])
        num_cols = list(fitted_cols["num"])

        cat_feature_count = len(cat_cols)
        num_feature_count = len(num_cols)
        date_feature_count = len(self.date_cols)

        X = np.asarray(df_proc)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # ColumnTransformer output layout: [num | cat | remainder (dates)]
        num_block = X[:, :num_feature_count]
        cat_block = X[:, num_feature_count:num_feature_count + cat_feature_count]
        date_start = num_feature_count + cat_feature_count
        date_block = X[:, date_start:date_start + date_feature_count]

        # SimpleImputer has no inverse with add_indicator=False, so only invert the encoder
        ordinal = cat_transformer.named_steps["ordinal"]
        # Generative models can emit non-integer or out-of-range codes; snap them to valid category indices
        max_codes = np.array([len(c) - 1 for c in ordinal.categories_])
        cat_codes = np.rint(np.nan_to_num(cat_block.astype(float)))
        cat_codes = np.clip(cat_codes, 0, max_codes)
        cat_inv = pd.DataFrame(
            ordinal.inverse_transform(cat_codes),
            columns=cat_cols
        )
        num_inv = pd.DataFrame(num_block, columns=num_cols)

        date_inv = pd.DataFrame(date_block, columns=self.date_cols)

        df_inv = pd.concat([cat_inv, num_inv, date_inv], axis=1)

        df_inv = df_inv.reindex(columns=self.original_columns_)

        # go through date_cols and convert dates back to original format
        df_inv = Utils.convert_back_dates(df_inv, self.date_cols)

        return df_inv
    

class RealData(Data):
    pass


class SyntheticData(Data):
    """Synthetic data class: In contrast to real data, it can be written to
    file.
    """

    def write(self, outputfile):
        """Write synthetic data to output to file.  Excel and CSV are supported.
        The default is CSV, if the given output file name isn't compatible with
        Excel file endings.
        """
        x = re.compile(r"[.]xls.?\Z", flags=re.IGNORECASE)
        if x.search(outputfile):
            self._data.to_excel(outputfile, index=False)
        else:
            self._data.to_csv(outputfile, index=False)
