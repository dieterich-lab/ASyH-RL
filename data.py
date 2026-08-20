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

from sklearn.preprocessing import StandardScaler, OrdinalEncoder # , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator
# import Pipeline
from sklearn.pipeline import Pipeline
# import SimpleImputer 
from sklearn.impute import SimpleImputer

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
            self._metadata = Metadata(metadata=STM.to_dict())
        else:
            self._metadata = metadata

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
    
    def _detect_discrete_columns(self, df, threshold=10):
        """内部自动推断分类变量，避免导入错误"""
        return [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() <= threshold]


    # def knn_impute_onehot(self, df, n_neighbors=5, cat_cols=None, num_cols=None):
    def preprocess_data(self, 
                        n_neighbors=5, 
                        cat_cols=None, num_cols=None,
                        keep_nans=False,
                        inplace=True):
        
        df = self.data.copy()

        # TODO: review and mb remake, fix the logic for filling in cat_cols and num_cols
        if cat_cols is None:
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if num_cols is None:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()

        mask_nan = df.isna()

        # self.preprocessor = ColumnTransformer(
        #         transformers=[
        #             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        #             ('impute', KNNImputer(n_neighbors=n_neighbors), num_cols + cat_cols),
        #             ('num', StandardScaler(), num_cols)
        #         ])

        # numeric_features = df.select_dtypes(include=["number"]).columns
        # categorical_features = df.select_dtypes(exclude=["number"]).columns

        self.numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        self.categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        self.preprocessor = ColumnTransformer([
            ("num", self.numeric_transformer, num_cols),
            ("cat", self.categorical_transformer, cat_cols),
        ], remainder="drop")

        self.original_columns_ = list(df.columns)

        df_proc = self.preprocessor.fit_transform(df)

        # TODO: FIX IN THE CODE BELOW HOW COLUMN NAMES ARE ASSIGNED !!
        column_names = self.preprocessor.get_feature_names_out()
        column_names = [name.split("__")[-1] for name in column_names]
        print(f"Column names after preprocessing: {column_names}")
        df_proc = pd.DataFrame(df_proc, columns=column_names)

        if keep_nans:
            # 将缺失值位置还原为 NaN
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


    def _inverse_transform_data(self, df_proc: pd.DataFrame) -> pd.DataFrame:
        """Inverse-transform data using the fitted sub-transformers."""
        if not hasattr(self, "preprocessor") or self.preprocessor is None:
            raise ValueError("No fitted preprocessor found. Call _preprocess_data() first.")

        if not hasattr(self, "original_columns_"):
            raise ValueError("Missing original_columns_. Store them before fitting.")

        cat_transformer = self.preprocessor.named_transformers_.get("cat")
        num_transformer = self.preprocessor.named_transformers_.get("num")

        if cat_transformer is None or num_transformer is None:
            raise ValueError("Expected both 'cat' and 'num' transformers in the preprocessor.")

        cat_cols = list(self.preprocessor.transformers_[0][2])
        num_cols = list(self.preprocessor.transformers_[1][2])

        cat_feature_count = len(cat_transformer.get_feature_names_out(cat_cols))
        num_feature_count = len(num_transformer.get_feature_names_out(num_cols))

        X = np.asarray(df_proc)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        cat_block = X[:, :cat_feature_count]
        num_block = X[:, -num_feature_count:]

        cat_inv = pd.DataFrame(
            cat_transformer.inverse_transform(cat_block),
            columns=cat_cols
        )
        num_inv = pd.DataFrame(
            num_transformer.inverse_transform(num_block),
            columns=num_cols
        )

        df_inv = pd.concat([cat_inv, num_inv], axis=1)
        return df_inv.reindex(columns=self.original_columns_)


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
