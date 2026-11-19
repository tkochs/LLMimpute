import pandas as pd
import numpy as np
from tqdm import tqdm

class MeanModeImputer:

    def __init__(self, data):
        self.data = data

    def impute(self, injected_df, rows_to_process):

        imputed_df = injected_df.copy()
        numerical_columns, categorical_columns = self.data.get_column_types(injected_df)

#<------caclulating mean  and mode of the columns from the whole injected dataset-------->
        numerical_means = {}
        for column in numerical_columns:
            numerical_means[column] = injected_df[column].mean()

        categorical_modes = {}
        for column in categorical_columns:
            mode = injected_df[column].mode(dropna=True)
            if not mode.empty:
                categorical_modes[column] = mode.iloc[0]# if it return a series of mode values, select first one
            else:
                categorical_modes[column] = None

#<-------imputing using mean and mode----------------------->

        # for index, row in rows_to_process.iterrows():
        for index, row in tqdm(rows_to_process.iterrows(),total=len(rows_to_process),desc="Mean/Mode Imputation",):

            for column in numerical_columns:

                if pd.isna(imputed_df.at[index, column]):
                    imputed_df.at[index, column] = numerical_means[column]
            for column in categorical_columns:
                 if pd.isna(imputed_df.at[index, column]):
                    imputed_df.at[index, column] = categorical_modes[column]
        return imputed_df