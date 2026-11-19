import pandas as pd
import numpy as np


class DatasetSampler:

    def __init__(self, sample_size, max_seed):
        self.sample_size = sample_size
        self.max_seed = max_seed

    def get_columns(self, df):
        numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        return numerical_columns, categorical_columns

    def numerical_difference(self, original_df, sample_df, numerical_columns):
        differences = []

        for col in numerical_columns:
            mean_diff = abs(original_df[col].mean() - sample_df[col].mean())
            std_diff = abs(original_df[col].std() - sample_df[col].std())
            column_range = (original_df[col].max() - original_df[col].min())
            if column_range == 0:
                column_range = 1
            differences.append(mean_diff / column_range)
            differences.append(std_diff / column_range)
        return np.mean(differences) if differences else 0

    def categorical_difference(self, original_df, sample_df, categorical_columns):

        differences = []
        for col in categorical_columns:
            original = original_df[col].value_counts(normalize=True,dropna=False,)
            sample = sample_df[col].value_counts(normalize=True,dropna=False,)
            categories = original.index.union(sample.index)
            original = original.reindex(categories, fill_value=0)
            sample = sample.reindex(categories, fill_value=0)
            differences.append(abs(original - sample).mean())
        return np.mean(differences) if differences else 0

    def find_best_seed(self, original_df):

        numerical_columns, categorical_columns = self.get_columns(original_df)

        best_seed = None
        best_score = float("inf")
        for seed in range(self.max_seed):
            sample_df = original_df.sample(n=self.sample_size,random_state=seed,)
            num_score = self.numerical_difference(original_df,sample_df,numerical_columns,)
            cat_score = self.categorical_difference(original_df,sample_df,categorical_columns,)
            score = num_score + cat_score
            if score < best_score:
                best_score = score
                best_seed = seed

        return best_seed