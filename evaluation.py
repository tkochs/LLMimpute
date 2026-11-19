import numpy as np
import pandas as pd

from sklearn.metrics import (mean_squared_error,mean_absolute_error,accuracy_score,f1_score,)


class Evaluator:

    def __init__(self, config, data):
        self.data = data
        evaluation = config["evaluation"]

        self.numerical_metrics = evaluation["numerical_metrics"]
        self.categorical_metrics = evaluation["categorical_metrics"]
        self.mixed_metrics = evaluation["mixed_metrics"]
        self.normalize_numerical = evaluation["normalize_numerical"]

    def evaluate(self,ground_truth_df,imputed_df,missing_mask,original_dataset,):
        results = {}

        #<---------------detect column types------------>

        numerical_columns, categorical_columns = self.data.get_column_types(ground_truth_df)
        #<---------------Numerical Metrics--------------->    

        rmse_values = []
        mae_values = []

        for column in numerical_columns:

            mask = missing_mask[column]
            y_true = ground_truth_df.loc[mask, column]
            y_pred = imputed_df.loc[mask, column]
            mask = missing_mask[column]
            # Normalize numerical values if enabled and extracting range from original dataset (min-max normalization)
            if self.normalize_numerical:

                column_min = original_dataset[column].min()
                column_max = original_dataset[column].max()
                column_range = column_max - column_min
                if column_range != 0:
                    y_true = (y_true - column_min) / column_range
                    y_pred = (y_pred - column_min) / column_range
            
            if len(y_true) == 0:
                continue
            if "rmse" in self.numerical_metrics:

                rmse_values.append(np.sqrt(mean_squared_error(y_true,y_pred,)))
            
            if "mae" in self.numerical_metrics:

                mae_values.append(mean_absolute_error(y_true,y_pred,))
                

        if "rmse" in self.numerical_metrics:

            results["rmse"] = (np.mean(rmse_values)#calclating mean
                if rmse_values
                else None
            )

        if "mae" in self.numerical_metrics:

            results["mae"] = (np.mean(mae_values)
                if mae_values
                else None
            )

        #<-----------------------Categorical Metrics--------------------->
       

        accuracy_values = []
        f1_values = []
        for column in categorical_columns:
            mask = missing_mask[column]
            y_true = ground_truth_df.loc[mask, column]
            y_pred = imputed_df.loc[mask, column]
            if len(y_true) == 0:
                continue
            if "accuracy" in self.categorical_metrics:
                accuracy_values.append(accuracy_score(y_true,y_pred,))
                    
            if "macro_f1" in self.categorical_metrics:

                f1_values.append(f1_score(y_true,y_pred,average="macro",))
                    
        if "accuracy" in self.categorical_metrics:

            results["accuracy"] = (np.mean(accuracy_values)
                if accuracy_values
                else None
            )

        if "macro_f1" in self.categorical_metrics:

            results["macro_f1"] = (np.mean(f1_values)
                if f1_values
                else None
            )
       
   

        # <----------------------- Gower Distance ---------------------->
        #Gower will do automatic normalization

        if "gower" in self.mixed_metrics:

            # Compute ranges for numerical columns ,extreacting range from original dataset
            ranges = {}
            for column in numerical_columns:
                column_range = (original_dataset[column].max() - original_dataset[column].min())
                ranges[column] = column_range if column_range != 0 else 1.0

            row_distances = []

            # Compute Gower distance for each row
            for idx in ground_truth_df.index:

                total = 0.0
                valid = 0

                # Numerical columns
                for column in numerical_columns:

                    if not missing_mask.at[idx, column]:
                        continue

                    x = ground_truth_df.at[idx, column]
                    y = imputed_df.at[idx, column]

                    if np.isnan(x) or np.isnan(y):
                        continue
                    distance = abs(x - y) / ranges[column]
                    distance = min(distance, 1.0)
                    total += distance
                    valid += 1

                # Categorical columns
                for column in categorical_columns:

                    if not missing_mask.at[idx, column]:
                        continue

                    x = ground_truth_df.at[idx, column]
                    y = imputed_df.at[idx, column]

                    if x is None or y is None:
                        continue

                    total += 0 if x == y else 1
                    valid += 1

                if valid > 0:
                    row_distances.append(total / valid)

            results["gower"] = (
                np.mean(row_distances)
                if row_distances
                else None
            )

        return results
