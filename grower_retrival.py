import numpy as np
import pandas as pd


class Retriever:

    def __init__(self, config,data):
        self.top_k = config["gower_retrieval"]["top_k"]
        self.data = data
     

    def retrieve(self, target_row, dataset):

        numerical_columns, categorical_columns = self.data.get_column_types(dataset)
        ranges = {}

        for column in numerical_columns:#selecting ranges from only completed rows
            column_range = dataset[column].max() - dataset[column].min()
            if column_range == 0:
                ranges[column] = 1.0
            else:
                ranges[column] = column_range

        distances = []
        # below we are converting row in dataframe into series , then we are accessing using column names
        for idx, candidate_row in dataset.iterrows():#comparing target row with completed rws

            total = 0.0
            valid = 0
            for column in numerical_columns:
                x = target_row[column]
                y = candidate_row[column]

                if pd.isna(x) or pd.isna(y):
                    continue
                distance = abs(x - y) / ranges[column]
                distance = min(distance, 1.0)#maintaining gowers definition
                total += distance
                valid += 1

            for column in categorical_columns:

                x = target_row[column]
                y = candidate_row[column]

                if pd.isna(x) or pd.isna(y):
                    continue
                total += 0 if x == y else 1
                valid += 1
            if valid == 0:
                distance = np.inf #represents a largest finite number
            else:
                distance = total / valid

            distances.append((idx, distance))

        # ---------- Sorting rows  by distance ----------
        distances.sort(key=lambda x: x[1])#sort w.r.t 1
        top_rows = distances[:self.top_k]
        top_indices = []
        for row in top_rows:
            top_indices.append(row[0])#assgining only index instead of distance
        similar_rows = dataset.loc[top_indices].copy()
        gower_distances = []
        for row in top_rows:
            gower_distances.append(row[1])
        similar_rows["gower_distance"] = gower_distances
        similar_rows["gower_similarity"] = 1 - similar_rows["gower_distance"]
        return similar_rows
        