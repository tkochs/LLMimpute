import pandas as pd


class HammingRetriever:

    def __init__(self, config):
        self.top_k = config["hamming_retrieval"]["top_k"]

    def retrieve(self, target_row, dataset, target_index=None):

        # Target missingness pattern
        # Missing = 1
        # Observed = 0
        target_pattern = target_row.isna().astype(int)

        results = []

        for index, row in dataset.iterrows():

            # Do not compare the target row with itself
            if target_index is not None and index == target_index:
                continue

            row_pattern = row.isna().astype(int)

            # Hamming distance:
            # count positions where missingness pattern differs
            hamming_distance = (target_pattern != row_pattern).sum()

            results.append({
                "index": index,
                "hamming_distance": hamming_distance,
            })

        # Smallest distance = most similar missingness pattern
        results = sorted(
            results,
            key=lambda x: x["hamming_distance"],
        )

        # Select top-k
        top_results = results[:self.top_k]

        top_indices = [item["index"] for item in top_results]

        similar_rows = dataset.loc[top_indices].copy()

        # Add distance only for inspection/debugging
        distance_map = {
            item["index"]: item["hamming_distance"]
            for item in top_results
        }

        similar_rows["hamming_distance"] = (
            similar_rows.index.map(distance_map)
        )

        return similar_rows