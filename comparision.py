import pandas as pd


class ImputationComparison:

    def create(self,ground_truth_df,injected_df,llm_df,meanmode_df,missing_mask,selected_indices,retrivalrows):
        rows = []

        for column in ground_truth_df.columns:

            # Missing rows for this column
            missing_indices = missing_mask.index[missing_mask[column]]

            # Only keep rows that were selected for imputation
            imputed_indices = missing_indices.intersection(selected_indices)

            if len(imputed_indices) == 0:
                continue

            column_comparison = pd.DataFrame({
                "Row": imputed_indices,
                "Column": column,
                "Ground Truth": ground_truth_df.loc[imputed_indices, column].values,
                "Injected": injected_df.loc[imputed_indices, column].values,
                "LLM": llm_df.loc[imputed_indices, column].values,
                "Mean/Mode": meanmode_df.loc[imputed_indices, column].values,
            })

            column_comparison["Retrieved Rows"] = column_comparison["Row"].map(
                lambda idx: retrivalrows.get(idx, [])
            )

            rows.append(column_comparison)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    
    def numerical_statistics(self, comparison_df, numerical_columns):

        stats = []

        # Only numerical columns that actually have imputed values
        imputed_columns = list(set(numerical_columns).intersection(set(comparison_df["Column"])))

        for column in imputed_columns:

            temp = comparison_df[comparison_df["Column"] == column].copy()#seprating column by column
            temp["Ground Truth"] = pd.to_numeric(temp["Ground Truth"], errors="coerce")#converting csv strings into numberes
            temp["LLM"] = pd.to_numeric(temp["LLM"], errors="coerce")#if conversion fails becomes nan
            temp["Mean/Mode"] = pd.to_numeric(temp["Mean/Mode"], errors="coerce")

            stats.append({
                "Column": column,
                "Count": len(temp),

                "GT Mean": temp["Ground Truth"].mean(),
                "LLM Mean": temp["LLM"].mean(),
                "MM Mean": temp["Mean/Mode"].mean(),

                "GT Std": temp["Ground Truth"].std(),
                "LLM Std": temp["LLM"].std(),
                "MM Std": temp["Mean/Mode"].std(),

                "LLM MAE": (temp["LLM"] - temp["Ground Truth"]).abs().mean(),
                "MM MAE": (temp["Mean/Mode"] - temp["Ground Truth"]).abs().mean(),
            })

        return pd.DataFrame(stats)

    def categorical_statistics(self, comparison_df, categorical_columns):

        results = []

        # Only categorical columns that actually have imputed values
        imputed_columns = list(set(categorical_columns).intersection(set(comparison_df["Column"])))

        for column in imputed_columns:
            temp = comparison_df[
                comparison_df["Column"] == column
            ]

            results.append({
                "Column": column,
                "Count": len(temp),
                "LLM Accuracy": (temp["Ground Truth"] == temp["LLM"]).mean(),
                "Mean/Mode Accuracy": (temp["Ground Truth"] == temp["Mean/Mode"]).mean(),
            })

        return pd.DataFrame(results)