import pandas as pd
import json
import pydantic
import numpy as np
# from schema import inputRow

class Datasetp:

    def loaddataset(self, path):
        return pd.read_csv(path)

    def split_rows(self, df):

        missing_rows = df[df.isnull().any(axis=1)]
        complete_rows = df[~df.isnull().any(axis=1)]
        return missing_rows, complete_rows

    def select_rows(self, df, alpha, max_rows):

        missing_rows, complete_rows = self.split_rows(df)

        if alpha == 0:
            rows = missing_rows

        elif alpha == 1:
            rows = df

        else:
            n_complete = int(len(complete_rows) * alpha)
            sampled_complete = complete_rows.sample(n=min(n_complete, len(complete_rows)),random_state=42)
            rows = pd.concat([missing_rows, sampled_complete])
            rows = rows.sort_index()

        if str(max_rows).lower() != "all":

            max_rows = int(max_rows)
            if max_rows <= 0:
                raise ValueError("max_rows must be greater than 0 or 'all'")
            
            rows = rows.head(max_rows)
        return rows

    def rowtojson(self, row):

        row_dict = row.where(pd.notna(row), None).to_dict()#replace with none if we have missing values
        return json.dumps(row_dict, indent=2)
        #return  inputRow(**row_dict).model_dump_json(indent=2)#<---- using pydantic--->
    # def rowtotext(self, row):#for converting row to plain text
    #     lines = []
    #     for column, value in row.items():
    #         if pd.isna(value):
    #             value = "Missing"
    #         lines.append(f"{column}: {value}")
    #     return "\n".join(lines)

    def rowtotext(self, row):

        feature_templates = {
            "Age": "The adult has an age of {}.",
            "EducationLevel": "The adult has an education level of {}.",
            "EmploymentType": "The adult has an employment type of {}.",
            "AnnualPersonalIncome": "The adult has an annual personal income of {}.",
            "HasDisability": "The adult has a disability status of {}.",
            "Retired": "The adult has a retired status of {}.",
            "FamilyMemberReceivingPension": "The adult has a family member receiving pension status of {}.",
            "PensionEligible": "The adult has a pension eligibility status of {}."
        }

        missing_templates = {
            "Age": "The adult's age is missing.",
            "EducationLevel": "The adult's education level is missing.",
            "EmploymentType": "The adult's employment type is missing.",
            "AnnualPersonalIncome": "The adult's annual personal income is missing.",
            "HasDisability": "The adult's disability status is missing.",
            "Retired": "The adult's retired status is missing.",
            "FamilyMemberReceivingPension": "The adult's family member receiving pension status is missing.",
            "PensionEligible": "The adult's pension eligibility status is missing."
        }

        available_lines = []
        missing_lines = []
        for column, value in row.items():

            if pd.isna(value):
                missing_lines.append(f"- {missing_templates[column]}")
            else:
                available_lines.append(f"- {feature_templates[column].format(value)}")

        return ("Observed Information\n\n"
        + "\n".join(available_lines)
            + "\n\n Missing Information\n\n"
            + "\n".join(missing_lines)
        )

    def dataframetojson(self, dataframe):#returns a list of JSON strings

        rows_json = []
        for _, row in dataframe.iterrows():
            rows_json.append(
                self.rowtojson(row))
        return  rows_json

    def dataframetodict(self, dataframe):

        rows_json = []

        for _, row in dataframe.iterrows():

            row_json = self.rowtojson(row)
            # Convert JSON string -> Python dictionary
            row_dict = json.loads(row_json)
            # Put each dictionary on one line
            rows_json.append(str(row_dict))

        return "\n".join(rows_json)

    def dataframetotext(self, df):#for converting gowers dataframe in to plain text
        rows = []
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            rows.append(f"Reference Row {i}")
            rows.append(self.rowtotext(row))
            rows.append("")
        return "\n".join(rows)
    
    def get_column_types(self, df):
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

        return numerical_columns, categorical_columns
    def analyze(self, df):

        numerical_columns, categorical_columns = self.get_column_types(df)

        dataset_summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "numerical_columns": len(numerical_columns),
            "categorical_columns": len(categorical_columns),
            "total_missing_values": int(df.isna().sum().sum()),
            "rows_with_missing": int(df.isna().any(axis=1).sum()),
            "complete_rows": int((~df.isna().any(axis=1)).sum()),
        }

        column_summary = pd.DataFrame({
            "missing_values": df.isna().sum(),
        })

        return dataset_summary, column_summary

    def rowtotext_gower(self, df):
        rows_text = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            row_text = []
            for column, value in row.items():
                if pd.isna(value):
                    value = "Missing"
                row_text.append(f"{column}: {value}")
            rows_text.append(f"Similar Row {i}:\n" + "\n".join(row_text))
        return "\n\n".join(rows_text)

    def similar_rows_to_text(self, similar_rows):

        feature_templates = {
            "Age": "The adult has an age of {}.",
            "EducationLevel": "The adult has an education level of {}.",
            "EmploymentType": "The adult has an employment type of {}.",
            "AnnualPersonalIncome": "The adult has an annual personal income of {}.",
            "HasDisability": "The adult has a disability status of {}.",
            "Retired": "The adult has a retired status of {}.",
            "FamilyMemberReceivingPension": "The adult has a family member receiving pension status of {}.",
            "PensionEligible": "The adult has a pension eligibility status of {}."
        }

        missing_templates = {
            "Age": "The adult's age is missing.",
            "EducationLevel": "The adult's education level is missing.",
            "EmploymentType": "The adult's employment type is missing.",
            "AnnualPersonalIncome": "The adult's annual personal income is missing.",
            "HasDisability": "The adult's disability status is missing.",
            "Retired": "The adult's retired status is missing.",
            "FamilyMemberReceivingPension": "The adult's family member receiving pension status is missing.",
            "PensionEligible": "The adult's pension eligibility status is missing."
        }

        paragraphs = []

        for number, (_, row) in enumerate(
            similar_rows.iterrows(),
            start=1):

            sentences = []

            for column in feature_templates:

                value = row[column]

                if pd.isna(value):
                    sentences.append(missing_templates[column])
                else:
                    sentences.append(
                        feature_templates[column].format(value))

            paragraph = (f"Similar Row {number}\n"    + " ".join(sentences))

            paragraphs.append(paragraph)

        return "\n\n".join(paragraphs)
    def prediction_to_markdown(self, prediction):

       # If generator returned JSON string
        if isinstance(prediction, str):
            import json
            prediction = json.loads(prediction)

        feature_templates = {
            "Age": "The adult has an age of {}.",
            "EducationLevel": "The adult has an education level of {}.",
            "EmploymentType": "The adult has an employment type of {}.",
            "AnnualPersonalIncome": "The adult has an annual personal income of {}.",
            "HasDisability": "The adult has a disability status of {}.",
            "Retired": "The adult has a retired status of {}.",
            "FamilyMemberReceivingPension": "The adult has a family member receiving pension status of {}.",
            "PensionEligible": "The adult has a pension eligibility status of {}."
        }

        lines = []

        for column, template in feature_templates.items():

            value = prediction.get(column)

            if value is None:
                lines.append(
                    f"{column}: Missing."
                )
            else:
                lines.append(
                    template.format(value)
                )

        return "\n".join(lines)