from pathlib import Path
import keyword
import re
import pandas as pd

#Generate a Pydantic schema from a pandas DataFrame


class SchemaGenerator:
    def __init__(self, dataframe: pd.DataFrame,config):
        self.df = dataframe
        self.enable_constraints=config["autoschema_constraints"]["enabled"]
        

    def save(self, output_file: str = "schema.py") -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        schema = self.generate_schema()

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(schema)
            
    def generate_schema(self) -> str:#we can also remove it ,just the function should be retun string

        lines = ["from typing import Literal","from pydantic import BaseModel, Field","","","class ProductRow(BaseModel):", "",]

        for column in self.df.columns:
            series = self.df[column].dropna()
            column_name = self.clean_column_name(column)

            # Numerical columns
            if pd.api.types.is_numeric_dtype(series):
                #returns true if it is numeric
                minimum = series.min()
                maximum = series.max()
                if pd.api.types.is_integer_dtype(series):
                    dtype = "int"
                else:
                    dtype = "float"
                
                if self.enable_constraints:
                    lines.append(f"    {column_name}: {dtype} = Field(..., ge={minimum}, le={maximum})")
                else:
                    lines.append(f"    {column_name}: {dtype}")
              
            
            else:# Categorical columns

                unique_values = sorted(series.astype(str).unique())
                if len(unique_values) <= 30:
                    literals = ", ".join(f'"{value}"' for value in unique_values)
                    lines.append(f"    {column_name}: Literal[{literals}]")
                else:
                    lines.append(f"    {column_name}: str")
            lines.append("")

            #    literals = ", ".join(f'"{value}"'for value in unique_values)
            #    lines.append(f"    {column_name}: Literal[{literals}]")
            # lines.append("")

        return "\n".join(lines)

    @staticmethod#no need to create object
    def clean_column_name(column: str) -> str:
        #Convert column names into valid Python identifiers

        column = re.sub(r"\W+", "_", column.strip())#replacing name space in columns with _
        if column[0].isdigit():
            column = "_" + column
        # if keyword.iskeyword(column):#if we have python keywords in cloumn names ,we wll replace with name_
        #     column += "_"
        if keyword.iskeyword(column):
            column += "_"

        return column