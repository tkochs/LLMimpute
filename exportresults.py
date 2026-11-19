import os
import pandas as pd
import yaml


class ResultExporter:

    def __init__(self, config):

        self.config = config
        export = config["export"]
        self.enabled = export["enabled"]
        self.output_dir = export["output_dir_path"]
        self.experiment_name = config["name"]
        self.check_experiment_name()

    def export(self, *tables):#any number of aruguments

        if not self.enabled:
            return

        experiment_dir = os.path.join(self.output_dir,self.experiment_name,)
        os.makedirs(experiment_dir, exist_ok=True)#creates folder
        output_file = os.path.join(experiment_dir,f"{self.experiment_name}.csv")

        with open(output_file, "w") as f:#if exists -->overwrites
            for title, df in tables:
                f.write(f"{title}\n")
                #df is an argument
                df.to_csv(f, index=True)#dataframe,include index to
                f.write("\n\n")
        print(f"\nResults saved to: {output_file}")

    def export_dataset(self, dataset_name, df):

        if not self.enabled:
            return

        experiment_dir = os.path.join(self.output_dir,self.experiment_name,)
        os.makedirs(experiment_dir, exist_ok=True)
        output_file = os.path.join(experiment_dir,f"{dataset_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"{dataset_name} saved to: {output_file}")



    def export_experiment_history(self,results):

        if not self.enabled:
            return
        
        history_file = os.path.join(
                    self.output_dir,
                    "experiment_history.csv",)

        if self.config["missingness"]["enabled"]:

            mechanism = self.config["missingness"]["mechanism"]
            missing_rate = self.config["missingness"]["missing_rate"]

        elif self.config["injected_dataset"]["enabled"]:
            
            mechanism = self.config["injected_dataset"]["mechanism"]
            missing_rate = self.config["injected_dataset"]["missing_rate"]

        else:
            mechanism = "None"
            missing_rate = 0

        rows = []

        for result in results:

            rows.append({
                            "experiment":self.config["name"],
                            "mechanism": mechanism,
                            "missing_rate": missing_rate,
                            "method": result["method"],
                            "rmse": result.get("rmse"),
                            "mae": result.get("mae"),
                            "accuracy": result.get("accuracy"),
                            "macro_f1": result.get("macro_f1"),
                            "gower": result.get("gower"),
                        })

        current_df = pd.DataFrame(rows)
              
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)

            if self.config["name"] in history_df["experiment"].values:

                # Find where this experiment starts
                start_idx = history_df.index[history_df["experiment"] == self.config["name"]][0]
                # Remove old rows
                history_df = history_df[history_df["experiment"] != self.config["name"]].reset_index(drop=True)
                # Insert the new rows back at the same position
                top = history_df.iloc[:start_idx]
                bottom = history_df.iloc[start_idx:]
                history_df = pd.concat([top, current_df, bottom],ignore_index=True)
            else:
                history_df = pd.concat([history_df, current_df],ignore_index=True)
        else:
            history_df = current_df

        print(f"Experiment history saved to: {history_file}")
        history_df.to_csv(history_file, index=False)
        return history_df

    def check_experiment_name(self):

        if not self.enabled:
            return
        while True:
            experiment_dir = os.path.join(self.output_dir, self.experiment_name)
            if not os.path.exists(experiment_dir):
                break
            print(f"\nExperiment name: '{self.experiment_name}' was already exists and found same name in the config file.")
            choice = input("Do you want to Overwrite or? (y/n): ").strip().lower()
            if choice == "y":
                break
            elif choice =="n":
                self.experiment_name = input("Enter new experiment name (e.g. exper11): ").strip()
            else:
                print("Please enter only 'y' or 'n'.")
        self.config["name"] = self.experiment_name

    def export_file(self, file_path):

        if not self.enabled:
            return

        experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        output_file = os.path.join(experiment_dir, f"{self.experiment_name}.yaml")
        
        with open(file_path, "r") as src, open(output_file, "w") as dst: #opens both files at a time and copies from source to dest
            dst.write(src.read())

        print(f"{os.path.basename(file_path)} saved to: {output_file}")

    def export_text(self, filename, documents, label):

        if not self.enabled:
            return

        experiment_dir = os.path.join(self.output_dir,self.experiment_name,)

        os.makedirs(experiment_dir, exist_ok=True)
        output_file = os.path.join(experiment_dir,filename,)

        with open(output_file, "w", encoding="utf-8") as f:
            for i, doc in enumerate(documents, start=1):
                f.write("=" * 80 + "\n")
                f.write(f"{label} {i}\n")
                f.write("=" * 80 + "\n")
                f.write(doc.page_content)
                f.write("\n\n")
                f.write(f"Metadata: {doc.metadata}\n\n")
        print(f"{filename} saved to: {output_file}")

    def export_config(self):

        if not self.enabled:
            return

        experiment_dir = os.path.join(self.output_dir,self.experiment_name)
        os.makedirs(experiment_dir,exist_ok=True)
        output_file = os.path.join(experiment_dir,f"{self.experiment_name}.yaml")
        with open(output_file,"w",encoding="utf-8") as f:
            yaml.safe_dump(self.config,f,sort_keys=False,allow_unicode=True)
        print(f"Experiment config saved to: "f"{output_file}")



