import yaml
import json
from autoschema import SchemaGenerator
from llmimputer import QWENModel
from meanmodeimputer import MeanModeImputer
from data_processor import Datasetp
from grower_retrival import Retriever
from inject import MissingValueGenerator
from evaluation import Evaluator
from exportresults import ResultExporter
from comparision import ImputationComparison
from ragpipeline.ragmain import HybridRAG
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from bestseed import DatasetSampler
from hamming_context import HammingRetriever
import traceback
from schema_manager import SchemaManager
import uuid

class ImputationPipeline:

    def __init__(self):
        self.results = []
        self.failed_validation_indices = []
        self.retrieval_history = {}
       

    def load_config(self):
        config_file = input("please Enter your config file name : ").strip()
        if not config_file.endswith(".yaml"): # Add .yaml only if the user didn't type it
            config_file += ".yaml"
        config_file=Path("configfiles")/config_file
        self.config_path = config_file 
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        print("config file loaded successfully")
        

    def initialize(self):
        self.dataset_path = self.config["dataset"]["path"]
        self.injected_dataset_path = self.config["injected_dataset"]["path"]
        self.use_injected_dataset = self.config["injected_dataset"]["enabled"]
        self.alpha = float(self.config["imputation"]["llm_input_complete_row_fraction"])
        self.max_rows = self.config["imputation"]["max_rows"]
        self.retrieval_enabled = self.config["gower_retrieval"]["enabled"]
        self.missing_enabled = self.config["missingness"]["enabled"]
        self.no_missingrows=self.config["random_missing_rows"]["num_rows"]
        self.summary_of_experiment =self.config["summary"]
        self.rag_context=self.config["Rag_retrieval"]["document_folder"]
        self.embedding_model_name=self.config["Rag_retrieval"]["embedding_model"]
        self.chunk_size=self.config["Rag_retrieval"]["chunk_size"]
        self.chunk_overlap=self.config["Rag_retrieval"]["chunk_overlap"]
        self.rag_prompt=self.config["Rag_retrieval"]["rag_prompt"]
        self.top_k_dense=self.config["Rag_retrieval"]["top_k_dense"]
        self.top_k_bm25=self.config["Rag_retrieval"]["top_k_bm25"]
        self.rerank_top_n=self.config["Rag_retrieval"]["rerank_top_n"]
        self.alpha_rrf=self.config["Rag_retrieval"]["dense_rrf_weight"]
        self.enable_web_search =self.config["web_search"]["enabled"]
        self.websearch_query = self.config["web_search"]["query"]
        self.websearch_max_results = self.config["web_search"]["max_results"]
        self.rag_prompt=self.config["Rag_retrieval"]["rag_prompt"]
        self.hamming_enabled = self.config["hamming_retrieval"]["enabled"]
        self.hamming_top_k = self.config["hamming_retrieval"]["top_k"]
        batch_config = self.config["inference"]["batching"]
        self.batch_enabled = batch_config["enabled"]
        self.batch_size = batch_config["batch_size"]
        chroma_dir = f"/tmp/lmangam_chroma_db/{self.experiment_id}"


        if self.config["Rag_retrieval"]["enabled"]  or self.config["hyde"]["enabled"]:
                    self.rag=HybridRAG(pdf_path=self.rag_context,embedding_model=self.embedding_model_name,
                    chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap,top_k_dense=self.top_k_dense,top_k_bm25=self.top_k_bm25,rerank_top_n=self.rerank_top_n,enable_web_search=self.enable_web_search,
                    websearch_query=self.websearch_query,websearch_max_results=self.websearch_max_results,alpha=self.alpha_rrf,   persist_directory=chroma_dir)

        self.exporter = ResultExporter(self.config)
    

    def prepare_dataset(self):

        self.data = Datasetp()
        a= self.data.loaddataset(self.dataset_path)
        original_df = a.dropna(axis=0).reset_index(drop=True)  # remove pre-existing NaN rows
        sampling_config = self.config["sampling"]
        
        if sampling_config["enabled"]:
            sample_size = sampling_config["sample_size"]
            sampler = DatasetSampler(sample_size=sample_size,max_seed=sampling_config["max_seed"],)
            best_seed = sampler.find_best_seed(original_df)
            print(f"Best sampling seed: {best_seed}")
            self.ground_truth_df = original_df.sample(n=500,random_state=best_seed,).reset_index(drop=True)
            self.df = self.ground_truth_df.copy()
        else:
            self.df= original_df.copy().reset_index(drop=True)
       
        # generator = SchemaGenerator(self.df)#for generating schema we gave orignal datset only
        # generator.save("schema.py")
        # print(generator.generate_schema())
        if self.missing_enabled:

            injector = MissingValueGenerator(self.config) 
            self.injected_df = injector.inject(self.df)# Dataset used for LLM imputation.

        elif self.use_injected_dataset:
            self.injected_df = self.data.loaddataset(self.injected_dataset_path).reset_index(drop=True)           
        else:
            self.injected_df =self.df.copy()# No missing value generation.

        if len(self.df) != len(self.injected_df):#validation check
                raise ValueError(f"Dataset validation failed: row count mismatch. "
                    f"Original dataset has {len(self.df)} rows, "
                    f"but injected dataset has {len(self.injected_df)} rows.")

            # Check number/names/order of features
        if list(self.df.columns) != list(self.injected_df.columns):
            raise ValueError(
                "Dataset validation failed: feature mismatch.\n"
                f"Original features: {list(self.df.columns)}\n"
                f"Injected features: {list(self.injected_df.columns)}")

        observed_mask = self.injected_df.notna()

                
        if not self.df.where(observed_mask).equals(self.injected_df.where(observed_mask)):
                raise ValueError(
                    "Dataset validation failed: "
                    "Some observed values in the injected dataset "
                    "are different from the original dataset.")


        print("Dataset validation successful.")
        print(f"Rows: {len(self.df)}")
        print(f"Features: {len(self.df.columns)}")

        generator = SchemaGenerator(self.injected_df, self.config)

        self.schema_model = SchemaManager.create(
            generator,
            self.experiment_id,)
        print("Dataset is being processed succesfully")
    
    def analyze_dataset(self):
        if self.config["datasetanalysis"]["enabled"]:
            self.dataset_summary,self.column_summary = self.data.analyze(self.df)
            self.injected_summary, self.injected_columns = self.data.analyze(self.injected_df)
        self.missing_mask = self.injected_df.isna()#Save original missing positions
        self.rows_to_process = self.data.select_rows(self.injected_df,alpha=self.alpha,max_rows=self.max_rows)
        self.selected_indices =self.rows_to_process.index
        # print("injected df",injected_df.loc[selected_indices])

    def prepare_rows(self):

        if self.retrieval_enabled: #gower retrival
            self.retriever = Retriever(self.config,self.data)
            _, self.complete_rows = self.data.split_rows(self.injected_df)
        if self.hamming_enabled:
            self.hamming_retriever = HammingRetriever(self.config)

    def process_llm_result(self, index, result):

        if result is None:
            self.failed_validation_indices.append(index)
            return

        # Get original row using index
        row = self.rows_to_process.loc[index]#check in selecetd row

        # Find missing columns
        missing_columns = row[row.isna()].index.tolist()

        if any(pd.isna(result.get(col)) for col in missing_columns):#checks each imputated value
            print(f"Row {index}: incomplete prediction")
            self.failed_validation_indices.append(index)
            return
        for column, value in result.items():
            self.llm_imputed_df.at[index, column] = value

      
    
    def run_llm_imputation(self):

        self.model = QWENModel(self.config,schema_model=self.schema_model)
        if self.config["chooseimputation method"]["llm"]["enabled"]:
            self.llm_imputed_df =self.injected_df.copy()
            self.hyde_documents = []
            batch = []

            
            # self.hyde_cache = {}

            # if self.config["hyde"]["enabled"]:

            #     hyde_items = []

            #     for index, row in self.rows_to_process.iterrows():

            #         row_json = self.data.rowtojson(row)

            #         missing_columns = (
            #             row[row.isna()]
            #             .index
            #             .tolist()
            #         )

            #         hyde_items.append({
            #             "index": index,
            #             "row_json": row_json,
            #             "missing_columns": missing_columns,
            #         })

            #     print("Generating HYDE documents concurrently...")

            #     self.hyde_cache = (
            #         self.model.generate_hyde_batch(hyde_items)
                # )

            for index, row in tqdm(self.rows_to_process.iterrows(),total=len(self.rows_to_process),desc="LLM Imputation",):
                    row_json = self.data.rowtojson(row)
                    print("=" * 80)
                    print("INPUT ROW")
                    print("=" * 80)
                    print(row_json)

                    contexts = {
                                    "row": row_json,
                                    "row_index": index,
                                    "rag_context": "",
                                    "gower_context": "",
                                    "random_rows_context": "",
                                    "hamming_context": "",
                                }
                    rag_docs = []
                    # row_text = self.data.rowtotext(row)
                    missing_columns = row[row.isna()].index.tolist()
                    known_values = row.dropna()
                    known_text = "\n".join(f"- {col}: {value}" for col, value in known_values.items())
                    print("=" * 80)
                    print("MISSING COLUMNS")
                    print("=" * 80)
                    print(missing_columns)

                    if self.config["Rag_retrieval"]["enabled"]:
                        rag_prompt = self.rag_prompt.format(row_text=row_json,missing_columns=", ".join(missing_columns),)
                        normal_rag_docs = self.rag.retrieve(rag_prompt,rag_prompt,)
                        rag_docs.extend(normal_rag_docs)



                     
                
                    if self.config["hyde"]["enabled"] : #getting doc from llm 
                        hyde_query =self.model.generate_hyde(row_text=row_json,missing_columns=", ".join(missing_columns),)
                        # hyde_query = self.hyde_cache[index]
                        self.hyde_documents.append({"row_index": index,"missing_columns": ", ".join(missing_columns),"row": row_json,"hyde_document": hyde_query,})
                        print("=" * 80)
                        print("hypothetical document from LLm")
                        print("=" * 80)
                        print(hyde_query)
                        hyde_docs = self.rag.retrieve(hyde_query,hyde_query)
                        rag_docs.extend(hyde_docs)

                    if rag_docs:
                        rag_docs = self.rag.merge_documents(rag_docs) #for filtering unique values

                        
                        # rag_docs = self.rag.retrieve(queries,rag_prompt,)
                        # hyde_docs = self.rag.retrieve(hyde_query,hyde_query)
                        # rag_docs = self.rag.merge_documents(rag_docs,hyde_docs,)

                        rag_context = "\n\n".join(f"### Retrieved Document {i}\n\n{doc.page_content}" for i, doc in enumerate(rag_docs, start=1))
                        contexts["rag_context"] = rag_context
                        print("=" * 80)
                        print("The documents retrived by rag")
                        print("=" * 80)
                        print(rag_context)

       
                    if self.retrieval_enabled:
                        # similar_rows = self.retriever.retrieve(target_row=row,complete_rows=self.complete_rows,) for complete rows only
                        similar_rows = self.retriever.retrieve(target_row=row,dataset=self.injected_df,)
                        similar_rows = similar_rows.drop(columns=["gower_distance","gower_similarity",],errors="ignore",)
                        self.retrieval_history[index] = similar_rows.index.tolist()
                        gower_context = self.data.dataframetojson(similar_rows)
                        #gower_context = self.data.similar_rows_to_text(similar_rows)
                        contexts["gower_context"] = gower_context

                    if self.hamming_enabled:

                        hamming_rows = self.hamming_retriever.retrieve(target_row=row,dataset=self.injected_df,target_index=index,)
                        print("=" * 80)
                        print("HAMMING MISSING-PATTERN NEIGHBORS")
                        print("=" * 80)
                        print(hamming_rows[["hamming_distance"]])
                        # Remove distance before sending rows to LLM
                        hamming_rows_for_context = hamming_rows.drop(columns=["hamming_distance"],errors="ignore",)
                        hamming_context = self.data.dataframetojson(hamming_rows_for_context)
                        print(hamming_context)
                        contexts["hamming_context"] = hamming_context
                                        
                        
                    if self.config["random_missing_rows"]["enabled"]:
                        missing_rows =self.injected_df[self.injected_df.isna().any(axis=1)]
                        available_missing_rows = missing_rows.drop(index=index, errors="ignore")
                        num_random_rows = min(self.no_missingrows,len(available_missing_rows),)
                        if num_random_rows > 0:
                            random_missing_rows =available_missing_rows.sample(n=num_random_rows,random_state=42,)
                            random_rows_context = self.data.dataframetojson(random_missing_rows)
                            contexts["random_rows_context"] = random_rows_context 

                    if self.batch_enabled:

                        batch.append(contexts,)

                        if len(batch) == self.batch_size:
                            results = self.model.generate_structured_batch(batch)
                            for item, result in zip(batch, results):
                                self.process_llm_result(index=item["row_index"],result=result,)
                            batch = []
                    else:

                        result = self.model.generate_structured(**contexts)
                        self.process_llm_result(index=index,result=result,)


    
    
            if self.batch_enabled and batch:#sending last reaming rows
                results = self.model.generate_structured_batch(batch)
                for item, result in zip(batch, results):
                    self.process_llm_result(index=item["row_index"],result=result,)
            #< checking the whether observed value is changed by llm or not--->
            if not self.model.prediction_comparison_history:
                print("No changes detected.")
            else:
                for change in self.model.prediction_comparison_history:
                    print(f"\nRow: {change['row_index']}")
                    print(f"Call: {change['call']}")
                    if change["imputed_changes"]:
                        print("Imputed values changed:")
                        for col, values in change["imputed_changes"].items():
                            print(f"  {col}: {values}")
                    if change["observed_changes"]:
                        print("Observed values changed:")
                        for col, values in change["observed_changes"].items():
                            print(f"  {col}: {values}")

            valid_indices = self.selected_indices.difference(self.failed_validation_indices)
            self.exporter.export_dataset("llm_imputed_dataset",self.llm_imputed_df,)
            prompt_history_df = pd.DataFrame(self.model.prompt_history)

            for call, group in prompt_history_df.groupby("call"):
                self.exporter.export_dataset(f"llm_prompt_call_{call}",group,)

            
 

            evaluator = Evaluator(self.config, self.data)
            llm_results = evaluator.evaluate(ground_truth_df=self.df.loc[valid_indices],
                            imputed_df=self.llm_imputed_df.loc[valid_indices],
                            missing_mask=self.missing_mask.loc[valid_indices],original_dataset=self.df,)
            llm_results["method"] = "LLM"
            self.results.append(llm_results)
            if self.failed_validation_indices:
                print("Failed row indices:", self.failed_validation_indices)

            token_usage_df = pd.DataFrame([
                        {
                            "input_tokens": self.model.total_input_tokens,
                            "output_tokens": self.model.total_output_tokens,
                            "total_tokens": self.model.total_tokens,
                        }])

            self.exporter.export_dataset("token_usage",token_usage_df)




    def run_mean_mode(self):
        if self.config["chooseimputation method"]["mean_mode"]["enabled"]:
            self.imputed_df = self.injected_df.copy()# Dataset that will be imputed()
            imputer = MeanModeImputer(self.data) #passsing data_processor.py object
            self.imputed_df = imputer.impute(injected_df=self.imputed_df,rows_to_process=self.rows_to_process,)

            evaluator = Evaluator(self.config, self.data)
            mean_mode_results = evaluator.evaluate(ground_truth_df=self.df.loc[self.selected_indices],
                            imputed_df=self.imputed_df.loc[self.selected_indices],
                            missing_mask=self.missing_mask.loc[self.selected_indices],
                            original_dataset=self.df,)
            mean_mode_results["method"] = "Mean/Mode" #adding key value pair to the dictionary
            self.results.append(mean_mode_results)
    
    def compare_results(self):
        comparison = ImputationComparison()
        self.comparison_df = comparison.create(ground_truth_df=self.df,injected_df=self.injected_df,llm_df=self.llm_imputed_df,
            meanmode_df=self.imputed_df,missing_mask=self.missing_mask,
            selected_indices=self.selected_indices,retrivalrows=self.retrieval_history,)
        self.numerical_columns, self.categorical_columns = self.data.get_column_types(self.df)
        self.numerical_stats = comparison.numerical_statistics(self.comparison_df,self.numerical_columns,)
        self.categorical_stats = comparison.categorical_statistics(self.comparison_df,self.categorical_columns,)
        print("=" * 80)
        print(self.numerical_stats)
        print("=" * 80)
        print(self.categorical_stats)
        print("=" * 80)


    def export_results(self):
       
        results_df = pd.DataFrame(self.results)
        print("=" * 80)
        print(results_df)
        print("=" * 80)
        if self.config["missingness"]["enabled"]:
            missingness_mechanism = self.config["missingness"]["mechanism"]
            missingness_percentage = self.config["missingness"]["missing_rate"]

        elif self.config["injected_dataset"]["enabled"]:
            missingness_mechanism = self.config["injected_dataset"]["mechanism"]
            missingness_percentage = self.config["injected_dataset"]["missing_rate"]
        else:
            missingness_mechanism = "None"
            missingness_percentage = 0

        self.exporter.export_experiment_history(self.results,)#for visualization purpose
        self.exporter.export_dataset("Imputed_output_comparision",self.comparison_df,)
        self.exporter.export_dataset("ground truth",self.df,)
        self.exporter.export_dataset("injected_dataset",self.injected_df,)
        self.exporter.export_file(self.config_path)#exporting config file too
        if self.config["Rag_retrieval"]["enabled"]:
            self.exporter.export_text("documents.txt",self.rag.loader.documents,"Document",)
            self.exporter.export_text("chunks.txt",self.rag.loader.chunks,"Chunk",)


        

        if self.config["chooseimputation method"]["mean_mode"]["enabled"]:
            self.exporter.export_dataset("mean_mode_imputed_dataset",self.imputed_df,)
        if self.config["hyde"]["enabled"]:
            hyde_df = pd.DataFrame(self.hyde_documents)
            self.exporter.export_dataset("hyde_documents",hyde_df,)

        self.exporter.export(
            ("About the experiment",pd.DataFrame([self.summary_of_experiment])),
            ("Evaluation Results", results_df),
            ("Original Dataset Summary", pd.DataFrame([self.dataset_summary])),
            ("Injected Dataset Summary", pd.DataFrame([self.injected_summary])),
            ("Numerical Statistics", self.numerical_stats),
            ("Categorical Statistics", self.categorical_stats),
            ("failed_validation_indices", pd.DataFrame(self.failed_validation_indices,columns=["Index"])),
            # ("Injected Column Summary", injected_columns),
        )
        

    def run(self):

        self.load_config()
        experiments = self.config["experiments"]
        for experiment in experiments:

            self.experiment_id = (f"{experiment['name']}_{uuid.uuid4().hex[:12]}")
            print("\n")
            print("=" * 100)
            print(f"STARTING EXPERIMENT: {experiment['name']}")
            print(f"DATASET: {experiment['injected_dataset_path']}")
            print(f"MISSING RATE: {experiment['missing_rate']}")
            print("=" * 100)

            self.config["name"] = experiment["name"]
            self.config["summary"] = experiment["summary"]
            self.config["injected_dataset"]["path"] = (experiment["injected_dataset_path"])
            self.config["injected_dataset"]["missing_rate"] = (experiment["missing_rate"])
            self.results = []
            self.failed_validation_indices = []
            self.retrieval_history = {}

            try:

                self.initialize()
                self.prepare_dataset()
                self.analyze_dataset()
                self.prepare_rows()
                if self.config["chooseimputation method"]["llm"]["enabled"]:
                    self.run_llm_imputation()
                if self.config["chooseimputation method"]["mean_mode"]["enabled"]:
                    self.run_mean_mode()
                self.compare_results()
                self.export_results()

                print("\n")
                print("=" * 100)
                print(f"COMPLETED SUCCESSFULLY: "f"{experiment['name']}")
                print("=" * 100)
                
            except Exception as e:

                print("\n")
                print("=" * 100)
                print(f"FAILED: {experiment['name']}")
                print(f"ERROR: {e}")
                print("=" * 100)
                traceback.print_exc()

            finally:

          
               if (hasattr(self, "rag") and self.config["Rag_retrieval"]["enabled"]):
              
                    try:
                        self.rag.cleanup()
                    except Exception as cleanup_error:
                        print(f"RAG cleanup warning: {cleanup_error}")
                    self.rag = None
































