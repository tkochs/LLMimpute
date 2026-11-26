import importlib.util
import json
import keyword
import re
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pydantic import Field, ValidationError, create_model
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import outlines

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from sentence_transformers import CrossEncoder

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

class LLMImputer(TransformerMixin, BaseEstimator):
    """Impute the missing cells of a pandas DataFrame with an LLM.

    Every argument corresponds to one key of ``configfiles/config27.yaml`` (named in the comment);
    defaults are the values used there.
    """
    def __init__(
        self,
        # logging
        verbose=False,
        # --- LLM backend --------------------------------
        use_local_llm=False,                              
        model_name="meta-llama/Llama-3.1-8B-Instruct",    
        torch_dtype="auto",                               
        device_map="auto",                                
        api_base_url="http://localhost:8001/v1",          
        api_model="Gemma-4-31B-IT",                       
        structured_output="lmfe",                         
        # --- generation ---------------------------------
        max_new_tokens=250,                               
        do_sample=False,                                  
        temperature=None,                                 
        top_p=None,                                       
        top_k=None,                                       
        outlines_max_new_tokens=300,
        # EXPERIMENTAL (disabled): batching, see LLMModel.generate_structured_batch
        # batching=True,
        # batch_size=10,
        # --- prompts ------------------------------------
        prompts=None,                                     
        validation_retry_prompt=None,                     
        # --- which rows are sent to the LLM -------------
        llm_input_complete_row_fraction=0,                
        max_rows="all",                                   
        # --- output schema ------------------------------
        schema_constraints=False,                         
        # --- context: RAG + HyDE ------------------------
        rag_retrieval=True,                               
        document_folder=None,                             
        embedding_model="Qwen/Qwen3-Embedding-0.6B",      
        chunk_size=1000,                                  
        chunk_overlap=200,                                
        rag_prompt=None,                                  
        top_k_dense=15,                                   
        top_k_bm25=7,                                     
        rerank_top_n=3,                                   
        dense_rrf_weight=0.7,                             
        hyde=True,                                        
        hyde_prompt=None,                                 
        hyde_json_output=False,                           
        rag_device="cpu",
        # web_search=False,                                 
        # websearch_query=None,                             
        # websearch_max_results=5,                          
        chroma_dir=None,
        # --- context: neighbouring rows -----------------
        gower_retrieval=True,                             
        gower_top_k=15,                                   
    ):
        self.verbose = verbose
        self.use_local_llm = use_local_llm
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.api_base_url = api_base_url
        self.api_model = api_model
        self.structured_output = structured_output
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.outlines_max_new_tokens = outlines_max_new_tokens
        # self.batching = batching
        # self.batch_size = batch_size
        self.prompts = prompts
        self.validation_retry_prompt = validation_retry_prompt
        self.llm_input_complete_row_fraction = llm_input_complete_row_fraction
        self.max_rows = max_rows
        self.schema_constraints = schema_constraints
        self.rag_retrieval = rag_retrieval
        self.document_folder = document_folder
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rag_prompt = rag_prompt
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.rerank_top_n = rerank_top_n
        self.dense_rrf_weight = dense_rrf_weight
        self.hyde = hyde
        self.hyde_prompt = hyde_prompt
        self.hyde_json_output = hyde_json_output
        # self.web_search = web_search
        # self.websearch_query = websearch_query
        # self.websearch_max_results = websearch_max_results
        self.chroma_dir = chroma_dir
        self.gower_retrieval = gower_retrieval
        self.gower_top_k = gower_top_k

    def fit(self, X, y=None):
        # input validation
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Please enter X as DataFrame")
        if self.rag_retrieval or self.hyde:
            if self.document_folder is None:
                raise ValueError("RAG is turned on but no document_folder is passed!")
            if self.chroma_dir is None:
                raise ValueError("RAG is turned on but no chroma_dir is passed!")
        self.cleanup()
        self.run_id_ = uuid.uuid4().hex[:12]
        self.data_ = Datasetp()

        # sklearn stuff
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        # prompts: argument or the default 
        self.prompts_ = self.prompts if self.prompts is not None else DEFAULT_PROMPTS
        self.validation_retry_prompt_ = (self.validation_retry_prompt if self.validation_retry_prompt is not None
                                         else DEFAULT_VALIDATION_RETRY_PROMPT)
        self.hyde_prompt_ = self.hyde_prompt if self.hyde_prompt is not None else DEFAULT_HYDE_PROMPT
        self.rag_prompt_ = self.rag_prompt if self.rag_prompt is not None else DEFAULT_RAG_PROMPT
        # self.websearch_query_ = self.websearch_query if self.websearch_query is not None else DEFAULT_WEBSEARCH_QUERY

        # output schema 
        # generator = SchemaGenerator(X, enable_constraints=self.schema_constraints)
        # self.schema_model_ = SchemaManager.create(generator, self.run_id_)
        # original column name -> schema field name (clean_column_name rewrites e.g. "blood pressure" -> "blood_pressure");
        # the LLM answers with the schema field names, so results are looked up through this table
        self.schema_field_names_ = SchemaGenerator.field_names(X.columns)
        self.schema_model_ = build_schema_model(X, self.schema_field_names_, self.schema_constraints)

        # RAG index 
        if self.rag_retrieval or self.hyde:
            self.rag_ = HybridRAG(pdf_path=self.document_folder, embedding_model=self.embedding_model,
                                  chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                                  top_k_dense=self.top_k_dense, top_k_bm25=self.top_k_bm25,
                                  rerank_top_n=self.rerank_top_n,
                                  # enable_web_search=self.web_search,
                                  # websearch_query=self.websearch_query_,
                                  # websearch_max_results=self.websearch_max_results,
                                  # web search is disabled; HybridRAG/DocumentLoader still require these args
                                  enable_web_search=False, websearch_query=None, websearch_max_results=0,
                                  alpha=self.dense_rrf_weight, persist_directory=Path(self.chroma_dir)/self.run_id_, device=self.rag_device)
        if self.gower_retrieval:
            self.reference_ = X.copy()
            self.retriever_ = Retriever(top_k=self.gower_top_k, data=self.data_)

        # the LLM
        self.model_ = LLMModel(
            schema_model=self.schema_model_,
            model_name=self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            use_local_llm=self.use_local_llm,
            api_base_url=self.api_base_url,
            api_model=self.api_model,
            method=self.structured_output,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tokens_for_outlines=self.outlines_max_new_tokens,
            refinement_prompts=self.prompts_,
            hyde_prompt=self.hyde_prompt_,
            validation_retry_prompt=self.validation_retry_prompt_,
            json_output=self.hyde_json_output,
            verbose=self.verbose,
        )
        # rows of the most recent transform() whose LLM output was unusable (they keep NaN).
        # Created here and only cleared/appended in transform, so transform sets no attributes.
        self.failed_rows_ = []
        self._is_fitted = True
        return self

    def transform(self, X):
        # input validation: same type and same columns (names and order) as seen in fit
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(f"X has columns {list(X.columns)}, but LLMImputer was fitted with "
                             f"{list(self.feature_names_in_)}")
        #! input validation !#

        rows_to_process = self.data_.select_rows(X, alpha=float(self.llm_input_complete_row_fraction),
                                                 max_rows=self.max_rows)
        llm_imputed_df = X.copy()
        self.failed_rows_.clear()
        # batch = []  # EXPERIMENTAL (disabled): batching

        for index, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="LLM Imputation",):
            row_json = self.data_.rowtojson(row)
            if self.verbose:
                print("=" * 80)
                print("INPUT ROW")
                print("=" * 80)
                print(row_json)

            contexts = {
                "row": row_json,
                "row_index": index,
                "rag_context": "",
                "gower_context": "",
            }
            rag_docs = []
            missing_columns = row[row.isna()].index.tolist()
            if self.verbose:
                print("=" * 80)
                print("MISSING COLUMNS")
                print("=" * 80)
                print(missing_columns)

            if self.rag_retrieval:
                rag_prompt = self.rag_prompt_.format(row_text=row_json, missing_columns=", ".join(missing_columns),)
                normal_rag_docs = self.rag_.retrieve(rag_prompt, rag_prompt,)
                rag_docs.extend(normal_rag_docs)

            if self.hyde:  # getting doc from llm
                hyde_query = self.model_.generate_hyde(row_text=row_json, missing_columns=", ".join(missing_columns),)
                if self.verbose:
                    print("=" * 80)
                    print("hypothetical document from LLm")
                    print("=" * 80)
                    print(hyde_query)
                hyde_docs = self.rag_.retrieve(hyde_query, hyde_query)
                rag_docs.extend(hyde_docs)

            if rag_docs:
                rag_docs = self.rag_.merge_documents(rag_docs)  # for filtering unique values
                rag_context = "\n\n".join(f"### Retrieved Document {i}\n\n{doc.page_content}"
                                          for i, doc in enumerate(rag_docs, start=1))
                contexts["rag_context"] = rag_context
                if self.verbose:
                    print("=" * 80)
                    print("The documents retrived by rag")
                    print("=" * 80)
                    print(rag_context)

            # OPTIONAL: FIX neighbours reach the LLM in slightly wrong format.
            # Also sends 'NaN' for missing data instead of 'null' which would be json convention.
            if self.gower_retrieval:
                similar_rows = self.retriever_.retrieve(target_row=row, dataset=self.reference_)
                similar_rows = similar_rows.drop(columns=["gower_distance", "gower_similarity",], errors="ignore",)
                gower_context = self.data_.dataframetojson(similar_rows)
                contexts["gower_context"] = gower_context

            # EXPERIMENTAL (disabled): batching
            # if self.batching:
            #     batch.append(contexts,)
            #     if len(batch) == self.batch_size:
            #         results = self.model_.generate_structured_batch(batch)
            #         for item, result in zip(batch, results):
            #             self._process_llm_result(rows_to_process, llm_imputed_df, index=item["row_index"], result=result)
            #         batch = []
            # else:
            result = self.model_.generate_structured(**contexts)
            self._process_llm_result(rows_to_process, llm_imputed_df, index=index, result=result)

        # EXPERIMENTAL (disabled): batching
        # if self.batching and batch:  # sending last remaining rows
        #     results = self.model_.generate_structured_batch(batch)
        #     for item, result in zip(batch, results):
        #         self._process_llm_result(rows_to_process, llm_imputed_df, index=item["row_index"], result=result)

        if self.failed_rows_:
            warnings.warn(f"{len(self.failed_rows_)} rows could not be imputed and keep NaN; see failed_rows_ "
                          f"(first: {self.failed_rows_[:10]})", RuntimeWarning, stacklevel=2)
        return llm_imputed_df

    def _process_llm_result(self, rows_to_process, imputed, index, result):
        """Write the LLM's values into the missing cells of row `index` of `imputed` (in place).

        An unusable result (failed validation after all retries, or a missing column absent
        from the output) is recorded in ``failed_rows_``; that row keeps its NaN.
        """
        if result is None:
            self.failed_rows_.append(index)
            return
        row = rows_to_process.loc[index]
        missing_columns = row[row.isna()].index.tolist()
        # result keys are schema field names -> translate back to the original column names
        values = {col: result.get(self.schema_field_names_[col]) for col in missing_columns}
        if any(pd.isna(value) for value in values.values()):
            self.failed_rows_.append(index)
            return
        for col, value in values.items():  # only the missing cells; observed values are never overwritten
            imputed.at[index, col] = value

    def cleanup(self):
        if getattr(self, "rag_", None) is not None:
            try:
                self.rag_.cleanup()
            except Exception as cleanup_error:
                print(f"RAG cleanup warning: {cleanup_error}")
            self.rag_ = None


# =============================================================================
#  LLM wrapper 
# =============================================================================
class LLMModel:

    def __init__(self, schema_model, model_name, torch_dtype, device_map, use_local_llm, api_base_url, api_model,
                 method, max_new_tokens, do_sample, temperature, top_p, top_k, tokens_for_outlines,
                 refinement_prompts, hyde_prompt, validation_retry_prompt, json_output, verbose):

        self.schema_model = schema_model

        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.tokens_for_outlines = tokens_for_outlines
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.method = method
        self.refinement_prompts = refinement_prompts
        self.hyde_prompt = hyde_prompt
        self.validation_retry_prompt = validation_retry_prompt
        self.json_output = json_output
        self.use_local_llm = use_local_llm
        self.predictions = {}
        self.prompt_history = []
        self.prediction_comparison_history = []
        # Token usage for the complete dataset/experiment
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.verbose=verbose

        if self.use_local_llm:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=self.torch_dtype, device_map=self.device_map,)
            self.original_embedding = self.model.model.embed_tokens
            self.model.eval()  # only inference mode
            print("HF device map:")
            print(self.model.hf_device_map)
            self.schema = self.schema_model.model_json_schema()
            self.parser = JsonSchemaParser(self.schema)
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, self.parser,)
            self.outline_model = outlines.from_transformers(self.model, self.tokenizer,)

        else:
            self.client = OpenAI(base_url=api_base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
            self.api_model = api_model

        if self.verbose:
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: "
                      f"allocated={torch.cuda.memory_allocated(i)/1024**3:.2f} GB, "
                      f"reserved={torch.cuda.memory_reserved(i)/1024**3:.2f} GB")

    def prepare_prompt(self, stage, predictions=None, **kwargs):
        if predictions:
            kwargs.update(predictions)
        mode = stage["mode"]
        if mode == "plain":
            return stage["prompt"].format(**kwargs)

        if mode == "chat":
            messages = [
                {
                    "role": "system",
                    "content": stage["system_prompt"],
                },
                {
                    "role": "user",
                    "content": stage["user_prompt"].format(**kwargs),
                },
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,) if self.use_local_llm else messages
        raise ValueError(f"Unknown mode: {mode}")

    # EXPERIMENTAL (disabled): batching. As written this was sequential (same speed as no batching).
    # To re-enable with a real speed-up:
    #   - make `self.predictions` a local variable in _generate_prompting (pass it to compare_predictions);
    #     it is shared per-row state, so concurrent rows would overwrite each other's predictions
    #   - API backend only: run rows through a ThreadPoolExecutor (vLLM batches concurrent requests);
    #     local HF generate is not thread-safe, keep it sequential
    #   - guard the token counters in _update_token_usage with a threading.Lock
    # def generate_structured_batch(self, batch):
    #     results = []
    #     for contexts in batch:
    #         result = self.generate_structured(**contexts)
    #         results.append(result)
    #
    #     return results

    def generate_structured(self, row, row_index=None, rag_context="", gower_context=""):

        if self.use_local_llm:
            if self.method == "lmfe":
                generator = self._generate_lmfe

            elif self.method == "outlines":
                generator = self._generate_outlines
            else:
                raise ValueError(f"Unknown structured method: {self.method}")
        else:
            generator = self._generate_api

        return self._generate_prompting(generator=generator, row=row, row_index=row_index, rag_context=rag_context,
                                        gower_context=gower_context)

    # <--------------------------- LM FORMAT ENFORCER --------------------------->

    def _generate_lmfe(self, prompt):

        formatted_prompt = prompt.strip()
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt",).to(self.model.device)

        with torch.inference_mode():  # only forward pass, no need of gradients
            outputs = self.model.generate(**inputs,
                                          prefix_allowed_tokens_fn=self.prefix_function,
                                          max_new_tokens=self.max_new_tokens,
                                          do_sample=self.do_sample,
                                          temperature=self.temperature,  # changes how confident the probabilities are
                                          top_p=self.top_p,
                                          top_k=self.top_k,
                                          pad_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,)

    # <--------------------------- OUTLINES ------------------------------------->

    def _generate_outlines(self, prompt):

        with torch.inference_mode():
            formatted_prompt = prompt.strip()
            result = self.outline_model(formatted_prompt, self.schema_model, max_new_tokens=self.tokens_for_outlines)
        return result

    # <--------------------------- API call ------------------------------------->

    def _generate_api(self, prompt):

        response = self.client.responses.parse(
            model=self.api_model,
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            text_format=self.schema_model,
        )

        self._update_token_usage(response)
        return response.output_parsed.model_dump_json()

    # <--------------------------- Validation ----------------------------------->

    def _validate_and_dump(self, data):
        try:
            # LM Format Enforcer -> JSON string
            if isinstance(data, str):
                validated = self.schema_model.model_validate_json(data)
            # Outlines -> ProductRow object (or dict)
            else:
                validated = self.schema_model.model_validate(data)
            return validated.model_dump()
        except ValidationError as e:
            print(f"Validation failed: {e}")
            return None

    # <--------------------------- HyDE ----------------------------------------->

    def generate_hyde(self, row_text, missing_columns,):

        prompt = self.prepare_prompt(self.hyde_prompt, row_text=row_text, missing_columns=missing_columns,).strip()

        if self.use_local_llm:
            return self._generate_hyde_local(prompt)
        else:
            return self._generate_hyde_api(prompt)

    def _generate_hyde_local(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt",).to(self.model.device)
        with torch.inference_mode():
            if self.json_output:
                outputs = self.model.generate(
                    **inputs,
                    prefix_allowed_tokens_fn=self.prefix_function,
                    max_new_tokens=200, # hard-coded max-new token limit
                    do_sample=False, # Deterministic result
                    temperature=0.0, # Deterministic result
                    pad_token_id=self.tokenizer.eos_token_id,)
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200, # hard-coded max-new token limit
                    do_sample=False, # Deterministic result
                    temperature=0.0, # Deterministic result
                    pad_token_id=self.tokenizer.eos_token_id,)

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,).strip()

    def _generate_hyde_api(self, prompt):

        if self.json_output:

            response = self.client.responses.parse(
                model=self.api_model,
                input=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                text_format=self.schema_model,
            )
            self._update_token_usage(response)

            return response.output_parsed.model_dump_json()

        else:

            response = self.client.responses.create(
                model=self.api_model,
                input=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            self._update_token_usage(response)
            return response.output_text.strip()

    # <--------------------------- staged prompting + retry -------------------->
    def _generate_prompting(self, generator, row, row_index=None, rag_context="", gower_context=""):

        self.predictions = {}
        for call_index, stage in enumerate(self.refinement_prompts, start=1):
            prompt = self.prepare_prompt(stage=stage, row=row, rag_context=rag_context, gower_context=gower_context, predictions=self.predictions,)
            if self.verbose:
                print("=" * 80)
                print(f"Stage: {stage.get('name', 'Unknown')}")
                print("=" * 80)
                print(prompt)
                print(f"Stage: {stage.get('name')} | Prompt length: {len(prompt)} characters", flush=True)
            start = time.perf_counter()
            try:
                prediction = generator(prompt)
                print(f"SUCCESS after {time.perf_counter()-start:.2f}s", flush=True)
            except Exception as e:
                print(f"FAILED after {time.perf_counter()-start:.2f}s | {type(e).__name__}: {e}", flush=True)
                raise
            prediction_key = f"prediction_{call_index}"
            self.predictions[prediction_key] = prediction

            self.prompt_history.append({
                "row_index": row_index,
                "call": call_index,
                "stage": stage.get("name", "Unknown"),
                "prompt": prompt,
                "output": prediction,
            })
        self.compare_predictions(row=row, row_index=row_index,)
        final_prediction = self.predictions[f"prediction_{len(self.refinement_prompts)}"]
        if self.verbose:
            print("=" * 80)
            print("Output")
            print("=" * 80)
            print(final_prediction)
        attempts = 0
        validated = self._validate_and_dump(final_prediction)
        # TODO: Make max_retries an argument for the user
        while validated is None and attempts < 3:
            retry_prompt = self.prepare_prompt(stage=self.validation_retry_prompt, row=row, prediction=final_prediction,
                                               rag_context=rag_context, gower_context=gower_context, predictions=self.predictions,)
            retry_prediction = generator(retry_prompt)
            self.prompt_history.append({
                "row_index": row_index,
                "call": f"validation_retry_{attempts + 1}",
                "stage": "validation_retry",
                "prompt": retry_prompt,
                "output": retry_prediction,
            })
            final_prediction = retry_prediction
            validated = self._validate_and_dump(final_prediction)
            attempts += 1

        return validated

    def compare_predictions(self, row, row_index):

        row = json.loads(row) if isinstance(row, str) else row

        predictions = {key: json.loads(value) if isinstance(value, str) else value for key, value in self.predictions.items()}

        missing_cols = [col for col, value in row.items() if value is None]
        observed_cols = [col for col, value in row.items() if value is not None]

        prediction_keys = list(predictions.keys())

        for i, current_key in enumerate(prediction_keys):

            current = predictions[current_key]

            # Check whether this call changed originally observed values
            observed_changes = {
                col: {
                    "original": row[col],
                    current_key: current.get(col),
                }
                for col in observed_cols
                if current.get(col) != row[col]}

            # Compare imputed values with previous call
            imputed_changes = {}

            if i > 0:
                previous_key = prediction_keys[i - 1]
                previous = predictions[previous_key]

                imputed_changes = {col: {
                    previous_key: previous.get(col),
                    current_key: current.get(col), }
                    for col in missing_cols
                    if previous.get(col) != current.get(col)}

            if imputed_changes or observed_changes:
                self.prediction_comparison_history.append({
                    "row_index": row_index,
                    "call": current_key,
                    "imputed_changes": imputed_changes,
                    "observed_changes": observed_changes,
                })

    def _update_token_usage(self, response):
        usage = getattr(response, "usage", None)

        if usage is not None:
            self.total_input_tokens += usage.input_tokens or 0
            self.total_output_tokens += usage.output_tokens or 0
            self.total_tokens += usage.total_tokens or 0



# =============================================================================
#  Output schema  
# =============================================================================
def build_schema_model(df, field_names, enable_constraints=False):
    fields = {}
    for column in df.columns:
        series = df[column].dropna()
        name = field_names[column]
        if pd.api.types.is_numeric_dtype(series):
            dtype = int if pd.api.types.is_integer_dtype(series) else float
            bounds = ({"ge": dtype(series.min()), "le": dtype(series.max())}
                      if enable_constraints and len(series) else {})
            fields[name] = (dtype, Field(..., **bounds))
        else:
            values = sorted(series.astype(str).unique())
            fields[name] = (Literal[tuple(values)] if 0 < len(values) <= 30 else str, ...)
    return create_model("ProductRow", **fields)


class SchemaGenerator:
    """Column-name helpers for the output schema (the code generation below is replaced by build_schema_model)."""
    @staticmethod  # no need to create object
    def clean_column_name(column: str) -> str:
        # Convert column names into valid Python identifiers

        column = re.sub(r"\W+", "_", column.strip())  # replacing name space in columns with _
        # if column[0].isdigit():
        #     column = "_" + column
        # pydantic treats names starting with "_" as private attributes, not fields -> they would vanish from the
        # schema. So prefix "f" instead (also covers empty names and names starting with a special character).
        if not column or column[0].isdigit() or column[0] == "_":
            column = "f" + column
        if keyword.iskeyword(column):
            column += "_"

        return column

    @classmethod
    def field_names(cls, columns) -> dict:
        """Original column name -> unique schema field name.

        Names that clean to the same identifier ("a b", "a_b") get a numeric suffix ("a_b", "a_b_2").
        """
        names, used = {}, set()
        for column in columns:
            name = base = cls.clean_column_name(column)
            suffix = 2
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            used.add(name)
            names[column] = name
        return names


# =============================================================================
#  Row <-> text helpers    
# =============================================================================
class Datasetp:
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
            sampled_complete = complete_rows.sample(n=min(n_complete, len(complete_rows)), random_state=42) # TODO: Make hard coded value a parameter
            rows = pd.concat([missing_rows, sampled_complete])
            rows = rows.sort_index()

        if str(max_rows).lower() != "all":

            max_rows = int(max_rows)
            if max_rows <= 0:
                raise ValueError("max_rows must be greater than 0 or 'all'")

            rows = rows.head(max_rows)
        return rows

    def rowtojson(self, row):

        row_dict = row.where(pd.notna(row), None).to_dict()  # replace with None if we have missing values
        return json.dumps(row_dict, indent=2)

    def dataframetojson(self, dataframe):  # returns a list of JSON strings

        rows_json = []
        for _, row in dataframe.iterrows():
            rows_json.append(
                self.rowtojson(row))
        return rows_json

    def get_column_types(self, df):

        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

        return numerical_columns, categorical_columns


# =============================================================================
#  Neighbouring rows # pure python is slow; vectorization might help otherwise call into a C/C++/rust api
# =============================================================================
class Retriever:

    def __init__(self, top_k, data):
        self.top_k = top_k
        self.data = data

    def retrieve(self, target_row, dataset):

        numerical_columns, categorical_columns = self.data.get_column_types(dataset)
        ranges = {}

        for column in numerical_columns:
            column_range = dataset[column].max() - dataset[column].min()
            if column_range == 0:
                ranges[column] = 1.0
            else:
                ranges[column] = column_range

        distances = []
        # below we are converting row in dataframe into series, then we are accessing using column names
        for idx, candidate_row in dataset.iterrows():

            total = 0.0
            valid = 0
            for column in numerical_columns:
                x = target_row[column]
                y = candidate_row[column]

                if pd.isna(x) or pd.isna(y):
                    continue
                distance = abs(x - y) / ranges[column]
                distance = min(distance, 1.0)  # maintaining gower's definition
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
                distance = np.inf
            else:
                distance = total / valid

            distances.append((idx, distance))

        # ---------- Sorting rows by distance ----------
        distances.sort(key=lambda x: x[1])
        top_rows = distances[:self.top_k]
        top_indices = []
        for row in top_rows:
            top_indices.append(row[0])  # assigning only index instead of distance
        similar_rows = dataset.loc[top_indices].copy()
        gower_distances = []
        for row in top_rows:
            gower_distances.append(row[1])
        similar_rows["gower_distance"] = gower_distances
        similar_rows["gower_similarity"] = 1 - similar_rows["gower_distance"]
        return similar_rows



# =============================================================================
#  Hybrid RAG
# =============================================================================

# class WebSearch:
#
#     def __init__(self, max_results):
#         self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
#
#     def search(self, query):
#         return self.wrapper.results(query, max_results=self.wrapper.max_results)


class DocumentLoader:

    def __init__(self, pdf_path, chunk_size, chunk_overlap, enable_web_search, websearch_query, websearch_max_results):

        self.pdf_path = Path(pdf_path)
        # self.enable_web_search = enable_web_search
        self.query = websearch_query
        # self.web_search = WebSearch(websearch_max_results)
        self.chunker = HierarchicalChunker()
        self.converter = DocumentConverter()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)

    def load(self):
        documents = []
        pdf_chunks = []
        for file in self.pdf_path.iterdir():
            if not file.is_file():
                continue
            try:
                if file.suffix.lower() == ".pdf":
                    result = self.converter.convert(file)
                    for chunk in self.chunker.chunk(result.document):
                        pdf_chunks.append(Document(page_content=chunk.text, metadata={
                            "source": str(file),
                            "filename": file.name, }))
                    print(f"Loaded: {file.name}")

                else:
                    result = self.converter.convert(file)
                    documents.append(Document(page_content=result.document.export_to_markdown(),
                                              metadata={"source": str(file), "filename": file.name}))
                    print(f"Loaded: {file.name}")

            except Exception as e:
                print(f"Skipping {file.name}: {e}")

            # no single extractor works on every website, sometimes docling does not work with javascript heavy websites
            # for web search we have duckduckgosearch and for data extraction we have webbaseloader and docling
        # if self.enable_web_search:
        #     # for web search we are not using HierarchicalChunker
        #     results = self.web_search.search(self.query)
        #     for result in results:
        #         url = result["link"]
        #
        #         try:
        #             doc = self.converter.convert(url)
        #             documents.append(Document(page_content=doc.document.export_to_markdown(),
        #                                       metadata={"source": url, "title": result["title"], "type": "web", },))
        #             print(f"Loaded Web (Docling): {url}")
        #         except Exception:
        #
        #             try:
        #                 loader = WebBaseLoader(url, header_template={"User-Agent": "Mozilla/5.0"}, requests_kwargs={"timeout": 20},
        #                                        raise_for_status=True, bs_get_text_kwargs={"separator": "\n", "strip": True, },)
        #                 for doc in loader.load():  # langchain provides loader.load function to convert data into langchain documents
        #                     text = "\n".join(line.strip() for line in doc.page_content.splitlines() if line.strip())
        #                     doc.page_content = text
        #                     doc.metadata.update({"title": result["title"], "type": "web", })
        #                     documents.append(doc)
        #                 print(f"Loaded Web (WebBaseLoader): {url}")
        #             except Exception as e:
        #                 print(f"Skipping {url}: {e}")
        self.documents = documents
        self.pdf_chunks = pdf_chunks

        return documents, pdf_chunks

    def split(self):
        documents, pdf_chunks = self.load()
        chunks = self.splitter.split_documents(documents)
        chunks.extend(pdf_chunks)
        self.chunks = chunks
        return chunks


class EmbeddingModel:
    def __init__(self, model_name, device):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
                                               model_kwargs={"device": device})
        print("loaded embedding model")


class VectorDatabase:
    def __init__(self, embedding_model, persist_directory, collection_name,):
        self.vector_store = Chroma(collection_name=collection_name, embedding_function=embedding_model,
                                   persist_directory=persist_directory,)
    def add_documents(self, documents):
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)

    # hard-coded values (might need to do hyperparam search at some point)
    def get_retriever(self, search_type, k):
        if search_type == "mmr":
            return self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.8, },)
        elif search_type == "similarity_score_threshold":
            return self.vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": 0.7, },)
        elif search_type == "similarity":
            return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k, },)

    def delete_collection(self):
        self.vector_store.delete_collection()


class HybridRAG:

    def __init__(self, pdf_path, embedding_model, chunk_size, chunk_overlap, top_k_dense, top_k_bm25, rerank_top_n,
                 enable_web_search, websearch_query, websearch_max_results, alpha, persist_directory="./chroma_db", device="cuda:0"):

        self.loader = DocumentLoader(pdf_path, chunk_size, chunk_overlap, enable_web_search, websearch_query, websearch_max_results)
        self.chunks = self.loader.split()
        # TODO: using different models than just Qwen
        self.reranker = CrossEncoder("Qwen/Qwen3-Reranker-4B", device=device, model_kwargs={"torch_dtype": torch.bfloat16})
        print("loaded reranker")

        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)  # builds its own inverted index
        self.bm25_retriever.k = top_k_bm25
        embedding_model = EmbeddingModel(embedding_model, device)
        self.vector_db = VectorDatabase(embedding_model=embedding_model.embedding, persist_directory=persist_directory,
                                        collection_name="rag_collection",)
        self.vector_db.add_documents(self.chunks)  # converts into vectors and stores them
        retriever = self.vector_db.get_retriever(search_type="mmr", k=top_k_dense,)
        # print(torch.cuda.memory_summary())
        # Reciprocal Rank Fusion of dense and BM25 results
        self.hybrid_retriever = EnsembleRetriever(retrievers=[retriever, self.bm25_retriever], weights=[alpha, 1 - alpha])
        print("Created vector db succesfully")
        self.top_n = rerank_top_n

    def retrieve(self, retrieval_query, rerank_query):
        # Retrieve documents
        docs = self.hybrid_retriever.invoke(retrieval_query)
        if not docs:
            return []
        # Remove duplicates
        docs = list({doc.page_content: doc for doc in docs}.values())
        # Prepare text for reranker
        document_texts = [doc.page_content for doc in docs]
        # Rerank retrieved documents
        rankings = self.reranker.rank(query=rerank_query, documents=document_texts,)
        # Return Top-N
        return [docs[item["corpus_id"]] for item in rankings[:self.top_n]]

    def merge_documents(self, *doc_lists):

        unique_docs = {}
        for docs in doc_lists:
            for doc in docs:
                unique_docs[doc.page_content] = doc

        return list(unique_docs.values())

    def cleanup(self):
        self.vector_db.delete_collection()
        print("Vector database deleted successfully")


# =============================================================================
#  Default prompts      
# =============================================================================
# Placeholders filled with str.format:
#   impute / judge stages : {row} {rag_context} {gower_context} 
#                           {prediction_1}, {prediction_2}, ... (output of earlier stages)
#   validation retry      : the above + {prediction}
#   HyDE / RAG query      : {row_text} {missing_columns}

DEFAULT_IMPUTE_PROMPT = """\


You are an expert data analyst specializing in tabular data and missing-value imputation.
  Your task is to impute **only the missing values** while keeping all observed values unchanged.

  #Before answering, reason carefully using the available evidence in this priority:

    1. **Provided context** — use retrieved domain knowledge, metadata, feature-specific information, policies, rules, relationships, constraints, and relevant information.
    2. **Neighboring similar records** — use the provided similar rows to identify relevant patterns.
    3. **Record-level reasoning** — consider relationships and dependencies among features. Infer multiple missing values jointly when necessary.
    4. **General/domain knowledge** — use your own knowledge only as a **last resort** when the provided evidence is insufficient.

    When evidence conflicts, prefer the information most relevant and strongly supported by the current record and context.
    Perform a final consistency check and output **only the most plausible completed record** in the required structured format.

  #You will receive:
    1. The original incomplete record.
    2. Retrieved context, which may contain domain knowledge, metadata, feature-specific information, relationships, constraints, rules, and other relevant information.
    3. Neighboring similar records from the dataset.

  # Internal Reasoning
  Before answering, reason carefully step-by-step internally.

  1. Identify exactly which values are missing and which are observed.
  2. Analyze the observed values and determine how they relate to each missing feature.
  3. Examine the neighboring similar rows and identify patterns relevant to the missing feature(s).
  4. Use the neighboring similar rows as reference evidence for imputation, not as values to copy directly.
  5. Examine the retrieved context and identify information relevant to the missing feature(s).
  6. Use the retrieved context and neighboring similar rows as supporting evidence, not as ground truth.
  7. Use the retrieved domain knowledge, metadata, feature-specific information, relationships, constraints, and rules to support known relationships among the features.
  8. Combine evidence from the current record, the neighboring similar rows, retrieved domain knowledge, metadata, feature-specific information, relationships, and constraints to infer the most plausible record-specific value.
  9. Do not blindly copy a neighboring similar row or mechanically calculate the mean, median, or mode.
  10. If multiple values are missing, infer them jointly so they remain mutually consistent.
  11. Check relationships among all features.
  12. Verify that the completed record is internally consistent and appropriate for the observed profile and retrieved context.
  13. Reconsider any questionable prediction internally before producing the final answer.

  ## Evidence Priority

  Current record's observed values > relevant relationships among features > retrieved domain knowledge, metadata, and feature-specific information > neighboring similar records.

  ## Rules

  * Never modify originally observed values.
  * Predict only originally missing values.
  * Use neighboring similar rows as reference evidence, not ground truth.
  * Use retrieved domain knowledge, metadata, feature-specific information, relationships, constraints, and rules to understand the features and their dependencies.
  * Do not blindly copy or average values from neighboring similar records.
  * Prefer record-specific evidence over generic assumptions.
  * Do not invent unsupported information.
  * Ensure all predicted values are realistic and mutually consistent.
  * Ensure all predicted features are consistent with the overall record and retrieved context.
  * Perform all reasoning internally and do not output your reasoning.

  # Original incomplete record
  {row}

  # CONTEXT
  Retrieved domain knowledge, metadata, feature-specific information, relationships, constraints, rules, and other relevant information:
  {rag_context}

  # Neighboring similar rows:
  {gower_context}

  Return only one complete valid JSON object containing all features.
"""


DEFAULT_JUDGE_PROMPT = """\
You are an expert data analyst specializing in tabular data and missing-value imputation. Your task is to independently evaluate and refine an imputed record.
Use the available evidence to determine the most plausible values for the missing features while preserving all originally observed values.

You will receive:

1. The original incomplete record.
2. The previous prediction.
3. Retrieved context containing relevant domain knowledge, metadata, feature-specific information, relationships, constraints, rules, and other relevant information.
4. Neighboring similar records from the dataset.

Use the original incomplete record, retrieved context, and neighboring similar records as the primary evidence for independently evaluating the previous prediction. Consider relationships among the observed and missing features and reason jointly when multiple values are missing. Use your general knowledge only as a last resort when the available evidence is insufficient.

## Your Task

Judge whether the imputed values in the previous prediction are the most plausible values supported by the available evidence, rather than merely whether they are possible or valid.

Before answering, reason carefully and internally:

1. Identify exactly which values are missing and which are observed.
2. Analyze the observed values and determine how they relate to each missing feature.
3. For each originally missing feature, independently determine the most plausible value using the original record, retrieved context, feature-specific information, relationships, constraints, and neighboring similar records.
4. Perform this independent assessment before deciding whether the previous prediction should be retained.
5. Compare the independently inferred value for each missing feature with the corresponding value in the previous prediction.
6. Evaluate whether each imputed numerical value is well supported by the available evidence and whether another realistic value is better supported.
7. Evaluate whether each imputed categorical value is well supported by the available evidence and whether another valid category is better supported.
8. Use neighboring similar records as reference evidence and examine whether the previous prediction is consistent with relevant local patterns.
9. Use retrieved domain knowledge, metadata, feature-specific information, relationships, constraints, and rules to evaluate the previous prediction.
10. Do not blindly accept the previously predicted values simply because they are plausible.
11. Distinguish between a value that is merely plausible and a value that is the most strongly supported by the available evidence.
12. If the evidence supports another value more strongly than the previous prediction, replace the previous prediction for that missing feature.
13. If multiple values are missing, infer and evaluate them jointly so they remain mutually consistent.
14. Check relationships among all features.
15. Verify that the completed record is internally consistent and appropriate for the observed profile and retrieved context.
16. Reconsider any questionable prediction internally before producing the final answer.

## Evidence Priority

Current record's observed values > relevant relationships among features > retrieved domain knowledge, metadata, and feature-specific information > neighboring similar records > previous prediction.

## Rules

* Never modify originally observed values.
* Modify only originally missing features.
* Treat the previous prediction as a candidate prediction to evaluate, not as ground truth.
* Do not give the previous prediction preferential treatment merely because it was generated earlier.
* Do not blindly accept predicted values.
* Do not change a prediction merely to produce a different answer.
* Keep a previous prediction when it remains the value most strongly supported by the available evidence.
* Change a previous prediction when another value is more strongly supported by the available evidence.
* Use neighboring similar rows as reference evidence, not ground truth.
* Use retrieved domain knowledge, metadata, feature-specific information, relationships, constraints, and rules to understand the features and their dependencies.
* Do not blindly copy or average values from neighboring similar records.
* Prefer record-specific relationships and observed evidence.
* Do not invent unsupported information.
* Ensure the final record is realistic and internally consistent.
* Perform all reasoning internally and do not output your reasoning.

Original incomplete record:
{row}

Previous prediction:
{prediction_1}

# CONTEXT
Retrieved domain knowledge, metadata, feature-specific information, relationships, constraints, rules, and other relevant information:
{rag_context}

# Neighboring similar rows:
{gower_context}

After independently evaluating every originally missing feature, compare the evidence-supported value with the previous prediction. Preserve a predicted value when it remains the most strongly supported value. Replace it only when another value is better supported by the available evidence.

Return only one complete valid JSON object containing all features, with no reasoning, explanation, Markdown, or additional text.
"""


DEFAULT_VALIDATION_RETRY_TEXT = """\


You are an expert data analyst specializing in tabular data and missing-value imputation.
Original incomplete record:
{row}

Previous prediction:
{prediction}

The previous prediction did not satisfy the required validation criteria. Carefully review and correct it to produce a complete, valid, and logically consistent record.
Before answering, reason internally about the observed values and relationships among the features.

Rules:
- Never modify values observed in the original record.
- Correct only missing, invalid, incomplete, or implausible predicted values.
- Preserve predicted values that are already valid and plausible.
- Impute every originally missing feature.
- Ensure all features are present.
- Ensure predicted values are mutually consistent with relationships among the features and available domain knowledge.
- Use general knowledge only when necessary and avoid unsupported assumptions.
- Do not output your reasoning.
- Return only one complete valid JSON object.
- Do not include explanations, Markdown, or additional text.
"""


DEFAULT_HYDE_TEXT = """\

You are generating a hypothetical document for Retrieval-Augmented Generation (HYDE).
 ## Available Record
 {row_text}
 ## Missing Features
 {missing_columns}
 ## Task
 Generate ONE short hypothetical retrieval document focused specifically on {missing_columns}.
 The primary objective is to generate semantic knowledge describing how {missing_columns} may relate to the observed features and values in {row_text}, and feature-specific information useful for understanding {missing_columns}.
 {missing_columns} must remain the PRIMARY and DOMINANT subject throughout the generated document.

 ## Internal Reasoning
 1. Focus first on {missing_columns}.
 2. Inspect the observed features and values in {row_text}.
 3. Determine which observed features may provide useful evidence about {missing_columns}.
 4. Identify possible direct or indirect relationships between {missing_columns} and those relevant observed features.
 5. Identify feature-specific information that may be useful for understanding {missing_columns}.
 6. Ignore observed features that do not provide useful information about {missing_columns}.
 7. Generate semantic text describing only relationships and feature-specific information useful for understanding {missing_columns}.

 ## Retrieval Keywords
 Use terminology strongly centered on {missing_columns}.
 Include:

 * The exact feature name(s) {missing_columns}.
 * Semantically related terminology for {missing_columns} when appropriate.
 * Observed feature names only when describing their relationship with {missing_columns}.
 * Observed categorical or numerical values only when they provide useful context for {missing_columns}.
 * Feature-specific information related to {missing_columns} when appropriate.
 * Relevant terms such as relationship, association, dependency, condition, constraint, range, threshold, status, eligibility, or pattern when useful.

 Use {missing_columns} explicitly and naturally throughout the document to strengthen semantic retrieval.

 ## Generation Rules

 * Generate a semantic retrieval document, NOT a completed record.
 * Focus almost entirely on {missing_columns}.
 * Every sentence must provide information useful for retrieving knowledge about {missing_columns}.
 * Include information useful for retrieving both relationships involving {missing_columns} and feature-specific information about {missing_columns}.
 * Discuss observed features ONLY when connecting them to {missing_columns}.
 * Do NOT describe observed features independently.
 * Do NOT discuss relationships only among observed features.
 * Do NOT summarize the entire record.
 * Do NOT introduce unrelated features.
 * Do NOT generate broad or generic pension information.
 * Do NOT predict or impute {missing_columns}.
 * Do NOT provide a final value for {missing_columns}.
 * Do NOT invent unsupported relationships or rules.
 * Use actual observed values when they improve semantic matching with {missing_columns}.
 * Keep the document under 200 tokens.

 ## Retrieval Priority
 {missing_columns} and feature-specific information >>> {missing_columns} and its relationships with observed values >>> conditions, constraints, ranges, thresholds, dependencies, and patterns directly involving {missing_columns} >>> everything else.

 Return only the hypothetical retrieval document with no reasoning, explanation, Markdown, or additional text.
"""


DEFAULT_RAG_PROMPT = """\

# RECORD

Available features and values:
{row_text}
## RETRIEVAL TARGET
Missing features:
{missing_columns}
{missing_columns} is the PRIMARY and DOMINANT retrieval target.
Retrieve only knowledge from the provided knowledge base that may help understand or estimate {missing_columns} using the observed information in {row_text}.
## RETRIEVAL INSTRUCTIONS
1. Search primarily for passages that explicitly discuss {missing_columns}.
2. Retrieve direct or indirect relationships between {missing_columns} and the AVAILABLE features in {row_text}.
3. Prioritize passages where {missing_columns} and relevant observed features are discussed together.
4. Retrieve relationships, dependencies, conditions, constraints, ranges, thresholds, patterns, or associations involving {missing_columns} when they are explicitly supported by the knowledge base.
5. Prioritize information relevant to the actual observed values or categories in {row_text}.
6. Use information about an observed feature only when it provides evidence about {missing_columns}.
7. Give highest relevance to passages that help distinguish plausible values of {missing_columns}.
## RESTRICTIONS
* Do NOT assume or predefine relationships between features.
* Do NOT infer relationships from general knowledge.
* Do NOT introduce rules, thresholds, ranges, or dependencies that are not present in the provided knowledge base.
* Do NOT retrieve information merely because it matches an observed feature.
* Do NOT retrieve standalone descriptions of observed features unless they establish a relationship with {missing_columns}.
* Do NOT retrieve relationships unrelated to {missing_columns}.
* Do NOT predict or impute {missing_columns}.
* Use the knowledge base only as supporting contextual evidence for imputation.

## RETRIEVAL PRIORITY
{missing_columns} + relationships with observed features >>> rules/constraints/patterns explicitly involving {missing_columns} >>> other directly relevant supporting information.
Return only knowledge from the provided knowledge base that is directly useful for understanding {missing_columns} in relation to the observed information in {row_text}.
"""


DEFAULT_WEBSEARCH_QUERY = """\
"Retrieve comprehensive evidence-based medical knowledge relevant to maternal health during pregnancy that may support
LLM-based imputation of missing values. Include clinical guidelines, physiological relationships, normal and abnormal ranges,
diagnostic criteria, maternal risk factors, and medically established correlations among maternal age, systolic blood pressure,
diastolic blood pressure, blood glucose, body temperature, heart rate, and maternal risk level."
"""


DEFAULT_PROMPTS = [
    {
        "name": "Imputation with LLM + Rag +gower(10)+cleveland_mnar_seed_90+change in prompt+chnaging in juding prompt+change in docuemnt parsing+pyampute injected",
        "mode": "plain",
        "prompt": DEFAULT_IMPUTE_PROMPT,
    },
    {
        "name": "Juding the prediciton",
        "mode": "plain",
        "prompt": DEFAULT_JUDGE_PROMPT,
    },
]

DEFAULT_VALIDATION_RETRY_PROMPT = {"mode": "plain", "prompt": DEFAULT_VALIDATION_RETRY_TEXT}

DEFAULT_HYDE_PROMPT = {"mode": "plain", "prompt": DEFAULT_HYDE_TEXT}
