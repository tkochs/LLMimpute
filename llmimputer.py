import json
from pydantic import ValidationError
from transformers import (AutoTokenizer,AutoModelForCausalLM,)
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (build_transformers_prefix_allowed_tokens_fn,)
import outlines
import torch
from peft import PeftModel
from adapter import InputAdapter
from adapter import EmbeddingWithAdapter
from data_processor import Datasetp
from openai import OpenAI
# from concurrent.futures import ThreadPoolExecutor




class QWENModel:

    def __init__(self, config,schema_model):

        self.config = config
        self.data = Datasetp()
        self.schema_model = schema_model

        self.model_name =self.config["model"]["name"]
        self.torch_dtype =self.config["model"]["torch_dtype"]
        self.device_map =self.config["model"]["device_map"]
        self.tokens_for_outlines=self.config["outlinestoken"]["max_new_tokens"]
        self.max_new_tokens =self.config["generation"]["max_new_tokens"]
        self.do_sample =self.config["generation"]["do_sample"]
        self.temperature =self.config["generation"]["temperature"]
        self.top_p =self.config["generation"]["top_p"]
        self.top_k =self.config["generation"]["top_k"]
        self.method =self.config["structured_output"]["method"]
        self.adapter_path = self.config["model"]["adapter_path"]
        self.use_adapter = self.config["model"]["use_adapter"]
        self.adapter_bottleneck = self.config["model"]["adapter_bottleneck"]
        self.trained_adapter = self.config["model"]["trained_adapter"]
        self.refinement_prompts = self.config["prompting"]["prompts"]
        self.hyde_prompt = self.config["hyde"]["prompt"]
        self.validation_retry_prompt = self.config["prompting"]["validation_retry"]
        self.json_output=self.config["Rag_retrieval"]["hyde_json output"]
        self.use_local_llm = self.config["model"]["use_local_llm"]
        self.max_workers = self.config["inference"]["batching"].get("max_workers", 5)
        self.predictions = {}
        self.prompt_history = []
        self.prediction_comparison_history = []
        # Token usage for the complete dataset/experiment
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0 
            
        


        if self.use_local_llm:

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,dtype=self.torch_dtype,device_map=self.device_map,)
            self.original_embedding = self.model.model.embed_tokens
            self.model.eval()#only inference mode
            print("HF device map:")
            print(self.model.hf_device_map)
            self.schema = self.schema_model.model_json_schema()
            self.parser = JsonSchemaParser(self.schema)
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer,self.parser,)
            self.outline_model = outlines.from_transformers(self.model,self.tokenizer,)

        else:

            self.client = OpenAI(base_url=self.config["api"]["base_url"],api_key="EMPTY",)
            self.api_model = self.config["api"]["model"]

        # print("HF device map:")
        # print(self.model.hf_device_map)
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: "
                f"allocated={torch.cuda.memory_allocated(i)/1024**3:.2f} GB, "
                f"reserved={torch.cuda.memory_reserved(i)/1024**3:.2f} GB") 
        
        if self.use_adapter:

            print("Loading trained adapter...")
            adapter = InputAdapter(hidden_size=self.model.config.hidden_size, bottleneck=self.adapter_bottleneck,)
            if self.trained_adapter:
                adapter.load_state_dict(torch.load(self.adapter_path,map_location="cpu",))#loading the adapter
            else:
                print("Using randomly initialized adapter.")

            adapter = adapter.to(device=self.model.device,dtype=next(self.model.parameters()).dtype,)#random
            self.adapter_embedding = EmbeddingWithAdapter(self.original_embedding,adapter,)#included adapter
            print("✓ Adapter loaded successfully.")
        else:
            print("Running with base model (no adapter).")

      
    def use_base_embedding(self):
        if self.use_adapter:
            self.model.model.embed_tokens = self.original_embedding
    def use_adapter_embedding(self):
        if self.use_adapter:
            self.model.model.embed_tokens = self.adapter_embedding 



    def prepare_prompt(self, stage,predictions=None,**kwargs):


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

            return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,)
        raise ValueError(f"Unknown mode: {mode}")
        
    def generate_structured_batch(self, batch):

        results = []

        for contexts in batch:
            result = self.generate_structured(**contexts)
            results.append(result)
        

        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     results = list(
        #         executor.map(
        #         lambda contexts: self.generate_structured(**contexts),
        #         batch
        #         )
        #     )


        return results


    def generate_structured(self,row,row_index=None,rag_context="",gower_context="",random_rows_context="",hamming_context=""):

        if self.use_local_llm:
            if self.method == "lmfe":
                generator = self._generate_lmfe

            elif self.method == "outlines":
                generator = self._generate_outlines
            else:
                raise ValueError(f"Unknown structured method: {self.method}")
        else:
            generator = self._generate_api

        return self._generate_prompting(generator=generator,row=row,row_index=row_index,rag_context=rag_context,gower_context=gower_context,random_rows_context=random_rows_context,hamming_context=hamming_context)
# <--------------------------------------------------LM FORMAT ENFORCER-------------------------------------------->


    def _generate_lmfe(self, prompt):

        formatted_prompt = prompt.strip()
        # print("="*80)
        # print("LLM prompt",formatted_prompt)
        # print("="*80)
        self.use_adapter_embedding()
        inputs = self.tokenizer(formatted_prompt,return_tensors="pt",).to(self.model.device)

        with torch.inference_mode(): #only farword pass , no need of gradients
            outputs = self.model.generate(**inputs,
                prefix_allowed_tokens_fn=self.prefix_function,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature, #changes model how confident those probolities are
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id)
       
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True,)
        # print("result",raw_text)
        #outputs-getting first sequence of the generation
        #inputids have shape of dimension and no of tokens in that dimension 
        #input prompt and output

        # return self._validate_and_dump(raw_text)


# <----------------------------------OUTLINES------------------------------------------->


    def _generate_outlines(self, prompt):

        self.use_adapter_embedding()
        with torch.inference_mode():
            formatted_prompt = prompt.strip()
            result =self.outline_model(formatted_prompt_llm,self.schema_model,max_new_tokens=self.tokens_for_outlines)
        return result.model_dump_json()

# <----------------------------------API call------------------------------------------->
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


#<------------------------------------Validation--------------------------------->
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

   
    def generate_hyde(self,row_text,missing_columns,):


        prompt = self.prepare_prompt(self.hyde_prompt,row_text=row_text,missing_columns=missing_columns,).strip()
        # print("="*80)
        # print("hyde prompt",prompt)
        # print("="*80)

        if self.use_local_llm:
            return self._generate_hyde_local(prompt)
        else:
            return self._generate_hyde_api(prompt)


    def _generate_hyde_local(self, prompt):

        inputs = self.tokenizer(prompt,return_tensors="pt",).to(self.model.device)
        with torch.inference_mode():
            if self.json_output:
                outputs = self.model.generate(
                    **inputs,
                    prefix_allowed_tokens_fn=self.prefix_function,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,)
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,)

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True,).strip()

    def _generate_hyde_api(self, prompt):

        if self.json_output:

            response = self.client.responses.parse(model=self.api_model,
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


    def _generate_prompting(self,generator,row,row_index=None,rag_context="", gower_context="", random_rows_context="",hamming_context=""):

        self.predictions = {}
       
        for call_index, stage in enumerate(self.refinement_prompts, start=1):

            prompt = self.prepare_prompt(stage=stage,row=row,rag_context=rag_context,gower_context=gower_context,random_rows_context=random_rows_context,hamming_context=hamming_context,predictions=self.predictions,)
            print("=" * 80)
            print(f"Stage: {stage.get('name', 'Unknown')}")
            print("=" * 80)
            print(prompt)
            import time
            start = time.perf_counter()
            print(f"Stage: {stage.get('name')} | Prompt length: {len(prompt)} characters", flush=True)
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
                }
            )
        self.compare_predictions(row=row,row_index=row_index,)
        final_prediction = self.predictions[f"prediction_{len(self.refinement_prompts)}"]
        print("=" * 80)
        print("Output")
        print("=" * 80)
        print(final_prediction)
        attempts = 0
        validated= self._validate_and_dump(final_prediction)
        while validated is None and attempts < 3:
            retry_prompt = self.prepare_prompt(stage=self.validation_retry_prompt,row=row,prediction=final_prediction,rag_context=rag_context,gower_context=gower_context,random_rows_context=random_rows_context,hamming_context=hamming_context,predictions=self.predictions,)
            retry_prediction = generator(retry_prompt)
            self.prompt_history.append(
            {
                "row_index": row_index,
                "call": f"validation_retry_{attempts + 1}",
                "stage": "validation_retry",
                "prompt": retry_prompt,
                "output": retry_prediction,
            }
            )
            final_prediction = retry_prediction
            validated= self._validate_and_dump(final_prediction)
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
                        current_key: current.get(col),}
                    for col in missing_cols
                    if previous.get(col) != current.get(col)}

            if imputed_changes or observed_changes:
                self.prediction_comparison_history.append({
                    "row_index": row_index,
                    "call": current_key,
                    "imputed_changes": imputed_changes,
                    "observed_changes": observed_changes,
                })




    # def generate_hyde_batch(self, items):

    #     def worker(item):

    #         hyde_query = self.generate_hyde(
    #             row_text=item["row_json"],
    #             missing_columns=", ".join(
    #                 item["missing_columns"]
    #             ),
    #         )

    #         return item["index"], hyde_query

    #     with ThreadPoolExecutor(
    #         max_workers=self.max_workers
    #     ) as executor:

    #         results = list(
    #             executor.map(worker, items)
    #         )

    #     return dict(results)

    def _update_token_usage(self, response):
        usage = getattr(response, "usage", None)

        if usage is not None:
            self.total_input_tokens += usage.input_tokens or 0
            self.total_output_tokens += usage.output_tokens or 0
            self.total_tokens += usage.total_tokens or 0