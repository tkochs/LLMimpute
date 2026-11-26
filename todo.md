---
tags: [llmimpute, todo]
created: 2026-11-26
---
# TODO

Details and code snippets: [[publication-todo]]

## Reproduce the experiments
- [ ] Add data to `datasets/`: `energy.csv` + `energy_mcar_seed_{1,90,173,193,210,310}.csv`; check the headers are plain identifiers
- [ ] Fill `rag_documents/` with the original RAG documents (`Rag_documents5`)
- [ ] `config27.yaml`: relative paths (`datasets/…`, `outputs`); point `Rag_retrieval.document_folder` (l.179) to `rag_documents` instead of `/hri/storage/…/Rag_documents5`; comment out `adapter_path` and `inference.batching`; fix the experiment descriptions (MCAR, seeds 193/210/310)
- [ ] Record the LLM server: checkpoint behind `Gemma-4-31B-IT`, vLLM version, launch command (the API calls send no sampling settings, so the server defaults apply)
- [ ] Pin the environment: commit `uv.lock`; declare `scikit-learn` and `langchain-classic`; drop the `hyperimpute` workspace member
- [ ] Recover the script that generated the amputed CSVs, or document how they were made
- [ ] Find the configs of every experiment in the paper (cleveland, maternal health, …); check none needs a removed feature (Hamming, random rows, web search, adapter)
- [ ] Run `config27.yaml` + `baselines.yaml` on the real setup and compare with the paper's numbers

## Bugs that change results
Decide per bug: keep it to reproduce the paper, or fix it and rerun before publishing.
- [ ] LLM is scored on fewer rows than mean/mode (failed rows are excluded for the LLM only)
- [ ] Gower: the query row is its own nearest neighbour
- [ ] Missing values are sent as `NaN` instead of `null` for all-numeric rows
- [ ] Neighbours are sent as a Python list repr (`dataframetojson`)
- [ ] API mode ignores the `generation` settings (temperature, top_p, …)
- [ ] "plain" prompts: the local model gets no chat template, the API applies one
- [ ] HyDE prompt still says "pension"

## Code (`src/`)
- [ ] Chat mode + API: `_generate_api` / `_generate_hyde_api` wrap the message list in another user message; `generate_hyde` calls `.strip()` on it
- [ ] Local chat mode: double BOS token → `add_special_tokens=False`
- [ ] `compare_predictions` doesn't use `schema_field_names_` (log is wrong for renamed columns)
- [ ] Bool columns are typed `float`; the schema class is still called `ProductRow`
- [ ] Hard-coded values → parameters: HyDE `max_new_tokens=200`, `random_state=42`, MMR / score threshold, reranker model, 3 retries
- [ ] Remove unused imports (`importlib.util`, `sys`, `torch.nn`, …)
- [ ] Stage-name typos (`chnaging`, `juding`, `docuemnt`, `prediciton`)

## Release
- [ ] `.gitignore`: `env`, `outputs/`, `__pycache__/`, chroma folders (check `env` holds no API key before committing)
- [ ] `LICENSE.md`: year range says 2007–2026; confirm the copyright holder (HRI / university?)
- [ ] Tests (sklearn `check_estimator`, schema, failed rows, runner) + CI
- [ ] README: install, `run_experiment.py` usage, LLM server setup
- [ ] `CITATION.cff`; package name and metadata in `pyproject.toml`
