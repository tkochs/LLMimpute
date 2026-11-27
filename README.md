# LLM Imputation

Impute missing values in tabular data with an LLM, using retrieved domain documents and similar rows as context.

## Usage

```python
from model import LLMImputer   # src/model.py, sklearn-style transformer

imp = LLMImputer(document_folder="docs/", chroma_dir="chroma/")
df_imputed = imp.fit(df).transform(df)
```

Experiments (LLM vs. mean/mode baseline, with metrics):

```
python src/run_experiment.py configfiles/config27.yaml
```

## Architecture

```
src/model.py           LLMImputer: the whole method
src/run_experiment.py  YAML config -> load/ampute data -> impute -> evaluate -> export to outputs/
src/ampute.py          inject MCAR / MAR / MNAR missingness
configfiles/           experiment configs (config27 = LLM, baselines = mean/mode only)
ragpipeline/           older standalone RAG code, not used by src/
```

For each row with missing values, `LLMImputer.transform` does:

1. **RAG**: hybrid retrieval (dense Chroma + BM25, RRF) over `document_folder`, then cross-encoder rerank
2. **HyDE**: the LLM writes a hypothetical document, which is used as an extra retrieval query
3. **Gower neighbours**: the top-k most similar rows from the fitted data
4. **Prompt stages**: impute, then judge/refine. The output is JSON constrained to a pydantic schema (lmfe / outlines / API), with a retry if validation fails

LLM backend: a local HF model, or an OpenAI-compatible API (`api.base_url`, key in `env`).
