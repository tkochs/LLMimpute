"""
Run imputation experiments from a YAML config: LLMImputer and/or the mean/mode baseline.
"""
import argparse
import copy
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

METHODS_SECTION = "chooseimputation method"
SHORT_LABELS = {"LLM": "LLM", "Mean/Mode": "MM"}  # column prefixes in "Numerical Statistics", as in pipeline.py

# pipeline.py switches for features LLMImputer does not have; enabling one is an error
UNSUPPORTED = [
    ("web_search", "enabled"),
    ("hamming_retrieval", "enabled"),
    ("random_missing_rows", "enabled"),
    ("model", "use_adapter"),
    ("model", "trained_adapter"),
]


# =============================================================================
#  Config
# =============================================================================

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_enabled(cfg, section, key="enabled"):
    return bool((cfg.get(section) or {}).get(key, False))


def experiment_configs(config, default_name):
    """One config per entry of `experiments:`, with that entry's overrides (as in pipeline.run).

    A config without an `experiments:` list is a single experiment, named by `name:` or the file name.
    """
    experiments = config.get("experiments")
    if not experiments:
        cfg = copy.deepcopy(config)
        cfg.setdefault("name", default_name)
        yield cfg
        return
    for experiment in experiments:
        cfg = copy.deepcopy(config)
        cfg["name"] = experiment["name"]
        cfg["summary"] = experiment["summary"]
        cfg["injected_dataset"]["path"] = experiment["injected_dataset_path"]
        cfg["injected_dataset"]["missing_rate"] = experiment["missing_rate"]
        yield cfg


def check_supported(cfg):
    enabled = [f"{section}.{key}" for section, key in UNSUPPORTED if is_enabled(cfg, section, key)]
    if enabled:
        raise ValueError(f"Not supported by LLMImputer: {', '.join(enabled)}. Disable it in the config.")


def imputer_kwargs(cfg, verbose=False):
    """Map the pipeline.py config sections onto LLMImputer's arguments."""
    model, api, gen, rag = cfg["model"], cfg["api"], cfg["generation"], cfg["Rag_retrieval"]
    return {
        "verbose": verbose,
        # LLM backend
        "use_local_llm": model["use_local_llm"],
        "model_name": model["name"],
        "torch_dtype": model["torch_dtype"],
        "device_map": model["device_map"],
        "api_base_url": api["base_url"],
        "api_model": api["model"],
        "structured_output": cfg["structured_output"]["method"],
        # generation
        "max_new_tokens": gen["max_new_tokens"],
        "do_sample": gen["do_sample"],
        "temperature": gen["temperature"],
        "top_p": gen["top_p"],
        "top_k": gen["top_k"],
        "outlines_max_new_tokens": cfg["outlinestoken"]["max_new_tokens"],
        # prompts
        "prompts": cfg["prompting"]["prompts"],
        "validation_retry_prompt": cfg["prompting"]["validation_retry"],
        # which rows are sent to the LLM
        "llm_input_complete_row_fraction": float(cfg["imputation"]["llm_input_complete_row_fraction"]),
        "max_rows": cfg["imputation"]["max_rows"],
        # output schema
        "schema_constraints": cfg["autoschema_constraints"]["enabled"],
        # RAG + HyDE
        "rag_retrieval": rag["enabled"],
        "document_folder": rag["document_folder"],
        "embedding_model": rag["embedding_model"],
        "chunk_size": rag["chunk_size"],
        "chunk_overlap": rag["chunk_overlap"],
        "rag_prompt": rag["rag_prompt"],
        "top_k_dense": rag["top_k_dense"],
        "top_k_bm25": rag["top_k_bm25"],
        "rerank_top_n": rag["rerank_top_n"],
        "dense_rrf_weight": rag["dense_rrf_weight"],
        "hyde": cfg["hyde"]["enabled"],
        "hyde_prompt": cfg["hyde"]["prompt"],
        "hyde_json_output": rag.get("hyde_json output", False),  # key name (with a space) as in config27.yaml
        "rag_device": rag.get("device", "cuda:0"),  # pipeline.py ran the embedding model and reranker on cuda:0
        "chroma_dir": rag.get("chroma_dir", os.path.join(tempfile.gettempdir(), "llmimpute_chroma")),
        # neighbouring rows
        "gower_retrieval": cfg["gower_retrieval"]["enabled"],
        "gower_top_k": cfg["gower_retrieval"]["top_k"],
    }


# =============================================================================
#  Data                                     <- pipeline.prepare_dataset, bestseed.py, data_processor.py
# =============================================================================

def best_sampling_seed(df, sample_size, max_seed):
    """Seed whose sample of `sample_size` rows best matches the full data (bestseed.DatasetSampler)."""
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    best_seed, best_score = None, float("inf")
    for seed in range(max_seed):
        sample = df.sample(n=sample_size, random_state=seed)
        num_diffs = []
        for col in numerical:
            col_range = df[col].max() - df[col].min()
            if col_range == 0:
                col_range = 1
            num_diffs.append(abs(df[col].mean() - sample[col].mean()) / col_range)
            num_diffs.append(abs(df[col].std() - sample[col].std()) / col_range)
        cat_diffs = []
        for col in categorical:
            full = df[col].value_counts(normalize=True, dropna=False)
            part = sample[col].value_counts(normalize=True, dropna=False)
            categories = full.index.union(part.index)
            cat_diffs.append(abs(full.reindex(categories, fill_value=0) - part.reindex(categories, fill_value=0)).mean())
        score = (np.mean(num_diffs) if num_diffs else 0) + (np.mean(cat_diffs) if cat_diffs else 0)
        if score < best_score:
            best_seed, best_score = seed, score
    return best_seed


def load_data(cfg):
    """Ground truth and the same table with missing values, validated as in pipeline.prepare_dataset."""
    original = pd.read_csv(cfg["dataset"]["path"]).dropna(axis=0).reset_index(drop=True)  # drop rows with NaN

    if is_enabled(cfg, "sampling"):
        sampling = cfg["sampling"]
        seed = best_sampling_seed(original, sampling["sample_size"], sampling["max_seed"])
        print(f"Best sampling seed: {seed}")
        # n=500 as in pipeline.py, which does not use sampling.sample_size here
        ground_truth = original.sample(n=500, random_state=seed).reset_index(drop=True)
    else:
        ground_truth = original.copy().reset_index(drop=True)

    if is_enabled(cfg, "missingness"):
        from ampute import MissingValueGenerator  # needs swiss_cheese; only imported when used
        m = cfg["missingness"]
        injected = MissingValueGenerator(mechanism=m["mechanism"], missing_rate=m["missing_rate"], mean=m["mean"],
                                         variance=m["variance"], mode=m["mode"], seed=m["seed"]).inject(ground_truth)
    elif is_enabled(cfg, "injected_dataset"):
        injected = pd.read_csv(cfg["injected_dataset"]["path"]).reset_index(drop=True)
    else:
        injected = ground_truth.copy()  # no missing values

    if len(ground_truth) != len(injected):
        raise ValueError(f"Dataset validation failed: row count mismatch. Original dataset has {len(ground_truth)} "
                         f"rows, but injected dataset has {len(injected)} rows.")
    if list(ground_truth.columns) != list(injected.columns):
        raise ValueError(f"Dataset validation failed: feature mismatch.\nOriginal features: {list(ground_truth.columns)}"
                         f"\nInjected features: {list(injected.columns)}")
    observed = injected.notna()
    if not ground_truth.where(observed).equals(injected.where(observed)):
        raise ValueError("Dataset validation failed: Some observed values in the injected dataset are different "
                         "from the original dataset.")
    print("Dataset validation successful.")
    print(f"Rows: {len(ground_truth)}")
    print(f"Features: {len(ground_truth.columns)}")
    return ground_truth, injected


def column_types(df):
    """Numerical and categorical column names (data_processor.get_column_types)."""
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    return numerical, categorical


def select_rows(df, alpha, max_rows):
    """Rows that get imputed (data_processor.select_rows; LLMImputer uses the same rule).

    alpha = fraction of the complete rows that is sent along with the rows that have missing values.
    """
    missing = df[df.isnull().any(axis=1)]
    complete = df[~df.isnull().any(axis=1)]
    if alpha == 0:
        rows = missing
    elif alpha == 1:
        rows = df
    else:
        n_complete = int(len(complete) * alpha)
        sampled = complete.sample(n=min(n_complete, len(complete)), random_state=42)
        rows = pd.concat([missing, sampled]).sort_index()
    if str(max_rows).lower() != "all":
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("max_rows must be greater than 0 or 'all'")
        rows = rows.head(max_rows)
    return rows


def analyze(df):
    """Dataset summary (data_processor.analyze)."""
    numerical, categorical = column_types(df)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_columns": len(numerical),
        "categorical_columns": len(categorical),
        "total_missing_values": int(df.isna().sum().sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "complete_rows": int((~df.isna().any(axis=1)).sum()),
    }


# =============================================================================
#  Methods                                  <- pipeline.run_llm_imputation, meanmodeimputer.py
# =============================================================================

def run_llm(cfg, injected, verbose=False):
    """Fit LLMImputer on the table with missing values and impute it; also collect its logs."""
    from model import LLMImputer  # torch / transformers / langchain are only needed when the LLM runs
    check_supported(cfg)
    imputer = LLMImputer(**imputer_kwargs(cfg, verbose))
    try:
        imputer.fit(injected)
        imputed = imputer.transform(injected)
        model = imputer.model_
        rag = getattr(imputer, "rag_", None)
        return {
            "imputed": imputed,
            "failed": list(imputer.failed_rows_),
            "retrieval_history": dict(imputer.retrieval_history_),
            "hyde_documents": list(imputer.hyde_documents_),
            "prompt_history": pd.DataFrame(model.prompt_history),
            "comparison_history": list(model.prediction_comparison_history),
            "token_usage": pd.DataFrame([{"input_tokens": model.total_input_tokens,
                                          "output_tokens": model.total_output_tokens,
                                          "total_tokens": model.total_tokens}]),
            "documents": list(getattr(rag.loader, "documents", [])) if rag is not None else [],
            "chunks": list(getattr(rag.loader, "chunks", [])) if rag is not None else [],
        }
    finally:
        imputer.cleanup()


def print_comparison_history(history):
    """Changes between the LLM's stages, and observed values it changed (as printed by pipeline.py)."""
    if not history:
        print("No changes detected.")
        return
    for change in history:
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


def mean_mode_impute(injected, rows):
    """Column mean (numerical) / mode (categorical) of the whole table, filled into `rows` only."""
    imputed = injected.copy()
    numerical, categorical = column_types(injected)
    means = {col: injected[col].mean() for col in numerical}
    modes = {}
    for col in categorical:
        mode = injected[col].mode(dropna=True)
        modes[col] = mode.iloc[0] if not mode.empty else None  # several modes -> the first
    for index in rows.index:
        for col in numerical:
            if pd.isna(imputed.at[index, col]):
                imputed.at[index, col] = means[col]
        for col in categorical:
            if pd.isna(imputed.at[index, col]):
                imputed.at[index, col] = modes[col]
    return imputed


# =============================================================================
#  Evaluation                               <- evaluation.py, comparision.py
# =============================================================================

def evaluate(cfg, ground_truth, imputed, missing_mask, original):
    """Metrics over the cells that were missing (evaluation.Evaluator.evaluate).

    rmse / mae per numerical column (min-max normalised with the range of `original` if
    evaluation.normalize_numerical), accuracy / macro_f1 per categorical column, each averaged over the
    columns; gower = mean over rows of the Gower distance between the true and imputed missing cells.
    """
    ev = cfg["evaluation"]
    numerical_metrics, categorical_metrics = ev["numerical_metrics"], ev["categorical_metrics"]
    numerical, categorical = column_types(ground_truth)
    results = {}

    rmse_values, mae_values = [], []
    for col in numerical:
        mask = missing_mask[col]
        y_true, y_pred = ground_truth.loc[mask, col], imputed.loc[mask, col]
        if ev["normalize_numerical"]:
            col_min, col_max = original[col].min(), original[col].max()
            col_range = col_max - col_min
            if col_range != 0:
                y_true = (y_true - col_min) / col_range
                y_pred = (y_pred - col_min) / col_range
        if len(y_true) == 0:
            continue
        if "rmse" in numerical_metrics:
            rmse_values.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        if "mae" in numerical_metrics:
            mae_values.append(mean_absolute_error(y_true, y_pred))
    if "rmse" in numerical_metrics:
        results["rmse"] = np.mean(rmse_values) if rmse_values else None
    if "mae" in numerical_metrics:
        results["mae"] = np.mean(mae_values) if mae_values else None

    accuracy_values, f1_values = [], []
    for col in categorical:
        mask = missing_mask[col]
        y_true, y_pred = ground_truth.loc[mask, col], imputed.loc[mask, col]
        if len(y_true) == 0:
            continue
        if "accuracy" in categorical_metrics:
            accuracy_values.append(accuracy_score(y_true, y_pred))
        if "macro_f1" in categorical_metrics:
            f1_values.append(f1_score(y_true, y_pred, average="macro"))
    if "accuracy" in categorical_metrics:
        results["accuracy"] = np.mean(accuracy_values) if accuracy_values else None
    if "macro_f1" in categorical_metrics:
        results["macro_f1"] = np.mean(f1_values) if f1_values else None

    if "gower" in ev["mixed_metrics"]:
        ranges = {}
        for col in numerical:
            col_range = original[col].max() - original[col].min()
            ranges[col] = col_range if col_range != 0 else 1.0
        row_distances = []
        for idx in ground_truth.index:
            total, valid = 0.0, 0
            for col in numerical:
                if not missing_mask.at[idx, col]:
                    continue
                x, y = ground_truth.at[idx, col], imputed.at[idx, col]
                if np.isnan(x) or np.isnan(y):
                    continue
                total += min(abs(x - y) / ranges[col], 1.0)
                valid += 1
            for col in categorical:
                if not missing_mask.at[idx, col]:
                    continue
                x, y = ground_truth.at[idx, col], imputed.at[idx, col]
                if x is None or y is None:
                    continue
                total += 0 if x == y else 1
                valid += 1
            if valid > 0:
                row_distances.append(total / valid)
        results["gower"] = np.mean(row_distances) if row_distances else None

    return results


def comparison_table(ground_truth, injected, imputed, missing_mask, selected_indices, retrieval_history):
    """One row per imputed cell: truth, injected value and each method's value (comparision.create)."""
    parts = []
    for col in ground_truth.columns:
        indices = missing_mask.index[missing_mask[col]].intersection(selected_indices)
        if len(indices) == 0:
            continue
        part = pd.DataFrame({"Row": indices, "Column": col,
                             "Ground Truth": ground_truth.loc[indices, col].values,
                             "Injected": injected.loc[indices, col].values})
        for label, df in imputed.items():
            part[label] = df.loc[indices, col].values
        part["Retrieved Rows"] = part["Row"].map(lambda idx: retrieval_history.get(idx, []))
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def numerical_statistics(comparison, numerical, methods):
    """Per numerical column: mean, std and MAE of each method vs the truth (comparision.numerical_statistics)."""
    stats = []
    present = set(comparison["Column"]) if "Column" in comparison else set()
    for col in [c for c in numerical if c in present]:
        temp = comparison[comparison["Column"] == col].copy()
        for name in ["Ground Truth", *methods]:
            temp[name] = pd.to_numeric(temp[name], errors="coerce")
        row = {"Column": col, "Count": len(temp), "GT Mean": temp["Ground Truth"].mean()}
        row.update({f"{SHORT_LABELS[m]} Mean": temp[m].mean() for m in methods})
        row["GT Std"] = temp["Ground Truth"].std()
        row.update({f"{SHORT_LABELS[m]} Std": temp[m].std() for m in methods})
        row.update({f"{SHORT_LABELS[m]} MAE": (temp[m] - temp["Ground Truth"]).abs().mean() for m in methods})
        stats.append(row)
    return pd.DataFrame(stats)


def categorical_statistics(comparison, categorical, methods):
    """Per categorical column: accuracy of each method (comparision.categorical_statistics)."""
    stats = []
    present = set(comparison["Column"]) if "Column" in comparison else set()
    for col in [c for c in categorical if c in present]:
        temp = comparison[comparison["Column"] == col]
        row = {"Column": col, "Count": len(temp)}
        row.update({f"{m} Accuracy": (temp["Ground Truth"] == temp[m]).mean() for m in methods})
        stats.append(row)
    return pd.DataFrame(stats)


# =============================================================================
#  Export                                   <- exportresults.py
# =============================================================================

class Exporter:
    """Writes into <export.output_dir_path>/<name>/ (exportresults.ResultExporter, minus the input() prompt)."""

    def __init__(self, cfg, overwrite=False):
        export = cfg["export"]
        self.cfg = cfg
        self.enabled = export["enabled"]
        self.output_dir = Path(export["output_dir_path"])
        self.name = cfg["name"]
        self.dir = self.output_dir / self.name
        if self.enabled and self.dir.exists() and not overwrite:
            raise FileExistsError(f"{self.dir} already exists. Use --overwrite or rename the experiment.")

    def _path(self, filename):
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / filename

    def dataset(self, name, df):
        if not self.enabled:
            return
        path = self._path(f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name} saved to: {path}")

    def tables(self, *tables):
        """Several titled tables (with index) in <name>.csv."""
        if not self.enabled:
            return
        path = self._path(f"{self.name}.csv")
        with open(path, "w") as f:
            for title, df in tables:
                f.write(f"{title}\n")
                df.to_csv(f, index=True)
                f.write("\n\n")
        print(f"\nResults saved to: {path}")

    def text(self, filename, documents, label):
        if not self.enabled:
            return
        path = self._path(filename)
        with open(path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(documents, start=1):
                f.write("=" * 80 + "\n")
                f.write(f"{label} {i}\n")
                f.write("=" * 80 + "\n")
                f.write(doc.page_content)
                f.write("\n\n")
                f.write(f"Metadata: {doc.metadata}\n\n")
        print(f"{filename} saved to: {path}")

    def config_file(self, config_path):
        """Copy of the config file as <name>.yaml."""
        if not self.enabled:
            return
        path = self._path(f"{self.name}.yaml")
        path.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"{Path(config_path).name} saved to: {path}")

    def history(self, results):
        """One row per method in <output_dir>/experiment_history.csv; a rerun replaces its rows in place."""
        if not self.enabled:
            return
        cfg = self.cfg
        if is_enabled(cfg, "missingness"):
            mechanism, missing_rate = cfg["missingness"]["mechanism"], cfg["missingness"]["missing_rate"]
        elif is_enabled(cfg, "injected_dataset"):
            mechanism, missing_rate = cfg["injected_dataset"]["mechanism"], cfg["injected_dataset"]["missing_rate"]
        else:
            mechanism, missing_rate = "None", 0
        current = pd.DataFrame([{"experiment": self.name, "mechanism": mechanism, "missing_rate": missing_rate,
                                 "method": r["method"], "rmse": r.get("rmse"), "mae": r.get("mae"),
                                 "accuracy": r.get("accuracy"), "macro_f1": r.get("macro_f1"),
                                 "gower": r.get("gower")} for r in results])
        path = self.output_dir / "experiment_history.csv"
        if path.exists():
            history = pd.read_csv(path)
            if self.name in history["experiment"].values:
                start = history.index[history["experiment"] == self.name][0]
                history = history[history["experiment"] != self.name].reset_index(drop=True)
                history = pd.concat([history.iloc[:start], current, history.iloc[start:]], ignore_index=True)
            else:
                history = pd.concat([history, current], ignore_index=True)
        else:
            history = current
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(path, index=False)
        print(f"Experiment history saved to: {path}")


# =============================================================================
#  One experiment
# =============================================================================

def run_experiment(cfg, config_path, overwrite=False, verbose=False):
    exporter = Exporter(cfg, overwrite)  # checks the output folder before anything runs
    ground_truth, injected = load_data(cfg)
    missing_mask = injected.isna()
    rows = select_rows(injected, float(cfg["imputation"]["llm_input_complete_row_fraction"]),
                       cfg["imputation"]["max_rows"])
    selected = rows.index

    imputed, results, failed, llm = {}, [], [], None
    if is_enabled(cfg[METHODS_SECTION], "llm"):
        llm = run_llm(cfg, injected, verbose)
        print_comparison_history(llm["comparison_history"])
        imputed["LLM"] = llm["imputed"]
        failed = llm["failed"]
        valid = selected.difference(failed)  # the LLM is scored only on rows it did not fail on
        scores = evaluate(cfg, ground_truth.loc[valid], llm["imputed"].loc[valid], missing_mask.loc[valid],
                          ground_truth)
        results.append(scores | {"method": "LLM"})
        if failed:
            print("Failed row indices:", failed)

    if is_enabled(cfg[METHODS_SECTION], "mean_mode"):
        imputed["Mean/Mode"] = mean_mode_impute(injected, rows)
        scores = evaluate(cfg, ground_truth.loc[selected], imputed["Mean/Mode"].loc[selected],
                          missing_mask.loc[selected], ground_truth)
        results.append(scores | {"method": "Mean/Mode"})

    if not imputed:
        raise ValueError(f"No method enabled under '{METHODS_SECTION}' (llm / mean_mode).")

    methods = list(imputed)
    comparison = comparison_table(ground_truth, injected, imputed, missing_mask, selected,
                                  llm["retrieval_history"] if llm else {})
    numerical, categorical = column_types(ground_truth)
    numerical_stats = numerical_statistics(comparison, numerical, methods)
    categorical_stats = categorical_statistics(comparison, categorical, methods)
    results_df = pd.DataFrame(results)
    for table in (numerical_stats, categorical_stats, results_df):
        print("=" * 80)
        print(table)
    print("=" * 80)

    # ---- export (same files as pipeline.py)
    exporter.history(results)
    exporter.dataset("Imputed_output_comparision", comparison)
    exporter.dataset("ground truth", ground_truth)
    exporter.dataset("injected_dataset", injected)
    exporter.config_file(config_path)
    if llm:
        exporter.dataset("llm_imputed_dataset", llm["imputed"])
        if not llm["prompt_history"].empty:
            for call, group in llm["prompt_history"].groupby("call"):
                exporter.dataset(f"llm_prompt_call_{call}", group)
        exporter.dataset("token_usage", llm["token_usage"])
        if cfg["Rag_retrieval"]["enabled"]:
            exporter.text("documents.txt", llm["documents"], "Document")
            exporter.text("chunks.txt", llm["chunks"], "Chunk")
        if cfg["hyde"]["enabled"]:
            exporter.dataset("hyde_documents", pd.DataFrame(llm["hyde_documents"]))
    if "Mean/Mode" in imputed:
        exporter.dataset("mean_mode_imputed_dataset", imputed["Mean/Mode"])

    tables = [("About the experiment", pd.DataFrame([cfg.get("summary") or {}])),
              ("Evaluation Results", results_df)]
    if is_enabled(cfg, "datasetanalysis"):
        tables += [("Original Dataset Summary", pd.DataFrame([analyze(ground_truth)])),
                   ("Injected Dataset Summary", pd.DataFrame([analyze(injected)]))]
    tables += [("Numerical Statistics", numerical_stats),
               ("Categorical Statistics", categorical_stats),
               ("failed_validation_indices", pd.DataFrame(failed, columns=["Index"]))]
    exporter.tables(*tables)
    return results_df


# =============================================================================
#  CLI
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run imputation experiments (LLMImputer and/or mean/mode) "
                                                 "from a YAML config.")
    parser.add_argument("config", type=Path,
                        help="YAML config, e.g. configfiles/config27.yaml or configfiles/baselines.yaml")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output folders")
    parser.add_argument("--verbose", action="store_true",
                        help="print every input row, context and LLM output (LLMImputer verbose)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    failed_experiments = []
    for cfg in experiment_configs(config, default_name=args.config.stem):
        name = cfg["name"]
        print("\n" + "=" * 100)
        print(f"STARTING EXPERIMENT: {name}")
        print(f"DATASET: {(cfg.get('injected_dataset') or {}).get('path')}")
        print(f"MISSING RATE: {(cfg.get('injected_dataset') or {}).get('missing_rate')}")
        print("=" * 100)
        try:
            run_experiment(cfg, args.config, overwrite=args.overwrite, verbose=args.verbose)
            print("\n" + "=" * 100)
            print(f"COMPLETED SUCCESSFULLY: {name}")
            print("=" * 100)
        except Exception as error:  # one failed experiment must not stop the others (as in pipeline.py)
            failed_experiments.append(name)
            print("\n" + "=" * 100)
            print(f"FAILED: {name}")
            print(f"ERROR: {error}")
            print("=" * 100)
            traceback.print_exc()

    if failed_experiments:
        print(f"\n{len(failed_experiments)} experiment(s) failed: {', '.join(failed_experiments)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
