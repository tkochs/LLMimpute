"""
Amputation: turn a complete table into one with missing values, so that an imputer can be scored
against the values that were removed.

Where each piece comes from:
    MissingValueGenerator               <- inject.py            (swiss_cheese: MCAR / MAR / MNAR)
    pyampute_amputation                 <- pyamp_injecting.py   (pyampute MultivariateAmputation)
    MATERNAL_HEALTH_*                   <- pyamp_injecting.py   (encoding, patterns and seeds used for the maternal health data)
    missingness_report                  <- pyamp_injecting.py   (sections 6.4 - 6.7, the printed statistics)
    check_amputation                    <- pipeline.py          (prepare_dataset, the checks after amputation)

Comments tagged ORIGINAL: describe what the original code did; comments tagged CHANGED: note every
difference from it.

ORIGINAL: how pipeline.py (prepare_dataset) produced the table with missing values:
    1. load the dataset and drop every row that already had a missing value (a.dropna(axis=0)),
       so the remaining table is a complete ground truth.
    2. optionally sample 500 rows (sampling.enabled; bestseed.DatasetSampler picks the seed whose sample is
       closest to the full table in mean / std / category frequencies). Disabled in config27.
    3. make the missing values in one of two ways:
         missingness.enabled: true      -> MissingValueGenerator (swiss_cheese), during the run
         injected_dataset.enabled: true -> load a CSV that was amputated beforehand
       config27 used the second way (missingness.enabled: false) with the files energy_mcar_seed_*.csv.
       The script that wrote those files is not in the repo (pyamp_injecting.py only writes
       Maternal_pyampute_MNAR_seed*.csv), so the code does not show which method produced them.
       The per-experiment `missing_rate` values in config27 (0.824, 0.855, ...) were only copied into the
       result export; they were not used to amputate.
    4. check the result against the ground truth (check_amputation below).
CHANGED: steps 1 and 2 are data preparation and are not in this file.

Usage:
    X_missing = MissingValueGenerator(mechanism="MCAR", missing_rate=0.2, seed=1).inject(X)
    X_missing = pyampute_amputation(X, MATERNAL_HEALTH_PATTERNS, seed=17,
                                    category_mappings=MATERNAL_HEALTH_CATEGORY_MAPPINGS)
    check_amputation(X, X_missing)
"""

import pandas as pd
from swiss_cheese import MCAR, MAR, MNAR
from pyampute.ampute import MultivariateAmputation


# =============================================================================
#  swiss_cheese                                   <- inject.py
# =============================================================================

# ORIGINAL: read its settings from config["missingness"]. config27 values: mechanism "MNAR",
#           missing_rate 0.1219, seed 90, mean 0.75, variance null, mode "GM" (but missingness.enabled: false,
#           so config27 did not use this class).
# CHANGED:  the settings are arguments instead of the config dict (defaults = the config27 values);
#           the unused `import yaml` is dropped. The config key missingness.n_workers was never read and is
#           not an argument.
class MissingValueGenerator:

    def __init__(self, mechanism="MNAR", missing_rate=0.1219, mean=0.75, variance=None, mode="GM", seed=90):

        self.mechanism = mechanism    # missingness.mechanism: "MCAR" | "MAR" | "MNAR"
        self.alpha = missing_rate     # missingness.missing_rate: "determines missing percentage (common for 3 mechanisms)"

        self.mean = mean              # missingness.mean:     only used for MAR / MNAR
        self.variance = variance      # missingness.variance: only used for MAR / MNAR
        self.mode = mode              # missingness.mode:     "GM" | "min" | "max", only used for MAR / MNAR
        self.seed = seed              # missingness.seed

    def inject(self, df):

        # Create a copy so that the original dataset
        # remains unchanged for evaluation.
        df = df.copy()

        if self.mechanism == "MCAR":

            df_missing = MCAR(
                random_seed=self.seed
            )(df, self.alpha)

        elif self.mechanism == "MAR":

            df_missing = MAR(
                mean=self.mean,
                variance=self.variance,
                mode=self.mode,
                random_seed=self.seed
            )(df, self.alpha)

        elif self.mechanism == "MNAR":

            df_missing = MNAR(
                mean=self.mean,
                variance=self.variance,
                mode=self.mode,
                random_seed=self.seed
            )(df, self.alpha)

        else:

            raise ValueError(
                f"Unknown mechanism: {self.mechanism}"
            )

        return df_missing


# =============================================================================
#  pyampute                                       <- pyamp_injecting.py
# =============================================================================

# ORIGINAL: pyamp_injecting.py is a stand-alone script (pipeline.py never calls it). For the maternal health
#           data and each of six seeds it:
#             1. encodes RiskLevel as numbers (low/mid/high risk -> 0/1/2), because pyampute computes
#                weighted scores numerically,
#             2. runs MultivariateAmputation(prop=0.60, patterns=patterns, std=True, seed=seed),
#             3. converts RiskLevel back to its labels (NaN remains NaN),
#             4. prints statistics (see missingness_report) and
#             5. saves Maternal_pyampute_MNAR_seed{seed}.csv into a hard-coded /hri/storage/... folder.
# CHANGED:  one call = one seed, and the table is returned instead of saved; reading the input, the loop over
#           seeds, the statistics and writing the CSV are left to the caller.
#           The RiskLevel encoding is generalised to `category_mappings` ({column: {label: code}}); pass
#           MATERNAL_HEALTH_CATEGORY_MAPPINGS for the original behaviour. The reverse mapping is built from it
#           (the script wrote it out by hand; same result).
#           As in the script, labels are stripped and lower-cased before mapping, so the mapping keys must be
#           lower case and the returned column holds the lower-case labels.
#           The input is copied first, so the caller's DataFrame is not changed (the script changed
#           df["RiskLevel"] in place).
def pyampute_amputation(df, patterns, seed, prop=0.60, std=True, category_mappings=None):

    df = df.copy()
    category_mappings = category_mappings or {}

    # encode text columns as numbers
    for column, mapping in category_mappings.items():
        df[column] = (
            df[column]
            .str.strip()
            .str.lower()
            .map(mapping)
        )

        # Make sure all categories were successfully converted
        if df[column].isna().any():
            raise ValueError(
                f"{column} contains categories that could not be mapped."
            )

    # prop: approximate proportion of ROWS that become incomplete.
    # This does NOT mean that this share of all cells becomes missing.
    amputer = MultivariateAmputation(
        prop=prop,
        patterns=patterns,
        std=std,
        seed=seed
    )

    missing_df = amputer.fit_transform(
        df.copy()
    )

    # convert the encoded columns back to their labels; NaN remains NaN
    for column, mapping in category_mappings.items():
        reverse_mapping = {code: label for label, code in mapping.items()}
        missing_df[column] = (
            missing_df[column]
            .map(reverse_mapping)
        )

    return missing_df


# ORIGINAL: risk_mapping in pyamp_injecting.py (section 4).
MATERNAL_HEALTH_CATEGORY_MAPPINGS = {
    "RiskLevel": {
        "low risk": 0,
        "mid risk": 1,
        "high risk": 2,
    }
}

# ORIGINAL: SEEDS in pyamp_injecting.py. Seed 42 was already used; these are six additional fixed seeds.
MATERNAL_HEALTH_SEEDS = [17, 23, 51, 73, 91, 123]

# ORIGINAL: patterns in pyamp_injecting.py (section 5), unchanged.
#   Pattern 1 -> Age
#   Pattern 2 -> SystolicBP + DiastolicBP
#   Pattern 3 -> Age + SystolicBP + HeartRate
#   Pattern 4 -> HeartRate + RiskLevel
#   Pattern 5 -> RiskLevel
# Each pattern has frequency 0.20. Variables that become missing are also involved in their missingness
# score, producing MNAR missingness.
MATERNAL_HEALTH_PATTERNS = [

    # PATTERN 1 - missing: Age
    # Age itself influences whether Age becomes missing (self-masked MNAR).
    # sigmoid-left: lower Age scores have higher missingness probability.
    {
        "incomplete_vars": ["Age"],
        "mechanism": "MNAR",
        "weights": {"Age": 1},
        "score_to_probability_func": "sigmoid-left",
        "freq": 0.20
    },

    # PATTERN 2 - missing: SystolicBP, DiastolicBP
    # Both variables participate in their missingness score.
    # sigmoid-mid: middle values of the weighted score have higher missingness probability.
    {
        "incomplete_vars": ["SystolicBP", "DiastolicBP"],
        "mechanism": "MNAR",
        "weights": {"SystolicBP": 1, "DiastolicBP": 1},
        "score_to_probability_func": "sigmoid-mid",
        "freq": 0.20
    },

    # PATTERN 3 - missing: Age, SystolicBP, HeartRate
    # score = 1*z(Age) + 1*z(SystolicBP) + 1*z(HeartRate)
    # All three variables participate in the score and can subsequently become missing.
    {
        "incomplete_vars": ["Age", "SystolicBP", "HeartRate"],
        "mechanism": "MNAR",
        "weights": {"Age": 1, "SystolicBP": 1, "HeartRate": 1},
        "score_to_probability_func": "sigmoid-mid",
        "freq": 0.20
    },

    # PATTERN 4 - missing: HeartRate, RiskLevel
    # Both variables influence their own missingness (RiskLevel encoded low=0, mid=1, high=2).
    # sigmoid-left: lower combined scores have higher missingness probability.
    {
        "incomplete_vars": ["HeartRate", "RiskLevel"],
        "mechanism": "MNAR",
        "weights": {"HeartRate": 1, "RiskLevel": 1},
        "score_to_probability_func": "sigmoid-left",
        "freq": 0.20
    },

    # PATTERN 5 - missing: RiskLevel
    # RiskLevel itself determines its probability of becoming missing.
    # sigmoid-left: lower RiskLevel scores are more likely to become missing.
    {
        "incomplete_vars": ["RiskLevel"],
        "mechanism": "MNAR",
        "weights": {"RiskLevel": 1},
        "score_to_probability_func": "sigmoid-left",
        "freq": 0.20
    },
]


# =============================================================================
#  Statistics                                     <- pyamp_injecting.py (sections 6.4 - 6.7)
# =============================================================================

# ORIGINAL: printed inside the seed loop of pyamp_injecting.py after every amputation.
# CHANGED:  a function of its own (works for the output of either method); the printed text no longer
#           contains the seed, and the numbers are also returned.
def missingness_report(missing_df):

    # missingness per column
    missing_counts = missing_df.isna().sum()
    missing_percentages = missing_df.isna().mean() * 100

    stats = pd.DataFrame({
        "Missing_Count": missing_counts,
        "Missing_Percentage": missing_percentages
    })

    print("\n" + "=" * 65)
    print("MISSINGNESS PER COLUMN")
    print("=" * 65)
    print(stats)

    # overall cell-wise missingness
    total_cells = missing_df.shape[0] * missing_df.shape[1]
    total_missing = missing_df.isna().sum().sum()
    overall_missing_percentage = (total_missing / total_cells) * 100

    print("\n" + "=" * 65)
    print("OVERALL MISSINGNESS")
    print("=" * 65)
    print("Total rows:", len(missing_df))
    print("Total columns:", missing_df.shape[1])
    print("Total cells:", total_cells)
    print("Total missing cells:", total_missing)
    print(f"Overall cell-wise missingness: {overall_missing_percentage:.2f}%")

    # incomplete row statistics
    incomplete_rows = missing_df.isna().any(axis=1).sum()
    incomplete_percentage = (incomplete_rows / len(missing_df)) * 100

    print("\n" + "=" * 65)
    print("INCOMPLETE ROWS")
    print("=" * 65)
    print("Incomplete rows:", incomplete_rows)
    print(f"Incomplete row percentage: {incomplete_percentage:.2f}%")

    # example incomplete rows
    print("\n" + "=" * 65)
    print("EXAMPLE INCOMPLETE ROWS")
    print("=" * 65)
    print(missing_df[missing_df.isna().any(axis=1)].head(10))

    return {
        "per_column": stats,
        "total_missing_cells": total_missing,
        "cell_missing_percentage": overall_missing_percentage,
        "incomplete_rows": incomplete_rows,
        "incomplete_row_percentage": incomplete_percentage,
    }


# =============================================================================
#  Checks after amputation                        <- pipeline.py (prepare_dataset)
# =============================================================================

# ORIGINAL: run in prepare_dataset() on self.df (ground truth) and self.injected_df, for both ways of
#           getting the missing values. Raises if the amputed table has a different number of rows,
#           different columns (names or order), or if any value that is still observed differs from the
#           ground truth.
# CHANGED:  a function of its own that takes the two tables as arguments.
def check_amputation(original_df, injected_df):

    if len(original_df) != len(injected_df):  # validation check
        raise ValueError(f"Dataset validation failed: row count mismatch. "
                         f"Original dataset has {len(original_df)} rows, "
                         f"but injected dataset has {len(injected_df)} rows.")

    # Check number/names/order of features
    if list(original_df.columns) != list(injected_df.columns):
        raise ValueError(
            "Dataset validation failed: feature mismatch.\n"
            f"Original features: {list(original_df.columns)}\n"
            f"Injected features: {list(injected_df.columns)}")

    observed_mask = injected_df.notna()

    if not original_df.where(observed_mask).equals(injected_df.where(observed_mask)):
        raise ValueError(
            "Dataset validation failed: "
            "Some observed values in the injected dataset "
            "are different from the original dataset.")

    print("Dataset validation successful.")
    print(f"Rows: {len(original_df)}")
    print(f"Features: {len(original_df.columns)}")
