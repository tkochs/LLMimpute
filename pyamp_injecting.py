import pandas as pd
from pyampute.ampute import MultivariateAmputation
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

INPUT_PATH = Path(
    "/hri/storage/user/lmangam/test/"
    "maternal_health_risk/"
    "Maternal Health Risk Data Set.csv"
)

OUTPUT_DIR = Path(
    "/hri/storage/user/lmangam/test/"
    "maternal_health_risk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. EXPERIMENT SETTINGS
# ============================================================

# Seed 42 was already used.
# These are six additional fixed seeds.
SEEDS = [
    17,
    23,
    51,
    73,
    91,
    123
]

# Approximate proportion of ROWS that become incomplete.
# This does NOT mean 60% of all cells become missing.
PROP = 0.60


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)

print("\nOriginal missing values:")
print(df.isna().sum())


# ============================================================
# 4. ENCODE RISK LEVEL
# ============================================================
#
# PyAmpute calculates weighted scores numerically.
#
# Low Risk  -> 0
# Mid Risk  -> 1
# High Risk -> 2
#
# After missingness injection, the observed values are
# converted back to their original labels.
# ============================================================

risk_mapping = {
    "low risk": 0,
    "mid risk": 1,
    "high risk": 2,
}

reverse_risk_mapping = {
    0: "low risk",
    1: "mid risk",
    2: "high risk",
}

df["RiskLevel"] = (
    df["RiskLevel"]
    .str.strip()
    .str.lower()
    .map(risk_mapping)
)


# Make sure all categories were successfully converted
if df["RiskLevel"].isna().any():
    raise ValueError(
        "RiskLevel contains categories that could not be mapped."
    )


# ============================================================
# 5. DEFINE MNAR MISSINGNESS PATTERNS
# ============================================================
#
# Pattern 1 -> Age
#
# Pattern 2 -> SystolicBP + DiastolicBP
#
# Pattern 3 -> Age + SystolicBP + HeartRate
#
# Pattern 4 -> HeartRate + RiskLevel
#
# Pattern 5 -> RiskLevel
#
# Each pattern has frequency 0.20.
#
# Variables that become missing are also involved in their
# missingness score, producing MNAR missingness.
# ============================================================

patterns = [

    # ========================================================
    # PATTERN 1
    #
    # Missing:
    #   Age
    #
    # Age itself influences whether Age becomes missing.
    # Therefore this is self-masked MNAR.
    #
    # sigmoid-left:
    # Lower Age scores have higher missingness probability.
    # ========================================================

    {
        "incomplete_vars": [
            "Age"
        ],

        "mechanism": "MNAR",

        "weights": {
            "Age": 1
        },

        "score_to_probability_func": "sigmoid-left",

        "freq": 0.20
    },


    # ========================================================
    # PATTERN 2
    #
    # Missing:
    #   SystolicBP
    #   DiastolicBP
    #
    # Both variables participate in their missingness score.
    #
    # sigmoid-mid:
    # Middle values of the weighted score have higher
    # missingness probability.
    # ========================================================

    {
        "incomplete_vars": [
            "SystolicBP",
            "DiastolicBP"
        ],

        "mechanism": "MNAR",

        "weights": {
            "SystolicBP": 1,
            "DiastolicBP": 1
        },

        "score_to_probability_func": "sigmoid-mid",

        "freq": 0.20
    },


    # ========================================================
    # PATTERN 3
    #
    # Missing:
    #   Age
    #   SystolicBP
    #   HeartRate
    #
    # score =
    #   1*z(Age)
    # + 1*z(SystolicBP)
    # + 1*z(HeartRate)
    #
    # All three variables participate in the score and can
    # subsequently become missing.
    # ========================================================

    {
        "incomplete_vars": [
            "Age",
            "SystolicBP",
            "HeartRate"
        ],

        "mechanism": "MNAR",

        "weights": {
            "Age": 1,
            "SystolicBP": 1,
            "HeartRate": 1
        },

        "score_to_probability_func": "sigmoid-mid",

        "freq": 0.20
    },


    # ========================================================
    # PATTERN 4
    #
    # Missing:
    #   HeartRate
    #   RiskLevel
    #
    # Both variables influence their own missingness.
    #
    # RiskLevel encoding:
    #   Low  = 0
    #   Mid  = 1
    #   High = 2
    #
    # sigmoid-left gives lower combined scores higher
    # missingness probability.
    # ========================================================

    {
        "incomplete_vars": [
            "HeartRate",
            "RiskLevel"
        ],

        "mechanism": "MNAR",

        "weights": {
            "HeartRate": 1,
            "RiskLevel": 1
        },

        "score_to_probability_func": "sigmoid-left",

        "freq": 0.20
    },


    # ========================================================
    # PATTERN 5
    #
    # Missing:
    #   RiskLevel
    #
    # RiskLevel itself determines its probability of
    # becoming missing.
    #
    # Low  = 0
    # Mid  = 1
    # High = 2
    #
    # sigmoid-left makes lower RiskLevel scores more likely
    # to become missing.
    # ========================================================

    {
        "incomplete_vars": [
            "RiskLevel"
        ],

        "mechanism": "MNAR",

        "weights": {
            "RiskLevel": 1
        },

        "score_to_probability_func": "sigmoid-left",

        "freq": 0.20
    }
]


# ============================================================
# 6. RUN EXPERIMENT FOR EACH SEED
# ============================================================

for seed in SEEDS:

    print("\n")
    print("=" * 80)
    print(f"MNAR INJECTION — SEED {seed}")
    print("=" * 80)


    # ========================================================
    # 6.1 CREATE PYAMPUTE OBJECT
    # ========================================================

    amputer = MultivariateAmputation(
        prop=PROP,
        patterns=patterns,
        std=True,
        seed=seed
    )


    # ========================================================
    # 6.2 INJECT MNAR MISSING VALUES
    # ========================================================

    missing_df = amputer.fit_transform(
        df.copy()
    )


    # ========================================================
    # 6.3 CONVERT RISKLEVEL BACK TO ORIGINAL LABELS
    # ========================================================
    #
    # 0 -> low risk
    # 1 -> mid risk
    # 2 -> high risk
    #
    # NaN remains NaN.
    # ========================================================

    missing_df["RiskLevel"] = (
        missing_df["RiskLevel"]
        .map(reverse_risk_mapping)
    )


    # ========================================================
    # 6.4 MISSINGNESS PER COLUMN
    # ========================================================

    missing_counts = (
        missing_df
        .isna()
        .sum()
    )

    missing_percentages = (
        missing_df
        .isna()
        .mean()
        * 100
    )

    stats = pd.DataFrame({
        "Missing_Count": missing_counts,
        "Missing_Percentage": missing_percentages
    })


    print("\n" + "=" * 65)
    print(f"MISSINGNESS PER COLUMN — SEED {seed}")
    print("=" * 65)

    print(stats)


    # ========================================================
    # 6.5 OVERALL CELL-WISE MISSINGNESS
    # ========================================================

    total_cells = (
        missing_df.shape[0]
        * missing_df.shape[1]
    )

    total_missing = (
        missing_df
        .isna()
        .sum()
        .sum()
    )

    overall_missing_percentage = (
        total_missing
        / total_cells
    ) * 100


    print("\n" + "=" * 65)
    print(f"OVERALL MISSINGNESS — SEED {seed}")
    print("=" * 65)

    print(
        "Total rows:",
        len(missing_df)
    )

    print(
        "Total columns:",
        missing_df.shape[1]
    )

    print(
        "Total cells:",
        total_cells
    )

    print(
        "Total missing cells:",
        total_missing
    )

    print(
        f"Overall cell-wise missingness: "
        f"{overall_missing_percentage:.2f}%"
    )


    # ========================================================
    # 6.6 INCOMPLETE ROW STATISTICS
    # ========================================================

    incomplete_rows = (
        missing_df
        .isna()
        .any(axis=1)
        .sum()
    )

    incomplete_percentage = (
        incomplete_rows
        / len(missing_df)
    ) * 100


    print("\n" + "=" * 65)
    print(f"INCOMPLETE ROWS — SEED {seed}")
    print("=" * 65)

    print(
        "Incomplete rows:",
        incomplete_rows
    )

    print(
        f"Incomplete row percentage: "
        f"{incomplete_percentage:.2f}%"
    )


    # ========================================================
    # 6.7 SHOW EXAMPLE INCOMPLETE ROWS
    # ========================================================

    print("\n" + "=" * 65)
    print(f"EXAMPLE INCOMPLETE ROWS — SEED {seed}")
    print("=" * 65)

    print(
        missing_df[
            missing_df
            .isna()
            .any(axis=1)
        ].head(10)
    )


    # ========================================================
    # 6.8 CREATE OUTPUT PATH
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / f"Maternal_pyampute_MNAR_seed{seed}.csv"
    )


    # ========================================================
    # 6.9 SAVE DATASET
    # ========================================================

    missing_df.to_csv(
        output_path,
        index=False
    )


    print("\nDataset saved to:")
    print(output_path)


# ============================================================
# 7. FINISHED
# ============================================================

print("\n")
print("=" * 80)
print("ALL MNAR SEED EXPERIMENTS COMPLETED")
print("=" * 80)

print(
    "Seeds used:",
    SEEDS
)

print(
    "Number of new datasets:",
    len(SEEDS)
)

print("\nOutput directory:")
print(OUTPUT_DIR)