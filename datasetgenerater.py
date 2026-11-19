import numpy as np
import pandas as pd
from pathlib import Path

N_ROWS = 1000
RANDOM_SEED = 42

OUTPUT_PATH = ("/hri/storage/user/lmangam/test/"
"synthetic_pension_dataset.csv")

rng = np.random.default_rng(RANDOM_SEED)

# 1. AGE
# Dataset population:
# Age 24 to 120
# Generation distribution:
# 24-59  -> 30%
# 60-64  -> 14%
# 65-69  -> 14%
# 70-74  -> 14%
# 75-79  -> 14%
# 80-120 -> 14%

age_groups = rng.choice([
        "24-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80-120",
    ],
    size=N_ROWS,
    p=[
        0.30,
        0.14,
        0.14,
        0.14,
        0.14,
        0.14,
    ],)

ages = np.empty(N_ROWS, dtype=int)#afterwords we gonna fill positions

for i, group in enumerate(age_groups):

    if group == "24-59":
        ages[i] = rng.integers(24, 60)
    elif group == "60-64":
        ages[i] = rng.integers(60, 65)
    elif group == "65-69":
        ages[i] = rng.integers(65, 70)
    elif group == "70-74":
        ages[i] = rng.integers(70, 75)
    elif group == "75-79":
        ages[i] = rng.integers(75, 80)
    else:
        ages[i] = rng.integers(80, 121)


# 2. EDUCATION LEVEL
# Primary   -> 34%
# Secondary -> 33%
# Degree    -> 33%


education_levels = rng.choice([
        "Primary",
        "Secondary",
        "Degree",
    ],
    size=N_ROWS,
    p=[
        0.34,
        0.33,
        0.33,
    ],)

# 3. EMPLOYMENT TYPE
# Exact relationship:
# Primary   <-> Blue Collar
# Secondary <-> White Collar
# Degree    <-> Government
#
# EmploymentType is therefore derived from EducationLevel.

education_to_employment = {
    "Primary": "Blue Collar",
    "Secondary": "White Collar",
    "Degree": "Government",}

employment_types = np.array(
    [
        education_to_employment[education]
        for education in education_levels
    ]
)


# 4. ANNUAL PERSONAL INCOME
# Exact permitted ranges:
# Blue Collar:
#     10000 - 15000
# White Collar:
#     40000 - 49999
# Government:
#     50000 - 100000
# The ranges DO NOT overlap.


annual_personal_incomes = np.empty(N_ROWS,dtype=int,)

for i, employment in enumerate(employment_types):
    if employment == "Blue Collar":
        annual_personal_incomes[i] = rng.integers(10000,15001,)       # inclusive maximum = 15000
       
    elif employment == "White Collar":
        annual_personal_incomes[i] = rng.integers(40000,50000,)  # inclusive maximum = 49999
    elif employment == "Government":
        annual_personal_incomes[i] = rng.integers(50000,100001,)# inclusive maximum = 100000


# 5. DISABILITY STATUS
# Experimental distribution:
# Yes -> 30%
# No  -> 70%
# Disability is generated independently of:
# Age
# EducationLevel
# EmploymentType
# AnnualPersonalIncome

has_disability = rng.choice(
    [
        "Yes",
        "No",
    ],
    size=N_ROWS,
    p=[
        0.30,
        0.70,
    ],
)


# 6. RETIREMENT STATUS

# Retirement depends on Age.
# Age < 60:
#     Retired = No
# Age 60-64:
#     70% Yes / 30% No
# Age 65-69:
#     80% Yes / 20% No
#     90% Yes / 10% No
# Age 75-79:
#     95% Yes / 5% No
# Age >= 80:
#     Retired = Yes
# These probabilities are DATA GENERATION settings.


retirement_status = np.empty(N_ROWS,dtype=object,)

for i, age in enumerate(ages):
    if age < 60:
        retirement_status[i] = "No"
    elif age <= 64:
        retirement_status[i] = rng.choice(["Yes", "No"],p=[0.70, 0.30],)
    elif age <= 69:
        retirement_status[i] = rng.choice(["Yes", "No"],p=[0.80, 0.20],)
    elif age <= 74:
        retirement_status[i] = rng.choice(["Yes", "No"],p=[0.90, 0.10],)
    elif age <= 79:
        retirement_status[i] = rng.choice(["Yes", "No"],p=[0.95, 0.05],)
    else:
        retirement_status[i] = "Yes"


# 7. FAMILY MEMBER RECEIVING PENSION
# Yes -> 30%
# No  -> 70%

# Meaning:
# Yes = another member of the family is already
#       receiving pension.
# No = no other family member is currently
#      receiving pension.
# IMPORTANT:
# Only one person in a family can receive pension.
# FamilyMemberReceivingPension = Yes             ↓
# PensionEligible = No
# This rule also applies when HasDisability = Yes.


family_member_receiving_pension = rng.choice(
    [
        "Yes",
        "No",
    ],
    size=N_ROWS,
    p=[
        0.30,
        0.70,
    ],
)



# 8. PENSION ELIGIBILITY
# PensionEligible is the TARGET variable.
# It is NOT randomly generated.
# RULE PRIORITY:
# RULE 1
# FamilyMemberReceivingPension = Yes
# -> PensionEligible = No
# RULE 2
# If no family member receives pension:
# HasDisability = Yes
# -> PensionEligible = Yes
# Disability overrides:
# - retirement requirement
# - income requirement

# For non-disabled people:
# Retired must be Yes.

# For non-disabled retired people:
# AnnualPersonalIncome must be > 40000.
# RULE 5
# If all normal requirements are satisfied:
# PensionEligible = Yes



def determine_pension_eligibility(family_member_receiving_pension,has_disability,retired,annual_personal_income,):

   
    if family_member_receiving_pension == "Yes":
        return "No"
    if has_disability == "Yes":
        return "Yes"
    if retired == "No":
        return "No"

    if annual_personal_income <= 40000:
        return "No"

    return "Yes"



pension_eligibility = np.array(
    [
        determine_pension_eligibility(
            family_member_receiving_pension=
                family_member_receiving_pension[i],

            has_disability=
                has_disability[i],

            retired=
                retirement_status[i],

            annual_personal_income=
                annual_personal_incomes[i],
        )
        for i in range(N_ROWS)
    ]
)

# 10. CREATE FINAL DATASET


df = pd.DataFrame(
    {
        "Age": ages,

        "EducationLevel":
            education_levels,

        "EmploymentType":
            employment_types,

        "AnnualPersonalIncome":
            annual_personal_incomes,

        "HasDisability":
            has_disability,

        "Retired":
            retirement_status,

        "FamilyMemberReceivingPension":
            family_member_receiving_pension,

        "PensionEligible":
            pension_eligibility,
    }
)



# 11. SHUFFLE FINAL DATASET
df = df.sample(frac=1,random_state=RANDOM_SEED,).reset_index(drop=True)
# 12. LOGICAL VALIDATION
print("\nRunning logical validation...")
assert df["Age"].between(24, 120).all()
assert (df.loc[df["EducationLevel"] == "Primary","EmploymentType",]== "Blue Collar").all()
assert (df.loc[df["EducationLevel"] == "Secondary","EmploymentType",]== "White Collar").all()
assert (df.loc[df["EducationLevel"] == "Degree","EmploymentType",]== "Government").all()
blue_collar_mask = (df["EmploymentType"] == "Blue Collar")
assert df.loc[blue_collar_mask,"AnnualPersonalIncome",].between(10000,15000,).all()
white_collar_mask = (df["EmploymentType"] == "White Collar")
assert df.loc[white_collar_mask,"AnnualPersonalIncome",].between(40000,49999,).all()
government_mask = (df["EmploymentType"] == "Government")
assert df.loc[government_mask,"AnnualPersonalIncome",].between(50000,100000,).all()
assert (df.loc[df["AnnualPersonalIncome"].between(10000, 15000),"EmploymentType",]== "Blue Collar").all()
assert (df.loc[df["AnnualPersonalIncome"].between(40000, 49999),"EmploymentType",]== "White Collar").all()
assert (df.loc[df["AnnualPersonalIncome"].between(50000, 100000),"EmploymentType",]== "Government").all()
assert (df.loc[df["Age"] < 60,"Retired",]== "No").all()
assert (df.loc[df["Age"] >= 80,"Retired",]== "Yes").all()
assert (df.loc[df["FamilyMemberReceivingPension"] == "Yes","PensionEligible",]== "No").all()
disabled_eligible_mask = ((df["HasDisability"] == "Yes")&(df["FamilyMemberReceivingPension"]== "No"))
assert (df.loc[disabled_eligible_mask,"PensionEligible",]== "Yes").all()
not_retired_mask = ((df["HasDisability"] == "No") &(df["Retired"] == "No"))
assert (df.loc[not_retired_mask,"PensionEligible",]== "No").all()
low_income_mask = ((df["HasDisability"] == "No")&(df["Retired"] == "Yes")&(df["AnnualPersonalIncome"] <= 40000))
assert (df.loc[low_income_mask,"PensionEligible",]== "No").all()
normal_eligible_mask = ((df["HasDisability"] == "No")&(df["Retired"] == "Yes")&(df["AnnualPersonalIncome"] > 40000)&
    (
        df["FamilyMemberReceivingPension"]
        == "No"
    )
)

assert (
    df.loc[
        normal_eligible_mask,
        "PensionEligible",
    ]
    == "Yes"
).all()
print("All logical validation checks passed.")

output_path = Path(OUTPUT_PATH)

output_path.parent.mkdir(parents=True,exist_ok=True,)

df.to_csv(output_path,index=False,)

