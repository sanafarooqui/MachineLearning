"""Data loading and cleaning for the hospital readmission project.

Encapsulates the cleaning/feature-engineering decisions made during EDA on
``diabetic_data.csv`` (see notebooks/01_eda.ipynb for the rationale behind
each step).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolved relative to this file (src/data_prep.py -> repo root / data / raw / ...)
# so that load_data() works regardless of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"

ID_COLUMNS = ["encounter_id", "patient_nbr"]

# Medication columns where >99% of values are a single category — provide
# little signal and were dropped after EDA (see 01_eda.ipynb, "informative_meds").
LOW_INFORMATION_COLUMNS = [
    "weight", "glyburide", "glimepiride", "nateglinide", "chlorpropamide",
    "acetohexamide", "tolbutamide", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped",
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab fac including rehab units of a hospital",
    23: "Discharged/transferred to a long term care hospital",
    24: "Discharged/transferred to a nursing facility certified under Medicaid but not Medicare",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred/referred to a psychiatric hospital",
    29: "Discharged/transferred to a Critical Access Hospital (CAH)",
    30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
}

ADMISSION_SOURCE_MAP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critial access hospital", 11: "Normal Delivery",
    12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
    15: "Not Available", 17: "NULL",
    18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency", 20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital inpt/same fac reslt in a sep claim",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}

ID_MAPPINGS = {
    "admission_type_id": ADMISSION_TYPE_MAP,
    "discharge_disposition_id": DISCHARGE_DISPOSITION_MAP,
    "admission_source_id": ADMISSION_SOURCE_MAP,
}

# Values that represent missingness once the ID columns are mapped to labels.
NULL_LABELS = {"NULL", "Not Available", "Not Mapped", "Unknown/Invalid"}

# Patients who died cannot be readmitted — keeping them would bias the
# negative class.
EXPIRED_DISCHARGE_LABELS = [
    "Expired",
    "Expired at home. Medicaid only, hospice.",
    "Expired in a medical facility. Medicaid only, hospice.",
    "Expired, place unknown. Medicaid only, hospice.",
]

TARGET_COLUMN = "readmitted"
BINARY_TARGET_COLUMN = "readmitted_30"


def load_data(path: "str | Path" = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw diabetic encounters dataset."""
    return pd.read_csv(path)


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop encounter/patient identifier columns that carry no predictive signal."""
    return df.drop(columns=ID_COLUMNS)


def map_icd9_to_category(code) -> str:
    """Group an ICD-9 diagnosis code into a clinical category.

    diag_1/2/3 have 700+ distinct codes — grouping into ranges per
    https://www.aapc.com/codes/icd9-codes-range/ keeps cardinality manageable.
    """
    if pd.isnull(code):
        return "Unknown"

    code = str(code).strip()

    if code.startswith("V"):
        return "Supplementary"
    if code.startswith("E"):
        return "External Causes"

    try:
        code_num = float(code)
    except ValueError:
        return "Unknown"

    if 390 <= code_num <= 459:
        return "Circulatory"
    elif 460 <= code_num <= 519:
        return "Respiratory"
    elif 520 <= code_num <= 579:
        return "Digestive"
    elif code_num == 250:
        return "Diabetes"
    elif 800 <= code_num <= 999:
        return "Injury"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 580 <= code_num <= 629:
        return "Genitourinary"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    elif 1 <= code_num <= 139:
        return "Infectious"
    elif 240 <= code_num <= 249 or 251 <= code_num <= 279:
        return "Endocrine/Metabolic"
    elif 280 <= code_num <= 289:
        return "Blood"
    elif 290 <= code_num <= 319:
        return "Mental"
    elif 320 <= code_num <= 389:
        return "Nervous System"
    elif 630 <= code_num <= 679:
        return "Pregnancy"
    elif 680 <= code_num <= 709 or code_num == 782:
        return "Skin"
    elif 740 <= code_num <= 759:
        return "Congenital"
    elif 760 <= code_num <= 779:
        return "Perinatal"
    elif 780 <= code_num <= 799:
        return "Symptoms/Ill-defined"
    else:
        return "Other"


def map_medical_specialty(spec) -> str:
    """Group the ~70 raw medical_specialty values into broader clinical groups."""
    if pd.isnull(spec):
        return "Unknown"

    spec = str(spec).strip()

    if "Intern" in spec or "Family" in spec or "General" in spec:
        return "General Practice"
    elif "Cardio" in spec:
        return "Cardiology"
    elif "Surg" in spec:
        return "Surgery"
    elif "Endocrin" in spec or "Metabolism" in spec:
        return "Endocrinology"
    elif "Nephro" in spec:
        return "Nephrology"
    elif "Neuro" in spec:
        return "Neurology"
    elif "Gastro" in spec or "Hepato" in spec:
        return "Gastroenterology"
    elif "Pulmo" in spec or "Thoracic" in spec or "Respiratory" in spec:
        return "Pulmonology"
    elif "Ortho" in spec:
        return "Orthopedics"
    elif "Onco" in spec or "Radiat" in spec:
        return "Oncology"
    elif "Psych" in spec:
        return "Psychiatry"
    elif "Obst" in spec or "Gynec" in spec:
        return "OB/GYN"
    elif "Pedia" in spec:
        return "Pediatrics"
    elif "Urolog" in spec:
        return "Urology"
    elif "Ophthal" in spec:
        return "Ophthalmology"
    elif "Hematol" in spec:
        return "Hematology"
    elif "Infect" in spec:
        return "Infectious Disease"
    elif "Emergency" in spec or "Trauma" in spec:
        return "Emergency"
    elif "Rehab" in spec or "Physical" in spec:
        return "Rehabilitation"
    elif "Anesthesio" in spec:
        return "Anesthesiology"
    elif "Radiol" in spec or "Diagnostic" in spec:
        return "Radiology"
    else:
        return "Other"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning / feature-engineering pipeline discovered in EDA.

    Expects ``df`` to already have identifier columns dropped (see
    :func:`drop_identifier_columns`). Steps, in order:

    1. Drop low-information medication columns.
    2. Map admission/discharge/source IDs to readable labels, and recode
       their "unknown" labels to NaN.
    3. Drop encounters where the patient was discharged as "Expired" — they
       cannot be readmitted, so keeping them would bias the negative class.
    4. Recode literal ``'?'`` placeholders to NaN.
    5. Group ICD-9 diagnosis codes and medical specialties into broader
       categories to control cardinality.
    """
    df = df.drop(columns=LOW_INFORMATION_COLUMNS)

    for col, mapping in ID_MAPPINGS.items():
        df[col] = df[col].map(mapping)
    for col in ID_MAPPINGS:
        df[col] = df[col].replace(NULL_LABELS, np.nan)

    df = df[~df["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_LABELS)].reset_index(drop=True)

    df = df.replace("?", np.nan)

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(map_icd9_to_category)

    df["medical_specialty"] = df["medical_specialty"].apply(map_medical_specialty)

    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 3-class ``readmitted`` target into a binary 30-day target.

    We only care about readmissions within 30 days, so 'NO' and '>30' are
    combined into class 0, and '<30' becomes class 1.
    """
    df = df.copy()
    df[BINARY_TARGET_COLUMN] = (df[TARGET_COLUMN] == "<30").astype(int)
    return df


def load_and_clean(path: "str | Path" = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load raw data and run the full cleaning pipeline."""
    df = load_data(path)
    df = drop_identifier_columns(df)
    df = clean_data(df)
    df = add_binary_target(df)
    return df
