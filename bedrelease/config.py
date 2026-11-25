"""Central settings for BedRelease. Every number the dashboard shows comes from here or from outputs/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "diabetic_data.csv"
IDS_CSV = ROOT / "data" / "raw" / "IDs_mapping.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

MIRROR_URL = ("https://raw.githubusercontent.com/lugooe/Diabetes/"
              "630243bb0ca527d7c4e059e4364215dfa2f27463/diabetic_data.csv")
EXPECTED_ROWS = 101766
EXPECTED_COLS = 50
RAW_SHA256 = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"

SEED = 42
MAX_LOS = 14                    # dataset only contains stays of 1..14 days
SPLIT = {"train": 0.70, "valid": 0.15, "test": 0.15}   # split by patient, never by encounter

# Discharge dispositions that mean the patient died or went to hospice (excluded from readmission cohort)
DEATH_HOSPICE = {11, 13, 14, 19, 20, 21}

# Features known when the patient is admitted (no stay-total counters, no in-stay lab/medication counts)
ADMISSION_FEATURES_CAT = ["race", "gender", "age", "admission_type", "admission_source", "payer_code", "ward"]
ADMISSION_FEATURES_NUM = ["number_outpatient", "number_emergency", "number_inpatient",
                          "prior_encounters", "prior_mean_los", "last_los"]   # last 3 = same patient's EARLIER encounters in the file
DIAG_FEATURE = "diag_group"     # assumes a working diagnosis is known on admission (ablation shown in the evaluation)
# Stay-total counters (they grow with the length of stay, so they leak the answer) -> used ONLY for the leakage test
LEAKY_NUM = ["num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]

# Decision-support defaults (assumptions, editable in the dashboard)
DEFAULT_WARD_BEDS = 30
DEFAULT_ER_ARRIVALS = 3.0       # expected emergency admissions needing a bed tomorrow (assumption)
DEFAULT_FREE_BEDS = 3           # empty beds right now (assumption)
DEFAULT_CONFIDENCE = 0.90
# colour bands for P(discharge in next 24h). Chosen from the test-set distribution (base rate ~23%, so 0.6 would never trigger):
# GREEN ~ top 20% of patient-days (validated alert threshold), YELLOW ~ operating threshold, RED below.
GREEN_P, YELLOW_P = 0.29, 0.21

ward_note = ("Ward = admitting-physician specialty group from the data (a proxy; the file has no ward or bed ids, "
             "no dates and no hospital census).")
