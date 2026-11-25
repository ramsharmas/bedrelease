"""Load, verify and clean the UCI Diabetes 130-US hospitals file (101,766 inpatient encounters)."""
from __future__ import annotations
import hashlib, json, urllib.request
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C


class DataError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(force: bool = False) -> Path:
    """Fetch the pinned GitHub mirror; refuse the file if size/hash do not match."""
    C.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if C.RAW_CSV.exists() and not force:
        return C.RAW_CSV
    tmp = C.RAW_CSV.with_suffix(".part")
    urllib.request.urlretrieve(C.MIRROR_URL, tmp)
    if sha256_of(tmp) != C.RAW_SHA256:
        tmp.unlink(missing_ok=True)
        raise DataError("Downloaded file hash does not match the verified copy; refusing to use it.")
    tmp.replace(C.RAW_CSV)
    return C.RAW_CSV


ADMISSION_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 5: "Not available", 6: "Not available",
                  7: "Trauma center", 8: "Not available"}


def _admission_source(i: int) -> str:
    if i in (1, 2, 3):
        return "Physician / clinic referral"
    if i == 7:
        return "Emergency room"
    if i in (4, 5, 6, 10, 18, 22, 25, 26):
        return "Transfer from another facility"
    return "Other / not available"


WARD_MAP = {  # admitting-physician specialty -> ward group (proxy)
    "Cardiology": "Cardiology", "Surgery-Cardiovascular": "Surgery", "Surgery-Cardiovascular/Thoracic": "Surgery",
    "Surgery-General": "Surgery", "Surgery-Neuro": "Surgery", "Surgery-Thoracic": "Surgery", "Surgery-Vascular": "Surgery",
    "Surgery-Plastic": "Surgery", "Surgeon": "Surgery", "Orthopedics": "Orthopedics", "Orthopedics-Reconstructive": "Orthopedics",
    "InternalMedicine": "Internal Medicine", "Family/GeneralPractice": "Family / General", "Emergency/Trauma": "Emergency / Trauma",
    "Nephrology": "Internal Medicine", "Pulmonology": "Internal Medicine", "Gastroenterology": "Internal Medicine",
    "Hematology/Oncology": "Internal Medicine", "Endocrinology": "Internal Medicine", "Neurology": "Internal Medicine",
    "ObstetricsandGynecology": "Women's health", "Obstetrics": "Women's health", "Gynecology": "Women's health",
    "Pediatrics": "Pediatrics", "Psychiatry": "Psychiatry", "Urology": "Surgery",
}


def disposition_group(i: int) -> str:
    if i == 1:
        return "Home"
    if i in (6, 8):
        return "Home with home-health"
    if i in (2, 3, 4, 5, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30):
        return "Another facility / SNF"
    if i == 7:
        return "Left against advice"
    if i in C.DEATH_HOSPICE:
        return "Died / hospice"
    return "Other / unknown"


def diag_group(code) -> str:
    """ICD-9 primary diagnosis -> broad group (grouping follows Strack et al. 2014, Table 2)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Unknown"
    s = str(code)
    if s[0] in "VE":
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if 250 <= v < 251:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


@dataclass
class Clean:
    df: pd.DataFrame
    report: dict


def load_clean(path: Path | None = None) -> Clean:
    path = Path(path or C.RAW_CSV)
    if not path.exists():
        raise DataError(f"Raw data not found at {path}. Run: python run_pipeline.py --download")
    raw = pd.read_csv(path, na_values="?", low_memory=False)
    rep: dict = {"raw_rows": int(len(raw)), "raw_cols": int(raw.shape[1]), "sha256": sha256_of(path)}
    if raw.shape != (C.EXPECTED_ROWS, C.EXPECTED_COLS):
        raise DataError(f"Expected {C.EXPECTED_ROWS} x {C.EXPECTED_COLS}, got {raw.shape}.")
    rep["duplicate_encounters"] = int(raw["encounter_id"].duplicated().sum())
    rep["patients"] = int(raw["patient_nbr"].nunique())
    rep["los_min"], rep["los_max"] = int(raw["time_in_hospital"].min()), int(raw["time_in_hospital"].max())
    rep["missing_share_pct"] = {c: round(float(raw[c].isna().mean() * 100), 1)
                                for c in ["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code", "race"]}
    df = raw.drop_duplicates("encounter_id").copy()
    df["los"] = df["time_in_hospital"].astype(int)
    df["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE).fillna("Not available")
    df["admission_source"] = df["admission_source_id"].map(_admission_source)
    df["ward"] = df["medical_specialty"].map(WARD_MAP)
    df["ward"] = df["ward"].fillna(pd.Series(np.where(df["medical_specialty"].isna(), "Not recorded", "Other specialty"),
                                             index=df.index))
    df["diag_group"] = df["diag_1"].map(diag_group)
    df["disposition"] = df["discharge_disposition_id"].astype(int).map(disposition_group)
    for c in ("race", "payer_code", "gender"):
        df[c] = df[c].fillna("Unknown")
    df["gender"] = df["gender"].replace("Unknown/Invalid", "Unknown")
    # Same-patient history from EARLIER encounters only. encounter_id order is chronological within a patient:
    # only 0.9% of 'NO readmission' encounters have a later encounter in the file, versus 72% of '<30' ones.
    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    g = df.groupby("patient_nbr")
    df["prior_encounters"] = g.cumcount()
    prior_sum = g["los"].cumsum() - df["los"]
    df["prior_mean_los"] = (prior_sum / df["prior_encounters"].replace(0, np.nan))
    df["last_los"] = g["los"].shift(1)
    df["readmit30"] = (df["readmitted"] == "<30").astype(int)
    df["died_or_hospice"] = df["discharge_disposition_id"].isin(C.DEATH_HOSPICE)
    rep["rows_after_cleaning"] = int(len(df))
    rep["readmit30_rate_pct"] = round(float(df["readmit30"].mean() * 100), 2)
    rep["died_or_hospice_rows"] = int(df["died_or_hospice"].sum())
    rep["ward_counts"] = {k: int(v) for k, v in df["ward"].value_counts().items()}
    return Clean(df.reset_index(drop=True), rep)


if __name__ == "__main__":
    c = load_clean()
    print(json.dumps(c.report, indent=2))
