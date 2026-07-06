"""
NHANES data loading utilities for the SIADS 699 Capstone project.

This module loads selected NHANES 2017-March 2020 XPT files from a local
data folder and merges them into one participant-level dataframe using SEQN.

Expected local folder structure:

data/
├── Lab/
├── Questionnaire/
├── Demographic/
└── Examination/

Raw data files should be downloaded locally from the shared Google Drive folder.
They should not be committed to GitHub.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# File configuration
# ---------------------------------------------------------------------

LAB_FILES: Dict[str, str] = {
    "P_GHB.xpt": "Glycohemoglobin",
    "P_GLU.xpt": "Fasting Glucose",
    "P_HDL.xpt": "HDL Cholesterol",
    "P_HSCRP.xpt": "High-Sensitivity C-Reactive Protein",
    "P_INS.xpt": "Insulin",
    "P_TCHOL.xpt": "Total Cholesterol",
    "P_TRIGLY.xpt": "Triglycerides",
    "P_BIOPRO.xpt": "Standard Biochemistry Profile",
    "P_FASTQX.xpt": "Fasting Questionnaire",
}

QUESTIONNAIRE_FILES: Dict[str, str] = {
    "P_PAQ.xpt": "Physical Activity Questionnaire",
    "P_DBQ.xpt": "Dietary Behavior Questionnaire",
    "P_DIQ.xpt": "Diabetes Questionnaire",
    "P_BPQ.xpt": "Blood Pressure and Cholesterol Questionnaire",
    "P_SMQ.xpt": "Smoking Questionnaire",
    "P_SLQ.xpt": "Sleep Disorders Questionnaire",
    "P_MCQ.xpt": "Medical Conditions Questionnaire",
    "P_ALQ.xpt": "Alcohol Use Questionnaire",
    "P_DPQ.xpt": "Depression Screener",
}

DEMOGRAPHIC_FILES: Dict[str, str] = {
    "P_DEMO.xpt": "Demographics",
}

EXAMINATION_FILES: Dict[str, str] = {
    "P_BMX.xpt": "Body Measures",
    "P_BPXO.xpt": "Blood Pressure - Oscillometric Measurement",
}


FILE_GROUPS: Dict[str, Dict[str, str]] = {
    "Lab": LAB_FILES,
    "Questionnaire": QUESTIONNAIRE_FILES,
    "Demographic": DEMOGRAPHIC_FILES,
    "Examination": EXAMINATION_FILES,
}


# ---------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------

COLUMN_NAMES: Dict[str, str] = {
    # Identifiers / survey design
    "SEQN": "ID",
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",
    "SDDSRVYR": "Survey Cycle",
    "WTSAFPRP": "Fasting Weight",
    "WTMECPRP": "MEC Weight",
    "SDMVPSU": "PSU",
    "SDMVSTRA": "Stratum",

    # Physical activity
    "PAQ605": "Vigorous Work Activity",
    "PAQ610": "Vigorous Work Days",
    "PAD615": "Vigorous Work Minutes",
    "PAQ620": "Moderate Work Activity",
    "PAQ625": "Moderate Work Days",
    "PAD630": "Moderate Work Minutes",
    "PAQ635": "Walk or Bicycle for Transportation",
    "PAQ640": "Walk/Bicycle Days",
    "PAD645": "Walk/Bicycle Minutes",
    "PAQ650": "Vigorous Recreational Activity",
    "PAQ655": "Vigorous Recreational Days",
    "PAD660": "Vigorous Recreational Minutes",
    "PAQ665": "Moderate Recreational Activity",
    "PAQ670": "Moderate Recreational Days",
    "PAD675": "Moderate Recreational Minutes",
    "PAD680": "Sedentary Minutes",

    # Core biomarkers
    "LBXGH": "Glycohemoglobin (%)",
    "LBXGLU": "Fasting Glucose (mg/dL)",
    "LBXIN": "Insulin (µU/mL)",
    "LBDHDD": "HDL Cholesterol (mg/dL)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBXTR": "Triglycerides (mg/dL)",
    "LBXHSCRP": "HS C-Reactive Protein (mg/L)",

    # Extended biomarkers
    "LBXSUA": "Uric Acid (mg/dL)",
    "LBXSATSI": "ALT (U/L)",
    "LBXSGTSI": "GGT (U/L)",

    # Body measures
    "BMXBMI": "BMI (kg/m^2)",
    "BMXWAIST": "Waist Circumference (cm)",

    # Blood pressure
    "BPXOSY1": "Systolic BP 1 (mmHg)",
    "BPXOSY2": "Systolic BP 2 (mmHg)",
    "BPXOSY3": "Systolic BP 3 (mmHg)",
    "BPXODI1": "Diastolic BP 1 (mmHg)",
    "BPXODI2": "Diastolic BP 2 (mmHg)",
    "BPXODI3": "Diastolic BP 3 (mmHg)",

    # Sleep
    "SLQ300": "Weekday Sleep Time",
    "SLQ310": "Weekday Wake Time",
    "SLD012": "Weekday Sleep Hours",
    "SLQ320": "Weekend Sleep Time",
    "SLQ330": "Weekend Wake Time",
    "SLD013": "Weekend Sleep Hours",
    "SLQ030": "Snoring Frequency",
    "SLQ040": "Stop Breathing During Sleep Frequency",
    "SLQ050": "Doctor Told Trouble Sleeping",
    "SLQ120": "Daytime Sleepiness Frequency",

    # Depression screener items
    "DPQ010": "Little Interest",
    "DPQ020": "Feeling Down",
    "DPQ030": "Sleep Problems",
    "DPQ040": "Feeling Tired",
    "DPQ050": "Appetite Problems",
    "DPQ060": "Feeling Bad About Self",
    "DPQ070": "Trouble Concentrating",
    "DPQ080": "Moving or Speaking Slowly",
    "DPQ090": "Thoughts Better Off Dead",
    "DPQ100": "Difficulty From Symptoms",
}


# ---------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------

def get_category_paths(data_dir: str | Path = "data") -> Dict[str, Path]:
    """
    Return expected NHANES category folders under the local data directory.
    """
    data_dir = Path(data_dir)

    category_paths = {
        "Lab": data_dir / "Lab",
        "Questionnaire": data_dir / "Questionnaire",
        "Demographic": data_dir / "Demographic",
        "Examination": data_dir / "Examination",
    }

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Could not find data folder at: {data_dir.resolve()}\n"
            "Download the shared NHANES data from Google Drive and place it in the local data/ folder."
        )

    missing_folders = [
        category for category, path in category_paths.items()
        if not path.exists()
    ]

    if missing_folders:
        raise FileNotFoundError(
            f"Missing expected data subfolders: {missing_folders}\n"
            "Expected: data/Lab, data/Questionnaire, data/Demographic, data/Examination"
        )

    return category_paths


def read_xpt_file(folder_path: str | Path, filename: str) -> pd.DataFrame:
    """
    Read one NHANES XPT file into a pandas dataframe.
    """
    file_path = Path(folder_path) / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    return pd.read_sas(file_path, format="xport")


def load_selected_files(
    folder_path: str | Path,
    file_descriptions: Dict[str, str],
    category_name: str,
    verbose: bool = True,
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Load selected NHANES files for one category.

    Returns:
        dataframes: list of loaded dataframes
        missing_files: list of expected files not found
    """
    dataframes = []
    missing_files = []

    for index, (filename, friendly_name) in enumerate(file_descriptions.items(), start=1):
        file_path = Path(folder_path) / filename

        if not file_path.exists():
            missing_files.append(filename)
            continue

        df = read_xpt_file(folder_path, filename)
        dataframes.append(df)

        if verbose:
            print(f"{category_name} DataFrame {index}: {friendly_name} ({filename}) {df.shape}")

    if missing_files and verbose:
        print(f"Missing {category_name} files: {missing_files}")

    return dataframes, missing_files


def load_all_selected_files(
    data_dir: str | Path = "data",
    verbose: bool = True,
) -> Tuple[Dict[str, List[pd.DataFrame]], Dict[str, List[str]]]:
    """
    Load all selected NHANES files grouped by category.
    """
    category_paths = get_category_paths(data_dir)

    loaded_data = {}
    missing_files = {}

    for category_name, file_descriptions in FILE_GROUPS.items():
        dataframes, missing = load_selected_files(
            folder_path=category_paths[category_name],
            file_descriptions=file_descriptions,
            category_name=category_name,
            verbose=verbose,
        )
        loaded_data[category_name] = dataframes
        missing_files[category_name] = missing

    return loaded_data, missing_files


def merge_on_participant_id(
    dataframes: List[pd.DataFrame],
    id_col: str = "SEQN",
    how: str = "outer",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge NHANES dataframes on participant identifier.

    Duplicate non-ID columns from later files are dropped before merging.
    """
    dataframes = [df for df in dataframes if id_col in df.columns]

    if verbose:
        print(f"DataFrames with {id_col}: {len(dataframes)}")

    if not dataframes:
        raise ValueError(f"No loaded dataframes contain {id_col}; cannot merge.")

    merged_df = dataframes[0].copy()

    for df in dataframes[1:]:
        overlap_cols = merged_df.columns.intersection(df.columns).drop(id_col, errors="ignore")
        merged_df = merged_df.merge(
            df.drop(columns=overlap_cols),
            on=id_col,
            how=how,
        )

    return merged_df


def load_master_dataframe(
    data_dir: str | Path = "data",
    rename_columns: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and merge selected NHANES files into one master dataframe.
    """
    loaded_data, missing_files = load_all_selected_files(
        data_dir=data_dir,
        verbose=verbose,
    )

    all_dataframes = (
        loaded_data["Demographic"]
        + loaded_data["Questionnaire"]
        + loaded_data["Lab"]
        + loaded_data["Examination"]
    )

    master_df = merge_on_participant_id(
        all_dataframes,
        id_col="SEQN",
        how="outer",
        verbose=verbose,
    )

    if rename_columns:
        master_df = master_df.rename(columns=COLUMN_NAMES)

    if verbose:
        print(f"Master dataframe shape: {master_df.shape}")

        total_missing = sum(len(files) for files in missing_files.values())
        if total_missing:
            print("Warning: some expected files were missing:")
            for category, files in missing_files.items():
                if files:
                    print(f"  {category}: {files}")

    return master_df