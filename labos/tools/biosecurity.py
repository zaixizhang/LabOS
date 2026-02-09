"""
LabOS Biosecurity Tools — biosafety data sanitisation.
"""

import json
import os
from typing import List

import pandas as pd
from smolagents import tool

LEVEL_KEYWORDS = {
    1: ["virus", "viral", "virion", "capsid"],
    2: ["SARS", "ebola", "bioweapon", "smallpox"],
    3: ["HIV", "COVID-19", "hepatitis B", "influenza"],
}


def _is_sensitive(text: str, keywords: List[str]) -> bool:
    if not isinstance(text, str):
        return False
    return any(w.lower() in text.lower() for w in keywords)


@tool
def sanitize_bio_dataset(
    file_path: str,
    output_dir: str = "./agent_outputs",
    fields: List[str] = None,
    level: int = 1,
) -> str:
    """Clean a CSV or JSON biological dataset by removing records containing sensitive virus-related keywords.

    Args:
        file_path: Path to the input file (CSV or JSON).
        output_dir: Directory to save the sanitised output file.
        fields: Fields to scan. If None, all text fields are scanned.
        level: Sensitivity level for filtering (1=Strict, 2=Moderate, 3=Loose).

    Returns:
        Path to the sanitised output file.
    """
    if level not in LEVEL_KEYWORDS:
        raise ValueError("Level must be 1, 2, or 3")

    keywords = LEVEL_KEYWORDS[level]
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[-1].lower()
    cleaned_path = os.path.join(output_dir, f"cleaned_{os.path.basename(file_path)}")

    if ext == ".csv":
        df = pd.read_csv(file_path)
        if fields is None:
            fields = df.select_dtypes(include="object").columns.tolist()
        mask = ~df[fields].apply(lambda col: col.apply(lambda x: _is_sensitive(x, keywords))).any(axis=1)
        df[mask].to_csv(cleaned_path, index=False)

    elif ext == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
        cleaned = []
        for record in data:
            check = fields or list(record.keys())
            combined = " ".join(str(record.get(f, "")) for f in check)
            if not _is_sensitive(combined, keywords):
                cleaned.append(record)
        with open(cleaned_path, "w") as f:
            json.dump(cleaned, f, indent=2)
    else:
        raise ValueError("Unsupported file format. Only .csv and .json are supported.")

    return cleaned_path
