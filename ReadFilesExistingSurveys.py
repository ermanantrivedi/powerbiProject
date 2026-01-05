#!/usr/bin/env python3
"""
Multi-Survey Content Quality Script (with Skip List)
Contains just TEXT and CONTENTQUALITY fields

- Iterates over multiple survey folders
- Skips specified surveys (comma-separated)
- Outputs one CSV per survey folder
"""

import re
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from docx import Document
from odf.opendocument import load
from odf import teletype

# ----------------------------
# ENV PATH
# ----------------------------
ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

env_path = Path(ENV_PATH)
if not env_path.exists():
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("", encoding="utf-8")

load_dotenv(env_path)

# ----------------------------
# PROJECT ROOT & FOLDERS
# ----------------------------
PROJECT_ROOT = env_path.parent

BASE_DOC_FOLDER = (
    PROJECT_ROOT
    / "qualtrics_downloaded_files"
    / "downloads"
)

OUTPUT_DIR = PROJECT_ROOT / "Clustered_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# SURVEYS TO SKIP (comma-separated)
# ----------------------------
SKIP_SURVEYS = "SV_0p7HfPlkL4Qu4g6, SV_3V0SgOjtT6ZqTxI"

SKIP_SET = {s.strip() for s in SKIP_SURVEYS.split(",") if s.strip()}

# ----------------------------
# MODEL LOAD
# ----------------------------
from models.nlp_models import ModelHub

MODEL_NAME = os.getenv("MODEL_NAME", "").lower()
model = ModelHub.load(MODEL_NAME)

# ----------------------------
# HELPERS
# ----------------------------
def extract_text(path: Path) -> str:
    try:
        suffix = path.suffix.lower()
    except Exception:
        return ""

    if suffix == ".docx":
        try:
            doc = Document(path)
            return "\n".join(
                p.text.strip() for p in doc.paragraphs if p.text.strip()
            )
        except Exception:
            return ""

    if suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    if suffix == ".odt":
        try:
            doc = load(str(path))
            return teletype.extractText(doc)
        except Exception:
            return ""

    return ""

def assess_content_quality(text, keywords):
    if not text or not text.strip():
        return "EMPTY"
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 10:
        return "LOW_CONTENT"
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    if unique_ratio < 0.3:
        return "GARBAGE"
    if not keywords:
        return "LOW_CONTENT"
    return "GOOD"

# ----------------------------
# MAIN LOOP
# ----------------------------
print(" Processing survey folders...")

for survey_folder in sorted(BASE_DOC_FOLDER.iterdir()):
    if not survey_folder.is_dir():
        continue

    survey_name = survey_folder.name

    #  SKIP LOGIC
    if survey_name in SKIP_SET:
        print(f"⏭ Skipping survey: {survey_name}")
        continue

    print(f"\n In-progress analysis of Survey: {survey_name}")

    records = []

    for p in sorted(survey_folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".docx", ".txt", ".odt"}:
            continue

        text = extract_text(p)
        keywords = model.extract_keywords(text)
        cq = assess_content_quality(text, keywords)

        records.append({
            "FileName": p.name,
            "Text": text,
            "ContentQuality": cq
        })

    if not records:
        print(f" No valid files in {survey_name}, skipping.")
        continue

    df = pd.DataFrame(records)

    output_csv = OUTPUT_DIR / f"{survey_name}_analysis_level1.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f" Saved: {output_csv}")

print("\n ALL SURVEYS PROCESSED")
