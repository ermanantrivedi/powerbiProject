#!/usr/bin/env python3
"""
Qualtrics ResponseId + File Tokens Exporter (X-API-TOKEN)
- Exports responses using export-responses API
- Creates two outputs:
    1. response_ids_with_files_<surveyId>.csv  → (responseId + fileTokens)
       → path saved in env as RESPONSE_CSV_PATH (small CSV, NO UUID)
    2. full_data_<surveyId>.csv → full Qualtrics export (NO UUID)
       → path saved in env as RESPONSE_FULL_CSV_PATH

Behavior:
- ENV_PATH is set to the project env constant required by your project.
- Files are placed under:
    <project_root>/qualtrics_downloaded_files/csv/full_data
    <project_root>/qualtrics_downloaded_files/csv/small_csv
- Those two folders are emptied at the start of each run.
"""

import requests
import time
import zipfile
import io
import csv
import sys
import re
import os
from pathlib import Path
from dotenv import load_dotenv

# ==========================================================
# ENV file constant required by project (do not remove)
ENV_PATH = Path(r"C:\Users\erman\PycharmProjects\PythonProject\bellini.ipynb\my_environment.env")
# ==========================================================

env_path = Path(ENV_PATH)

if not env_path.exists():
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("", encoding="utf-8")

load_dotenv(env_path)

# ========== LOAD ENV VARIABLES ==========
API_TOKEN   = os.getenv("API_TOKEN")
DATACENTER  = os.getenv("DATACENTER")
TARGET_NAMES = ["Who Am I Today? Self Assessment - 1.1"]
# ========================================

if not API_TOKEN:
    print("ERROR: API_TOKEN missing in .env")
    sys.exit(1)
if not DATACENTER:
    print("ERROR: DATACENTER missing in .env")
    sys.exit(1)

TIMEOUT = 60
POLL_INTERVAL = 2
EXPORT_TIMEOUT = 300
API_BASE = f"https://{DATACENTER}.qualtrics.com/API/v3"
F_TOKEN_RE = re.compile(r"(F_[A-Za-z0-9_]+)")


# ---------- WRITE ENV VARIABLE WITHOUT QUOTES ----------
def write_env_key_no_quotes(env_path: Path, key: str, value: str) -> None:
    key_prefix = f"{key}="
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except:
        lines = []
    new_lines = []
    written = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        existing_key = line.split("=", 1)[0].strip()
        if existing_key == key:
            new_lines.append(f"{key_prefix}{value}")
            written = True
        else:
            new_lines.append(line)

    if not written:
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.append(f"{key_prefix}{value}")

    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


# --------------------------------------------------------
# PROJECT ROOT (derived from ENV_PATH.parent) - avoids hardcoding other paths
PROJECT_ROOT = env_path.parent  # folder that contains this script

FULL_DATA_DIR = PROJECT_ROOT / "qualtrics_downloaded_files" / "csv" / "full_data"
SMALL_CSV_DIR = PROJECT_ROOT / "qualtrics_downloaded_files" / "csv" / "small_csv"

FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)
SMALL_CSV_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------


# ----- CLEAN FOLDERS BEFORE NEW DOWNLOAD -----
def empty_folder(folder: Path):
    """
    Remove all files and subfolders inside `folder`, but do not remove the folder itself.
    Safe guard: requires folder to be inside PROJECT_ROOT to avoid accidental deletes.
    """
    try:
        folder = folder.resolve()
    except Exception:
        return

    # Safety check: ensure folder is under project root
    try:
        if PROJECT_ROOT.resolve() not in folder.parents and folder.resolve() != PROJECT_ROOT.resolve():
            # refuse to delete anything outside project root
            print(f"Skipping cleanup for {folder} — not inside project root.")
            return
    except Exception:
        return

    for item in folder.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                # remove directory tree
                for sub in sorted(item.rglob("*"), reverse=True):
                    try:
                        if sub.is_file():
                            sub.unlink()
                        elif sub.is_dir():
                            sub.rmdir()
                    except Exception:
                        # ignore individual failures
                        pass
                try:
                    item.rmdir()
                except Exception:
                    pass
        except Exception:
            # ignore and continue
            pass


# --------------------------------------------------------
# ----------------- EXPORT FULL SURVEY CSV --------------
# --------------------------------------------------------
def export_full_survey_csv(survey_id: str, output_dir: Path = None) -> Path:
    """Exports full survey CSV and saves it as full_data_<surveyId>.csv
       If output_dir is None uses FULL_DATA_DIR inside project root.
    """
    out_dir = Path(output_dir) if output_dir else FULL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # START EXPORT
    url = f"{API_BASE}/surveys/{survey_id}/export-responses"
    r = requests.post(url, headers={"X-API-TOKEN": API_TOKEN}, json={"format": "csv"})
    r.raise_for_status()
    prog_id = r.json()["result"]["progressId"]

    # POLL EXPORT STATUS
    url = f"{API_BASE}/surveys/{survey_id}/export-responses/{prog_id}"
    waited = 0
    while True:
        r = requests.get(url, headers={"X-API-TOKEN": API_TOKEN})
        r.raise_for_status()
        status = r.json()["result"]["status"]

        if status == "complete":
            file_id = r.json()["result"]["fileId"]
            break

        if status == "failed":
            raise RuntimeError("Full survey export failed")

        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        if waited > EXPORT_TIMEOUT:
            raise RuntimeError("Full survey export timeout")

    # DOWNLOAD FILE
    url = f"{API_BASE}/surveys/{survey_id}/export-responses/{file_id}/file"
    r = requests.get(url, headers={"X-API-TOKEN": API_TOKEN})
    r.raise_for_status()
    zip_bytes = io.BytesIO(r.content)

    # Extract CSV
    try:
        z = zipfile.ZipFile(zip_bytes)
        csv_file = None
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                csv_file = io.TextIOWrapper(z.open(name), encoding="utf-8")
                break
        if csv_file is None:
            raise RuntimeError("No CSV found in export zip")
    except zipfile.BadZipFile:
        zip_bytes.seek(0)
        csv_file = io.TextIOWrapper(zip_bytes, encoding="utf-8")

    # SAVE with stable name (UUID removed)
    out_name = f"full_data_{survey_id}.csv"
    out_path = out_dir / out_name

    # Overwrite if exists (folder cleanup earlier ensures fresh output)
    with out_path.open("w", encoding="utf-8", newline="") as out:
        for line in csv_file:
            out.write(line)

    # SAVE PATH INTO ENV
    write_env_key_no_quotes(env_path, "RESPONSE_FULL_CSV_PATH", str(out_path))

    return out_path


# --------------------------------------------------------
# ----------------- SMALL CSV LOGIC -------------
# --------------------------------------------------------
class QualtricsResponseIdExporter:

    def __init__(self, api_token, datacenter):
        self.api_token = api_token
        self.datacenter = datacenter
        self.api_base = API_BASE

        self.s = requests.Session()
        self.s.headers.update({
            "X-API-TOKEN": self.api_token,
            "Content-Type": "application/json"
        })

    # --- fetch surveys ---
    def fetch_all_surveys(self):
        url = f"{self.api_base}/surveys"
        surveys = []

        while url:
            r = self.s.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            j = r.json()
            surveys.extend(j["result"]["elements"])
            next_page = j["result"].get("nextPage")

            if not next_page:
                break
            url = next_page if next_page.startswith("http") else f"https://{self.datacenter}.qualtrics.com{next_page}"

        return surveys

    def pick_surveys(self, all_surveys, target_names):
        selected = {}
        for t in target_names:
            selected[t] = [s["id"] for s in all_surveys if t.lower() in s["name"].lower()]
        return selected

    # --- export responses ---
    def start_export(self, survey_id):
        url = f"{self.api_base}/surveys/{survey_id}/export-responses"
        r = self.s.post(url, json={"format": "csv"}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()["result"]["progressId"]

    def poll_export(self, survey_id, progress_id):
        url = f"{self.api_base}/surveys/{survey_id}/export-responses/{progress_id}"
        waited = 0
        while True:
            r = self.s.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            status = r.json()["result"]["status"]
            if status == "complete":
                return r.json()["result"]["fileId"]
            if status == "failed":
                raise RuntimeError("Export failed")
            time.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            if waited > EXPORT_TIMEOUT:
                raise RuntimeError("Timed out")

    def download_export_file(self, survey_id, file_id):
        url = f"{self.api_base}/surveys/{survey_id}/export-responses/{file_id}/file"
        r = self.s.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return io.BytesIO(r.content)

    # --- extract small CSV ---
    def extract_csv(self, zip_io):
        try:
            z = zipfile.ZipFile(zip_io)
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    return io.TextIOWrapper(z.open(name), encoding="utf-8")
        except zipfile.BadZipFile:
            zip_io.seek(0)
            return io.TextIOWrapper(zip_io, encoding="utf-8")

    # --- collect tokens ---
    def collect_response_and_file_tokens(self, csv_file):
        reader = csv.reader(csv_file)
        header1 = next(reader, None)
        header2 = next(reader, None)

        if header2 and "ResponseId" in header2:
            rid_idx = header2.index("ResponseId")
        else:
            rid_idx = header1.index("ResponseId")

        rows = []
        for row in reader:
            if len(row) <= rid_idx:
                continue
            response_id = row[rid_idx].strip()
            if not response_id.startswith("R_"):
                continue

            tokens = set()
            for cell in row:
                for match in F_TOKEN_RE.findall(str(cell)):
                    tokens.add(match)

            rows.append((response_id, sorted(tokens)))

        return rows

    # --- save small CSV (NO UUID) ---
    def save_ids_and_tokens(self, survey_id, rows):
        filename = f"response_ids_with_files_{survey_id}.csv"
        file_path = SMALL_CSV_DIR / filename
        SMALL_CSV_DIR.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["responseId", "fileTokens"])
            for rid, toks in rows:
                writer.writerow([rid, ";".join(toks)])

        print(f"Saved {len(rows)} small CSV rows")

        # SAVE SMALL CSV PATH (DO NOT OVERWRITE)
        write_env_key_no_quotes(env_path, "RESPONSE_CSV_PATH", str(file_path))

    # --- master run ---
    def run(self, target_names):
        surveys = self.fetch_all_surveys()
        selected = self.pick_surveys(surveys, target_names)

        for name, ids in selected.items():
            if not ids:
                print(f"No survey found: {name}")
                continue

            survey_id = ids[0]
            write_env_key_no_quotes(env_path, "SURVEY_ID", survey_id)

            print("\nProcessing survey ...")

            # GENERATE FULL SURVEY CSV (saved into project subfolder, stable name)
            export_full_survey_csv(survey_id)

            # PROCESS SMALL CSV
            prog_id = self.start_export(survey_id)
            file_id = self.poll_export(survey_id, prog_id)
            zip_io = self.download_export_file(survey_id, file_id)
            csv_file = self.extract_csv(zip_io)

            rows = self.collect_response_and_file_tokens(csv_file)
            self.save_ids_and_tokens(survey_id, rows)


# ---------------- MAIN ----------------
def main():
    # CLEAN old CSVs inside the project subfolders before creating new ones
    empty_folder(FULL_DATA_DIR)
    empty_folder(SMALL_CSV_DIR)

    QualtricsResponseIdExporter(API_TOKEN, DATACENTER).run(TARGET_NAMES)


if __name__ == "__main__":
    main()
