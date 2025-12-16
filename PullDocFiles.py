"""
Qualtrics uploaded-files downloader (incremental)
- Input CSV columns: responseId = R_... fileTokens = semicolon-separated F_... tokens
- Saves each file into survey-specific folders: downloads/<survey_id>/
- Skips already downloaded files using in-memory + filesystem index
- Handles 429 rate limits
- Reads CSV with utf-8-sig
- NOW: processes ALL CSVs inside /small_csv/
- Extracts survey_id from filename: SV_<id>
- Skipped F_SUMMARY,F_Mindset,F_DavisC,F_SMART,F_INTERVIEW which was slowing down process
"""

import csv
import os
import re
import time
import mimetypes
from urllib.parse import unquote
import requests
from pathlib import Path
from dotenv import load_dotenv

# ==========================================================
ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
# ==========================================================

env_path = Path(ENV_PATH)
print(env_path)
if not env_path.exists():
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("", encoding="utf-8")

load_dotenv(env_path)

# ========= CONFIG =========
PROJECT_ROOT = env_path.parent
API_TOKEN = os.getenv("API_TOKEN")
DATACENTER = os.getenv("DATACENTER")

# NEW → read ignore tokens from .env
ignore_tokens_raw = os.getenv("IGNORE_FILE_TOKENS", "")
IGNORE_TOKENS = {t.strip() for t in ignore_tokens_raw.split(",") if t.strip()}

# Folder containing multiple CSVs (process all)
INPUT_CSV_FOLDER = PROJECT_ROOT / "qualtrics_downloaded_files" / "csv" / "small_csv"

BASE_URL = f"https://{DATACENTER}.qualtrics.com"

session = requests.Session()
session.headers.update({
    "X-API-TOKEN": API_TOKEN,
    "Accept": "*/*, application/json"
})

RESPID_COL  = "responseId"
FILETOK_COL = "fileTokens"

MAX_RETRIES     = 3
BASE_SLEEP      = 5
PER_FILE_DELAY  = 0.2
# ==========================


# --------------------
# helpers (unchanged)
# --------------------
def clean_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*;\x00-\x1F]', "_", name)
    name = re.sub(r"_+", "_", name).strip("._ ")
    return name[:200] or "file"


def parse_content_disposition(cd: str) -> str | None:
    if not cd:
        return None
    parts = [p.strip() for p in cd.split(";")]
    filename = None
    filename_star = None
    for p in parts[1:]:
        if p.lower().startswith("filename*="):
            value = p.split("=", 1)[1].strip().strip('"').strip("'")
            if "''" in value:
                _, encoded = value.split("''", 1)
                try:
                    value = unquote(encoded)
                except Exception:
                    pass
            filename_star = value
        elif p.lower().startswith("filename="):
            value = p.split("=", 1)[1].strip().strip('"').strip("'")
            filename = value
    return filename_star or filename


def guess_filename(resp, file_token, response_id):
    cd = resp.headers.get("content-disposition", "")
    original = parse_content_disposition(cd)

    ext = ""
    if original:
        original = unquote(original)
        _, ext = os.path.splitext(original)

    if not ext:
        ctype = resp.headers.get("content-type", "").split(";")[0].strip()
        ext = mimetypes.guess_extension(ctype) or ""

    return clean_filename(f"{file_token}{ext}")


def build_file_url(survey_id, response_id, file_token):
    return f"{BASE_URL}/API/v3/surveys/{survey_id}/responses/{response_id}/uploaded-files/{file_token}"


# --------------------
# Index helpers (unchanged)
# --------------------
_TOKEN_RE = re.compile(r'^(F_[A-Za-z0-9]+)')


def load_index_from_folder(folder: Path) -> dict:
    idx = {}
    try:
        for p in folder.iterdir():
            if not p.is_file():
                continue
            m = _TOKEN_RE.match(p.name)
            if m:
                token = m.group(1)
                if token not in idx:
                    idx[token] = p.name
    except Exception as e:
        print(f"!! Warning: failed to scan {folder}: {e}")
    return idx


def existing_file_for_token(file_token: str, index: dict, folder: Path):
    if file_token in index:
        candidate = folder / index[file_token]
        if candidate.exists():
            return candidate

    for p in folder.glob(f"{file_token}*"):
        if p.is_file():
            return p

    return None


# --------------------
# download logic (unchanged)
# --------------------
def download_file(file_token, response_id, index, folder, survey_id):

    existing = existing_file_for_token(file_token, index, folder)
    if existing:
        print(f"⏭️  Skipping {file_token} — already present in {folder.name}")
        return "skipped"

    url = build_file_url(survey_id, response_id, file_token)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {file_token} ({survey_id}) for response {response_id} ... attempt {attempt}")
            r = session.get(url, stream=True)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else BASE_SLEEP * attempt
                print(f" 429 Too Many Requests → sleeping {sleep_for}s")
                time.sleep(sleep_for)
                continue

            r.raise_for_status()

            filename = guess_filename(r, file_token, response_id)
            out_path = folder / filename

            if out_path.exists():
                index[file_token] = filename
                return "skipped"

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    if chunk:
                        f.write(chunk)

            print(f"✔ Saved {file_token} → {out_path}")
            index[file_token] = filename
            time.sleep(PER_FILE_DELAY)
            return "downloaded"

        except Exception as e:
            print(f"!! Error downloading {file_token}: {e}")
            if attempt == MAX_RETRIES:
                print(f"!! Giving up on {file_token}")
                return "error"
            time.sleep(BASE_SLEEP * attempt)

    return "error"


# --------------------
# main — with final overall summary + ignore support
# --------------------
def main():

    if not INPUT_CSV_FOLDER.exists():
        print(f"ERROR: CSV folder not found: {INPUT_CSV_FOLDER}")
        return

    csv_files = list(INPUT_CSV_FOLDER.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in small_csv folder.")
        return

    overall_summary = {}

    for csv_file in csv_files:

        print("\n===================================")
        print(f"📌 Processing CSV: {csv_file.name}")
        print("===================================\n")

        m = re.search(r"(SV_[A-Za-z0-9]+)", csv_file.name)
        if not m:
            print(f"⚠️  No survey ID found in filename: {csv_file.name}")
            continue

        survey_id = m.group(1)
        print(f"Detected survey_id: {survey_id}")

        with open(csv_file, newline="", encoding="utf-8-sig") as f:
            reader = list(csv.DictReader(f))

        if not reader:
            print("⚠️ CSV is empty, skipping.")
            continue

        resp_col = RESPID_COL
        tok_col = FILETOK_COL

        survey_folder = PROJECT_ROOT / "qualtrics_downloaded_files" / "downloads" / survey_id
        survey_folder.mkdir(parents=True, exist_ok=True)

        index = load_index_from_folder(survey_folder)
        print(f"Index for {survey_id}: {len(index)} files\n")

        download_count = 0
        skip_count = 0

        for row in reader:
            response_id = (row.get(resp_col) or "").strip()
            if not response_id.startswith("R_"):
                continue

            toks = (row.get(tok_col) or "").strip()
            if not toks:
                continue

            tokens = [t.strip() for t in toks.split(";") if t.strip()]

            for token in tokens:

                # NEW → skip ignore tokens from .env
                if token in IGNORE_TOKENS:
                    print(f"⏭️ Ignoring token (configured): {token}")
                    continue

                if not token.startswith("F_"):
                    continue

                status = download_file(token, response_id, index, survey_folder, survey_id)

                if status == "downloaded":
                    download_count += 1
                elif status == "skipped":
                    skip_count += 1

        print(f"\nSummary for {survey_id}:")
        print(f"  Downloaded: {download_count}")
        print(f"  Skipped:    {skip_count}\n")

        overall_summary[survey_id] = {
            "downloaded": download_count,
            "skipped": skip_count
        }

    print("\n===================================")
    print("📊 FINAL SUMMARY FOR ALL SURVEYS")
    print("===================================\n")

    for sid, stats in overall_summary.items():
        print(f"Survey: {sid}")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  Skipped:    {stats['skipped']}\n")


if __name__ == "__main__":
    main()
