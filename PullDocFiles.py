"""
Qualtrics uploaded-files downloader (incremental)

- Input CSV columns:
    responseId   = R_...
    fileTokens   = semicolon-separated F_... tokens (e.g. "F_aaa;F_bbb")
- For each (responseId, F_...) pair, calls:
    GET /API/v3/surveys/{surveyId}/responses/{responseId}/uploaded-files/{fileId}
- Saves each file into OUTPUT_DIR as: F_token.<ext> (uses server extension or content-type fallback)
- Pre-checks: builds an in-memory index from files present in OUTPUT_DIR at startup (Option 2).
- On re-run, if a file for that token already exists (index or filesystem), API call is skipped.
- Handles 429 (Too Many Requests) with retries + backoff.
- Reads CSV with utf-8-sig to avoid BOM issues in headers.
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
# ALWAYS USE THIS ENV PATH (dynamic, no hardcoding)
ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
# ==========================================================

env_path = Path(ENV_PATH)
print(env_path)
if not env_path.exists():
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("", encoding="utf-8")

load_dotenv(env_path)

# ========= CONFIG =========
API_TOKEN = os.getenv("API_TOKEN")   # <- don't keep real token in source control
DATACENTER = os.getenv("DATACENTER")                     # e.g. "pdx1", "usf.pdx1", "eu1"
SURVEY_ID   = os.getenv("SURVEY_ID")               # surveyId for this CSV
INPUT_CSV   = os.getenv(r"RESPONSE_CSV_PATH")   # CSV: responseId,fileTokens
# Project root (dynamic)
PROJECT_ROOT = env_path.parent

# Save files to:
OUTPUT_DIR = PROJECT_ROOT / "qualtrics_downloaded_files" / "downloads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)                        # where files will be saved

RESPID_COL  = "responseId"                       # column name in the CSV
FILETOK_COL = "fileTokens"                       # column with "F_..." tokens

MAX_RETRIES     = 5          # max retries for a single file
BASE_SLEEP      = 5          # base seconds for backoff
PER_FILE_DELAY  = 0.2        # small delay after each successful download
# ==========================

BASE_URL = f"https://{DATACENTER}.qualtrics.com"

session = requests.Session()
session.headers.update({
    "X-API-TOKEN": API_TOKEN,
    "Accept": "*/*, application/json"
})


# --------------------
# helpers
# --------------------
def clean_filename(name: str) -> str:
    """Remove bad characters for filesystem and limit length."""
    name = re.sub(r'[<>:"/\\|?*;\x00-\x1F]', "_", name)
    name = re.sub(r"_+", "_", name).strip("._ ")
    return name[:200] or "file"


def parse_content_disposition(cd: str) -> str | None:
    """
    Parse Content-Disposition and return filename or filename* value, if present.
    """
    if not cd:
        return None

    parts = [p.strip() for p in cd.split(";")]
    filename = None
    filename_star = None

    for p in parts[1:]:
        if p.lower().startswith("filename*="):
            value = p.split("=", 1)[1].strip().strip('"').strip("'")
            if "''" in value:
                _, encoded_name = value.split("''", 1)
                try:
                    value = unquote(encoded_name)
                except Exception:
                    pass
            filename_star = value
        elif p.lower().startswith("filename="):
            value = p.split("=", 1)[1].strip().strip('"').strip("'")
            filename = value

    return filename_star or filename


def guess_filename(resp: requests.Response, file_token: str, response_id: str | None) -> str:
    """
    Build filename using ONLY the file token and extension provided by server (or guessed from content-type).
    Example output: F_abc123.pdf
    """
    cd = resp.headers.get("content-disposition", "")
    original = parse_content_disposition(cd)

    ext = ""
    if original:
        original = unquote(original)
        _, ext = os.path.splitext(original)

    # If server did NOT provide extension, fallback to content-type detection
    if not ext:
        content_type = resp.headers.get("content-type", "").split(";")[0].strip()
        ext = mimetypes.guess_extension(content_type) or ""

    final_name = f"{file_token}{ext}"
    return clean_filename(final_name)


def build_file_url(response_id: str, file_token: str) -> str:
    """
    GET /API/v3/surveys/{surveyId}/responses/{responseId}/uploaded-files/{fileId}
    """
    return f"{BASE_URL}/API/v3/surveys/{SURVEY_ID}/responses/{response_id}/uploaded-files/{file_token}"


# --------------------
# Index: Option 2 — build in-memory index from existing files at startup
# --------------------
_TOKEN_RE = re.compile(r'^(F_[A-Za-z0-9]+)')  # extracts leading token like F_abc123

def load_index_from_folder() -> dict:
    """
    Scan OUTPUT_DIR for files starting with 'F_' and build a map:
      { "F_token": "filename.ext", ... }
    This is performed once at startup and kept in-memory for the run.
    """
    idx = {}
    try:
        for p in OUTPUT_DIR.iterdir():
            if not p.is_file():
                continue
            name = p.name
            m = _TOKEN_RE.match(name)
            if m:
                token = m.group(1)
                # prefer the first seen filename for a token
                if token not in idx:
                    idx[token] = name
    except Exception as e:
        print(f"!! Warning: failed to scan OUTPUT_DIR for index: {e}")
    return idx


def existing_file_for_token(file_token: str, index: dict) -> Path | None:
    """
    1) Check in-memory index for exact filename.
    2) Fallback to filesystem glob for any file starting with token.
    Returns Path if found, else None.
    """
    # 1) index lookup
    if file_token in index:
        candidate = OUTPUT_DIR / index[file_token]
        if candidate.exists():
            return candidate
        # if indexed filename missing (external deletion), fallthrough to glob

    # 2) filesystem fallback: find any file that starts with token (handles unknown ext or different naming)
    for p in OUTPUT_DIR.glob(f"{file_token}*"):
        if p.is_file():
            return p

    return None


# --------------------
# download logic
# --------------------
def download_file(file_token: str, response_id: str, index: dict) -> str:
    """
    Attempts to download. Returns:
      - "downloaded" if file was downloaded this run
      - "skipped" if file already existed and we did not call API
      - "error" on failure after retries
    """
    # pre-check: skip API if file already exists (index or filesystem)
    existing = existing_file_for_token(file_token, index)
    if existing:
        print(f"⏭️  Skipping {file_token} — already present: {existing}")
        return "skipped"

    url = build_file_url(response_id, file_token)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {file_token} for response {response_id} (attempt {attempt}) ...")
            r = session.get(url, stream=True)

            # Handle rate limiting
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_for = int(retry_after)
                else:
                    sleep_for = BASE_SLEEP * attempt
                print(f"  Got 429 Too Many Requests, sleeping {sleep_for} seconds before retry...")
                time.sleep(sleep_for)
                continue

            r.raise_for_status()

            # Build final filename and write
            filename = guess_filename(r, file_token, response_id)
            out_path = OUTPUT_DIR / filename

            # race-safe double-check
            if out_path.exists():
                print(f"⏭️  Skipping {file_token} (already exists: {out_path})")
                # update in-memory index for run
                index[file_token] = filename
                return "skipped"

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    if chunk:
                        f.write(chunk)

            print(f"✔ Saved {file_token} → {out_path}")

            # update in-memory index for this run
            index[file_token] = filename

            time.sleep(PER_FILE_DELAY)
            return "downloaded"

        except requests.HTTPError as e:
            print(f"!! HTTP error for {file_token} / {response_id}: {e}")
            break

        except Exception as e:
            print(f"!! Error downloading {file_token} for {response_id}: {e}")
            if attempt == MAX_RETRIES:
                print(f"!! Giving up on {file_token} after {MAX_RETRIES} attempts.")
                return "error"
            else:
                sleep_for = BASE_SLEEP * attempt
                print(f"  Sleeping {sleep_for} seconds before retry...")
                time.sleep(sleep_for)

    return "error"


# --------------------
# CSV helpers
# --------------------
def find_best_column_match(headers, desired):
    desired_l = desired.lower()
    for h in headers:
        if h.lower() == desired_l:
            return h
    for h in headers:
        if desired_l in h.lower():
            return h
    return None


# --------------------
# main
# --------------------
def main():
    if not INPUT_CSV:
        print("ERROR: RESPONSE_CSV_PATH not set in env.")
        return

    # Build in-memory index once at startup (Option 2)
    index = load_index_from_folder()
    print(f"Index built from folder: {len(index)} tokens found")

    download_count = 0
    skip_count = 0

    # Use 'utf-8-sig' so a leading BOM (e.g. '\ufeff') in the CSV header is removed automatically.
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # header matching (case-insensitive / substring)
        headers = reader.fieldnames or []
        resp_col = find_best_column_match(headers, RESPID_COL) or RESPID_COL
        token_col = find_best_column_match(headers, FILETOK_COL) or FILETOK_COL

        for row in reader:
            response_id = (row.get(resp_col) or "").strip()
            if not response_id.startswith("R_"):
                continue

            tokens_field = (row.get(token_col) or "").strip()
            if not tokens_field:
                continue

            # fileTokens are like "F_aaa;F_bbb;F_ccc"
            tokens = [t.strip() for t in tokens_field.split(";") if t.strip()]

            for token in tokens:
                if not token.startswith("F_"):
                    continue
                status = download_file(token, response_id, index)
                if status == "downloaded":
                    download_count += 1
                elif status == "skipped":
                    skip_count += 1
                # errors are logged inline

    print("\nSummary:")
    print(f"Total files downloaded: {download_count}")
    print(f"Total files skipped: {skip_count}")


if __name__ == "__main__":
    main()
