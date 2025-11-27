"""
Qualtrics uploaded-files downloader (incremental)

- Input CSV columns:
    responseId   = R_...
    fileTokens   = semicolon-separated F_... tokens (e.g. "F_aaa;F_bbb")
- For each (responseId, F_...) pair, calls:
    GET /API/v3/surveys/{surveyId}/responses/{responseId}/uploaded-files/{fileId}
- Saves each file into OUTPUT_DIR
- On re-run, if a file with the same final name already exists, it is skipped.
- Handles 429 (Too Many Requests) with retries + backoff.
"""

import csv
import os
import re
import time
import mimetypes
from urllib.parse import urlparse, unquote
import requests
from pathlib import Path
from dotenv import load_dotenv

# ==========================================================
# ALWAYS USE THIS ENV PATH (dynamic, no hardcoding)
ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
# ==========================================================

env_path = Path(ENV_PATH)

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

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_filename(name: str) -> str:
    """Remove bad characters for filesystem and limit length."""
    # Also strip semicolons which caused issues in your path
    name = re.sub(r'[<>:"/\\|?*;\x00-\x1F]', "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("._ ")
    # Limit to something safe-ish
    return name[:200] or "file"


def parse_content_disposition(cd: str) -> str | None:
    """
    Parse Content-Disposition and return filename or filename* value, if present.
    Handles headers like:
      attachment; filename="foo.docx"; filename*=utf-8''foo.docx
    """
    if not cd:
        return None

    parts = [p.strip() for p in cd.split(";")]
    filename = None
    filename_star = None

    for p in parts[1:]:
        if p.lower().startswith("filename*="):
            value = p.split("=", 1)[1].strip()
            value = value.strip('"').strip("'")
            # filename*=utf-8''encoded-name
            if "''" in value:
                _, encoded_name = value.split("''", 1)
                try:
                    value = unquote(encoded_name)
                except Exception:
                    pass
            filename_star = value
        elif p.lower().startswith("filename="):
            value = p.split("=", 1)[1].strip()
            value = value.strip('"').strip("'")
            filename = value

    return filename_star or filename


def guess_filename(resp: requests.Response, file_token: str, response_id: str | None) -> str:
    """
    Build a nice filename from headers / URL / token.
    Prefix with responseId for traceability.
    """
    cd = resp.headers.get("content-disposition", "")
    fn = parse_content_disposition(cd)

    if not fn:
        # Fallback to URL path or token
        path = urlparse(resp.url).path
        fn = os.path.basename(path) or file_token

    fn = unquote(fn)

    # Ensure extension if missing
    content_type = resp.headers.get("content-type", "").split(";")[0].strip()
    if "." not in fn:
        ext = mimetypes.guess_extension(content_type) or ""
        fn = fn + ext

    if response_id:
        fn = f"{response_id}_{fn}"

    return clean_filename(fn)


def build_file_url(response_id: str, file_token: str) -> str:
    """
    Use the uploaded-files endpoint from Qualtrics docs:

    GET /API/v3/surveys/{surveyId}/responses/{responseId}/uploaded-files/{fileId}
    where fileId is the file token (F_...)
    """
    return f"{BASE_URL}/API/v3/surveys/{SURVEY_ID}/responses/{response_id}/uploaded-files/{file_token}"


def download_file(file_token: str, response_id: str):
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

            # Build final filename and check if it already exists
            filename = guess_filename(r, file_token, response_id)
            out_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(out_path):
                print(f"⏭️  Skipping {file_token} (already exists: {out_path})")
                return

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    if chunk:
                        f.write(chunk)

            print(f"✔ Saved {file_token} → {out_path}")
            time.sleep(PER_FILE_DELAY)
            return

        except requests.HTTPError as e:
            # Non-429 HTTP error; usually not worth retrying many times
            print(f"!! HTTP error for {file_token} / {response_id}: {e}")
            break

        except Exception as e:
            print(f"!! Error downloading {file_token} for {response_id}: {e}")
            if attempt == MAX_RETRIES:
                print(f"!! Giving up on {file_token} after {MAX_RETRIES} attempts.")
            else:
                sleep_for = BASE_SLEEP * attempt
                print(f"  Sleeping {sleep_for} seconds before retry...")
                time.sleep(sleep_for)


def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            response_id = (row.get(RESPID_COL) or "").strip()
            if not response_id.startswith("R_"):
                continue

            tokens_field = (row.get(FILETOK_COL) or "").strip()
            if not tokens_field:
                continue

            # fileTokens are like "F_aaa;F_bbb;F_ccc"
            tokens = [t.strip() for t in tokens_field.split(";") if t.strip()]

            for token in tokens:
                if not token.startswith("F_"):
                    continue
                download_file(token, response_id)


if __name__ == "__main__":
    main()
