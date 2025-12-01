#!/usr/bin/env python3
"""
Qualtrics ResponseId + File Tokens Exporter (X-API-TOKEN)
- Exports responses using export-responses API
- Creates two outputs:
    1. response_ids_with_files_<surveyId>.csv  → (responseId + fileTokens)
       → path saved in env as RESPONSE_CSV_PATH (small CSV, NO UUID)
    2. full_data_<surveyId>.csv → full Qualtrics export (NO UUID)
       → path saved in env as RESPONSE_FULL_CSV_PATH

This updated version additionally computes sentiment columns:
- Sentiment_Score (float rounded to 3 decimals)
- Sentiment_Label (string: Strongly Positive / Positive / Neutral / Negative / Strongly Negative)

Behavior:
- ENV_PATH is set to the project env constant required by your project.
- Files are placed under:
    <project_root>/qualtrics_downloaded_files/csv/full_data
    <project_root>/qualtrics_downloaded_files/csv/small_csv

Important:
- The sentiment annotation and Q6 normalization happen ONLY for full_data_<surveyId>.csv.
- The small CSV (response ids + file tokens) is NOT modified.
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
from typing import Optional

# ==========================================================
# ENV file constant required by project (do not remove)
ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
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

# -----------------------------
# Optional analysis libs
# -----------------------------
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# -----------------------------
# Sentiment thresholds and lexicons (kept consistent with Runner)
# -----------------------------
THRESHOLD_POSITIVE = 0.10
THRESHOLD_NEGATIVE = 0.00
STRONG_POSITIVE = 0.40
STRONG_NEGATIVE = -0.40

_POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "love", "liked", "like",
    "pleasant", "happy", "satisfied", "enjoy", "best", "positive",
    "amazing", "fantastic", "helpful", "help", "useful", "easy",
    "recommend", "recommended", "works", "working"
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "dislike", "worse", "worst",
    "poor", "unsatisfied", "angry", "frustrate", "problem", "issue",
    "not", "no", "never", "difficult", "disappoint", "complain",
    "didn't", "didnt", "not_good", "not_helpful"
}

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
                # Qualtrics CSV is typically UTF-8, use utf-8 for reading the file content
                csv_file = io.TextIOWrapper(z.open(name), encoding="utf-8")
                break
        if csv_file is None:
            raise RuntimeError("No CSV found in export zip")
    except zipfile.BadZipFile:
        zip_bytes.seek(0)
        csv_file = io.TextIOWrapper(zip_bytes, encoding="utf-8")

    # SAVE with stable name (UUID removed) - ***MODIFIED ENCODING HERE***
    out_name = f"full_data_{survey_id}.csv"
    out_path = out_dir / out_name

    # Overwrite if exists (folder cleanup earlier ensures fresh output)
    # Changed encoding to 'utf-8-sig' to save as CSV UTF-8 (Comma delimited)
    with out_path.open("w", encoding="utf-8-sig", newline="") as out:
        for line in csv_file:
            out.write(line)

    # SAVE PATH INTO ENV
    write_env_key_no_quotes(env_path, "RESPONSE_FULL_CSV_PATH", str(out_path))

    # After writing the CSV, compute sentiment columns (if pandas available)
    try:
        compute_and_write_sentiment_columns(out_path)
    except Exception:
        # don't crash the export if sentiment fails
        print("Warning: sentiment processing failed (see traceback).")

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

        # ***MODIFIED ENCODING HERE***
        # Changed encoding to 'utf-8-sig' to save as CSV UTF-8 (Comma delimited)
        with file_path.open("w", encoding="utf-8-sig", newline="") as f:
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


# -----------------------------
# Sentiment helpers (adapted from Runner.py)
# -----------------------------
def normalize_column_name(col: str) -> str:
    col = col.replace("\ufeff", "").strip().lower()
    return re.sub(r"[^0-9a-z]+", "_", col)


def find_q6_column(df: "pd.DataFrame") -> Optional[str]:
    if df is None:
        return None
    # 1) prefer exact q6.1 (case-insensitive)
    for col in df.columns:
        if str(col).strip().lower() == "q6.1":
            return col
    # 2) common variants
    for col in df.columns:
        if str(col).strip().lower() in {"q6_1", "q6-1", "q6.1", "q6 1", "q6"}:
            return col
    # 3) normalized search
    norm_map = {normalize_column_name(c): c for c in df.columns}
    if "q6_1" in norm_map:
        return norm_map["q6_1"]
    for norm, original in norm_map.items():
        if norm.startswith("q6") and "1" in norm:
            return original
    return None


def lexicon_score(text: Optional[str]) -> Optional[float]:
    if pd is not None and pd.isna(text):
        return None
    if text is None:
        return None
    s = str(text).strip()
    if s == "":
        return None
    words = [w.strip(".,!?:;\"'()[]").lower() for w in s.split()]
    if not words:
        return None
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    score = (pos - neg) / max(len(words), 1)
    score = max(-1.0, min(1.0, score))
    return float(score)


def score_with_textblob(text: Optional[str]) -> Optional[float]:
    if TextBlob is None:
        return None
    if pd is not None and pd.isna(text):
        return None
    if text is None:
        return None
    s = str(text).strip()
    if s == "":
        return None
    try:
        return float(TextBlob(s).sentiment.polarity)
    except Exception:
        return None


def label_from_score(s: Optional[float]) -> str:
    if s is None or (pd is not None and pd.isna(s)):
        return "Neutral"
    if s >= STRONG_POSITIVE:
        return "Strongly Positive"
    if s >= THRESHOLD_POSITIVE:
        return "Positive"
    if s <= STRONG_NEGATIVE:
        return "Strongly Negative"
    if s < THRESHOLD_NEGATIVE:
        return "Negative"
    return "Neutral"


def compute_and_write_sentiment_columns(csv_path: Path) -> None:
    """
    Read CSV at csv_path, map the comments/suggestion column to Q6_1 (placed next to Q6),
    compute Sentiment_Score and Sentiment_Label based on Q6_1,
    and overwrite the CSV atomically. If pandas is not available, this is a no-op.
    NOTE: This function is only invoked for the full_data CSV (export_full_survey_csv).
    """
    if pd is None:
        print("pandas not installed; skipping sentiment annotation.")
        return
    if not csv_path.exists():
        print(f"CSV not found for sentiment processing: {csv_path}")
        return

    try:
        # Use 'utf-8-sig' for reading back the CSV if it was just written with it
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as fh:
            df = pd.read_csv(fh, dtype=str)
    except Exception as ex:
        # Fallback to plain utf-8 if utf-8-sig read fails, or if pandas fails
        print(f"Failed reading CSV with utf-8-sig for sentiment: {ex}. Retrying with utf-8.")
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
                df = pd.read_csv(fh, dtype=str)
        except Exception as ex_f:
            print("Failed reading CSV for sentiment (final attempt):", ex_f)
            return

    # ---------------------------
    # 1) Prefer to map a comments/suggestion column -> Q6_1
    # ---------------------------
    comment_cols = [col for col in df.columns if re.search(r"comment|suggest", str(col), re.I)]
    if comment_cols:
        # rename the first matching comment/suggestion column
        chosen = comment_cols[0]
        if chosen != "Q6_1":
            # if Q6_1 already exists, overwrite its values with the chosen column (but keep original if needed)
            df = df.rename(columns={chosen: "Q6_1"})
    else:
        # ---------------------------
        # 2) Fallback: pick first Q6 variant (conservative — do not rename multiple)
        # ---------------------------
        variants_order = ["q6.1", "q6-1", "q6 1", "q6_1", "q6. 1", "q6 .1"]
        found = None
        for v in variants_order:
            for col in df.columns:
                if str(col).strip().lower() == v:
                    found = col
                    break
            if found:
                break
        if found and found != "Q6_1":
            df = df.rename(columns={found: "Q6_1"})
        # if nothing found, ensure Q6_1 exists (empty)
        if "Q6_1" not in df.columns:
            df["Q6_1"] = pd.NA

    # ---------------------------
    # 3) If Q6 exists and Q6_1 exists, make sure Q6_1 is placed immediately after Q6
    # ---------------------------
    cols = list(df.columns)
    if "Q6" in cols and "Q6_1" in cols:
        # remove Q6_1 then insert after Q6
        cols = [c for c in cols if c != "Q6_1"]
        q6_index = cols.index("Q6")
        cols.insert(q6_index + 1, "Q6_1")
        # reorder
        df = df.loc[:, cols]
    else:
        # ensure consistent casing: if found variant but named differently (rare), ensure column exists
        if "Q6_1" not in df.columns:
            df["Q6_1"] = pd.NA

    original_cols = list(df.columns)

    # detect Q6 column for downstream logic (function remains tolerant)
    q6_col = find_q6_column(df)
    if q6_col is None:
        # ensure Q6_1 exists for downstream logic
        if "Q6_1" not in df.columns:
            df["Q6_1"] = pd.NA
    else:
        # if find_q6_column found something not named Q6_1, do not overwrite Q6_1 — keep as-is
        pass

    # ensure sentiment columns exist
    if "Sentiment_Score" not in df.columns:
        df["Sentiment_Score"] = pd.NA
    if "Sentiment_Label" not in df.columns:
        df["Sentiment_Label"] = pd.NA

    non_empty_mask = df["Q6_1"].notna() & df["Q6_1"].astype(str).str.strip().ne("")
    non_empty_count = int(non_empty_mask.sum())

    # decide whether to prefer lexicon
    use_lexicon_global = False
    if TextBlob is None:
        use_lexicon_global = True
    elif non_empty_count > 0:
        nonzero = 0
        total = 0
        for i in df.loc[non_empty_mask].index:
            tbp = score_with_textblob(df.at[i, "Q6_1"])
            total += 1
            if tbp is not None and abs(tbp) > 1e-9:
                nonzero += 1
        if total > 0 and (nonzero / total) < 0.15:
            use_lexicon_global = True

    scores = []
    labels = []
    for idx in df.index:
        cur_score = df.at[idx, "Sentiment_Score"] if "Sentiment_Score" in df.columns else None
        cur_label = df.at[idx, "Sentiment_Label"] if "Sentiment_Label" in df.columns else None

        need_s = bool(pd.isna(cur_score) or str(cur_score).strip() == "")
        need_l = bool(pd.isna(cur_label) or str(cur_label).strip() == "")

        if not need_s and not need_l:
            scores.append(cur_score)
            labels.append(cur_label)
            continue

        text = df.at[idx, "Q6_1"] if "Q6_1" in df.columns else None

        lex = lexicon_score(text)
        tb = None if TextBlob is None else score_with_textblob(text)

        if use_lexicon_global:
            chosen = lex
        else:
            if tb is not None and abs(tb) >= 0.05:
                chosen = tb
            elif lex is not None and abs(lex) >= 0.10:
                chosen = lex
            else:
                chosen = tb if tb is not None else lex

        if chosen is None or (pd is not None and pd.isna(chosen)):
            final_score = 0.0
        else:
            final_score = round(float(chosen), 3)

        final_label = label_from_score(chosen)

        scores.append(final_score)
        labels.append(final_label)

    df["Sentiment_Score"] = scores
    df["Sentiment_Label"] = labels

    # ensure numeric and defaults
    try:
        df["Sentiment_Score"] = pd.to_numeric(df["Sentiment_Score"], errors="coerce").round(3)
    except Exception:
        pass
    df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0.0)
    df["Sentiment_Label"] = df["Sentiment_Label"].fillna("Neutral").astype(str)

    # reorder columns: append sentiment cols if not present originally
    final_cols = original_cols.copy()
    if "Sentiment_Score" not in final_cols:
        final_cols.append("Sentiment_Score")
    if "Sentiment_Label" not in final_cols:
        final_cols.append("Sentiment_Label")
    for c in df.columns:
        if c not in final_cols:
            final_cols.append(c)
    df = df.loc[:, final_cols]

    # atomic write
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        # ***MODIFIED ENCODING HERE***
        # Use 'utf-8-sig' for the final write to ensure Excel compatibility
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        tmp_path.replace(csv_path)
        print(f"Updated sentiment columns in {csv_path}")
    except Exception as ex:
        print("Failed writing updated CSV for sentiment:", ex)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ---------------- MAIN ----------------
def main():
    # CLEAN old CSVs inside the project subfolders before creating new ones
    def empty_folder(folder: Path):
        try:
            folder = folder.resolve()
        except Exception:
            return
        try:
            if PROJECT_ROOT.resolve() not in folder.parents and folder.resolve() != PROJECT_ROOT.resolve():
                print(f"Skipping cleanup for {folder} — not inside project root.")
                return
        except Exception:
            return
        for item in folder.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for sub in sorted(item.rglob("*"), reverse=True):
                        try:
                            if sub.is_file():
                                sub.unlink()
                            elif sub.is_dir():
                                sub.rmdir()
                        except Exception:
                            pass
                    try:
                        item.rmdir()
                    except Exception:
                        pass
            except Exception:
                pass

    empty_folder(FULL_DATA_DIR)
    empty_folder(SMALL_CSV_DIR)

    QualtricsResponseIdExporter(API_TOKEN, DATACENTER).run(TARGET_NAMES)


if __name__ == "__main__":
    main()