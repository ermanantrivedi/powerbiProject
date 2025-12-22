#!/usr/bin/env python3
"""
Qualtrics Multi-Survey Exporter (Final Merged Version)
Includes:
- Full original sentiment engine (Q6 mapping, lexicon/TextBlob hybrid, atomic write, etc.)
- Multi-level execution (LEVEL1_TARGETS, LEVEL2_TARGETS…)
- RUN_LEVEL override with comma-separated levels (e.g., RUN_LEVEL=1,3)
- Small CSV for ALL surveys (response_ids_with_files_<surveyid>.csv)
- Full CSV for ALL surveys (full_data_<surveyid>.csv)
- Only special survey gets sentiment ("Who Am I Today? Self Assessment - 1.1")
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
# ENV + PATH SETUP
# ==========================================================

ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
env_path = Path(ENV_PATH)
if not env_path.exists():
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("", encoding="utf-8")

load_dotenv(env_path)

API_TOKEN = os.getenv("API_TOKEN")
DATACENTER = os.getenv("DATACENTER")

if not API_TOKEN:
    print("ERROR: API_TOKEN missing in .env")
    sys.exit(1)
if not DATACENTER:
    print("ERROR: DATACENTER missing in .env")
    sys.exit(1)

API_BASE = f"https://{DATACENTER}.qualtrics.com/API/v3"
TIMEOUT = 60
POLL_INTERVAL = 2
EXPORT_TIMEOUT = 300
F_TOKEN_RE = re.compile(r"(F_[A-Za-z0-9_]+)")

# Storage folders
PROJECT_ROOT = env_path.parent
CSV_BASE = PROJECT_ROOT / "qualtrics_downloaded_files" / "csv"
FULL_DATA_DIR = CSV_BASE / "full_data"
SMALL_CSV_DIR = CSV_BASE / "small_csv"

FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)
SMALL_CSV_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LEVEL SYSTEM
# ==========================================================

def load_level_targets(level_num):
    key = f"LEVEL{level_num}_TARGETS"
    raw = os.getenv(key, "")
    return [x.strip() for x in raw.split(",") if x.strip()]

def get_all_levels():
    levels = []
    lvl = 1
    while True:
        targets = load_level_targets(lvl)
        if not targets:
            break
        levels.append((lvl, targets))
        lvl += 1
    return levels

def parse_run_levels(run_level_value):
    try:
        return [int(v.strip()) for v in run_level_value.split(",") if v.strip()]
    except:
        print("Invalid RUN_LEVEL. Use: 1 or 1,3 or 2,4")
        return []

RUN_LEVEL = os.getenv("RUN_LEVEL")

# ==========================================================
# ENV WRITE HELPER
# ==========================================================

def write_env_key_no_quotes(env_path: Path, key: str, value: str):
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except:
        lines = []

    new_lines = []
    written = False

    for line in lines:
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            new_lines.append(line)
            continue

        existing_key = line.split("=", 1)[0].strip()
        if existing_key == key:
            new_lines.append(f"{key}={value}")
            written = True
        else:
            new_lines.append(line)

    if not written:
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

# ==========================================================
# EXPORT FULL SURVEY CSV
# ==========================================================

def export_full_survey_csv(survey_id: str):
    url = f"{API_BASE}/surveys/{survey_id}/export-responses"
    r = requests.post(url, headers={"X-API-TOKEN": API_TOKEN}, json={"format": "csv"})
    r.raise_for_status()
    prog_id = r.json()["result"]["progressId"]

    # Poll export
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
            raise RuntimeError("Full export failed")

        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        if waited > EXPORT_TIMEOUT:
            raise RuntimeError("Full export timeout")

    # Download file
    url = f"{API_BASE}/surveys/{survey_id}/export-responses/{file_id}/file"
    r = requests.get(url, headers={"X-API-TOKEN": API_TOKEN})
    r.raise_for_status()
    zip_bytes = io.BytesIO(r.content)

    # Extract CSV
    try:
        z = zipfile.ZipFile(zip_bytes)
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                return io.TextIOWrapper(z.open(name), encoding="utf-8")
    except zipfile.BadZipFile:
        zip_bytes.seek(0)
        return io.TextIOWrapper(zip_bytes, encoding="utf-8")
# ==========================================================
# SMALL CSV (responseId + fileTokens)
# ==========================================================

def extract_ids_and_tokens(csv_file):
    reader = csv.reader(csv_file)
    header1 = next(reader, None)
    header2 = next(reader, None)

    rid_idx = (
        header2.index("ResponseId")
        if header2 and "ResponseId" in header2
        else header1.index("ResponseId")
    )

    rows = []
    for row in reader:
        if len(row) <= rid_idx:
            continue
        rid = row[rid_idx].strip()
        if not rid.startswith("R_"):
            continue

        # Collect file tokens
        tokens = {m for cell in row for m in F_TOKEN_RE.findall(str(cell))}
        rows.append((rid, sorted(tokens)))

    return rows


def save_small_csv(survey_id, rows):
    filename = f"response_ids_with_files_{survey_id}.csv"
    out_path = SMALL_CSV_DIR / filename

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["responseId", "fileTokens"])

        for rid, toks in rows:
            w.writerow([rid, ";".join(toks)])

    print(f"Small CSV saved successfully")
    return out_path


# ==========================================================
# FULL ORIGINAL SENTIMENT ENGINE (RESTORED EXACTLY)
# ==========================================================

# Load optional libs
try:
    import pandas as pd
except:
    pd = None

try:
    from textblob import TextBlob
except:
    TextBlob = None

# Lexicons (same as original)
_POSITIVE = {
    "good","great","excellent","awesome","amazing","fantastic","love","loved","lovely",
    "like","liked","likes","pleasant","happy","joy","joyful","satisfied","satisfying",
    "enjoy","enjoyed","enjoyable","best","positive","perfect","helpful","help","helped",
    "useful","easy","easier","easiest","recommend","recommended","recommendation",
    "works","working","worked","smooth","smoothly","brilliant","wonderful","exceptional",
    "beneficial","productive","effective","efficient","friendly","responsive","quick",
    "fast","improved","improving","reliable","trustworthy","affordable","valuable",
    "value","great job","well done","thank you","appreciate","appreciated","appreciation"
}

_NEGATIVE = {
    "bad","terrible","awful","horrible","poor","hate","hated","dislike","disliked",
    "worse","worst","unpleasant","unsatisfied","dissatisfied","unhappy","angry","mad",
    "frustrated","frustrating","frustration","problem","problems","issue","issues","bug",
    "bugs","not","never","don't think","difficult","hard","complicated","confusing",
    "disappoint","disappointed","disappointing","complain","complained","complaint",
    "complaints","didn't","didnt","don't","dont","can't","cant","won't","wont",
    "not good","not helpful","not working","slow","slower","slowest","expensive",
    "costly","unhelpful","useless","ineffective","unreliable","broken","fail","failed",
    "failing","crash","crashed","crashing","refund","return","returned","wrong",
    "incorrect","missing","incomplete","invalid","denied","lag","laggy","freeze","freezing"
}

THRESHOLD_POSITIVE = 0.10
THRESHOLD_NEGATIVE = 0.00
STRONG_POSITIVE = 0.40
STRONG_NEGATIVE = -0.40


# ---------- Helpers (exactly from your original script) ----------

def normalize_column_name(col: str) -> str:
    col = col.replace("\ufeff", "").strip().lower()
    return re.sub(r"[^0-9a-z]+", "_", col)


def find_q6_column(df: "pd.DataFrame") -> Optional[str]:
    if df is None:
        return None

    # Prefer exact Q6.1
    for col in df.columns:
        if str(col).strip().lower() == "q6.1":
            return col

    # Common variants
    for col in df.columns:
        if str(col).strip().lower() in {"q6_1", "q6-1", "q6.1", "q6 1", "q6"}:
            return col

    # Normalized search
    norm = {normalize_column_name(c): c for c in df.columns}
    if "q6_1" in norm:
        return norm["q6_1"]

    for key, orig in norm.items():
        if key.startswith("q6") and "1" in key:
            return orig

    return None


def lexicon_score(text: Optional[str]) -> Optional[float]:
    if pd is not None and pd.isna(text):
        return None
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    words = [w.strip(".,!?:;\"'()[]").lower() for w in s.split()]
    if not words:
        return None

    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)

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
    if not s:
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


# Full original sentiment engine — EXACT logic preserved
def compute_and_write_sentiment_columns(csv_path: Path) -> None:
    if pd is None:
        print("pandas not installed → skipping sentiment.")
        return
    if not csv_path.exists():
        print(f"CSV not found for sentiment: {csv_path}")
        return

    # Read CSV (UTF-8-SIG preferred)
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as fh:
            df = pd.read_csv(fh, dtype=str)
    except:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(fh, dtype=str)

    # --- 1) Identify comments/suggestion column
    comment_cols = [c for c in df.columns if re.search(r"comment|suggest", c, re.I)]
    if comment_cols:
        chosen = comment_cols[0]
        if chosen != "Q6_1":
            df = df.rename(columns={chosen: "Q6_1"})
    else:
        # 2) Fallback Q6 detection
        variants = ["q6.1", "q6-1", "q6 1", "q6_1", "q6. 1", "q6 .1"]
        found = None
        for v in variants:
            for col in df.columns:
                if str(col).strip().lower() == v:
                    found = col
                    break
            if found:
                break

        if found and found != "Q6_1":
            df = df.rename(columns={found: "Q6_1"})
        if "Q6_1" not in df.columns:
            df["Q6_1"] = pd.NA

    # --- 3) Ensure Q6_1 positioned after Q6
    cols = list(df.columns)
    if "Q6" in cols and "Q6_1" in cols:
        cols = [c for c in cols if c != "Q6_1"]
        q6_idx = cols.index("Q6")
        cols.insert(q6_idx + 1, "Q6_1")
        df = df.loc[:, cols]

    original_cols = list(df.columns)

    # detect Q6 column
    q6_col = find_q6_column(df)
    if q6_col is None and "Q6_1" not in df.columns:
        df["Q6_1"] = pd.NA

    # Ensure sentiment fields exist
    if "Sentiment_Score" not in df.columns:
        df["Sentiment_Score"] = pd.NA
    if "Sentiment_Label" not in df.columns:
        df["Sentiment_Label"] = pd.NA

    # Determine global scoring strategy
    non_empty = df["Q6_1"].notna() & df["Q6_1"].astype(str).str.strip().ne("")
    non_empty_count = int(non_empty.sum())

    use_lexicon_global = False
    if TextBlob is None:
        use_lexicon_global = True
    elif non_empty_count > 0:
        nonzero = 0
        for idx in df.loc[non_empty].index:
            tbp = score_with_textblob(df.at[idx, "Q6_1"])
            if tbp is not None and abs(tbp) > 1e-9:
                nonzero += 1
        if nonzero / max(non_empty_count,1) < 0.15:
            use_lexicon_global = True

    # Perform sentiment computation
    scores, labels = [], []

    for idx in df.index:
        cur_s = df.at[idx, "Sentiment_Score"]
        cur_l = df.at[idx, "Sentiment_Label"]

        need_s = pd.isna(cur_s) or str(cur_s).strip() == ""
        need_l = pd.isna(cur_l) or str(cur_l).strip() == ""

        if not need_s and not need_l:
            scores.append(cur_s)
            labels.append(cur_l)
            continue

        text = df.at[idx, "Q6_1"]
        lex = lexicon_score(text)
        tb = score_with_textblob(text)

        if use_lexicon_global:
            chosen = lex
        else:
            if tb is not None and abs(tb) >= 0.05:
                chosen = tb
            elif lex is not None and abs(lex) >= 0.10:
                chosen = lex
            else:
                chosen = tb if tb is not None else lex

        if chosen is None or pd.isna(chosen):
            final_score = 0.0
        else:
            final_score = round(float(chosen), 3)

        final_label = label_from_score(chosen)
        scores.append(final_score)
        labels.append(final_label)

    df["Sentiment_Score"] = scores
    df["Sentiment_Label"] = labels

    # Clean numeric
    try:
        df["Sentiment_Score"] = pd.to_numeric(df["Sentiment_Score"], errors="coerce").round(3)
    except:
        pass

    df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0.0)
    df["Sentiment_Label"] = df["Sentiment_Label"].fillna("Neutral").astype(str)

    # Reorder
    final_cols = original_cols.copy()
    if "Sentiment_Score" not in final_cols:
        final_cols.append("Sentiment_Score")
    if "Sentiment_Label" not in final_cols:
        final_cols.append("Sentiment_Label")

    for c in df.columns:
        if c not in final_cols:
            final_cols.append(c)

    df = df.loc[:, final_cols]

    # Atomic write
    tmp_path = csv_path.with_suffix(".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(csv_path)

    print("Sentiment columns updated →", csv_path)
# ==========================================================
# SURVEY EXPORTER CLASS
# ==========================================================

class QualtricsExporter:

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({
            "X-API-TOKEN": API_TOKEN,
            "Content-Type": "application/json"
        })

    # ------------------------------------------------------
    # Fetch surveys
    # ------------------------------------------------------
    def fetch_surveys(self):
        url = f"{API_BASE}/surveys"
        out = []

        while url:
            r = self.s.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            j = r.json()

            out.extend(j["result"]["elements"])

            nxt = j["result"].get("nextPage")
            if nxt:
                url = nxt if nxt.startswith("http") else f"https://{DATACENTER}.qualtrics.com{nxt}"
            else:
                break

        return out

    # ------------------------------------------------------
    # Pick surveys matching target names (case-insensitive)
    # ------------------------------------------------------
    def pick_surveys(self, all_surveys, target_names):
        matched = []
        for s in all_surveys:
            name = s["name"].lower()
            for t in target_names:
                if t.lower() in name:
                    matched.append((s["id"], s["name"]))
        return matched

    # ------------------------------------------------------
    # Main runner for a given target list
    # ------------------------------------------------------
    def run_targets(self, target_names):
        all_surveys = self.fetch_surveys()
        matched = self.pick_surveys(all_surveys, target_names)

        collected_ids = []

        for survey_id, survey_name in matched:
            collected_ids.append(survey_id)

            print(f"\n=== Exporting survey: {survey_name} ({survey_id}) ===")

            # --------------------------------------------------
            # FULL CSV
            # --------------------------------------------------
            out_path = FULL_DATA_DIR / f"full_data_{survey_id}.csv"

            csv_file = export_full_survey_csv(survey_id)
            with out_path.open("w", encoding="utf-8-sig", newline="") as out:
                for line in csv_file:
                    out.write(line)

            print(f"Full CSV saved successfully")

            # --------------------------------------------------
            # SMALL CSV for ALL surveys (Option A)
            # --------------------------------------------------
            csv_file_small = export_full_survey_csv(survey_id)
            rows = extract_ids_and_tokens(csv_file_small)
            save_small_csv(survey_id, rows)

           # Apply sentiment to ALL exported CSVs
            print("Applying sentiment engine to CSV")
            compute_and_write_sentiment_columns(out_path)

        return collected_ids
# ==========================================================
# MAIN EXECUTION LOGIC (Levels + RUN_LEVEL)
# ==========================================================

def main():
    exporter = QualtricsExporter()

    # ------------------------------------------------------
    # If RUN_LEVEL exists → override default behavior
    # ------------------------------------------------------
    if RUN_LEVEL:
        print(f"\nRUN_LEVEL detected → executing selected levels: {RUN_LEVEL}")

        run_levels = parse_run_levels(RUN_LEVEL)
        all_ids = []

        for lvl in run_levels:
            targets = load_level_targets(lvl)
            if not targets:
                print(f"⚠️  No targets found for Level {lvl}. Skipping.")
                continue

            print(f"\n=== Running Level {lvl} ===")
            ids = exporter.run_targets(targets)
            all_ids.extend(ids)

        if all_ids:
            write_env_key_no_quotes(env_path, "SURVEY_ID", ",".join(all_ids))
            print("\nSurvey IDs written to env:", all_ids)

        return  # STOP — do not run remaining levels


    # ------------------------------------------------------
    # DEFAULT: run ALL levels sequentially
    # ------------------------------------------------------
    print("\n=== RUN_LEVEL not set → running ALL levels sequentially ===")

    all_ids = []
    levels = get_all_levels()

    if not levels:
        print("⚠️  No LEVEL<n>_TARGETS found in environment. Nothing to run.")
        return

    for lvl, targets in levels:
        print(f"\n=== Running Level {lvl} ===")
        ids = exporter.run_targets(targets)
        all_ids.extend(ids)

    if all_ids:
        write_env_key_no_quotes(env_path, "SURVEY_ID", ",".join(all_ids))
        print("\nSurvey IDs written to env:", all_ids)



# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    main()
