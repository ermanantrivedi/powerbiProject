# Runner.py — final: re-includes previously-removed steps (PullDocFiles, ExtractContentForClustering)
# Environment file path constant (required by project guidelines)

import os
import sys
import subprocess
import logging
import shutil
import re
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

# optional libs
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# -----------------------------
# Environment
# -----------------------------
# create env file if missing (keeps parity with your prior setup)
BASE_DIR = Path(__file__).resolve().parent

# Load environment file if present (same-dir)
ENV_PATH = BASE_DIR / "my_environment.env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# -----------------------------
# Base & folders
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Re-included scripts (ordered)
SCRIPTS = [
    BASE_DIR / "PullSurveyRecordsFromQualtrics.py",      # produce CSV
    BASE_DIR / "PullDocFiles.py",                        # pull doc files
    BASE_DIR / "ExtractContentForClustering.py",         # extract content
]

SRC_FULL_FOLDER = Path(
    os.getenv(
        "SRC_FULL_FOLDER",
        BASE_DIR / "qualtrics_downloaded_files/csv/full_data",
    )
)

ONEDRIVE_FULL_DIR = Path(
    os.getenv(
        "ONEDRIVE_FULL_DIR",
        r"C:\Users\erman\OneDrive - University of South Florida\PowerBISource\CSVs\full_csv",
    )
)

CSV_PATHS = os.getenv("CSV_PATHS", "").strip()

# -----------------------------
# Logging
# -----------------------------
LOG_FILE = BASE_DIR / "nightly_runner.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)

# -----------------------------
# Sentiment thresholds and lexicons
# -----------------------------
THRESHOLD_POSITIVE = 0.10
THRESHOLD_NEGATIVE = 0.00
STRONG_POSITIVE = 0.40
STRONG_NEGATIVE = -0.40

# lexicon: helpful positive, 'nope' intentionally not in negative list
_POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "love", "liked", "like",
    "pleasant", "happy", "satisfied", "enjoy", "best", "positive",
    "amazing", "fantastic", "helpful", "help", "useful", "easy",
    "recommend", "recommended", "works", "working"
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "dislike", "worse", "worst",
    "poor", "unsatisfied", "angry", "frustrat", "problem", "issue",
    "not", "no", "never", "difficult", "disappoint", "complain",
    "didn't", "didnt", "not_good", "not_helpful"
}

# -----------------------------
# Utilities
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


def latest_csv_in_folder(folder: Path) -> Optional[Path]:
    if not folder.exists() or not folder.is_dir():
        logging.warning("Folder does not exist: %s", folder)
        return None
    csv_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not csv_files:
        logging.warning("No CSV files found in: %s", folder)
        return None
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return csv_files[0]


def run_script(script_path: Path, timeout: int | None = None) -> int:
    if not script_path.exists():
        logging.error("Script not found: %s", script_path)
        return 2

    python_exe = sys.executable
    cmd = [python_exe, str(script_path)]
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8:replace"
    child_env["PYTHONUTF8"] = "1"

    logging.info("Running: %s", script_path)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        logging.exception("Timeout when running %s", script_path)
        return 3
    except Exception:
        logging.exception("Failed to run: %s", script_path)
        return 4

    if result.stdout:
        logging.info("=== STDOUT (%s) ===\n%s", script_path.name, result.stdout.strip())
    if result.stderr:
        logging.error("=== STDERR (%s) ===\n%s", script_path.name, result.stderr.strip())

    return result.returncode


# -----------------------------
# Sentiment helpers
# -----------------------------
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


# -----------------------------
# Core: ensure sentiment columns
# -----------------------------
def ensure_sentiment_columns(csv_path: Path) -> None:
    logging.info("Processing sentiment for: %s", csv_path)
    if pd is None:
        logging.error("pandas is required but not installed.")
        return
    if not csv_path.exists():
        logging.warning("CSV not found: %s", csv_path)
        return

    # read safely
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(fh, dtype=str)
    except Exception:
        logging.exception("Failed reading CSV: %s", csv_path)
        return

    original_cols = list(df.columns)

    # detect Q6 column (prefer Q6.1)
    q6_col = find_q6_column(df)
    if q6_col is None:
        logging.warning("Q6 column not detected; creating 'Q6_1' empty column.")
        df["Q6_1"] = pd.NA
    else:
        logging.info("Detected Q6 source column: '%s' -> using as 'Q6_1'.", q6_col)
        df["Q6_1"] = df[q6_col]

    # ensure sentiment columns exist
    if "Sentiment_Score" not in df.columns:
        df["Sentiment_Score"] = pd.NA
    if "Sentiment_Label" not in df.columns:
        df["Sentiment_Label"] = pd.NA

    # show debug samples
    non_empty_mask = df["Q6_1"].notna() & df["Q6_1"].astype(str).str.strip().ne("")
    non_empty_count = int(non_empty_mask.sum())
    logging.info("Rows: %d, non-empty Q6_1: %d", len(df), non_empty_count)
    sample_texts = df.loc[non_empty_mask, "Q6_1"].astype(str).head(20).tolist()
    if sample_texts:
        logging.info("Sample Q6.1 texts (up to 20):")
        for i, t in enumerate(sample_texts, 1):
            preview = (t[:300] + "…") if len(t) > 300 else t
            logging.info("  %2d) %s", i, preview)

    # decide whether to prefer lexicon
    use_lexicon_global = False
    if TextBlob is None:
        logging.info("TextBlob not available — using lexicon primarily.")
        use_lexicon_global = True
    elif non_empty_count > 0:
        nonzero = 0
        total = 0
        for i in df.loc[non_empty_mask].index:
            tbp = score_with_textblob(df.at[i, "Q6_1"])
            total += 1
            if tbp is not None and abs(tbp) > 1e-9:
                nonzero += 1
        logging.info("TextBlob: non-empty=%d non-zero=%d", total, nonzero)
        if total > 0 and (nonzero / total) < 0.15:
            logging.info("TextBlob has low signal — preferring lexicon for many rows.")
            use_lexicon_global = True

    # compute per-row
    sample_debug = []
    scores = []
    labels = []
    for idx in df.index:
        cur_score = df.at[idx, "Sentiment_Score"]
        cur_label = df.at[idx, "Sentiment_Label"]

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

        if len(sample_debug) < 20 and text is not None and str(text).strip() != "":
            sample_debug.append((str(text)[:300], final_score, final_label, {"tb": tb, "lex": lex}))

    df["Sentiment_Score"] = scores
    df["Sentiment_Label"] = labels

    # ensure numeric and defaults
    try:
        df["Sentiment_Score"] = pd.to_numeric(df["Sentiment_Score"], errors="coerce").round(3)
    except Exception:
        logging.exception("Failed coercing Sentiment_Score to numeric.")
    df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0.0)
    df["Sentiment_Label"] = df["Sentiment_Label"].fillna("Neutral").astype(str)

    if sample_debug:
        logging.info("Sample computed sentiment (text preview, score, label, sources) up to 20:")
        for i, (t, s, l, srcs) in enumerate(sample_debug, 1):
            logging.info("  %2d) %s | score=%s | label=%s | tb=%s lex=%s",
                         i, (t + ("…" if len(t) > 300 else "")), s, l, srcs.get("tb"), srcs.get("lex"))

    # reorder columns
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
        df.to_csv(tmp_path, index=False, encoding="utf-8")
        tmp_path.replace(csv_path)
        logging.info("Wrote/updated sentiment columns in %s", csv_path)
    except Exception:
        logging.exception("Failed writing updated CSV for %s", csv_path)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# -----------------------------
# Resolve sources and helpers
# -----------------------------
def resolve_sources() -> List[Path]:
    if CSV_PATHS:
        parts = [p.strip() for p in CSV_PATHS.split(",") if p.strip()]
        resolved = []
        for p in parts:
            path = Path(p)
            if not path.is_absolute():
                path = (BASE_DIR / path).resolve()
            resolved.append(path)
        logging.info("Using explicit CSV_PATHS: %s", resolved)
        return resolved

    full_csv = latest_csv_in_folder(SRC_FULL_FOLDER)
    resolved = [full_csv] if full_csv else []
    logging.info("Resolved source CSVs: %s", resolved)
    return resolved


def clear_folder(folder: Path) -> None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
        logging.info("Cleared folder: %s", folder)
    except Exception:
        logging.exception("Failed clearing folder: %s", folder)


def copy_to_target(src: Path, target_dir: Path) -> Optional[Path]:
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.exception("Cannot create target dir: %s", target_dir)
        return None
    dest = target_dir / src.name
    try:
        shutil.copy2(str(src), str(dest))
        logging.info("Copied: %s -> %s", src, dest)
        return dest
    except Exception:
        logging.exception("Failed copying %s to %s", src, dest)
        return None


# -----------------------------
# MAIN
# -----------------------------
def main():
    start = datetime.now(timezone.utc)
    logging.info("=== Runner started ===")

    # Run the scripts in order
    for script in SCRIPTS:
        rc = run_script(script)
        if rc != 0:
            logging.error("Script failed (%s). Stopping pipeline.", rc)
            return

    try:
        sources = resolve_sources()
        if not sources:
            logging.warning("No resolved CSVs, skipping sentiment and copy.")
            return

        for src in sources:
            try:
                ensure_sentiment_columns(src)
            except Exception:
                logging.exception("Error ensuring sentiment for %s", src)

        clear_folder(ONEDRIVE_FULL_DIR)
        copy_to_target(sources[0], ONEDRIVE_FULL_DIR)

    except Exception:
        logging.exception("Error resolving/copying CSV files.")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("=== Finished in %.1f seconds ===", elapsed)


if __name__ == "__main__":
    main()
