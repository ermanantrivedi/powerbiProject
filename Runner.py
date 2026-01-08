# runner.py — runs nightly jobs & copies latest CSVs to OneDrive folders (with folder clearing)
# ENV file constant required by project (do not remove)

import os
import sys
import subprocess
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

# -----------------------------------------------------
# BASE DIRECTORY (same folder where this runner lives)
# -----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Load .env from runner folder (relative path)
ENV_PATH = BASE_DIR / "my_environment.env"
if not ENV_PATH.exists():
    ENV_PATH.write_text("", encoding="utf-8")
load_dotenv(ENV_PATH)


# -----------------------------------------------------
# SCRIPTS TO RUN (relative paths)
# -----------------------------------------------------
SCRIPTS = [
    BASE_DIR / "PullSurveyRecordsFromQualtrics.py",
    BASE_DIR / "PullDocFiles.py",
    BASE_DIR / "ExtractContentForClustering.py",
    BASE_DIR / "SMART_ContentAnalysis_level1.py",
    BASE_DIR / "ReadFilesExistingSurveys.py"
]

# -----------------------------------------------------
# LOCAL SOURCE FOLDERS (where your scripts write outputs)
# -----------------------------------------------------
SRC_FULL_FOLDER = Path(
    os.getenv(
        "SRC_FULL_FOLDER",
        BASE_DIR / "qualtrics_downloaded_files/csv/full_data",
    )
)

SRC_CLUSTERED_FOLDER = Path(
    os.getenv(
        "SRC_CLUSTERED_FOLDER",
        BASE_DIR / "Clustered_output",
    )
)

# -----------------------------------------------------
# ONEDRIVE TARGET FOLDERS (your OneDrive paths)
# -----------------------------------------------------
ONEDRIVE_FULL_DIR = Path(
    os.getenv(
        "ONEDRIVE_FULL_DIR",
        r"C:\Users\erman\OneDrive - University of South Florida\PowerBISource\CSVs\full_csv",
    )
)

ONEDRIVE_KEYWORD_DIR = Path(
    os.getenv(
        "ONEDRIVE_KEYWORD_DIR",
        r"C:\Users\erman\OneDrive - University of South Florida\PowerBISource\CSVs\keyword_csv",
    )
)

# Optional: explicit CSV paths (comma-separated): "fullpath1,fullpath2"
CSV_PATHS = os.getenv("CSV_PATHS", "").strip()

# -----------------------------------------------------
# LOGGING
# -----------------------------------------------------
LOG_FILE = BASE_DIR / "nightly_runner.log"
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# -----------------------------------------------------
# RUN SCRIPT WITH UTF-8 FIX
# -----------------------------------------------------
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
        logging.exception("Timeout: %s", script_path)
        return 3
    except Exception:
        logging.exception("Failed to run: %s", script_path)
        return 4

    if result.stdout:
        logging.info("=== STDOUT (%s) ===\n%s", script_path.name, result.stdout.strip())
    if result.stderr:
        logging.error("=== STDERR (%s) ===\n%s", script_path.name, result.stderr.strip())

    return result.returncode


# -----------------------------------------------------
# Clear folder before writing new files
# -----------------------------------------------------
def clear_folder(folder: Path) -> None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
        logging.info("Cleared folder: %s", folder)
    except Exception:
        logging.exception("Failed clearing folder: %s", folder)


# -----------------------------------------------------
# Copy ANY file (no filtering)
# -----------------------------------------------------
def copy_all_files(src_folder: Path, target_folder: Path) -> None:
    try:
        target_folder.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.exception("Cannot create target dir: %s", target_folder)
        return

    if not src_folder.exists() or not src_folder.is_dir():
        logging.warning("Source folder missing: %s", src_folder)
        return

    files = [p for p in src_folder.iterdir() if p.is_file()]
    if not files:
        logging.warning("No files found in: %s", src_folder)
        return

    for file in files:
        try:
            dest = target_folder / file.name
            shutil.copy2(str(file), str(dest))
            logging.info("Copied: %s -> %s", file, dest)
        except Exception:
            logging.exception("Failed copying %s to %s", file, dest)


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    start = datetime.now(timezone.utc)
    logging.info("=== Nightly runner started ===")

    # Run required scripts
    for script in SCRIPTS:
        rc = run_script(script)
        if rc != 0:
            logging.error("Script failed (%s). Stopping.", rc)
            break

    # Copy ALL files from both source folders
    try:
        # Clear destination folders first
        clear_folder(ONEDRIVE_FULL_DIR)
        clear_folder(ONEDRIVE_KEYWORD_DIR)

        # Copy ALL files (no filtering)
        copy_all_files(SRC_FULL_FOLDER, ONEDRIVE_FULL_DIR)
        copy_all_files(SRC_CLUSTERED_FOLDER, ONEDRIVE_KEYWORD_DIR)

    except Exception:
        logging.exception("Error copying files.")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("=== Finished in %.1f seconds ===", elapsed)


if __name__ == "__main__":
    main()
