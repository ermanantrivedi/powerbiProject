#!/usr/bin/env python3
import re
import os
import numpy as np
import pandas as pd
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from models.nlp_models import ModelHub

# Analysis & Clustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Document libraries
from docx import Document
from odf import teletype
from odf.opendocument import load
from odf import teletype
from odf.opendocument import load

# ----------------------------
# 1. CONFIG & DIRECTORY SETUP
# ----------------------------
ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

PROJECT_ROOT = Path(__file__).resolve().parent
DOC_FOLDER = PROJECT_ROOT / "qualtrics_downloaded_files" / "downloads" / "SV_3V0SgOjtT6ZqTxI"
OUTPUT_DIR = PROJECT_ROOT / "Clustered_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOP_PHRASES = {
    "specific", "measurable", "achievable", "relevant", "time-bound", "time bound",
    "achieve", "success", "track", "possible", "resources", "matter", "timeline",
    "goal", "smart", "want", "learn", "need", "final", "reflection", "framework",
    "question", "answer", "draft", "fundamentals", "sentences", "12 weeks", "weeks",
    "someday", "actual", "just", "making", "better", "process", "part", "literally",
    "think", "thought", "helps", "helped", "originally", "idea", "going", "would"
}

# ----------------------------
# 2. NLP MODELS
# ----------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2").lower()
model = ModelHub.load(MODEL_NAME)


# ----------------------------
# 3. HELPERS & QUALITY ASSESSMENT
# ----------------------------

def extract_text(path):
    """Robust text extraction for .docx, .odt, .txt with safe fallbacks."""
    try:
        suffix = path.suffix.lower()

        # ---------- DOCX ----------
        if suffix == ".docx":
            try:
                doc = Document(path)
                texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                return "\n".join(texts)
            except Exception:
                # Fallback: try reading as text / html
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                    return raw
                except Exception:
                    return ""

        # ---------- TXT ----------
        elif suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")

        # ---------- ODT ----------
        elif suffix == ".odt":
            try:
                doc = load(str(path))
                return teletype.extractText(doc)
            except Exception:
                # HARD fallback: extract text.xml directly
                try:
                    with zipfile.ZipFile(path) as z:
                        with z.open("content.xml") as f:
                            xml = f.read().decode("utf-8", errors="ignore")
                            return re.sub(r"<[^>]+>", " ", xml)
                except Exception as e:
                    print(f"⚠️ Failed ODT fallback for {path.name}: {e}")
                    return ""

    except Exception as e:
        print(f"Error reading {path.name}: {e}")
        return ""



def assess_content_quality(text, keywords):
    if not text or not text.strip():
        return "EMPTY", "No content"
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 10:
        return "LOW_CONTENT", "Very few words"
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    if unique_ratio < 0.3:
        return "GARBAGE", "Highly repetitive"
    if len(keywords) == 0:
        return "LOW_CONTENT", "No meaningful keywords"
    return "GOOD", "OK"


def choose_k(X, max_k=10):
    n = X.shape[0]
    if n < 3: return 1
    best_k, best_score = 1, -1
    for k in range(2, min(max_k, n - 1) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)
            if len(set(labels)) < 2: continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        except:
            continue
    return best_k if best_k > 1 else 1


def deduplicate_keywords(keywords, model_hub, threshold=0.70):
    if not keywords: return []
    keywords = [k.strip() for k in keywords if len(k) > 3 and k.lower() not in STOP_PHRASES]
    if not keywords: return []

    embeddings = model_hub.embed(keywords)
    sim_matrix = cosine_similarity(embeddings)
    kept, indices_to_ignore = [], set()

    for i in range(len(keywords)):
        if i in indices_to_ignore: continue
        for j in range(i + 1, len(keywords)):
            if sim_matrix[i, j] > threshold:
                if len(keywords[j]) > len(keywords[i]):
                    indices_to_ignore.add(i)
                else:
                    indices_to_ignore.add(j)
        if i not in indices_to_ignore: kept.append(keywords[i])
    return kept


def get_clean_smart_sections(text):
    # We simplify the regex to focus on the labels regardless of what symbols (:, ?, -) follow them.
    # We look for the label and capture everything until the next SMART label or the end of the text.
    patterns = {
        "Specific": r"(?i)Specific\b(.*?)(?=Measurable|Achievable|Relevant|Timebound|Time Bound|Final|$)",
        "Measurable": r"(?i)Measurable\b(.*?)(?=Specific|Achievable|Relevant|Timebound|Time Bound|Final|$)",
        "Achievable": r"(?i)Achievable\b(.*?)(?=Specific|Measurable|Relevant|Timebound|Time Bound|Final|$)",
        "Relevant": r"(?i)Relevant\b(.*?)(?=Specific|Measurable|Achievable|Timebound|Time Bound|Final|$)",
        "Timebound": r"(?i)(?:Time[- ]Bound|Timeline)\b(.*?)(?=Specific|Measurable|Achievable|Relevant|Final|$)"
    }

    sections = {}
    found_any = False

    # Pre-process text to remove excessive whitespace/newlines that break re.DOTALL in some environments
    normalized_text = " ".join(text.split())

    for label, pattern in patterns.items():
        # Using re.IGNORECASE and re.DOTALL to be safe
        match = re.search(pattern, normalized_text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()

            # Clean up leading punctuation like ':', '-', or '?' often left behind
            content = re.sub(r"^[?:\-—\s.12345]+", "", content).strip()

            if content:
                sections[label] = content
                found_any = True

    return sections if found_any else None


# ----------------------------
# 4. MAIN EXECUTION
# ----------------------------
def run_analysis():
    print(f"🚀 Running Analysis with Quality Checks & Smart Clustering...")

    total_files = 0
    analyzed_files = 0
    skipped_files = 0

    records = []


    if not DOC_FOLDER.exists():
        print(f"Error: {DOC_FOLDER} not found.")
        return

    # STAGE 1: File Processing
    for p in sorted(DOC_FOLDER.iterdir()):
        if p.suffix.lower() not in {".docx", ".odt", ".txt"}:
            continue

        total_files += 1
        raw_text = extract_text(p)

        if raw_text.strip():
            analyzed_files += 1
        else:
            skipped_files += 1

        # Keyword Extraction for Quality Assessment
        all_kws = model.extract_keywords(raw_text)
        cq, cqr = assess_content_quality(raw_text, all_kws)

        # SMART Sectioning
        sections = get_clean_smart_sections(raw_text)
        smart_data = {}

        labels = ["Specific", "Measurable", "Achievable", "Relevant", "Timebound"]
        if sections:
            for label in labels:
                content = sections.get(label, "")
                raw_kws = model.extract_keywords(content)
                clean_kws = deduplicate_keywords(raw_kws, model)
                smart_data[f"{label}_Keywords"] = ", ".join(clean_kws)
                smart_data[f"{label}_Score"] = len(clean_kws)
        else:
            for label in labels:
                smart_data[f"{label}_Keywords"], smart_data[f"{label}_Score"] = "", 0

        records.append({
            "FileName": p.stem,
            "Text": raw_text[:1000],  # Storing more text for potential clustering
            "ContentQuality": cq,
            "ContentQualityReason": cqr,
            **smart_data,
            "SMART_Total_Score": sum(smart_data.get(f"{l}_Score", 0) for l in labels)
        })

    df = pd.DataFrame(records)

    # Optional: Example of using choose_k if you have embeddings for the whole dataset
    # if not df.empty:
    #    embeddings = model.embed(df['Text'].tolist())
    #    optimal_k = choose_k(embeddings)
    #    print(f"Suggested clusters: {optimal_k}")

    # Save
    FINAL_CSV = OUTPUT_DIR / "smart_analysis_level1.csv"
    df.to_csv(str(FINAL_CSV), index=False, encoding="utf-8-sig")
    print("\n📊 File Processing Summary")
    print(f"   Total files found   : {total_files}")
    print(f"   Files analyzed      : {analyzed_files}")
    print(f"   Files skipped/empty : {skipped_files}\n")
    print(f"✅ DONE! Results saved to: {FINAL_CSV}")


if __name__ == "__main__":
    run_analysis()