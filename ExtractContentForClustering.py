#!/usr/bin/env python3
"""
Final script — cleaned path handling (no hardcoded project paths).

Behavior:
- ENV_PATH is set to the environment file constant required by the user.
- PROJECT_ROOT = folder that contains this script (env file) if possible.
- DOC_FOLDER = PROJECT_ROOT/qualtrics_downloaded_files/downloads
- OUTPUT_DIR  = PROJECT_ROOT/Clustered_output
- OUTPUT_CSV  = OUTPUT_DIR/final_analysis_text_personality_<timestamp>.csv

Note: a sample user-uploaded file is available at: /mnt/data/Who_Am_I.csv
(kept here as UPLOADED_FILE_PATH for convenience/inspection.)
"""
import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Document libraries
from docx import Document
from odf.opendocument import load
from odf import teletype

# NLP / embedding / clustering libs
from keybert import KeyBERT
import yake
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from dotenv import load_dotenv

# ensure vader lexicon present
nltk.download('vader_lexicon')

# ----------------------------
# ENV PATH constant (user request)
# ----------------------------
# Per user preference, provide the environment file path constant at the top:
ENV_PATH = Path(__file__).resolve().parent / "my_environment.env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
env_path = Path(ENV_PATH)

# If for some reason the env file doesn't exist, create it (keeps prior dynamic behavior safe)
if not env_path.exists():
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("", encoding="utf-8")

load_dotenv(env_path)

# Project root (folder containing the env file)
PROJECT_ROOT = env_path.parent

# Input docs folder (relative to project root)
DOC_FOLDER = PROJECT_ROOT / "qualtrics_downloaded_files" / "downloads"/"SV_0p7HfPlkL4Qu4g6"

# Output folder (relative to project root)
OUTPUT_DIR = PROJECT_ROOT / "Clustered_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Clear OUTPUT_DIR before writing new file -----
def empty_folder(folder: Path):
    """
    Safely remove all files and subfolders inside `folder`.
    Does NOT delete the folder itself.
    Safety: checks that folder is inside PROJECT_ROOT to avoid accidental deletes.
    """
    try:
        folder = folder.resolve()
    except Exception:
        return

    try:
        proj = PROJECT_ROOT.resolve()
    except Exception:
        return

    # safety check: folder must be under project root
    if proj not in folder.parents and folder != proj:
        print(f"Skipping cleanup for {folder} — not inside project root.")
        return

    if not folder.exists():
        return

    for item in sorted(folder.iterdir(), reverse=True):
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                # remove dir tree contents first
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

# Clear old outputs before creating the new timestamped file
empty_folder(OUTPUT_DIR)
# ----------------------------------------------------

# Final CSV path (timestamped)
OUTPUT_CSV = OUTPUT_DIR / f"final_analysis_text_personality.csv"

# (Optional) path to user-uploaded sample available in session
UPLOADED_FILE_PATH = Path("/mnt/data/Who_Am_I.csv")

# ----------------------------
# Config / Models / Helpers
# ----------------------------
kw_model = KeyBERT("all-MiniLM-L6-v2")
yake_extractor = yake.KeywordExtractor(top=20, stopwords=None)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment = SentimentIntensityAnalyzer()

USELESS_WORDS = {
    "trait", "traits", "helps", "help", "thing", "things", "stuff",
    "someone", "something", "way", "people", "person"
}

BAD_GENERIC = {
    "overall", "however", "summary", "assessment", "reflection", "assignment",
    "activity", "question", "response", "identity", "thing"
}

def is_valid_trait(w):
    w_clean = w.lower().strip()
    if w_clean in USELESS_WORDS:
        return False
    if len(w_clean) < 3:
        return False
    if not re.match(r"^[a-zA-Z]+$", w_clean):
        return False
    return True

def extract_record_id(filename):
    m = re.match(r"(R_[A-Za-z0-9]+)", filename)
    return m.group(1) if m else None

def extract_text(path):
    """
    Support .docx, .txt, .odt
    Returns extracted text or "" on failure.
    """
    try:
        suffix = path.suffix.lower()
    except Exception:
        return ""

    # DOCX
    if suffix == ".docx":
        try:
            doc = Document(path)
            texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n".join(texts)
        except Exception:
            return ""

    # TXT
    if suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    # ODT (odfpy required; imports are at top)
    if suffix == ".odt":
        try:
            doc = load(str(path))
            return teletype.extractText(doc)
        except Exception:
            return ""

    return ""

def extract_keywords(text):
    if not text or len(text.split()) < 5:
        return []

    try:
        kb = [x[0] for x in kw_model.extract_keywords(text, top_n=10)]
    except:
        kb = []

    try:
        yk = [x[0] for x in yake_extractor.extract_keywords(text)]
    except:
        yk = []

    combined = list(set(kb + yk))
    cleaned = []
    for w in combined:
        w_clean = w.lower().strip()
        if len(w_clean) < 3:
            continue
        if w_clean in BAD_GENERIC:
            continue
        cleaned.append(w.strip())

    return cleaned

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
    if n < 3:
        return 1
    best_k, best_score = 1, -1
    for k in range(2, min(max_k, n - 1) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        except:
            continue
    return best_k if best_k > 1 else 1

# ----------------------------
# STAGE 1 — READ FILES
# ----------------------------
records = []
print("\n📄 Reading files (.docx, .odt, .txt)...")

if not DOC_FOLDER.exists():
    print(f"Warning: DOC_FOLDER does not exist: {DOC_FOLDER}")
else:
    for p in sorted(DOC_FOLDER.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".docx", ".odt", ".txt"}:
            continue
        fname = p.name
        text = extract_text(p)
        keywords = extract_keywords(text)
        cq, cqr = assess_content_quality(text, keywords)
        records.append({
            "FileName": fname,
            "RecordID": extract_record_id(fname),
            "Text": text,
            "Keywords": keywords,
            "ContentQuality": cq,
            "ContentQualityReason": cqr
        })

df = pd.DataFrame(records)
print("✔ Files processed.")

# ----------------------------
# STAGE 2 — KEYWORD CONCEPT CLUSTERS
# ----------------------------
all_keywords = sorted({k for sub in df["Keywords"] for k in sub}) if not df.empty else []

if len(all_keywords) > 5:
    kw_vecs = embed_model.encode(all_keywords)
    pca = PCA(n_components=min(50, kw_vecs.shape[1]))
    red = pca.fit_transform(kw_vecs)

    k_concepts = choose_k(red, max_k=10)
    km = KMeans(n_clusters=k_concepts, random_state=42, n_init="auto")
    ids = km.fit_predict(red)

    keyword_to_cluster = {kw: cid for kw, cid in zip(all_keywords, ids)}

    # Create concept labels
    dfreq = Counter()
    for _, r in df.iterrows():
        for k in r["Keywords"]:
            dfreq[k] += 1

    def idf(k):
        return math.log((len(df) + 1) / (dfreq[k] + 0.5))

    concept_labels = {}
    for cid in range(k_concepts):
        words = [kw for kw, c in keyword_to_cluster.items() if c == cid]
        ranked = sorted(words, key=lambda w: -idf(w))
        concept_labels[cid] = ", ".join(ranked[:5]) if ranked else f"Concept {cid}"
else:
    keyword_to_cluster = {}
    concept_labels = {0: "General"}
    k_concepts = 1

# ----------------------------
# STAGE 3 — STUDENT CONCEPT VECTORS
# ----------------------------
concept_ids = sorted(concept_labels.keys())
cmap = {cid: idx for idx, cid in enumerate(concept_ids)}

vectors = []
valid = []
themes = []

for idx, row in df.iterrows():
    vec = np.zeros(len(concept_ids))
    theme = []
    for kw in row["Keywords"]:
        if kw in keyword_to_cluster:
            cid = keyword_to_cluster[kw]
            vec[cmap[cid]] += 1
            theme.append(concept_labels[cid])
    themes.append(", ".join(sorted(set(theme))))
    if vec.sum() > 0:
        vectors.append(vec)
        valid.append(idx)

df["ConceptThemes"] = themes

# ----------------------------
# STAGE 4 — PERSONALITY CLUSTERING
# ----------------------------
if len(vectors) > 1:
    vec_arr = np.vstack(vectors)
    if len({tuple(v) for v in vec_arr}) == 1:
        df["PersonalityClusterID"] = None
        df["PersonalityCluster"] = "General"
    else:
        k_students = choose_k(vec_arr, max_k=8)
        km_s = KMeans(n_clusters=k_students, random_state=42, n_init="auto")
        sids = km_s.fit_predict(vec_arr)

        df["PersonalityClusterID"] = None
        for rid, cid in zip(valid, sids):
            df.at[rid, "PersonalityClusterID"] = cid

        # Generate readable labels
        labels = {}
        for cid in range(k_students):
            rows = df[df["PersonalityClusterID"] == cid]["ConceptThemes"]
            all_t = []
            for t in rows:
                if t:
                    all_t.extend([x.strip() for x in t.split(",")])
            top = [w for w, _ in Counter(all_t).most_common(3)]
            labels[cid] = ", ".join(top) if top else f"Cluster {cid}"
        df["PersonalityCluster"] = df["PersonalityClusterID"].map(labels)
else:
    df["PersonalityClusterID"] = None
    df["PersonalityCluster"] = None

# ----------------------------
# STAGE 5 — MERGED STRENGTHS & WEAKNESSES
# ----------------------------
print("🧠 Applying AI classification for Good/Bad traits...")

positive_vec = embed_model.encode(
    ["strength", "discipline", "creativity", "kindness", "leadership"]
).mean(axis=0)
negative_vec = embed_model.encode(
    ["weakness", "struggle", "problem", "procrastination", "fear"]
).mean(axis=0)

GOODQ, BADQ, GSC, BSC = [], [], [], []

for _, row in df.iterrows():
    kws = row["Keywords"]
    good, bad = [], []
    for kw in kws:
        wclean = kw.lower().strip()
        if not is_valid_trait(wclean):
            continue
        sent = sentiment.polarity_scores(kw)["compound"]
        emb = embed_model.encode([kw])[0]
        sim_pos = np.dot(emb, positive_vec)
        sim_neg = np.dot(emb, negative_vec)
        if sent > 0.25 or sim_pos > sim_neg:
            good.append(kw)
        else:
            bad.append(kw)
    GOODQ.append(", ".join(sorted(set(good))))
    BADQ.append(", ".join(sorted(set(bad))))
    GSC.append(len(set(good)))
    BSC.append(len(set(bad)))

df["GoodQualities"] = GOODQ
df["NeedsImprovement"] = BADQ
df["GoodScore"] = GSC
df["BadScore"] = BSC

# Cleanup optional columns
for col in ("OtherIndicators", "OtherScore"):
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ----------------------------
# STAGE 6 — SEMANTIC CATEGORY MAPPING
# ----------------------------
print("🔍 Grouping similar traits into unified categories...")

def build_trait_categories(traits_list):
    words = []
    for x in traits_list:
        if pd.isna(x) or not x:
            continue
        if not isinstance(x, str):
            continue
        words.extend([w.strip().lower() for w in x.split(",") if w.strip()])
    words = sorted(list(set(words)))
    if len(words) == 0:
        return {}, {}, 1
    emb = embed_model.encode(words)
    k = choose_k(emb, max_k=min(10, len(words)))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(emb)
    kw_to_cluster = {w: int(c) for w, c in zip(words, labels)}
    reps = {}
    for cid in range(k):
        group_words = [w for w, c in kw_to_cluster.items() if c == cid]
        reps[cid] = group_words[0].title() if group_words else f"Group {cid}"
    return kw_to_cluster, reps, k

all_good = df["GoodQualities"].tolist() if "GoodQualities" in df else []
good_map, good_names, k_good = build_trait_categories(all_good)
all_bad = df["NeedsImprovement"].tolist() if "NeedsImprovement" in df else []
bad_map, bad_names, k_bad = build_trait_categories(all_bad)

GOOD_CAT, BAD_CAT = [], []
for idx, row in df.iterrows():
    gwords = []
    if isinstance(row.get("GoodQualities", ""), str) and row["GoodQualities"].strip():
        gwords = [w.strip().lower() for w in row["GoodQualities"].split(",") if w.strip()]
    bwords = []
    if isinstance(row.get("NeedsImprovement", ""), str) and row["NeedsImprovement"].strip():
        bwords = [w.strip().lower() for w in row["NeedsImprovement"].split(",") if w.strip()]
    g_cats = sorted({good_names[good_map[w]] for w in gwords if w in good_map})
    b_cats = sorted({bad_names[bad_map[w]] for w in bwords if w in bad_map})
    GOOD_CAT.append(", ".join(g_cats))
    BAD_CAT.append(", ".join(b_cats))

df["GoodCategory"] = GOOD_CAT
df["BadCategory"] = BAD_CAT

print("✔ Semantic grouping completed.")

# ----------------------------
# FINAL SAVE
# ----------------------------
df["Keywords"] = df["Keywords"].apply(lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else (x or ""))
df.to_csv(str(OUTPUT_CSV), index=False, encoding="utf-8-sig")

print("\n🎉 ALL DONE!")
print(f"📁 Cleaned dataset saved to:\n{OUTPUT_CSV}")
