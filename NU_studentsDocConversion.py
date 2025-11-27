"""
Updated pipeline script
- Uses Method 2 (SentenceTransformer auto-download) with deepseek-ai/deepseek-embedding-v2
- Adds similarity-margin thresholds and neutral class
- Normalizes embeddings for cosine similarity
- Keeps your original pipeline structure; cleans & improves some steps
- Saves final CSV to OUTPUT_CSV

Adjust DOC_FOLDER and OUTPUT_CSV paths as needed.
"""

import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from docx import Document
from keybert import KeyBERT
import yake
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# =========================
# CONFIG
# =========================
DOC_FOLDER = r"C:\Users\erman\OneDrive\Desktop\Bellini_Centre\USF_Qualtrics_Downloads\test"
OUTPUT_CSV = r"C:\Users\erman\OneDrive\Desktop\Bellini_Centre\USF_Qualtrics_Downloads\final_analysis_text_personality_deepseek.csv"

# optional sample/eval CSV (you uploaded this file)
SAMPLE_CSV = "/mnt/data/Who_Am_I.csv"  # <-- included for convenience; used only if exists

# =========================
# MODELS
# =========================
# Keybert may internally use a small transformer; leave as-is for keyword extraction
kw_model = KeyBERT("all-MiniLM-L6-v2")
yake_extractor = yake.KeywordExtractor(top=20, stopwords=None)

# Auto-download DeepSeek embedding model (Method 2)
print("📥 Loading embedding model (this will auto-download if needed)...")
embed_model = SentenceTransformer("deepseek-ai/deepseek-embedding-v2", device="cpu")
print("✔ Embedding model ready.")

# Sentiment
sentiment = SentimentIntensityAnalyzer()

# =========================
# CLEANING & TRAIT FILTERS
# =========================
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
    # allow multi-word traits that are letters and spaces (e.g., "team player")
    if not re.match(r"^[a-zA-Z\s\-]+$", w_clean):
        return False
    return True

# =========================
# HELPERS
# =========================
def extract_record_id(filename):
    m = re.match(r"(R_[A-Za-z0-9]+)", filename)
    return m.group(1) if m else None

def extract_text(path):
    try:
        doc = Document(path)
        texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(texts)
    except Exception:
        return ""

def extract_keywords(text):
    if not text or len(text.split()) < 5:
        return []
    combined = []
    try:
        kb = [x[0] for x in kw_model.extract_keywords(text, top_n=10)]
    except Exception:
        kb = []
    try:
        yk = [x[0] for x in yake_extractor.extract_keywords(text)]
    except Exception:
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
        except Exception:
            continue
    return best_k if best_k > 1 else 1

# =========================
# STAGE 1 — READ FILES
# =========================
records = []
print("\n📄 Reading DOCX files...")

for fname in os.listdir(DOC_FOLDER):
    if not fname.lower().endswith(".docx"):
        continue
    path = os.path.join(DOC_FOLDER, fname)
    text = extract_text(path)
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
print(f"✔ Files processed: {len(df)}")

# =========================
# STAGE 2 — KEYWORD CONCEPT CLUSTERS
# =========================
all_keywords = sorted({k for sub in df["Keywords"] for k in sub})

if len(all_keywords) > 5:
    kw_vecs = embed_model.encode(all_keywords, convert_to_numpy=True, show_progress_bar=True)
    # normalize for stable cosine similarity
    kw_vecs = util.normalize_embeddings(kw_vecs)
    pca = PCA(n_components=min(50, kw_vecs.shape[1]))
    red = pca.fit_transform(kw_vecs)
    k_concepts = choose_k(red, max_k=10)
    km = KMeans(n_clusters=k_concepts, random_state=42, n_init="auto")
    ids = km.fit_predict(red)
    keyword_to_cluster = {kw: cid for kw, cid in zip(all_keywords, ids)}
    # frequency for idf-like ranking
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

# =========================
# STAGE 3 — STUDENT CONCEPT VECTORS
# =========================
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

# =========================
# STAGE 4 — PERSONALITY CLUSTERING
# =========================
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

# =========================
# STAGE 5 — MERGED STRENGTHS & WEAKNESSES
# =========================
print("🧠 Applying AI classification for Good/Bad/Neutral traits...")

# Build positive/negative reference vectors using the same model
pos_ref_texts = ["strength", "discipline", "creativity", "kindness", "leadership"]
neg_ref_texts = ["weakness", "struggle", "problem", "procrastination", "fear"]

pos_ref = embed_model.encode(pos_ref_texts, convert_to_numpy=True, show_progress_bar=False).mean(axis=0)
neg_ref = embed_model.encode(neg_ref_texts, convert_to_numpy=True, show_progress_bar=False).mean(axis=0)
pos_ref = util.normalize_embeddings(pos_ref.reshape(1, -1))[0]
neg_ref = util.normalize_embeddings(neg_ref.reshape(1, -1))[0]

GOODQ, BADQ, GSC, BSC = [], [], [], []

# threshold settings
SIM_MARGIN = 0.06   # cosine margin required to classify as good/bad (tune 0.05-0.10)
SENT_POS = 0.25
SENT_NEG = -0.25

for _, row in df.iterrows():
    kws = row["Keywords"]
    good, bad = [], []
    for kw in kws:
        wclean = kw.lower().strip()
        if not is_valid_trait(wclean):
            continue
        # sentiment
        sent = sentiment.polarity_scores(kw)["compound"]
        # semantic similarity (use normalized cosine)
        emb = embed_model.encode([kw], convert_to_numpy=True, show_progress_bar=False)[0]
        emb_norm = util.normalize_embeddings(emb.reshape(1, -1))[0]
        sim_pos = float(np.dot(emb_norm, pos_ref))
        sim_neg = float(np.dot(emb_norm, neg_ref))
        diff = sim_pos - sim_neg
        # Classification with margin and sentiment
        if (sent >= SENT_POS) or (diff > SIM_MARGIN):
            good.append(kw)
        elif (sent <= SENT_NEG) or (diff < -SIM_MARGIN):
            bad.append(kw)
        else:
            # neutral/uncertain -> ignore for Good/Bad lists (or record as neutral if you want)
            pass
    GOODQ.append(", ".join(sorted(set(good))))
    BADQ.append(", ".join(sorted(set(bad))))
    GSC.append(len(set(good)))
    BSC.append(len(set(bad)))

df["GoodQualities"] = GOODQ
df["NeedsImprovement"] = BADQ
df["GoodScore"] = GSC
df["BadScore"] = BSC

# =========================
# STAGE 6 — AUTOMATIC SEMANTIC CATEGORY MAPPING
# =========================
print("🔍 Grouping similar traits into unified categories...")

def build_trait_categories(traits_list, min_clusters=1, max_clusters=10):
    # Flatten & clean
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
    # Embed and normalize
    emb = embed_model.encode(words, convert_to_numpy=True, show_progress_bar=False)
    emb = util.normalize_embeddings(emb)
    # Pick number of clusters automatically
    k = choose_k(emb, max_k=min(max_clusters, len(words)))
    # Cluster
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(emb)
    # Build mapping: keyword → cluster_id
    kw_to_cluster = {w: int(c) for w, c in zip(words, labels)}
    # Generate readable category names (use most frequent / first word)
    reps = {}
    for cid in range(k):
        group_words = [w for w, c in kw_to_cluster.items() if c == cid]
        reps[cid] = group_words[0].title() if group_words else f"Group {cid}"
    return kw_to_cluster, reps, k

# Build mappings for GOOD qualities
all_good = df["GoodQualities"].tolist()
good_map, good_names, k_good = build_trait_categories(all_good)

# Build mappings for BAD qualities
all_bad = df["NeedsImprovement"].tolist()
bad_map, bad_names, k_bad = build_trait_categories(all_bad)

# Assign category per student
GOOD_CAT, BAD_CAT = [], []

for idx, row in df.iterrows():
    gwords = []
    if isinstance(row["GoodQualities"], str) and row["GoodQualities"].strip():
        gwords = [w.strip().lower() for w in row["GoodQualities"].split(",") if w.strip()]
    bwords = []
    if isinstance(row["NeedsImprovement"], str) and row["NeedsImprovement"].strip():
        bwords = [w.strip().lower() for w in row["NeedsImprovement"].split(",") if w.strip()]
    # GOOD category names
    g_cats = sorted({good_names[good_map[w]] for w in gwords if w in good_map})
    # BAD category names
    b_cats = sorted({bad_names[bad_map[w]] for w in bwords if w in bad_map})
    GOOD_CAT.append(", ".join(g_cats))
    BAD_CAT.append(", ".join(b_cats))

df["GoodCategory"] = GOOD_CAT
df["BadCategory"] = BAD_CAT

print("✔ Semantic grouping completed.")

# =========================
# FINAL SAVE
# =========================
df["Keywords"] = df["Keywords"].apply(lambda x: ", ".join(x))
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("\n🎉 ALL DONE!")
print(f"📁 Cleaned dataset saved to:\n{OUTPUT_CSV}")

# =========================
# OPTIONAL: quick sample read / eval if your uploaded CSV exists
# =========================
if os.path.exists(SAMPLE_CSV):
    try:
        sample_df = pd.read_csv(SAMPLE_CSV)
        print(f"\nℹ Found sample CSV at {SAMPLE_CSV} — {len(sample_df)} rows (not automatically evaluated).")
        # If you want automated evaluation, we can add a small routine here (requires human labels).
    except Exception as e:
        print("⚠️ Could not read sample CSV:", e)
