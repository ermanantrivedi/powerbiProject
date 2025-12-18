import json
import numpy as np
import nltk
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from keybert import KeyBERT
import yake
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

# Gemini optional
try:
    from google import genai
except:
    pass

# Groq
from groq import Groq

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)


# ======================================================================
# ========================== BASE ABSTRACT MODEL ========================
# ======================================================================

class BaseNLPModel(ABC):
    @abstractmethod
    def extract_keywords(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def embed(self, texts: List[str]):
        pass

    @abstractmethod
    def sentiment(self, text: str):
        pass

    @abstractmethod
    def classify_traits(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        pass


# ======================================================================
# ============================= MiniLM MODEL ============================
# ======================================================================

class MiniLMModel(BaseNLPModel):
    def __init__(self):
        self.kw_model = KeyBERT("all-MiniLM-L6-v2")
        self.yake_extractor = yake.KeywordExtractor(top=20, stopwords=None)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentiment_model = SentimentIntensityAnalyzer()

        self.BAD_GENERIC = {
            "overall", "however", "summary", "assessment", "reflection",
            "assignment", "activity", "question", "response", "identity", "thing"
        }

    def extract_keywords(self, text):
        if not text or len(text.split()) < 5:
            return []

        kb = [x[0] for x in self.kw_model.extract_keywords(text, top_n=10)]
        yk = [x[0] for x in self.yake_extractor.extract_keywords(text)]
        return list({w.strip() for w in kb + yk if w.lower() not in self.BAD_GENERIC})

    def embed(self, texts):
        return self.embed_model.encode(texts)

    def sentiment(self, text):
        return self.sentiment_model.polarity_scores(text)["compound"]

    def classify_traits(self, text, keywords):
        good, bad = [], []
        for kw in keywords:
            (good if self.sentiment(kw) >= 0 else bad).append(kw)

        all_traits = sorted(set(good + bad))

        return {
            "GoodTraits": sorted(set(good)),
            "BadTraits": sorted(set(bad)),
            "Keywords": all_traits
        }


# ======================================================================
# ======================= BGE FREE CPU MODEL ============================
# ======================================================================

class BGEFreeCPUModel(BaseNLPModel):
    """
    Free, CPU-only, sentence-semantic trait extraction.
    """

    def __init__(self):
        self.kw_model = KeyBERT("all-MiniLM-L6-v2")
        self.yake_extractor = yake.KeywordExtractor(top=20, stopwords=None)
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.sentiment_model = SentimentIntensityAnalyzer()

        self.GOOD_TRAITS = [
            "hard working and disciplined",
            "intelligent and capable",
            "resilience and perseverance",
            "responsible and dependable",
            "conscientious and detail oriented",
            "social and communicative",
            "self awareness and reflection",
        ]

        self.BAD_TRAITS = [
            "burnout and overworking",
            "poor work life balance",
            "overthinking and anxiety",
            "self pressure and exhaustion",
        ]

        self.good_vecs = self.embed_model.encode(self.GOOD_TRAITS, normalize_embeddings=True)
        self.bad_vecs = self.embed_model.encode(self.BAD_TRAITS, normalize_embeddings=True)

    # -------------------------
    # Normalization helper
    # -------------------------
    def _normalize_trait_label(self, label: str) -> List[str]:
        if " and " in label:
            return [x.strip() for x in label.split(" and ") if x.strip()]
        return [label.strip()]

    # -------------------------
    # Standard interface
    # -------------------------
    def extract_keywords(self, text):
        if not text or len(text.split()) < 5:
            return []
        kb = [x[0] for x in self.kw_model.extract_keywords(text, top_n=10)]
        yk = [x[0] for x in self.yake_extractor.extract_keywords(text)]
        return list(set(kb + yk))

    def embed(self, texts):
        return self.embed_model.encode(texts, normalize_embeddings=True)

    def sentiment(self, text):
        return self.sentiment_model.polarity_scores(text)["compound"]

    # -------------------------
    # Sentence-semantic traits
    # -------------------------
    def classify_traits(self, text, keywords):
        if not text or len(text.strip()) < 20:
            return {"GoodTraits": [], "BadTraits": [], "Keywords": []}

        sentences = nltk.sent_tokenize(text)
        sent_vecs = self.embed(sentences)

        good_found, bad_found = set(), set()

        for vec in sent_vecs:
            g_sim = cosine_similarity([vec], self.good_vecs)[0]
            b_sim = cosine_similarity([vec], self.bad_vecs)[0]

            if g_sim.max() > 0.35 and g_sim.max() > b_sim.max():
                for t in self._normalize_trait_label(self.GOOD_TRAITS[g_sim.argmax()]):
                    good_found.add(t)

            elif b_sim.max() > 0.35 and b_sim.max() > g_sim.max():
                for t in self._normalize_trait_label(self.BAD_TRAITS[b_sim.argmax()]):
                    bad_found.add(t)

        keywords_out = sorted(good_found | bad_found)

        return {
            "GoodTraits": sorted(good_found),
            "BadTraits": sorted(bad_found),
            "Keywords": keywords_out
        }


# ======================================================================
# ============================= GEMINI MODEL ============================
# ======================================================================

class GeminiModel(BaseNLPModel):
    def __init__(self, gemini_client, model_name="gemini-2.5-flash"):
        self.client = gemini_client
        self.model_name = model_name
        self.core = BGEFreeCPUModel()

    def extract_keywords(self, text):
        return self.core.extract_keywords(text)

    def embed(self, texts):
        return self.core.embed(texts)

    def sentiment(self, text):
        return self.core.sentiment(text)

    def classify_traits(self, text, keywords):
        prompt = f"""
Extract abstract personality traits from the reflection below.
Classify them into GoodTraits and BadTraits.
Return STRICT JSON.

TEXT:
{text}
"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt]
            )
            data = json.loads(resp.text)

            good = data.get("GoodTraits", [])
            bad = data.get("BadTraits", [])

            keywords_out = sorted(set(good + bad))

            return {
                "GoodTraits": good,
                "BadTraits": bad,
                "Keywords": keywords_out
            }

        except Exception:
            return self.core.classify_traits(text, keywords)


# ======================================================================
# ========================== LLAMA GROQ MODEL ===========================
# ======================================================================

class LlamaGroqModel(BaseNLPModel):
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.core = BGEFreeCPUModel()

    def extract_keywords(self, text):
        return self.core.extract_keywords(text)

    def embed(self, texts):
        return self.core.embed(texts)

    def sentiment(self, text):
        return self.core.sentiment(text)

    def classify_traits(self, text, keywords):
        prompt = f"""
Extract abstract personality traits from the reflection below.
Classify them into GoodTraits and BadTraits.
Return ONLY valid JSON.

TEXT:
{text}
"""

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            data = json.loads(resp.choices[0].message["content"])

            good = data.get("GoodTraits", [])
            bad = data.get("BadTraits", [])

            return {
                "GoodTraits": good,
                "BadTraits": bad,
                "Keywords": sorted(set(good + bad))
            }

        except Exception:
            return self.core.classify_traits(text, keywords)


# ======================================================================
# =============================== MODEL HUB =============================
# ======================================================================

class ModelHub:
    @staticmethod
    def load(model_name: str, gemini_client=None):
        name = model_name.lower()

        if name == "minilm":
            return MiniLMModel()

        if name in {"bge", "free_cpu"}:
            return BGEFreeCPUModel()

        if name == "gemini":
            if gemini_client is None:
                raise ValueError("Gemini client required.")
            return GeminiModel(gemini_client)

        if name == "llama_groq":
            api_key = os.getenv("GROQ_API_KEY")
            return LlamaGroqModel(api_key)

        raise ValueError(f"Unknown model: {model_name}")
