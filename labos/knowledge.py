"""
LabOS Knowledge Base — TF-IDF template storage with optional Mem0 enhanced memory.

Stores successful problem-solving templates and retrieves similar ones
using TF-IDF cosine similarity.  Falls back gracefully when Mem0 is unavailable.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

try:
    from mem0 import Memory, MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False


# ---------------------------------------------------------------------------
# Traditional TF-IDF Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """TF-IDF knowledge base for storing and retrieving problem-solving templates."""

    def __init__(self, gemini_model=None, storage_dir: Optional[str] = None):
        self.templates: List[Dict] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.template_vectors = None
        self.gemini_model = gemini_model

        if storage_dir is None:
            storage_dir = os.path.join(Path.home(), ".labos", "knowledge")
        self.knowledge_file = Path(storage_dir) / "knowledge_base.json"
        self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)

        self.load_knowledge_base()

    # -- LLM-assisted summarisation ----------------------------------------

    def summarize_reasoning_process(self, question: str, reasoning: str, answer: str) -> str:
        """Summarise key reasoning steps using the LLM (if available)."""
        if not self.gemini_model:
            return self._manual_summary(question, reasoning)

        prompt = (
            "Summarise the key reasoning steps from the following analysis. "
            "Focus on principles and methodology, not the specific answer.\n\n"
            f"Task: {question}\n\nReasoning:\n{reasoning}\n\n"
            "Provide 4-5 key reasoning principles (concise)."
        )
        try:
            response = self.gemini_model([{"role": "user", "content": prompt}])
            text = getattr(response, "content", None) or (
                response.get("content") if isinstance(response, dict) else str(response)
            )
            if text and len(text.strip()) > 50:
                return text.strip()[:800]
        except Exception as exc:
            logger.warning("LLM summarisation failed: %s", exc)

        return self._manual_summary(question, reasoning)

    @staticmethod
    def _manual_summary(question: str, reasoning: str) -> str:
        q = question.lower()
        if any(t in q for t in ("data", "analysis", "csv", "plot")):
            return "Applied systematic data analysis with visualisation and statistical interpretation."
        if any(t in q for t in ("code", "script", "programming")):
            return "Used systematic programming with modular design and error handling."
        if any(t in q for t in ("biomedical", "biology", "medical")):
            return "Applied biomedical reasoning with evidence-based analysis."
        return "Applied systematic problem-solving with logical reasoning."

    # -- Template CRUD -----------------------------------------------------

    def add_template(self, task: str, thought: str, outcome: str, domain: str = "general"):
        key_reasoning = self.summarize_reasoning_process(task, thought, outcome)
        template = {
            "task": task,
            "key_reasoning": key_reasoning,
            "domain": domain,
            "keywords": self.extract_keywords(task),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.templates.append(template)
        self._rebuild_vectors()
        if len(self.templates) > 1000:
            self.templates = self.templates[-1000:]
            self._rebuild_vectors()
        self.save_knowledge_base()
        return {"success": True, "message": f"Template added. Total: {len(self.templates)}"}

    def retrieve_similar_templates(self, task: str, top_k: int = 3) -> List[Dict]:
        if not self.templates or self.template_vectors is None:
            return []
        try:
            vec = self.vectorizer.transform([task])
            sims = cosine_similarity(vec, self.template_vectors).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_idx:
                if sims[idx] > 0.1:
                    t = self.templates[idx].copy()
                    t["similarity"] = float(sims[idx])
                    results.append(t)
            return results
        except Exception as exc:
            logger.warning("Template retrieval failed: %s", exc)
            return []

    def search_templates_by_keyword(self, keyword: str) -> List[Dict]:
        kw = keyword.lower()
        return [
            t for t in self.templates
            if kw in t["task"].lower()
            or kw in t.get("key_reasoning", "").lower()
            or kw in " ".join(t.get("keywords", [])).lower()
        ]

    # -- Keyword extraction ------------------------------------------------

    def extract_keywords(self, text: str) -> List[str]:
        if self.gemini_model:
            try:
                prompt = (
                    "Extract 3-8 important keywords from the text below. "
                    "Return only keywords separated by commas.\n\n"
                    f"Text: {text}\n\nKeywords:"
                )
                resp = self.gemini_model([{"role": "user", "content": prompt}])
                raw = getattr(resp, "content", None) or (
                    resp.get("content") if isinstance(resp, dict) else str(resp)
                )
                kws = [k.strip().lower() for k in raw.split(",") if len(k.strip()) > 2]
                if kws:
                    return kws
            except Exception:
                pass

        # Fallback: static keyword list
        bio_keywords = [
            "data", "analysis", "code", "script", "algorithm", "model",
            "biomedical", "biology", "medical", "research", "genomics",
            "protein", "enzyme", "drug", "pathway", "sequencing",
            "machine_learning", "deep_learning", "prediction", "simulation",
        ]
        text_lower = text.lower()
        return [kw for kw in bio_keywords if kw in text_lower]

    # -- Persistence -------------------------------------------------------

    def _rebuild_vectors(self):
        if not self.templates:
            return
        texts = [
            f"{t['task']} {t.get('key_reasoning', '')} {' '.join(t.get('keywords', []))}"
            for t in self.templates
        ]
        try:
            self.template_vectors = self.vectorizer.fit_transform(texts)
        except Exception:
            self.template_vectors = None

    def save_knowledge_base(self):
        try:
            with open(self.knowledge_file, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to save knowledge base: %s", exc)

    def load_knowledge_base(self):
        try:
            if self.knowledge_file.exists():
                with open(self.knowledge_file, "r", encoding="utf-8") as f:
                    self.templates = json.load(f)
                self._rebuild_vectors()
                logger.info("Loaded knowledge base with %d templates", len(self.templates))
            else:
                logger.info("No existing knowledge base — starting fresh")
        except Exception as exc:
            logger.warning("Failed to load knowledge base: %s", exc)
            self.templates = []


# ---------------------------------------------------------------------------
# Mem0 Enhanced Knowledge Base
# ---------------------------------------------------------------------------

class Mem0EnhancedKnowledgeBase:
    """Wraps Mem0 semantic memory with a TF-IDF fallback."""

    def __init__(
        self,
        gemini_model=None,
        use_mem0_platform: bool = False,
        mem0_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        storage_dir: Optional[str] = None,
    ):
        self.gemini_model = gemini_model
        self.mem0_enabled = False
        self.memory = None
        self.fallback_kb: Optional[KnowledgeBase] = None

        if MEM0_AVAILABLE:
            try:
                if use_mem0_platform and mem0_api_key:
                    self.memory = MemoryClient(api_key=mem0_api_key)
                else:
                    cfg = {
                        "embedder": {
                            "provider": "openai",
                            "config": {
                                "model": "text-embedding-3-small",
                                "api_key": openrouter_api_key,
                                "openai_base_url": "https://openrouter.ai/api/v1",
                            },
                        },
                        "llm": {
                            "provider": "openai",
                            "config": {
                                "model": "gpt-4o-mini",
                                "api_key": openrouter_api_key,
                                "openai_base_url": "https://openrouter.ai/api/v1",
                            },
                        },
                        "vector_store": {
                            "provider": "chroma",
                            "config": {
                                "collection_name": "labos_knowledge",
                                "path": str(
                                    Path(storage_dir or Path.home() / ".labos" / "mem0_db")
                                ),
                            },
                        },
                    }
                    self.memory = Memory.from_config(cfg)
                self.mem0_enabled = True
                logger.info("Mem0 knowledge base initialised")
            except Exception as exc:
                logger.warning("Mem0 init failed (%s) — using TF-IDF fallback", exc)
                self._init_fallback(gemini_model, storage_dir)
        else:
            self._init_fallback(gemini_model, storage_dir)

    def _init_fallback(self, gemini_model, storage_dir):
        self.fallback_kb = KnowledgeBase(gemini_model=gemini_model, storage_dir=storage_dir)

    # -- public API (same interface as KnowledgeBase) ----------------------

    def add_template(self, task, thought, outcome, domain="general", user_id="agent_team"):
        if self.mem0_enabled:
            try:
                convo = [
                    {"role": "user", "content": f"Task: {task}"},
                    {"role": "assistant", "content": f"Reasoning: {thought}"},
                    {"role": "user", "content": f"Outcome: {outcome}"},
                ]
                meta = {
                    "domain": domain,
                    "type": "problem_solving_template",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self.memory.add(convo, user_id=user_id, metadata=meta)
                return {"success": True}
            except Exception as exc:
                logger.warning("Mem0 add failed: %s", exc)
                if self.fallback_kb:
                    return self.fallback_kb.add_template(task, thought, outcome, domain)
        elif self.fallback_kb:
            return self.fallback_kb.add_template(task, thought, outcome, domain)
        return {"success": False, "message": "No backend available"}

    def retrieve_similar_templates(self, task, top_k=3, user_id="agent_team"):
        if self.mem0_enabled:
            try:
                results = self.memory.search(query=task, user_id=user_id, limit=top_k)
                return [
                    {
                        "task": task,
                        "key_reasoning": r.get("memory", ""),
                        "similarity": r.get("score", 0.0),
                    }
                    for r in results.get("results", [])
                ]
            except Exception:
                if self.fallback_kb:
                    return self.fallback_kb.retrieve_similar_templates(task, top_k)
        elif self.fallback_kb:
            return self.fallback_kb.retrieve_similar_templates(task, top_k)
        return []

    def extract_keywords(self, text):
        if self.fallback_kb:
            return self.fallback_kb.extract_keywords(text)
        return [w.lower() for w in text.split() if len(w) > 3]

    def get_memory_stats(self, user_id="agent_team"):
        if self.mem0_enabled:
            return {"backend": "Mem0 Enhanced", "total_memories": "N/A"}
        if self.fallback_kb:
            return {
                "backend": "Traditional KnowledgeBase",
                "total_memories": len(self.fallback_kb.templates),
            }
        return {"backend": "None", "total_memories": 0}
