"""
LabOS Memory Manager — Three-tier memory system.

Tiers:
  1. KnowledgeMemory  – problem-solving templates (TF-IDF / Mem0)
  2. CollaborationMemory – shared multi-agent workspace (Mem0-only)
  3. SessionMemory     – per-session conversation context (Mem0-only)

Plus a lightweight AutoMemory for runtime task/tool tracking.
"""

import time
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from mem0 import Memory, MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

from labos.knowledge import KnowledgeBase


# ---------------------------------------------------------------------------
# Lightweight AutoMemory (no external deps)
# ---------------------------------------------------------------------------

class AutoMemory:
    """Tracks agent task history and tool usage at runtime."""

    def __init__(self):
        self.task_history: deque = deque(maxlen=50)
        self.tool_usage: Dict[str, Dict] = {}
        self.agent_performance: Dict[str, Dict] = {}

    def record_task(self, agent_name: str, task: str, result: str, success: bool, duration: float):
        self.task_history.append({
            "agent": agent_name,
            "task": task[:100],
            "success": success,
            "duration": duration,
            "timestamp": time.time(),
        })
        stats = self.agent_performance.setdefault(
            agent_name, {"total": 0, "success": 0, "avg_duration": 0.0}
        )
        stats["total"] += 1
        if success:
            stats["success"] += 1
        old = stats["avg_duration"]
        stats["avg_duration"] = (old * (stats["total"] - 1) + duration) / stats["total"]

    def record_tool_use(self, tool_name: str, success: bool):
        entry = self.tool_usage.setdefault(tool_name, {"uses": 0, "success": 0})
        entry["uses"] += 1
        if success:
            entry["success"] += 1

    def get_similar_tasks(self, task: str, limit: int = 3) -> List[Dict]:
        keywords = set(task.lower().split())
        matches = []
        for h in self.task_history:
            if h["success"]:
                score = len(keywords & set(h["task"].lower().split()))
                if score > 0:
                    matches.append((score, h))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    def get_best_agent(self, task: str) -> Optional[str]:
        similar = self.get_similar_tasks(task)
        if not similar:
            return None
        counts: Dict[str, int] = {}
        for t in similar:
            counts[t["agent"]] = counts.get(t["agent"], 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Base Memory Component (Mem0 integration)
# ---------------------------------------------------------------------------

class _BaseMemoryComponent:
    """Shared Mem0 init logic for memory tiers."""

    def __init__(self, name: str, mem0_config: Optional[Dict] = None):
        self.component_name = name
        self.mem0_enabled = False
        self.memory = None

        if MEM0_AVAILABLE and mem0_config:
            try:
                if mem0_config.get("use_platform"):
                    self.memory = MemoryClient(api_key=mem0_config["api_key"])
                else:
                    cfg = {
                        "embedder": {
                            "provider": "openai",
                            "config": {
                                "model": "text-embedding-3-small",
                                "api_key": mem0_config.get("openrouter_api_key"),
                                "openai_base_url": "https://openrouter.ai/api/v1",
                            },
                        },
                        "llm": {
                            "provider": "openai",
                            "config": {
                                "model": "gpt-4o-mini",
                                "api_key": mem0_config.get("openrouter_api_key"),
                                "openai_base_url": "https://openrouter.ai/api/v1",
                            },
                        },
                        "vector_store": {
                            "provider": "chroma",
                            "config": {
                                "collection_name": f"labos_{name}",
                                "path": str(Path.home() / ".labos" / "mem0_db" / name),
                            },
                        },
                    }
                    self.memory = Memory.from_config(cfg)
                self.mem0_enabled = True
                logger.info("%s: Mem0 initialised", name)
            except Exception as exc:
                logger.warning("%s: Mem0 init failed: %s", name, exc)


# ---------------------------------------------------------------------------
# Tier 1 — Knowledge Memory
# ---------------------------------------------------------------------------

class KnowledgeMemory(_BaseMemoryComponent):
    """Problem-solving template memory with TF-IDF fallback."""

    def __init__(self, gemini_model=None, mem0_config: Optional[Dict] = None):
        super().__init__("knowledge", mem0_config)
        self.fallback_kb: Optional[KnowledgeBase] = None
        if not self.mem0_enabled:
            self.fallback_kb = KnowledgeBase(gemini_model=gemini_model)

    def add_template(self, task, thought, outcome, domain="general", user_id="agent_team"):
        if self.mem0_enabled:
            try:
                convo = [
                    {"role": "user", "content": f"Task: {task}"},
                    {"role": "assistant", "content": f"Reasoning: {thought}"},
                    {"role": "user", "content": f"Outcome: {outcome}"},
                ]
                meta = {"type": "template", "domain": domain, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                result = self.memory.add(convo, user_id=user_id, metadata=meta)
                return {"success": True, "memory_id": result.get("id", "")}
            except Exception:
                if self.fallback_kb:
                    return self.fallback_kb.add_template(task, thought, outcome, domain)
        elif self.fallback_kb:
            return self.fallback_kb.add_template(task, thought, outcome, domain)
        return {"success": False}

    def search_templates(self, task, top_k=3, user_id="agent_team"):
        if self.mem0_enabled:
            try:
                results = self.memory.search(query=task, user_id=user_id, limit=top_k)
                templates = [
                    {"key_reasoning": r.get("memory", ""), "similarity": r.get("score", 0.0)}
                    for r in results.get("results", [])
                ]
                return {"success": True, "templates": templates}
            except Exception:
                if self.fallback_kb:
                    return {"success": True, "templates": self.fallback_kb.retrieve_similar_templates(task, top_k)}
        elif self.fallback_kb:
            return {"success": True, "templates": self.fallback_kb.retrieve_similar_templates(task, top_k)}
        return {"success": False, "templates": []}

    def get_stats(self, user_id="agent_team"):
        if self.mem0_enabled:
            return {"component": "KnowledgeMemory", "backend": "Mem0 Enhanced"}
        if self.fallback_kb:
            return {
                "component": "KnowledgeMemory",
                "backend": "Traditional KnowledgeBase",
                "total_templates": len(self.fallback_kb.templates),
            }
        return {"component": "KnowledgeMemory", "backend": "None", "total_templates": 0}


# ---------------------------------------------------------------------------
# Tier 2 — Collaboration Memory (Mem0 only)
# ---------------------------------------------------------------------------

class CollaborationMemory(_BaseMemoryComponent):
    """Shared workspace memory for multi-agent collaboration."""

    def __init__(self, mem0_config: Optional[Dict] = None):
        super().__init__("collaboration", mem0_config)

    def create_workspace(self, workspace_id: str, task: str, agents: Optional[List[str]] = None):
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 required for collaboration memory"}
        agents = agents or ["researcher_agent", "manager_agent", "critic_agent"]
        try:
            convo = [
                {"role": "system", "content": f"Workspace '{workspace_id}' created"},
                {"role": "assistant", "content": f"Task: {task}\nAgents: {', '.join(agents)}"},
            ]
            meta = {"type": "workspace", "workspace_id": workspace_id, "status": "active",
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            result = self.memory.add(convo, user_id="shared_workspace", metadata=meta)
            return {"success": True, "workspace_id": workspace_id, "memory_id": result.get("id", "")}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def add_observation(self, workspace_id: str, agent_name: str, content: str):
        if not self.mem0_enabled:
            return {"success": False}
        try:
            entry = [
                {"role": "user", "content": f"Agent: {agent_name}"},
                {"role": "assistant", "content": content},
            ]
            meta = {"type": "observation", "workspace_id": workspace_id,
                     "agent_name": agent_name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.memory.add(entry, user_id="shared_workspace", metadata=meta)
            return {"success": True}
        except Exception:
            return {"success": False}


# ---------------------------------------------------------------------------
# Tier 3 — Session Memory (Mem0 only)
# ---------------------------------------------------------------------------

class SessionMemory(_BaseMemoryComponent):
    """Per-session conversation context."""

    def __init__(self, mem0_config: Optional[Dict] = None):
        super().__init__("session", mem0_config)

    def add_turn(self, session_id: str, user_id: str, user_msg: str, assistant_msg: str):
        if not self.mem0_enabled:
            return {"success": False}
        try:
            convo = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            meta = {"type": "turn", "session_id": session_id,
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.memory.add(convo, user_id=user_id, metadata=meta)
            return {"success": True}
        except Exception:
            return {"success": False}

    def get_context(self, session_id: str, user_id: str, limit: int = 10):
        if not self.mem0_enabled:
            return {"success": False, "context": []}
        try:
            results = self.memory.search(query=f"session {session_id}", user_id=user_id, limit=limit)
            ctx = [
                {"content": r.get("memory", ""), "timestamp": r.get("metadata", {}).get("timestamp", "")}
                for r in results.get("results", [])
                if r.get("metadata", {}).get("session_id") == session_id
            ]
            return {"success": True, "context": ctx}
        except Exception:
            return {"success": False, "context": []}


# ---------------------------------------------------------------------------
# Unified Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Coordinates all three memory tiers."""

    def __init__(
        self,
        gemini_model=None,
        use_mem0: bool = False,
        mem0_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        mem0_cfg: Optional[Dict] = None
        if use_mem0:
            mem0_cfg = {
                "use_platform": bool(mem0_api_key),
                "api_key": mem0_api_key,
                "openrouter_api_key": openrouter_api_key,
            }

        logger.info("Initialising LabOS memory system ...")
        self.knowledge = KnowledgeMemory(gemini_model, mem0_cfg)
        self.collaboration = CollaborationMemory(mem0_cfg)
        self.session = SessionMemory(mem0_cfg)
        logger.info("Memory system ready")

    def get_overall_stats(self):
        return {
            "knowledge": self.knowledge.get_stats(),
            "collaboration_enabled": self.collaboration.mem0_enabled,
            "session_enabled": self.session.mem0_enabled,
        }

    # Convenience aliases
    def add_template(self, *a, **kw):
        return self.knowledge.add_template(*a, **kw)

    def retrieve_similar_templates(self, *a, **kw):
        r = self.knowledge.search_templates(*a, **kw)
        return r.get("templates", []) if r["success"] else []
