"""
LabOS Core Engine — Multi-agent orchestration with self-evolution.

Uses a single model (Gemini 3 via OpenRouter) for all agents:
  - manager_agent    : orchestrates tasks, delegates to sub-agents
  - researcher_agent : code execution, environment management, data analysis
  - critic_agent     : quality evaluation and improvement recommendations
  - toolmaker_agent  : dynamic tool creation when existing tools are insufficient
"""

import os
import sys
import re
import time
import json
import yaml
import hashlib
import threading
import importlib.util
import inspect
from pathlib import Path
from functools import lru_cache, wraps
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# smolagents imports
# ---------------------------------------------------------------------------
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# ---------------------------------------------------------------------------
# LabOS internal imports
# ---------------------------------------------------------------------------
from labos.memory import AutoMemory, MemoryManager
from labos.knowledge import KnowledgeBase, Mem0EnhancedKnowledgeBase

# ---------------------------------------------------------------------------
# Tool imports
# ---------------------------------------------------------------------------
from labos.tools.search import (
    enhanced_google_search,
    search_with_serpapi,
    multi_source_search,
    enhanced_knowledge_search,
    query_arxiv,
    query_pubmed,
    query_scholar,
    search_github_repositories,
    search_github_code,
    get_github_repository_info,
)
from labos.tools.web import (
    visit_webpage,
    extract_url_content,
    extract_pdf_content,
    fetch_supplementary_info_from_doi,
)
from labos.tools.devenv import (
    run_shell_command,
    create_conda_environment,
    install_packages_conda,
    install_packages_pip,
    check_gpu_status,
    create_script,
    run_script,
    create_and_run_script,
    create_requirements_file,
    monitor_training_logs,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEPS = 50
DEFAULT_MODEL = "google/gemini-3"

# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

def get_api_key(key_name: str, required: bool = True) -> str:
    """Get API key from environment with error handling."""
    val = os.getenv(key_name)
    if required and not val:
        print(f"Missing required API key: {key_name}")
        print(f"Set {key_name} in your .env file")
        if key_name == "OPENROUTER_API_KEY":
            print("Get your key at: https://openrouter.ai/")
            sys.exit(1)
    return val or ""


def _require_api_key() -> str:
    """Return the OpenRouter API key or exit."""
    return get_api_key("OPENROUTER_API_KEY", required=True)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
auto_memory = AutoMemory()
dynamic_tools_registry: dict = {}
tool_loading_cache: dict = {}
tool_loading_lock = threading.Lock()

# These are populated by ``initialize()``
manager_agent = None
researcher_agent = None
critic_agent = None
toolmaker_agent = None
global_memory_manager: MemoryManager | None = None
use_templates = False
_gemini_model = None

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_on_failure(max_retries=3, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Memory-enhanced agent wrapper
# ---------------------------------------------------------------------------

def _wrap_agent_with_memory(agent, agent_name: str):
    """Wrap agent.run to automatically record performance."""
    original_run = agent.run

    def run_with_memory(*args, **kwargs):
        start = time.time()
        success = False
        task_str = str(args[0]) if args else str(kwargs.get("task", ""))
        result = ""
        try:
            similar = auto_memory.get_similar_tasks(task_str, 2)
            if similar:
                print(f"  {agent_name}: Found {len(similar)} similar past tasks")
            result = original_run(*args, **kwargs)
            success = True
            return result
        except Exception:
            raise
        finally:
            auto_memory.record_task(agent_name, task_str, str(result)[:100], success, time.time() - start)

    agent.run = run_with_memory
    return agent


# ===================================================================
# Tools that reference global agents (defined before agent creation)
# ===================================================================

@tool
def auto_recall_experience(task_description: str) -> str:
    """Recall similar past tasks and their outcomes.

    Args:
        task_description: Current task description

    Returns:
        Similar successful tasks with recommended agent
    """
    similar = auto_memory.get_similar_tasks(task_description, 3)
    if not similar:
        return "No similar past tasks found."
    lines = [f"Found {len(similar)} similar tasks:"]
    for i, t in enumerate(similar, 1):
        lines.append(f"  {i}. {t['task']} — {t['duration']:.1f}s")
    best = auto_memory.get_best_agent(task_description)
    if best:
        lines.append(f"Recommended agent: {best}")
    return "\n".join(lines)


@tool
def check_agent_performance() -> str:
    """Check agent performance statistics.

    Returns:
        Performance stats for all agents
    """
    if not auto_memory.agent_performance:
        return "No performance data yet."
    lines = ["Agent Performance:"]
    for name, s in auto_memory.agent_performance.items():
        rate = s["success"] / s["total"] if s["total"] else 0
        lines.append(f"  {name}: {rate:.0%} success, avg {s['avg_duration']:.1f}s ({s['total']} tasks)")
    return "\n".join(lines)


@tool
def quick_tool_stats() -> str:
    """Overview of tool effectiveness.

    Returns:
        Tool success rate rankings
    """
    if not auto_memory.tool_usage:
        return "No tool usage data yet."
    stats = sorted(
        [(s["success"] / s["uses"], n, s["uses"]) for n, s in auto_memory.tool_usage.items() if s["uses"]],
        reverse=True,
    )
    lines = ["Top tools:"]
    for rate, name, uses in stats[:5]:
        lines.append(f"  {name}: {rate:.0%} ({uses} uses)")
    return "\n".join(lines)


@tool
def evaluate_with_critic(task_description: str, current_result: str, expected_outcome: str = "") -> str:
    """Use the critic agent to evaluate task quality.

    Args:
        task_description: Original task
        current_result: Current result
        expected_outcome: Expected outcome (optional)

    Returns:
        Critic evaluation
    """
    global critic_agent
    if critic_agent is None:
        return "Critic agent not initialised."
    prompt = (
        f"Evaluate task completion:\n\nTASK: {task_description}\n"
        f"RESULT: {current_result[:1500]}\n"
        f"EXPECTED: {expected_outcome or 'High quality output'}\n\n"
        "Provide: status, quality_score (1-10), gaps, recommendations."
    )
    try:
        return str(critic_agent.run(prompt))
    except Exception as exc:
        return f"Critic evaluation error: {exc}"


@tool
def list_dynamic_tools() -> str:
    """List all dynamically created tools.

    Returns:
        List of tools with purposes
    """
    if not dynamic_tools_registry:
        return "No dynamic tools created yet."
    lines = [f"Dynamic Tools ({len(dynamic_tools_registry)}):"]
    for name, info in dynamic_tools_registry.items():
        lines.append(f"  {name}: {info.get('purpose', '')[:60]}")
    return "\n".join(lines)


@tool
def load_dynamic_tool(tool_name: str, add_to_agents: bool = True) -> str:
    """Load a tool from the new_tools directory.

    Args:
        tool_name: Tool module name
        add_to_agents: Whether to add to agents

    Returns:
        Loading status
    """
    global researcher_agent, toolmaker_agent
    try:
        os.makedirs("./new_tools", exist_ok=True)
        path = f"./new_tools/{tool_name}.py"
        if not os.path.exists(path):
            return f"Tool file not found: {path}"

        spec = importlib.util.spec_from_file_location(tool_name, path)
        if not spec or not spec.loader:
            return f"Cannot load module: {tool_name}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[tool_name] = module
        spec.loader.exec_module(module)

        result = f"Loaded tool '{tool_name}'"
        if add_to_agents and researcher_agent and toolmaker_agent:
            fns = [obj for _, obj in inspect.getmembers(module)
                   if inspect.isfunction(obj) and hasattr(obj, "__smolagents_tool__")]
            for fn in fns:
                if fn not in researcher_agent.tools:
                    researcher_agent.tools.append(fn)
                if fn not in toolmaker_agent.tools:
                    toolmaker_agent.tools.append(fn)
            result += f" — added {len(fns)} functions to agents"
        return result
    except Exception as exc:
        return f"Error loading '{tool_name}': {exc}"


@tool
def create_new_tool(tool_name: str, tool_purpose: str, tool_category: str, technical_requirements: str) -> str:
    """Use the toolmaker agent to create a new tool.

    Args:
        tool_name: Name of tool to create
        tool_purpose: What the tool should do
        tool_category: Category (analysis, visualisation, etc.)
        technical_requirements: Technical details

    Returns:
        Creation result
    """
    global toolmaker_agent
    if toolmaker_agent is None:
        return "Toolmaker agent not initialised."
    try:
        task = (
            f"Create tool '{tool_name}' in ./new_tools/{tool_name}.py\n"
            f"Purpose: {tool_purpose}\nCategory: {tool_category}\n"
            f"Requirements: {technical_requirements}\n"
            "Use @tool decorator from smolagents. Include docstrings, type hints, error handling."
        )
        result = str(toolmaker_agent.run(task))
        dynamic_tools_registry[tool_name] = {
            "purpose": tool_purpose, "category": tool_category,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        load_result = load_dynamic_tool(tool_name, add_to_agents=True)
        return f"Tool created!\n{result}\nLoading: {load_result}"
    except Exception as exc:
        return f"Tool creation error: {exc}"


@tool
def refresh_agent_tools() -> str:
    """Refresh tools by scanning new_tools directory.

    Returns:
        Refresh status
    """
    import glob as _glob
    d = "./new_tools"
    if not os.path.exists(d):
        return "new_tools directory does not exist."
    files = _glob.glob(os.path.join(d, "*.py"))
    if not files:
        return "No tool files found."
    loaded = 0
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        r = load_dynamic_tool(name, add_to_agents=True)
        if "Loaded" in r:
            loaded += 1
    return f"Refreshed: {loaded}/{len(files)} tools loaded."


@tool
def get_tool_signature(tool_name: str) -> str:
    """Get the signature of a loaded tool.

    Args:
        tool_name: Tool name

    Returns:
        Tool signature and documentation
    """
    global manager_agent
    if manager_agent is None or tool_name not in manager_agent.tools:
        return f"Tool '{tool_name}' not found."
    try:
        func = manager_agent.tools[tool_name]
        sig = inspect.signature(getattr(func, "forward", func))
        doc = inspect.getdoc(func) or "No docs"
        return f"{tool_name}{sig}\n\n{doc[:500]}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def execute_tools_in_parallel(tool_calls: list, max_workers: int = 3, timeout: int = 30) -> str:
    """Execute multiple tool calls in parallel.

    Args:
        tool_calls: List of dicts with 'tool_name' and 'args'
        max_workers: Max parallel workers (default: 3)
        timeout: Timeout per call in seconds (default: 30)

    Returns:
        Formatted parallel execution results
    """
    import concurrent.futures
    global manager_agent
    if not tool_calls or not isinstance(tool_calls, list):
        return "Invalid tool_calls parameter."

    def _exec(call):
        name = call["tool_name"]
        args = call["args"]
        t0 = time.time()
        try:
            func = manager_agent.tools.get(name) if hasattr(manager_agent.tools, "get") else None
            if func is None:
                for t in manager_agent.tools:
                    if getattr(t, "name", getattr(t, "__name__", "")) == name:
                        func = t
                        break
            if func is None:
                raise ValueError(f"Tool '{name}' not found")
            return {"tool_name": name, "success": True, "result": func(**args), "duration": time.time() - t0}
        except Exception as exc:
            return {"tool_name": name, "success": False, "error": str(exc), "duration": time.time() - t0}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_exec, c): c for c in tool_calls}
        for f in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                results.append(f.result(timeout=5))
            except Exception as exc:
                c = futures[f]
                results.append({"tool_name": c["tool_name"], "success": False, "error": str(exc), "duration": 0})

    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    out = f"Parallel: {len(tool_calls)} tools, {len(ok)} ok, {len(fail)} failed\n"
    for r in ok:
        out += f"  OK {r['tool_name']} ({r['duration']:.1f}s): {str(r['result'])[:100]}\n"
    for r in fail:
        out += f"  FAIL {r['tool_name']}: {r['error']}\n"
    return out


@tool
def analyze_query_and_load_relevant_tools(user_query: str, max_tools: int = 10) -> str:
    """Analyse query and load the most relevant biomedical tools.

    Args:
        user_query: User's task description
        max_tools: Max tools to load (default: 10)

    Returns:
        Tool loading status
    """
    global manager_agent, toolmaker_agent
    try:
        qhash = hashlib.md5(user_query.encode()).hexdigest()
        ckey = f"{qhash}_{max_tools}"
        with tool_loading_lock:
            if ckey in tool_loading_cache:
                cached, ts = tool_loading_cache[ckey]
                if time.time() - ts < 300:
                    return f"(cached) {cached}"

        # Try to find tool modules in the labos.tools package
        tools_dir = os.path.join(os.path.dirname(__file__), "tools")
        tool_files = {
            "database": os.path.join(tools_dir, "database.py"),
            "screening": os.path.join(tools_dir, "screening.py"),
            "sequence": os.path.join(tools_dir, "sequence.py"),
        }

        available = {}
        for mod_name, fpath in tool_files.items():
            if not os.path.exists(fpath):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"labos.tools.{mod_name}", fpath)
                if not spec or not spec.loader:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for name, obj in inspect.getmembers(mod):
                    if hasattr(obj, "__class__") and "SimpleTool" in str(type(obj)):
                        desc = getattr(obj, "description", "") or ""
                        available[name] = {"function": obj, "description": desc[:100], "module": mod_name}
            except Exception:
                continue

        if not available:
            return "No biomedical tools found in tools directory."

        # Simple keyword matching for tool selection
        query_lower = user_query.lower()
        scored = []
        for tname, tinfo in available.items():
            text = f"{tname.replace('_', ' ')} {tinfo['description']}".lower()
            score = sum(1 for w in query_lower.split() if len(w) > 2 and w in text)
            scored.append((tname, score, tinfo))
        scored.sort(key=lambda x: x[1], reverse=True)

        loaded = 0
        result_lines = []
        for tname, score, tinfo in scored[:max_tools]:
            if score == 0:
                continue
            func = tinfo["function"]
            if manager_agent and tname not in manager_agent.tools:
                manager_agent.tools[tname] = func
                if hasattr(manager_agent, "python_executor") and hasattr(manager_agent.python_executor, "custom_tools"):
                    manager_agent.python_executor.custom_tools[tname] = func
                loaded += 1
            result_lines.append(f"  {tname} [{tinfo['module']}]")

        result = f"Loaded {loaded} tools for: '{user_query[:50]}...'\n" + "\n".join(result_lines)

        with tool_loading_lock:
            tool_loading_cache[ckey] = (result, time.time())
        return result

    except Exception as exc:
        return f"Error loading tools: {exc}"


# ---------------------------------------------------------------------------
# Knowledge base tools
# ---------------------------------------------------------------------------

@tool
def retrieve_similar_templates(task_description: str, top_k: int = 3) -> str:
    """Retrieve similar problem-solving templates from memory.

    Args:
        task_description: Current task
        top_k: Number of templates (default: 3)

    Returns:
        Similar templates
    """
    global global_memory_manager, use_templates
    if not use_templates or global_memory_manager is None:
        return "Template system not enabled."
    try:
        result = global_memory_manager.knowledge.search_templates(task_description, top_k)
        if result["success"] and result["templates"]:
            lines = [f"Found {len(result['templates'])} templates:"]
            for i, t in enumerate(result["templates"], 1):
                lines.append(f"  {i}. {t.get('key_reasoning', '')[:80]} (sim: {t.get('similarity', 0):.2f})")
            return "\n".join(lines)
        return "No similar templates found."
    except Exception as exc:
        return f"Template retrieval error: {exc}"


@tool
def save_successful_template(
    task_description: str, reasoning_process: str, solution_outcome: str, domain: str = "general"
) -> str:
    """Save a successful problem-solving approach.

    Args:
        task_description: Solved task
        reasoning_process: Reasoning that led to success
        solution_outcome: Successful outcome
        domain: Domain category (default: general)

    Returns:
        Save status
    """
    global global_memory_manager, use_templates
    if not use_templates or global_memory_manager is None:
        return "Template system not enabled."
    try:
        r = global_memory_manager.knowledge.add_template(task_description, reasoning_process, solution_outcome, domain)
        return "Template saved." if r.get("success") else f"Save failed: {r.get('message', '')}"
    except Exception as exc:
        return f"Save error: {exc}"


# ===================================================================
# Agent and model creation
# ===================================================================

# Common authorised imports for CodeAgent
AUTHORIZED_IMPORTS = [
    "sys", "os", "subprocess", "pathlib", "shutil", "glob", "tempfile",
    "io", "json", "csv", "pickle", "sqlite3", "gzip", "zipfile",
    "math", "statistics", "random", "decimal", "cmath",
    "time", "datetime", "calendar", "re", "string", "collections",
    "itertools", "functools", "operator", "heapq", "bisect", "queue",
    "threading", "multiprocessing", "concurrent", "asyncio",
    "typing", "dataclasses", "enum", "abc", "contextlib", "inspect",
    "argparse", "logging", "warnings", "traceback",
    "numpy", "pandas", "matplotlib", "scipy", "sklearn",
    "requests", "Bio", "yaml", "tqdm", "joblib",
    "torch", "torchvision", "huggingface_hub", "seaborn", "plotly", "PIL",
]


def _create_model(api_key: str, model_id: str = DEFAULT_MODEL, temperature: float = 0.1):
    """Create an OpenAIServerModel instance."""
    return OpenAIServerModel(
        model_id=model_id,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=temperature,
    )


def initialize(
    use_template: bool = True,
    use_mem0: bool = False,
    enable_tool_creation: bool = True,
    model_id: str = DEFAULT_MODEL,
):
    """Initialise the LabOS multi-agent system.

    Args:
        use_template: Enable knowledge base templates
        use_mem0: Enable Mem0 enhanced memory
        enable_tool_creation: Enable toolmaker agent
        model_id: OpenRouter model ID (default: google/gemini-3)

    Returns:
        The manager_agent instance (or None on failure)
    """
    global manager_agent, researcher_agent, critic_agent, toolmaker_agent
    global global_memory_manager, use_templates, _gemini_model

    api_key = _require_api_key()
    mem0_key = get_api_key("MEM0_API_KEY", required=False)
    use_templates = use_template

    print("LabOS: Initialising multi-agent system ...")
    print(f"  Model: {model_id}")

    # --- Create the single shared model ---
    _gemini_model = _create_model(api_key, model_id)

    # --- researcher_agent (ToolCallingAgent) ---
    researcher_tools = [
        extract_url_content, query_arxiv, query_scholar, query_pubmed,
        extract_pdf_content, fetch_supplementary_info_from_doi,
        multi_source_search, search_github_repositories, search_github_code,
        get_github_repository_info, run_shell_command, create_conda_environment,
        install_packages_conda, install_packages_pip, check_gpu_status,
        create_and_run_script, run_script, create_requirements_file,
        monitor_training_logs, list_dynamic_tools, load_dynamic_tool,
        refresh_agent_tools, auto_recall_experience, quick_tool_stats,
    ]

    researcher_agent = ToolCallingAgent(
        tools=researcher_tools,
        model=_gemini_model,
        max_steps=30,
        name="researcher_agent",
        description=(
            "Specialist agent for code execution, environment management, "
            "and biomedical data analysis. Has unrestricted Python execution."
        ),
    )
    researcher_agent.prompt_templates["managed_agent"]["task"] += (
        "\nSave outputs to './agent_outputs'. "
        "ALWAYS search GitHub for existing implementations before coding from scratch."
    )
    researcher_agent = _wrap_agent_with_memory(researcher_agent, "researcher_agent")
    print("  researcher_agent ready")

    # --- toolmaker_agent (ToolCallingAgent) ---
    if enable_tool_creation:
        toolmaker_tools = [
            extract_url_content, query_arxiv, query_scholar, query_pubmed,
            multi_source_search, search_github_repositories, search_github_code,
            get_github_repository_info, run_shell_command, create_conda_environment,
            install_packages_pip, check_gpu_status, create_and_run_script,
            run_script, create_requirements_file, monitor_training_logs,
        ]
        toolmaker_agent = ToolCallingAgent(
            tools=toolmaker_tools,
            model=_gemini_model,
            max_steps=25,
            name="toolmaker_agent",
            description="Specialist for creating new Python tools with @tool decorator.",
        )
        toolmaker_agent.prompt_templates["managed_agent"]["task"] += (
            "\nCreate tools in './new_tools/'. Use @tool from smolagents. "
            "Search GitHub first. Test after creation."
        )
        toolmaker_agent = _wrap_agent_with_memory(toolmaker_agent, "toolmaker_agent")
        print("  toolmaker_agent ready")

    # --- critic_agent (ToolCallingAgent) ---
    critic_tools = [extract_url_content, run_shell_command]
    critic_agent = ToolCallingAgent(
        tools=critic_tools,
        model=_gemini_model,
        max_steps=10,
        name="critic_agent",
        description="Expert evaluator of task quality and completion.",
    )
    critic_agent.prompt_templates["managed_agent"]["task"] += (
        "\nFocus on ACTUAL PERFORMANCE METRICS. Completion != Success."
    )
    critic_agent = _wrap_agent_with_memory(critic_agent, "critic_agent")
    print("  critic_agent ready")

    # --- Manager tools ---
    mgr_tools = [
        multi_source_search, search_github_repositories, search_github_code,
        get_github_repository_info, run_shell_command,
        analyze_query_and_load_relevant_tools, execute_tools_in_parallel,
        evaluate_with_critic, list_dynamic_tools, load_dynamic_tool,
        get_tool_signature, auto_recall_experience, check_agent_performance,
        quick_tool_stats, extract_url_content, query_arxiv, query_pubmed,
        extract_pdf_content, fetch_supplementary_info_from_doi,
    ]
    if enable_tool_creation:
        mgr_tools.append(create_new_tool)

    if use_template:
        mgr_tools.extend([retrieve_similar_templates, save_successful_template])

    # --- Managed agents ---
    managed = [researcher_agent, critic_agent]
    if enable_tool_creation and toolmaker_agent:
        managed.append(toolmaker_agent)

    # --- Load optional prompt templates ---
    prompt_templates = _load_prompt_templates()

    # --- manager_agent (CodeAgent) ---
    try:
        mgr_kwargs = dict(
            tools=mgr_tools,
            model=_gemini_model,
            managed_agents=managed,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            executor_kwargs={"additional_functions": {"globals": lambda: {"__name__": "__main__"}}},
            name="manager_agent",
            description=(
                "LabOS Manager — orchestrates tasks across researcher, critic"
                f"{', and toolmaker' if enable_tool_creation else ''} agents."
            ),
        )
        if prompt_templates:
            rendered = _render_templates(prompt_templates, managed, mgr_tools)
            mgr_kwargs["prompt_templates"] = rendered

        manager_agent = CodeAgent(**mgr_kwargs)
        manager_agent = _wrap_agent_with_memory(manager_agent, "manager_agent")
        print(f"  manager_agent ready ({len(mgr_tools)} tools)")
    except Exception as exc:
        print(f"  Error creating manager: {exc}")
        manager_agent = None
        return None

    # --- Memory system ---
    if use_template:
        try:
            global_memory_manager = MemoryManager(
                gemini_model=_gemini_model,
                use_mem0=use_mem0,
                mem0_api_key=mem0_key,
                openrouter_api_key=api_key,
            )
            print("  Memory system ready")
        except Exception as exc:
            print(f"  Memory init failed: {exc}")
            use_templates = False

    print("LabOS: Initialisation complete")
    return manager_agent


def _load_prompt_templates():
    """Try to load custom YAML prompt templates."""
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    for name in ("manager.yaml",):
        path = os.path.join(prompts_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
    return None


def _render_templates(templates, managed_agents, tools):
    """Render Jinja templates with variables."""
    try:
        from jinja2 import Template
    except ImportError:
        return templates

    variables = {
        "code_block_opening_tag": "```python",
        "code_block_closing_tag": "```",
        "custom_instructions": "",
        "authorized_imports": ", ".join(AUTHORIZED_IMPORTS[:30]),
        "managed_agents": {
            getattr(a, "name", f"agent_{i}"): a for i, a in enumerate(managed_agents)
        },
        "tools": {
            getattr(t, "name", str(t)): t for t in tools
        },
    }

    rendered = {}
    for key, val in templates.items():
        if isinstance(val, str):
            rendered[key] = Template(val).render(**variables)
        elif isinstance(val, dict):
            rendered[key] = {
                sk: Template(sv).render(**variables) if isinstance(sv, str) else sv
                for sk, sv in val.items()
            }
        else:
            rendered[key] = val
    return rendered


# ---------------------------------------------------------------------------
# Convenience: run a task
# ---------------------------------------------------------------------------

def run_task(task: str, reset: bool = False) -> str:
    """Run a task through the manager agent.

    Args:
        task: Task description
        reset: Whether to reset agent state

    Returns:
        Agent response as string
    """
    global manager_agent
    if manager_agent is None:
        raise RuntimeError("LabOS not initialised. Call labos.initialize() first.")
    return str(manager_agent.run(task, reset=reset))
