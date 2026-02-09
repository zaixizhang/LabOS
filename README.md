<div align="center">

# LabOS

### The AI-XR Co-Scientist That Sees and Works With Humans

<p>
    <em>Self-evolving multi-agent framework for biomedical research — uniting computational reasoning with physical experimentation through multimodal perception, self-evolving agents, and XR-enabled human-AI collaboration.</em>
</p>

<a href="https://arxiv.org/abs/2510.14861" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2510.14861-b31b1b?logo=arxiv&logoColor=white" height="25" />
</a>
<a href="https://ai4labos.com/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/Website-ai4labos.com-blue?logo=googlechrome&logoColor=white" height="25" />
</a>
<a href="https://pypi.org/project/labos/" target="_blank">
    <img alt="PyPI" src="https://img.shields.io/badge/PyPI-labos-3775A9?logo=pypi&logoColor=white" height="25" />
</a>
<a href="https://github.com/labos-ai/labos" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-labos--ai-181717?logo=github&logoColor=white" height="25" />
</a>
<a href="LICENSE" target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green?logo=opensourceinitiative&logoColor=white" height="25" />
</a>

<br><br>

<p>
Le Cong<sup>1,2*</sup>, Zaixi Zhang<sup>3*</sup>, Xiaotong Wang<sup>1,2*</sup>, Yin Di<sup>1,2*</sup>, Ruofan Jin<sup>3</sup>, Michal Gerasimiuk<sup>1,2</sup>, Yinkai Wang<sup>1,2</sup>,
<br>
Ravi K. Dinesh<sup>1,2</sup>, David Smerkous<sup>4</sup>, Alex Smerkous<sup>5</sup>, Xuekun Wu<sup>2,6</sup>, Shilong Liu<sup>3</sup>, Peishan Li<sup>1,2</sup>,
<br>
Yi Zhu<sup>1,2</sup>, Simran Serrao<sup>1,2</sup>, Ning Zhao<sup>1,2</sup>, Imran A. Mohammad<sup>2,7</sup>,
<br>
John B. Sunwoo<sup>2,7</sup>, Joseph C. Wu<sup>2,6</sup>, Mengdi Wang<sup>3&#8224;</sup>
</p>

<p>
<sup>1</sup>Stanford School of Medicine &middot; <sup>2</sup>Stanford University &middot; <sup>3</sup>Princeton University &middot; <sup>4</sup>Oregon State University &middot; <sup>5</sup>University of Washington &middot; <sup>6</sup>Stanford Cardiovascular Institute &middot; <sup>7</sup>Stanford Cancer Institute
<br>
<sup>*</sup>Equal Contribution &nbsp;&nbsp; <sup>&#8224;</sup>Corresponding Author
</p>

</div>

---

## Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Tool Library (98 Tools)](#tool-library-98-tools)
- [Package Structure](#package-structure)
- [API Keys](#api-keys)
- [Optional Dependencies](#optional-dependencies)
- [Citation](#citation)
- [Related Projects](#related-projects)
- [License](#license)

## Overview

Modern science advances fastest when thought meets action. **LabOS** is the first AI co-scientist that unites computational reasoning with physical experimentation. It connects multi-model AI agents, smart glasses, and robots so that AI can perceive scientific work, understand context, and assist experiments in real time.

This repository provides the **Dry-Lab computational core** — a pip-installable Python package with a self-evolving multi-agent system purpose-built for biomedical research. Four specialised agents collaborate through a shared Tool Ocean of 98 biomedical tools, continuously expanding their capabilities at runtime.

<div align="center">

| Component | Description |
|-----------|-------------|
| **Manager Agent** | Decomposes scientific objectives, orchestrates sub-agents, plans multi-step workflows |
| **Researcher Agent** | Executes bioinformatics analyses, runs code, queries databases and literature |
| **Critic Agent** | Evaluates result quality, identifies gaps, recommends improvements |
| **Toolmaker Agent** | Autonomously creates new tools when existing ones are insufficient |

</div>

Built on the [STELLA](https://github.com/zaixizhang/STELLA) framework, LabOS uses **Gemini 3** (`google/gemini-3` via OpenRouter) for all agents with a unified, single-model architecture.

## Key Results

LabOS consistently establishes a new state of the art across biomedical benchmarks:

<div align="center">

| Benchmark | Score | vs. Next Best |
|-----------|:-----:|:-------------:|
| **Humanity's Last Exam: Biomedicine** | **32%** | +8% |
| **LAB-Bench: DBQA** | **61%** | Top |
| **LAB-Bench: LitQA** | **65%** | Top |
| **Wet-Lab Error Detection** | **>90%** | vs. ~40% commercial |

</div>

Across applications — from cancer immunotherapy target discovery (CEACAM6) to stem cell engineering and cell fusion mechanism investigation (ITSN1) — LabOS demonstrates that AI can move beyond computational design to active participation in the laboratory.

## System Architecture

```
 LabOS: Dry-Lab Computational Core
╔══════════════════════════════════════════════════════════════════╗
║                        Manager Agent                            ║
║            (orchestration · planning · delegation)              ║
╠════════════════════╦════════════════════╦════════════════════════╣
║   Researcher       ║     Critic         ║     Toolmaker          ║
║   bioinformatics   ║     evaluation     ║     self-evolution     ║
║   code execution   ║     quality check  ║     tool creation      ║
╠════════════════════╩════════════════════╩════════════════════════╣
║                     Tool Ocean (98 tools)                        ║
║   search · database · sequence · screening · web · devenv · …   ║
╠══════════════════════════════════════════════════════════════════╣
║                   3-Tier Memory System                           ║
║   knowledge templates · collaboration workspace · session ctx   ║
╚══════════════════════════════════════════════════════════════════╝
```

**Self-Evolution Loop:** When existing tools are insufficient, the Toolmaker Agent autonomously identifies resources from the web and literature, generates new Python tools, validates them, and registers them into the shared Tool Ocean — all at runtime.

## Quick Start

### 1. Install

```bash
pip install labos
```

Or install from source:

```bash
git clone https://github.com/labos-ai/labos.git
cd labos
pip install -e .
```

### 2. Set Your API Key

LabOS requires an [OpenRouter](https://openrouter.ai/) API key to power all agents.

```bash
export OPENROUTER_API_KEY=your-key-here
```

Or use a `.env` file:

```bash
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your-key-here
```

### 3. Run

**Interactive CLI:**

```bash
labos run
```

**Web UI (Gradio):**

```bash
labos web
```

**Programmatic:**

```python
import labos

agent = labos.initialize()
result = labos.run_task("Find recent papers on CRISPR-Cas9 off-target effects")
print(result)
```

## CLI Reference

```
labos run              # Interactive chat
labos web              # Launch Gradio web UI
labos web --port 8080  # Custom port
labos web --share      # Public Gradio link
labos --version        # Show version
labos --help           # Show help
```

<details>
<summary><b>All Flags</b></summary>

| Flag | Description |
|------|-------------|
| `--use-template` | Enable knowledge-base templates (default: on) |
| `--no-template` | Disable templates |
| `--use-mem0` | Enable Mem0 enhanced memory |
| `--model MODEL` | OpenRouter model ID (default: `google/gemini-3`) |
| `--port PORT` | Web UI port (default: 7860) |
| `--share` | Create public Gradio link |

</details>

## Tool Library (98 Tools)

LabOS ships with **98 ready-to-use biomedical tools** across 7 categories:

| Category | Count | Coverage | Examples |
|----------|:-----:|----------|---------|
| **Database** | 30 | UniProt, KEGG, PDB, AlphaFold, BLAST, Ensembl, gnomAD, ClinVar, GWAS, OpenTargets, STRING, Reactome, and 18 more | `query_uniprot`, `blast_sequence` |
| **Screening** | 24 | Virtual screening, drug-gene networks, pathway search, survival analysis, disease-gene associations, biomarker discovery | `kegg_pathway_search`, `drug_gene_network_search` |
| **Sequence** | 19 | Protein structure prediction (Boltz2), enzyme kinetics (CataPro), mutation scoring (ESM, FoldX, Rosetta), phylogenetics (IQ-TREE), protein redesign (LigandMPNN, Chroma) | `run_boltz_protein_structure_prediction` |
| **Search** | 10 | Google, SerpAPI, arXiv, PubMed, Google Scholar, GitHub code & repos | `multi_source_search`, `query_pubmed` |
| **DevEnv** | 10 | Shell commands, conda/pip, GPU status, script creation and execution, training log monitoring | `run_shell_command`, `create_and_run_script` |
| **Web** | 4 | URL content extraction, PDF parsing, DOI supplementary lookup | `extract_url_content`, `extract_pdf_content` |
| **Biosecurity** | 1 | Sensitive data sanitisation with configurable strictness levels | `sanitize_bio_dataset` |

## Package Structure

```
labos/
├── __init__.py          # Version, public API
├── core.py              # Multi-agent orchestration engine
├── memory.py            # 3-tier memory system
├── knowledge.py         # TF-IDF + Mem0 knowledge base
├── ui.py                # Gradio web interface
├── cli.py               # CLI entry point
├── prompts/             # Agent prompt templates (YAML)
│   ├── manager.yaml
│   ├── researcher.yaml
│   ├── critic.yaml
│   └── toolmaker.yaml
└── tools/               # Biomedical tool library (98 tools)
    ├── __init__.py      # Tool registry
    ├── llm.py           # LLM helper for tool-internal calls
    ├── search.py        # Web + academic search (10 tools)
    ├── web.py           # URL / PDF content extraction (4 tools)
    ├── database.py      # Biomedical database queries (30 tools)
    ├── sequence.py      # Protein / enzyme analysis (19 tools)
    ├── screening.py     # Virtual screening & discovery (24 tools)
    ├── biosecurity.py   # Biosafety data sanitisation (1 tool)
    └── devenv.py        # Shell, conda, pip, scripts (10 tools)
```

## API Keys

| Key | Required | Purpose |
|-----|:--------:|---------|
| `OPENROUTER_API_KEY` | **Yes** | Powers all LLM agents via [OpenRouter](https://openrouter.ai/) |
| `SERPAPI_API_KEY` | No | Enhanced web search results |
| `MEM0_API_KEY` | No | [Mem0](https://mem0.ai/) platform for enhanced memory |

## Optional Dependencies

```bash
pip install labos[mem0]       # Mem0 enhanced memory
pip install labos[screening]  # RDKit for virtual screening
pip install labos[all]        # Everything
```

## Citation

If you find LabOS useful for your research, please cite:

```bibtex
@article{cong2025labos,
  title   = {LabOS: The AI-XR Co-Scientist That Sees and Works With Humans},
  author  = {Cong, Le and Zhang, Zaixi and Wang, Xiaotong and Di, Yin and Jin, Ruofan and Gerasimiuk, Michal and Wang, Yinkai and Dinesh, Ravi K. and Smerkous, David and Smerkous, Alex and Wu, Xuekun and Liu, Shilong and Li, Peishan and Zhu, Yi and Serrao, Simran and Zhao, Ning and Mohammad, Imran A. and Sunwoo, John B. and Wu, Joseph C. and Wang, Mengdi},
  journal = {arXiv preprint arXiv:2510.14861},
  year    = {2025}
}
```

## Related Projects

- **[STELLA](https://github.com/zaixizhang/STELLA)** — The foundational self-evolving agent framework that LabOS builds upon
- **[LabSuperVision](https://arxiv.org/abs/2510.14861)** — The first VLM benchmark spanning biomedical and materials science laboratories
- **[ai4labos.com](https://ai4labos.com/)** — Official LabOS project website

## License

Apache 2.0 — see [LICENSE](LICENSE).
