"""
LabOS Quick Start — minimal programmatic usage.

Prerequisites:
  pip install labos
  export OPENROUTER_API_KEY=your-key-here
"""

import labos

# Initialise the multi-agent system
agent = labos.initialize(
    use_template=True,       # Enable knowledge-base templates
    use_mem0=False,          # Set True if you have Mem0 installed
    enable_tool_creation=True,
)

# Run a simple task
result = labos.run_task(
    "Search PubMed for recent papers on CRISPR-Cas9 gene editing "
    "and summarise the top 3 findings."
)
print(result)
