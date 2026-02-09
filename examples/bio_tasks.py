"""
LabOS Biomedical Task Examples.

Demonstrates various biomedical research queries that LabOS can handle.

Prerequisites:
  pip install labos
  export OPENROUTER_API_KEY=your-key-here
"""

import labos

# Initialise once
agent = labos.initialize()

# ---- Example 1: Literature Search ----------------------------------------
print("=" * 60)
print("Example 1: Literature Search")
print("=" * 60)

result = labos.run_task(
    "Find recent publications on spatial transcriptomics methods "
    "for tissue mapping. Summarise key computational approaches."
)
print(result)

# ---- Example 2: Protein Analysis -----------------------------------------
print("\n" + "=" * 60)
print("Example 2: Protein Analysis")
print("=" * 60)

result = labos.run_task(
    "Query UniProt for the human TP53 protein. "
    "Summarise its function, known mutations, and disease associations."
)
print(result)

# ---- Example 3: Pathway Search -------------------------------------------
print("\n" + "=" * 60)
print("Example 3: Pathway Search")
print("=" * 60)

result = labos.run_task(
    "Search KEGG for the apoptosis signalling pathway. "
    "List the key genes involved and their roles."
)
print(result)

# ---- Example 4: Drug Discovery -------------------------------------------
print("\n" + "=" * 60)
print("Example 4: Drug Discovery")
print("=" * 60)

result = labos.run_task(
    "Search for recent research on GLP-1 receptor agonists for "
    "type 2 diabetes treatment. Compare efficacy of major candidates."
)
print(result)
