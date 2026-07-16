# DRIFT: Detection and Reasoning for Identified Fabrication Traces

DRIFT is a **zero-shot, black-box framework** for **span-level hallucination detection** in Retrieval-Augmented Generation (RAG) systems. Instead of assigning a single hallucination label to an entire response, DRIFT decomposes responses into atomic factual claims (DriftSpans) and evaluates each claim individually against retrieved evidence.

---

## Overview

DRIFT operates in four stages:

1. **DriftSpan Decomposition** – Splits an LLM-generated response into atomic factual claims.
2. **Retrieval Alignment** – Retrieves the most relevant evidence chunks for each claim.
3. **Tri-Signal Extraction** – Computes three complementary grounding signals:
   - **NLI Entailment Score**
   - **Lexical Divergence Rate**
   - **Cross-Chunk Contradiction Flag**
4. **Drift Index Computation** – Fuses the three signals into a per-span **Drift Index (DI)** to classify grounded and hallucinated claims.

---

## Experimental Results

Evaluated on the **RAGTruth** benchmark:

| Metric | Value |
|---------|------:|
| Dataset | RAGTruth Test Split |
| Samples | 2,574 |
| LLMs | 6 |
| Response-Level F1 | **0.5559** |
| Best News Summarization F1 | **0.3812** |
| Best Model (News) | Mistral-7B (**0.6429**) |

DRIFT outperforms all existing **zero-shot** baselines, including GPT-4 prompted hallucination detection.

---

# Installation

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for all dependencies.

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/DRIFT.git
cd DRIFT
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

# Dataset

DRIFT is evaluated using the **RAGTruth** benchmark.

Please follow the instructions in

```
data/README.md
```

to download and prepare the dataset.

---

# Usage

## Run Full Evaluation

```python
from run_drift import run_experiment

run_experiment(
    split="test",
    max_samples=None,
    task_type=None,
    k=3,
    threshold=0.3,
    alpha=0.1,
    save_results=True
)
```

---

## Run a Specific Task

```python
run_experiment(
    split="test",
    task_type="Summary",      # QA | Summary | Data2Text
    save_results=True
)
```

---

# Project Structure

```
DRIFT/
│
├── data/                    # Dataset loader and preprocessing
├── retriever/               # Retrieval alignment
├── transformations/         # DriftSpan decomposition & signal extraction
├── detector/                # Drift Index computation
├── evaluation/              # Evaluation metrics
├── results/                 # Experimental outputs
├── run_drift.py             # Main experiment pipeline
└── requirements.txt         # Project dependencies
```

---

# Models

| Component | Model |
|----------|------------------------------|
| NLI | cross-encoder/nli-deberta-v3-base |
| Embeddings | all-MiniLM-L6-v2 |
| Dependency Parsing | spaCy en_core_web_sm |

---

## Citation

Citation information will be added after publication.
```

(Update the citation after publication.)

---

# License

Specify the project license here (e.g., MIT, Apache-2.0, or BSD-3-Clause).

---

# Acknowledgements

This work was developed as part of the **CoDMAV Research Internship** at **PES University**.