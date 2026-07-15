# DRIFT: Detection and Reasoning for Identified Fabrication Traces

A zero-shot, black-box framework for span-level hallucination 
detection in Retrieval-Augmented Generation (RAG) systems.

## Overview

DRIFT decomposes LLM-generated answers into atomic factual 
units called DriftSpans and evaluates each against retrieved 
evidence using three complementary signals:
- NLI Entailment Score
- Lexical Divergence Rate  
- Cross-Chunk Contradiction Flag

These signals are fused into a per-span Drift Index (DI) score.

## Results

Evaluated on RAGTruth test split (2,574 samples, 6 LLMs):
- Response-level F1: 0.5559
- Outperforms all zero-shot baselines including GPT-4 (F1=0.476)

## Setup

### Requirements
- Python 3.11
- See requirements.txt for dependencies

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/DRIFT.git
cd DRIFT
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Dataset
See data/README.md for dataset download instructions.

## Usage

### Run DRIFT on RAGTruth
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

### Run on specific domain
```python
run_experiment(
    split="test",
    task_type="Summary",  # QA, Summary, or Data2Text
    save_results=True
)
```

## Project Structure
DRIFT/
├── data/                  # Dataset loader
├── retriever/             # Retrieval alignment
├── transformations/       # Decomposer + signal extractor
├── detector/              # Drift Index computation
├── evaluation/            # Metrics
├── results/               # Experimental results
└── run_drift.py           # Main pipeline

## Models Used
- NLI: cross-encoder/nli-deberta-v3-base
- Embeddings: all-MiniLM-L6-v2
- Decomposition: spaCy en_core_web_sm

## Citation
[Paper citation will go here after publication]