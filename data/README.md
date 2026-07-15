# Data

DRIFT uses the RAGTruth dataset.

## Download Instructions

1. Clone the official RAGTruth repository:
git clone https://github.com/ParticleMedia/RAGTruth

2. Copy the dataset files to this folder:
cp RAGTruth/dataset/response.jsonl data/
cp RAGTruth/dataset/source_info.jsonl data/

## Dataset Details
- 17,790 RAG responses from 6 LLMs
- Human annotated at span level
- Three task types: QA, Summary, Data2Text
- Paper: RAGTruth (ACL 2024)