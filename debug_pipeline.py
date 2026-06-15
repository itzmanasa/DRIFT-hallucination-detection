import json
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

from data.ragtruth_loader import load_ragtruth
from transformations.decomposer import decompose_into_driftspans
from retriever.retrieval_alignment import embed_chunks, get_top_k_chunks
from transformations.signal_extractor import extract_signals
from detector.drift_index import compute_drift_index

# Load one QA sample
samples = load_ragtruth(split="test", max_samples=5, task_type="QA")

if not samples:
    print("No QA samples found")
else:
    sample = samples[0]
    print(f"Question: {sample['question'][:100]}")
    print(f"Answer: {sample['answer'][:200]}")
    print(f"Number of chunks: {len(sample['chunks'])}")
    print(f"Is hallucinated: {sample['is_hallucinated']}")
    print(f"Labels: {sample['labels']}")
    print()
    
    spans = decompose_into_driftspans(sample['answer'])
    print(f"DriftSpans extracted: {len(spans)}")
    for s in spans:
        print(f"  - {s}")
    
    print()
    chunk_embeddings = embed_chunks(sample['chunks'])
    
    # Check first span
    if spans:
        span = spans[0]
        di_chunks = get_top_k_chunks(span, sample['chunks'], chunk_embeddings, k=3)
        signals = extract_signals(span, di_chunks)
        di = compute_drift_index(signals['e_i'], signals['l_i'], signals['c_i'], alpha=0.1)
        
        print(f"First span: {span}")
        print(f"Top chunks retrieved:")
        for i, c in enumerate(di_chunks):
            print(f"  [{i+1}] {c[:100]}")
        print(f"Signals: e_i={signals['e_i']}, l_i={signals['l_i']}, c_i={signals['c_i']}")
        print(f"DI score: {di}")