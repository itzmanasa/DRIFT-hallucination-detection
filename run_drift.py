import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import json
import time
from tqdm import tqdm

from data.ragtruth_loader import load_ragtruth
from transformations.decomposer import decompose_into_driftspans
from retriever.retrieval_alignment import embed_chunks, get_top_k_chunks
from transformations.signal_extractor import extract_signals
from detector.drift_index import compute_drift_index, compute_response_drift_score
from detector.blindspot_map import BlindspotMap
from evaluation.metrics import (
    match_span_to_labels,
    compute_response_level_metrics,
    print_metrics
)


def run_drift_on_sample(
    sample: dict,
    k: int = 3,
    threshold: float = 0.5,
    alpha: float = 0.5
) -> dict:
    """
    Runs full DRIFT pipeline on one RAGTruth sample.
    
    Args:
        sample: DRIFT-ready sample from ragtruth_loader
        k: number of top chunks for retrieval alignment
        threshold: DI threshold for hallucination verdict
        alpha: lexical dampening coefficient
    
    Returns:
        result dict with predictions and scores
    """
    question = sample["question"]
    chunks = sample["chunks"]
    answer = sample["answer"]
    labels = sample["labels"]
    
    # Skip if answer is empty
    if not answer.strip():
        return None
    
    # TRANSFORMATION 1 — Decompose answer into DriftSpans
    spans = decompose_into_driftspans(answer)
    
    if not spans:
        return None
    
    # TRANSFORMATION 2 — Precompute chunk embeddings once
    if not chunks:
        return None
    
    chunk_embeddings = embed_chunks(chunks)
    
    span_results = []
    
    for span in spans:
        # TRANSFORMATION 2 — Retrieval alignment
        di_chunks = get_top_k_chunks(span, chunks, chunk_embeddings, k=k)
        
        # TRANSFORMATION 3 — Signal extraction
        signals = extract_signals(span, di_chunks)
        
        # BOX 3 — Drift Index
        di = compute_drift_index(
            signals["e_i"],
            signals["l_i"],
            signals["c_i"],
            alpha=alpha
        )
        
        # Match against ground truth
        gt = match_span_to_labels(span, labels)
        
        span_results.append({
            "span": span,
            "e_i": signals["e_i"],
            "l_i": signals["l_i"],
            "c_i": signals["c_i"],
            "di_score": di,
            "prediction": 1 if di < threshold else 0,
            "ground_truth": gt
        })
    
    # Response level
    di_scores = [s["di_score"] for s in span_results]
    rds = compute_response_drift_score(di_scores)
    response_prediction = 1 if rds < threshold else 0
    response_gt = sample["is_hallucinated"]
    
    return {
        "query_id": sample["query_id"],
        "model": sample["model"],
        "task_type": sample["task_type"],
        "rds": rds,
        "response_prediction": response_prediction,
        "response_ground_truth": response_gt,
        "span_results": span_results,
        "chunks": chunks
    }


def run_experiment(
    split: str = "test",
    max_samples: int = 100,
    k: int = 3,
    threshold: float = 0.5,
    alpha: float = 0.5,
    save_results: bool = True
):
    """
    Runs DRIFT on RAGTruth and computes evaluation metrics.
    
    Args:
        split: dataset split to use
        max_samples: number of samples to run
        k: retrieval alignment top-k
        threshold: hallucination threshold
        alpha: lexical dampening coefficient
        save_results: whether to save results to JSON
    """
    print("=" * 60)
    print("DRIFT EXPERIMENT")
    print("=" * 60)
    print(f"Split      : {split}")
    print(f"Samples    : {max_samples}")
    print(f"k          : {k}")
    print(f"Threshold  : {threshold}")
    print(f"Alpha      : {alpha}")
    print("=" * 60)
    
    # Load dataset
    samples = load_ragtruth(split=split, max_samples=max_samples)
    
    # Initialize Blindspot Map
    bmap = BlindspotMap()
    
    # Results storage
    all_results = []
    response_predictions = []
    response_ground_truths = []
    span_predictions = []
    span_ground_truths = []
    
    # Run DRIFT on each sample
    start_time = time.time()
    
    for sample in tqdm(samples, desc="Running DRIFT"):
        result = run_drift_on_sample(
            sample, k=k, threshold=threshold, alpha=alpha
        )
        
        if result is None:
            continue
        
        all_results.append(result)
        
        # Collect response-level metrics
        response_predictions.append(result["response_prediction"])
        response_ground_truths.append(result["response_ground_truth"])
        
        # Collect span-level metrics
        for span_result in result["span_results"]:
            span_predictions.append(span_result["prediction"])
            span_ground_truths.append(span_result["ground_truth"])
        
        # Update Blindspot Map
        for span_result in result["span_results"]:
            bmap.record_query(
                query_id=result["query_id"],
                span=span_result["span"],
                di_chunks=result["chunks"][:k],
                is_hallucinated=span_result["prediction"] == 1
            )
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    print(f"\nCompleted {len(all_results)} samples in {elapsed:.1f}s")
    
    response_metrics = compute_response_level_metrics(
        response_predictions, response_ground_truths
    )
    span_metrics = compute_response_level_metrics(
        span_predictions, span_ground_truths
    )
    
    print_metrics(response_metrics, "Response")
    print_metrics(span_metrics, "Span")
    
    # Print Blindspot Map
    print("\n" + "=" * 60)
    print("SOURCE BLINDSPOT MAP")
    print("=" * 60)
    bmap_summary = bmap.get_summary()
    scores = bmap_summary["blindspot_scores"]
    available = bmap_summary["available_counts"]
    ignored = bmap_summary["ignored_counts"]
    
    print(f"\n{'Type':<12} {'Available':<12} {'Ignored':<12} {'BS Score':<12}")
    print("-" * 48)
    for etype in ["numerical", "negation", "hedged"]:
        print(f"{etype:<12} {available[etype]:<12} "
              f"{ignored[etype]:<12} {scores[etype]:<12}")
    
    # Save results
    if save_results:
        os.makedirs("results", exist_ok=True)
        
        output = {
            "config": {
                "split": split,
                "max_samples": max_samples,
                "k": k,
                "threshold": threshold,
                "alpha": alpha
            },
            "response_metrics": response_metrics,
            "span_metrics": span_metrics,
            "blindspot_map": bmap_summary,
            "num_samples": len(all_results)
        }
        
        results_path = f"results/drift_results_{split}_{max_samples}.json"
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    return all_results, response_metrics, span_metrics, bmap_summary


if __name__ == "__main__":
    run_experiment(
        split="test",
        max_samples=100,
        k=3,
        threshold=0.5,
        alpha=0.5,
        save_results=True
    )