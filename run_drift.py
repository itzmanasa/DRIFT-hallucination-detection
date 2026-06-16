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
from detector.drift_index import (
    compute_drift_index,
    compute_response_drift_score,
    get_span_verdicts
)
from evaluation.metrics import (
    match_span_to_labels,
    compute_metrics,
    print_metrics
)


def run_drift_on_sample(
    sample: dict,
    k: int = 3,
    threshold: float = 0.3,
    alpha: float = 0.1
) -> dict:
    """
    Runs full DRIFT pipeline on one RAGTruth sample.

    Pipeline:
    T1: Decompose answer into DriftSpans
    T2: Embed chunks once, align each span to top-k chunks
    T3: Extract three signals per span
    B3: Compute DI per span, RDS for response
    Eval: Match against ground truth labels

    Args:
        sample: DRIFT-ready sample from ragtruth_loader
        k: top-k chunks for retrieval alignment (default 3)
        threshold: DI threshold for hallucination verdict (default 0.3)
        alpha: lexical dampening coefficient (default 0.1)

    Returns:
        result dict or None if sample cannot be processed
    """
    answer = sample["answer"]
    chunks = sample["chunks"]
    labels = sample["labels"]

    if not answer.strip() or not chunks:
        return None

    # TRANSFORMATION 1 — Decompose answer into DriftSpans
    spans = decompose_into_driftspans(answer)
    if not spans:
        return None

    # TRANSFORMATION 2 — Precompute chunk embeddings once per sample
    chunk_embeddings = embed_chunks(chunks)
    if len(chunk_embeddings) == 0:
        return None

    span_results = []

    for span in spans:
        # TRANSFORMATION 2 — Retrieval alignment for this span
        di_chunks = get_top_k_chunks(
            span, chunks, chunk_embeddings, k=k
        )

        # TRANSFORMATION 3 — Extract three signals
        signals = extract_signals(span, di_chunks)

        # BOX 3 — Compute Drift Index
        di = compute_drift_index(
            signals["e_i"],
            signals["l_i"],
            signals["c_i"],
            alpha=alpha
        )

        # Ground truth matching
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

    # Response level scoring
    di_scores = [s["di_score"] for s in span_results]
    rds = compute_response_drift_score(di_scores)
    response_prediction = 1 if rds < threshold else 0

    return {
        "query_id": sample["query_id"],
        "model": sample["model"],
        "task_type": sample["task_type"],
        "rds": rds,
        "response_prediction": response_prediction,
        "response_ground_truth": sample["is_hallucinated"],
        "num_spans": len(spans),
        "span_results": span_results
    }


def run_experiment(
    split: str = "test",
    max_samples: int = None,
    task_type: str = "QA",
    k: int = 3,
    threshold: float = 0.3,
    alpha: float = 0.1,
    save_results: bool = True
) -> dict:
    """
    Runs DRIFT on RAGTruth and computes evaluation metrics.

    Args:
        split: dataset split — "train", "test", or "all"
        max_samples: limit samples (None = all available)
        task_type: "QA", "Summary", "Data2Text", or None
        k: retrieval alignment top-k
        threshold: hallucination decision threshold
        alpha: lexical dampening coefficient
        save_results: save output to results/ folder

    Returns:
        dict with response_metrics, span_metrics, config
    """
    print("=" * 60)
    print("DRIFT — Detection and Reasoning for")
    print("        Identified Fabrication Traces")
    print("=" * 60)
    print(f"Split      : {split}")
    print(f"Task       : {task_type or 'all'}")
    print(f"Samples    : {max_samples or 'all'}")
    print(f"k          : {k}")
    print(f"Threshold  : {threshold}")
    print(f"Alpha      : {alpha}")
    print("=" * 60)

    # Load dataset
    samples = load_ragtruth(
        split=split,
        max_samples=max_samples,
        task_type=task_type
    )

    if not samples:
        print("No samples loaded. Check split and task_type.")
        return {}

    # Results storage
    all_results = []
    response_predictions = []
    response_ground_truths = []
    span_predictions = []
    span_ground_truths = []

    start_time = time.time()

    for sample in tqdm(samples, desc="Running DRIFT"):
        result = run_drift_on_sample(
            sample,
            k=k,
            threshold=threshold,
            alpha=alpha
        )

        if result is None:
            continue

        all_results.append(result)

        # Response level
        response_predictions.append(result["response_prediction"])
        response_ground_truths.append(result["response_ground_truth"])

        # Span level
        for sr in result["span_results"]:
            span_predictions.append(sr["prediction"])
            span_ground_truths.append(sr["ground_truth"])

    elapsed = time.time() - start_time
    print(f"\nCompleted {len(all_results)} samples in {elapsed:.1f}s")

    # Compute metrics
    response_metrics = compute_metrics(
        response_predictions, response_ground_truths
    )
    span_metrics = compute_metrics(
        span_predictions, span_ground_truths
    )

    print_metrics(response_metrics, "Response")
    print_metrics(span_metrics, "Span")

    # Per-model breakdown
    print("\nPer-model Response F1:")
    models = set(r["model"] for r in all_results)
    model_metrics = {}
    for model in sorted(models):
        model_results = [r for r in all_results if r["model"] == model]
        m_preds = [r["response_prediction"] for r in model_results]
        m_truth = [r["response_ground_truth"] for r in model_results]
        m_metrics = compute_metrics(m_preds, m_truth)
        model_metrics[model] = m_metrics
        print(f"  {model:<30} F1={m_metrics['f1']:.4f} "
              f"P={m_metrics['precision']:.4f} "
              f"R={m_metrics['recall']:.4f}")

    # Save results
    output = {
        "config": {
            "split": split,
            "task_type": task_type,
            "max_samples": max_samples,
            "k": k,
            "threshold": threshold,
            "alpha": alpha,
            "num_processed": len(all_results),
            "runtime_seconds": round(elapsed, 1)
        },
        "response_metrics": response_metrics,
        "span_metrics": span_metrics,
        "per_model_metrics": model_metrics
    }

    if save_results:
        os.makedirs("results", exist_ok=True)
        tag = f"{split}_{task_type or 'all'}_{len(all_results)}"
        results_path = f"results/drift_{tag}.json"
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {results_path}")

    return output


if __name__ == "__main__":
    run_experiment(
        split="test",
        max_samples=20,
        task_type="QA",
        k=3,
        threshold=0.35,
        alpha=0.1,
        save_results=True
    )