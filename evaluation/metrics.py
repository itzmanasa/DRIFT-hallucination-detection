import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score
)
import numpy as np


def match_span_to_labels(
    span: str,
    labels: list[dict],
    threshold: float = 0.3
) -> int:
    """
    Matches a DriftSpan to RAGTruth ground truth labels.
    
    Method: string containment check.
    If any hallucinated label text appears in the span
    or the span appears in any label text → match found.
    
    Args:
        span: DriftSpan text
        labels: RAGTruth label dicts with 'text' field
        threshold: word overlap threshold for fuzzy matching
    
    Returns:
        1 if span matches a hallucinated label, 0 otherwise
    """
    span_lower = span.lower().strip()
    span_words = set(span_lower.split())
    
    for label in labels:
        label_text = label.get("text", "").lower().strip()
        if not label_text:
            continue
        label_words = set(label_text.split())
        
        # Check 1 — direct containment
        if label_text in span_lower or span_lower in label_text:
            return 1
        
        # Check 2 — word overlap above threshold
        if span_words and label_words:
            overlap = len(span_words & label_words)
            ratio = overlap / min(len(span_words), len(label_words))
            if ratio >= threshold:
                return 1
    
    return 0


def compute_span_level_metrics(
    predictions: list[int],
    ground_truth: list[int]
) -> dict:
    """
    Computes precision, recall, F1 at span level.
    
    Args:
        predictions: list of 0/1 DRIFT verdicts per span
        ground_truth: list of 0/1 ground truth labels per span
    
    Returns:
        dict with precision, recall, f1, accuracy
    """
    if not predictions or not ground_truth:
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
    
    precision = precision_score(
        ground_truth, predictions, zero_division=0
    )
    recall = recall_score(
        ground_truth, predictions, zero_division=0
    )
    f1 = f1_score(
        ground_truth, predictions, zero_division=0
    )
    accuracy = accuracy_score(ground_truth, predictions)
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4)
    }


def compute_response_level_metrics(
    predictions: list[int],
    ground_truth: list[int]
) -> dict:
    """
    Computes precision, recall, F1 at response level.
    
    Args:
        predictions: list of 0/1 DRIFT response verdicts
        ground_truth: list of 0/1 ground truth response labels
    
    Returns:
        dict with precision, recall, f1, accuracy
    """
    return compute_span_level_metrics(predictions, ground_truth)


def print_metrics(metrics: dict, level: str = "Response"):
    """Pretty prints metrics."""
    print(f"\n{level}-level Metrics:")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    # Test with dummy predictions
    dummy_predictions = [1, 0, 1, 1, 0, 0, 1, 0]
    dummy_ground_truth = [1, 0, 1, 0, 0, 1, 1, 0]
    
    metrics = compute_response_level_metrics(
        dummy_predictions, 
        dummy_ground_truth
    )
    print_metrics(metrics, "Response")
    
    # Test span matching
    test_span = "The drug was approved by FDA in 2022"
    test_labels = [
        {"text": "approved by FDA in 2022"},
        {"text": "no serious side effects"}
    ]
    
    match = match_span_to_labels(test_span, test_labels)
    print(f"\nSpan matching test:")
    print(f"  Span: {test_span}")
    print(f"  Match found: {match} (expected 1)")