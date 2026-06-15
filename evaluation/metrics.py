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


def match_span_to_labels(
    span: str,
    labels: list[dict],
    overlap_threshold: float = 0.3
) -> int:
    """
    Matches a DriftSpan to RAGTruth ground truth labels.

    Method: string containment + word overlap.

    Why not exact match: DriftSpans are extracted claims that
    may paraphrase the original hallucinated text slightly.
    Exact match would miss valid detections.

    Why not semantic similarity: that would require running
    another model and introduces circular dependency with
    our own signals.

    Two-stage matching:
    1. Direct containment — label text inside span or vice versa
    2. Word overlap — at least overlap_threshold fraction of
       smaller text's words appear in larger text

    Args:
        span: DriftSpan text
        labels: RAGTruth label dicts with 'text' field
        overlap_threshold: minimum word overlap ratio

    Returns:
        1 if span matches a hallucinated label, 0 otherwise
    """
    if not labels:
        return 0

    span_lower = span.lower().strip()
    span_words = set(span_lower.split())

    for label in labels:
        label_text = label.get("text", "").lower().strip()
        if not label_text:
            continue
        label_words = set(label_text.split())

        # Stage 1 — direct containment
        if label_text in span_lower or span_lower in label_text:
            return 1

        # Stage 2 — word overlap
        if span_words and label_words:
            overlap = len(span_words & label_words)
            smaller = min(len(span_words), len(label_words))
            if smaller > 0 and (overlap / smaller) >= overlap_threshold:
                return 1

    return 0


def compute_metrics(
    predictions: list[int],
    ground_truth: list[int]
) -> dict:
    """
    Computes precision, recall, F1, accuracy.

    Args:
        predictions: list of 0/1 DRIFT verdicts
        ground_truth: list of 0/1 ground truth labels

    Returns:
        dict with precision, recall, f1, accuracy
    """
    if not predictions or not ground_truth:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0
        }

    return {
        "precision": round(
            float(precision_score(
                ground_truth, predictions, zero_division=0
            )), 4),
        "recall": round(
            float(recall_score(
                ground_truth, predictions, zero_division=0
            )), 4),
        "f1": round(
            float(f1_score(
                ground_truth, predictions, zero_division=0
            )), 4),
        "accuracy": round(
            float(accuracy_score(ground_truth, predictions)
            ), 4)
    }


def print_metrics(metrics: dict, level: str = "Response"):
    """Pretty prints metrics table."""
    print(f"\n{level}-level Metrics:")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    # Test metrics
    preds = [1, 0, 1, 1, 0, 0, 1, 0]
    truth = [1, 0, 1, 0, 0, 1, 1, 0]

    metrics = compute_metrics(preds, truth)
    print_metrics(metrics, "Test")

    # Test span matching
    span = "The drug was approved by FDA in 2022"
    labels = [
        {"text": "approved by FDA in 2022"},
        {"text": "no serious side effects"}
    ]
    match = match_span_to_labels(span, labels)
    print(f"\nSpan match test: {match} (expected 1)")

    span2 = "The treatment was effective for adults"
    match2 = match_span_to_labels(span2, labels)
    print(f"No match test: {match2} (expected 0)")