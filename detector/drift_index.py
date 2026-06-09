import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")


def compute_drift_index(
    e_i: float,
    l_i: float,
    c_i: int,
    alpha: float = 0.5
) -> float:
    """
    Computes the Drift Index for one DriftSpan.
    
    Formula: DI(si) = e_i * (1 - alpha*l_i) * (1 - c_i)
    
    alpha: dampening coefficient for lexical divergence
    - alpha=1.0: full lexical penalty (original formula)
    - alpha=0.5: dampened lexical penalty (default)
    - alpha=0.0: lexical signal ignored entirely
    
    Rationale: LLMs often paraphrase source text using different
    words while preserving meaning. Full lexical penalty would
    over-penalize semantically valid paraphrases. Alpha dampens
    this without removing the signal entirely.
    
    Args:
        e_i: NLI entailment score in [0, 1]
        l_i: lexical divergence rate in [0, 1]
        c_i: cross-chunk contradiction flag in {0, 1}
        alpha: lexical dampening coefficient in [0, 1]
    
    Returns:
        DI score in [0, 1]
    """
    di = e_i * (1 - alpha * l_i) * (1 - c_i)
    return round(di, 4)


def compute_response_drift_score(drift_indices: list[float]) -> float:
    """
    Computes the Response-level Drift Score for a full answer.
    
    Formula: RDS(A) = average of all DI(si) scores
    
    Args:
        drift_indices: list of DI scores for all spans in answer
    
    Returns:
        RDS score in [0, 1], higher = more grounded
    """
    if not drift_indices:
        return 0.0
    
    rds = sum(drift_indices) / len(drift_indices)
    return round(rds, 4)


def get_hallucinated_spans(
    spans: list[str],
    drift_indices: list[float],
    threshold: float = 0.5
) -> list[dict]:
    """
    Returns spans flagged as hallucinated based on DI threshold.
    
    Args:
        spans: list of DriftSpan strings
        drift_indices: corresponding DI scores
        threshold: spans below this are flagged (default 0.5)
    
    Returns:
        list of dicts with span text and DI score
    """
    hallucinated = []
    for span, di in zip(spans, drift_indices):
        if di < threshold:
            hallucinated.append({
                "span": span,
                "di_score": di,
                "verdict": "HALLUCINATED"
            })
    return hallucinated


def analyze_response(
    spans: list[str],
    signals: list[dict],
    threshold: float = 0.5,
    alpha: float = 0.5
) -> dict:
    """
    Full analysis of one LLM response.
    Takes spans and their signals, returns complete DRIFT output.
    
    Args:
        spans: list of DriftSpan strings
        signals: list of signal dicts from signal_extractor
        threshold: hallucination threshold for span verdicts
    
    Returns:
        complete analysis dict with DI scores, RDS, and verdicts
    """
    drift_indices = []
    span_results = []
    
    for signal in signals:
        di = compute_drift_index(
            signal["e_i"],
            signal["l_i"],
            signal["c_i"],
            alpha=alpha
        )
        drift_indices.append(di)
        
        span_results.append({
            "span": signal["span"],
            "e_i": signal["e_i"],
            "l_i": signal["l_i"],
            "c_i": signal["c_i"],
            "di_score": di,
            "verdict": "HALLUCINATED" if di < threshold else "GROUNDED"
        })
    
    rds = compute_response_drift_score(drift_indices)
    hallucinated_spans = get_hallucinated_spans(spans, drift_indices, threshold)
    
    return {
        "rds": rds,
        "response_verdict": "HALLUCINATED" if rds < threshold else "GROUNDED",
        "num_spans": len(spans),
        "num_hallucinated": len(hallucinated_spans),
        "span_results": span_results
    }


if __name__ == "__main__":
    # Test with signals from signal_extractor output
    test_signals = [
        # Grounded span
        {"span": "The FDA granted approval to the drug in 2022",
         "e_i": 0.9977, "l_i": 0.6522, "c_i": 0},
        # Hallucinated span
        {"span": "The drug was approved in Europe in 2021",
         "e_i": 0.0, "l_i": 0.7333, "c_i": 1},
        # Another grounded span
        {"span": "Clinical trials reported no serious complications",
         "e_i": 0.8901, "l_i": 0.4210, "c_i": 0},
    ]
    
    spans = [s["span"] for s in test_signals]
    result = analyze_response(spans, test_signals)
    
    print("=" * 50)
    print("DRIFT ANALYSIS RESULT")
    print("=" * 50)
    print(f"Response Drift Score (RDS): {result['rds']}")
    print(f"Response Verdict: {result['response_verdict']}")
    print(f"Total spans: {result['num_spans']}")
    print(f"Hallucinated spans: {result['num_hallucinated']}")
    print()
    print("Span-level results:")
    for span_result in result["span_results"]:
        print(f"\n  Span: {span_result['span']}")
        print(f"  DI Score: {span_result['di_score']}")
        print(f"  Verdict: {span_result['verdict']}")