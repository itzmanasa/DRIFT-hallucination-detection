import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")


def compute_drift_index(
    e_i: float,
    l_i: float,
    c_i: int,
    alpha: float = 0.1
) -> float:
    """
    Computes the Drift Index for one DriftSpan.

    Formula: DI(si) = e_i * (1 - alpha * l_i) * (1 - c_i)

    Design decisions:
    - Multiplicative: all three signals must be favorable
      for DI to be high. One bad signal drags score down.
    - Contradiction veto: c_i=1 makes DI=0 exactly.
      A contradiction is a hard failure, not a soft penalty.
    - Alpha dampening: lexical divergence is weighted by alpha
      because LLMs naturally paraphrase source text. Full
      lexical penalty would over-penalize valid paraphrases.
      alpha=0.1 keeps lexical as weak signal, NLI dominates.

    Properties:
    - DI in [0, 1]
    - DI close to 1 = well grounded
    - DI close to 0 = likely hallucinated
    - DI = 0 exactly when c_i = 1 (contradiction found)

    Args:
        e_i: NLI entailment score in [0, 1]
        l_i: lexical divergence rate in [0, 1]
        c_i: cross-chunk contradiction flag in {0, 1}
        alpha: lexical dampening coefficient (default 0.1)

    Returns:
        DI score in [0, 1]
    """
    di = e_i * (1 - alpha * l_i) * (1 - c_i)
    return round(float(di), 4)


def compute_response_drift_score(drift_indices: list[float]) -> float:
    """
    Computes Response-level Drift Score for a full answer.

    Formula: RDS(A) = mean of all DI(si) scores

    Why mean and not minimum:
    - Minimum is too punishing — one outlier span destroys score
    - Mean reflects overall grounding quality of the response
    - Partial hallucination produces intermediate score, not zero

    Why mean and not proportion below threshold:
    - Proportion is threshold-sensitive and ignores magnitude
    - Mean preserves full information about DI distribution

    Args:
        drift_indices: list of DI scores for all spans in answer

    Returns:
        RDS in [0, 1], higher = more grounded
    """
    if not drift_indices:
        return 0.0
    rds = sum(drift_indices) / len(drift_indices)
    return round(float(rds), 4)


def get_span_verdicts(
    spans: list[str],
    drift_indices: list[float],
    threshold: float = 0.3
) -> list[dict]:
    """
    Assigns HALLUCINATED or GROUNDED verdict to each span.

    Threshold tau=0.3 chosen because:
    - Grounded spans with alpha=0.1 typically score 0.7-0.95
    - Hallucinated spans typically score 0.0-0.2
    - Natural separation falls around 0.3-0.4
    - tau=0.3 minimizes overlap between distributions

    Args:
        spans: list of DriftSpan strings
        drift_indices: corresponding DI scores
        threshold: spans below this flagged as hallucinated

    Returns:
        list of verdict dicts
    """
    verdicts = []
    for span, di in zip(spans, drift_indices):
        verdicts.append({
            "span": span,
            "di_score": di,
            "verdict": "HALLUCINATED" if di < threshold else "GROUNDED"
        })
    return verdicts


if __name__ == "__main__":
    # Test DI formula
    print("DI Formula Tests:")
    print()

    # Grounded span — high NLI, low divergence, no contradiction
    di = compute_drift_index(e_i=0.99, l_i=0.15, c_i=0, alpha=0.1)
    print(f"Grounded span: DI = {di} (expect ~0.97)")

    # Hallucinated — low NLI, high divergence, no contradiction
    di = compute_drift_index(e_i=0.05, l_i=0.85, c_i=0, alpha=0.1)
    print(f"Hallucinated span: DI = {di} (expect ~0.046)")

    # Contradiction — any values, c_i=1 vetoes everything
    di = compute_drift_index(e_i=0.99, l_i=0.10, c_i=1, alpha=0.1)
    print(f"Contradiction span: DI = {di} (expect 0.0)")

    # Response level
    print()
    dis = [0.95, 0.88, 0.05, 0.91]
    rds = compute_response_drift_score(dis)
    print(f"RDS for {dis} = {rds}")

    verdicts = get_span_verdicts(
        ["span1", "span2", "span3", "span4"],
        dis,
        threshold=0.3
    )
    print()
    print("Span verdicts:")
    for v in verdicts:
        print(f"  {v['span']}: DI={v['di_score']} → {v['verdict']}")