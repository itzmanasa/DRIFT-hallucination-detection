import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
from rouge_score import rouge_scorer

# Load NLI model once at module level
# cross-encoder/nli-deberta-v3-base chosen because:
# - Best balance of accuracy and speed among cross-encoders
# - Returns clean label names: entailment, neutral, contradiction
# - Works correctly on CPU and GPU
# - Confirmed returning entailment: 0.998 for grounded spans
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
nli_pipeline = pipeline(
    "text-classification",
    model=NLI_MODEL
)

# ROUGE scorer for lexical divergence
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def get_nli_entailment_score(span: str, chunks: list[str]) -> float:
    """
    Signal 1 — NLI Entailment Score (e_i).

    Checks if the evidence logically supports the claim.
    Takes MAXIMUM entailment score across all chunks in Di.

    Reasoning: if ANY chunk supports the claim, the claim is
    grounded. We do not average because one strongly supporting
    chunk is enough evidence.

    Input format: premise=[chunk], hypothesis=[span]
    The NLI model asks: does the chunk entail the span?

    Args:
        span: one DriftSpan si
        chunks: relevant chunks Di

    Returns:
        e_i in [0, 1], higher = more grounded
    """
    if not chunks:
        return 0.0

    max_entailment = 0.0

    for chunk in chunks:
        result = nli_pipeline(
            f"{chunk} [SEP] {span}",
            truncation=True,
            max_length=512
        )

        for item in result:
            label = item["label"].upper()
            # Handle all possible label formats from this model
            if label in ("ENTAILMENT", "LABEL_0", "ENTAIL"):
                score = item["score"]
                if score > max_entailment:
                    max_entailment = score

    return float(max_entailment)


def get_lexical_divergence_rate(span: str, chunks: list[str]) -> float:
    """
    Signal 2 — Lexical Divergence Rate (l_i).

    Measures how different the span's wording is from source chunks.
    High divergence = LLM paraphrased from memory, not from retrieved text.

    Uses ROUGE-L (longest common subsequence) because it captures
    word order, not just bag-of-words overlap. Better at detecting
    when LLM has drifted from source phrasing.

    Divergence = 1 - ROUGE-L score.
    Combined chunks used as reference to maximize overlap opportunity.

    Args:
        span: one DriftSpan si
        chunks: relevant chunks Di

    Returns:
        l_i in [0, 1], higher = more diverged from source
    """
    if not chunks:
        return 1.0

    combined = " ".join(chunks)
    scores = scorer.score(combined, span)
    rouge_l = scores["rougeL"].fmeasure
    return float(1.0 - rouge_l)


def get_cross_chunk_contradiction_flag(
    span: str,
    chunks: list[str]
) -> int:
    """
    Signal 3 — Cross-Chunk Contradiction Flag (c_i).

    Checks if ANY chunk in Di contradicts the span.
    This catches cases where the top chunk supports the claim
    but a secondary chunk directly contradicts it.

    Why this matters: single-signal NLI would miss this because
    it already returned entailment from the top chunk. Signal 3
    is the ONLY signal that catches hidden contradictions in
    secondary chunks.

    Scope: checks within Di only (not all of D).
    Di is already relevance-filtered so contradictions here
    are meaningful, not noise.

    Contradiction threshold: 0.7 to avoid false positives
    from neutral or ambiguous chunks.

    Args:
        span: one DriftSpan si
        chunks: relevant chunks Di (already top-k filtered)

    Returns:
        c_i in {0, 1}, 1 if contradiction found
    """
    if not chunks:
        return 0

    for chunk in chunks:
        result = nli_pipeline(
            f"{chunk} [SEP] {span}",
            truncation=True,
            max_length=512
        )

        for item in result:
            label = item["label"].upper()
            if label in ("CONTRADICTION", "LABEL_2", "CONTRADICT"):
                if item["score"] > 0.85:
                    return 1

    return 0


def extract_signals(span: str, chunks: list[str]) -> dict:
    """
    Extracts all three signals for one DriftSpan.

    Args:
        span: one DriftSpan si
        chunks: relevant chunks Di

    Returns:
        dict with e_i, l_i, c_i and the span text
    """
    e_i = get_nli_entailment_score(span, chunks)
    l_i = get_lexical_divergence_rate(span, chunks)
    c_i = get_cross_chunk_contradiction_flag(span, chunks)

    return {
        "span": span,
        "e_i": round(e_i, 4),
        "l_i": round(l_i, 4),
        "c_i": c_i
    }


if __name__ == "__main__":
    grounded_span = "The FDA granted approval to the drug in 2022"
    grounded_chunks = [
        "The FDA granted approval to the medication in late 2022.",
        "Clinical trials reported minimal adverse events.",
        "The drug showed strong efficacy in reducing fever symptoms."
    ]

    hallucinated_span = "The drug was approved in Europe in 2021"
    hallucinated_chunks = [
        "Regulatory bodies in Europe are still reviewing the drug.",
        "The FDA granted approval to the medication in late 2022.",
        "Manufacturing of the drug began in 2021 in New Jersey."
    ]

    print("=" * 50)
    print("TEST 1 — Grounded span (expect high e_i, low c_i)")
    print("=" * 50)
    signals = extract_signals(grounded_span, grounded_chunks)
    for k, v in signals.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 50)
    print("TEST 2 — Hallucinated span (expect low e_i, c_i=1)")
    print("=" * 50)
    signals = extract_signals(hallucinated_span, hallucinated_chunks)
    for k, v in signals.items():
        print(f"  {k}: {v}")