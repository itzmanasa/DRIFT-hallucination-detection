import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
from rouge_score import rouge_scorer

# Load NLI model once at module level
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
nli_pipeline = pipeline("text-classification", model=NLI_MODEL)

# Load ROUGE scorer once
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def get_nli_entailment_score(span: str, chunks: list[str]) -> float:
    """
    Signal 1 — NLI Entailment Score.
    Checks if the evidence (chunks) logically supports the claim (span).
    
    Takes the maximum entailment score across all chunks in Di.
    Reasoning: if ANY chunk supports the claim, it is grounded.
    
    Args:
        span: one DriftSpan (si)
        chunks: relevant chunks (Di)
    
    Returns:
        e_i — entailment score in [0, 1]
    """
    max_entailment = 0.0
    
    for chunk in chunks:
        # NLI input: premise = chunk (evidence), hypothesis = span (claim)
        result = nli_pipeline(
            f"{chunk} [SEP] {span}",
            truncation=True,
            max_length=512
        )
        
        # Extract entailment score
        for item in result:
            if item["label"].upper() == "ENTAILMENT":
                score = item["score"]
                if score > max_entailment:
                    max_entailment = score
    
    return max_entailment


def get_lexical_divergence_rate(span: str, chunks: list[str]) -> float:
    """
    Signal 2 — Lexical Divergence Rate.
    Measures how different the span's wording is from the source chunks.
    High divergence = LLM paraphrased from memory, not from retrieved text.
    
    Uses ROUGE-L: measures longest common subsequence overlap.
    Divergence = 1 - ROUGE-L score.
    
    Args:
        span: one DriftSpan (si)
        chunks: relevant chunks (Di)
    
    Returns:
        l_i — divergence rate in [0, 1], higher = more diverged
    """
    # Combine all chunks into one reference text
    combined_chunks = " ".join(chunks)
    
    # Compute ROUGE-L between span and combined chunks
    scores = scorer.score(combined_chunks, span)
    rouge_l = scores["rougeL"].fmeasure
    
    # Divergence is inverse of overlap
    divergence = 1.0 - rouge_l
    
    return divergence


def get_cross_chunk_contradiction_flag(
    span: str, 
    chunks: list[str]
) -> int:
    """
    Signal 3 — Cross-Chunk Contradiction Flag.
    Checks if ANY chunk in Di contradicts the span.
    Even if top chunk supports the span, another chunk may contradict it.
    
    Args:
        span: one DriftSpan (si)
        chunks: relevant chunks (Di)
    
    Returns:
        c_i — binary flag, 1 if contradiction found, 0 otherwise
    """
    for chunk in chunks:
        result = nli_pipeline(
            f"{chunk} [SEP] {span}",
            truncation=True,
            max_length=512
        )
        
        for item in result:
            if item["label"].upper() == "CONTRADICTION":
                # Contradiction found — flag immediately
                if item["score"] > 0.7:
                    return 1
    
    return 0


def extract_signals(
    span: str,
    chunks: list[str]
) -> dict:
    """
    Main function. Extracts all three signals for one DriftSpan.
    
    Args:
        span: one DriftSpan (si)
        chunks: relevant chunks Di
    
    Returns:
        dictionary with e_i, l_i, c_i
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
    # Test with grounded span
    grounded_span = "The FDA granted approval to the drug in 2022"
    grounded_chunks = [
        "The FDA granted approval to the medication in late 2022 after reviewing trial data.",
        "Clinical trials reported minimal adverse events with no serious complications observed.",
        "The drug showed strong efficacy in reducing fever symptoms across age groups."
    ]
    
    # Test with hallucinated span
    hallucinated_span = "The drug was approved in Europe in 2021"
    hallucinated_chunks = [
        "Regulatory bodies in Europe are still reviewing the drug for approval.",
        "The FDA granted approval to the medication in late 2022 after reviewing trial data.",
        "Manufacturing of the drug began in 2021 at facilities in New Jersey."
    ]
    
    print("=" * 50)
    print("TEST 1 — Grounded span (should have high DI)")
    print("=" * 50)
    signals = extract_signals(grounded_span, grounded_chunks)
    for key, val in signals.items():
        print(f"  {key}: {val}")
    
    print()
    print("=" * 50)
    print("TEST 2 — Hallucinated span (should have low DI)")
    print("=" * 50)
    signals = extract_signals(hallucinated_span, hallucinated_chunks)
    for key, val in signals.items():
        print(f"  {key}: {val}")