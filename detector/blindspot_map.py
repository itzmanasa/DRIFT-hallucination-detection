import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import re
import json
from transformers import pipeline

# Load zero-shot classifier for ambiguous chunks
# Only loads when needed — lazy loading
_classifier = None

def get_classifier():
    """Lazy load classifier only when needed."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    return _classifier


# Evidence type labels
EVIDENCE_TYPES = ["numerical", "negation", "hedged"]

# Rule-based patterns
NUMERICAL_PATTERN = re.compile(r'\b\d+\.?\d*\s*(%|percent|mg|kg|km|ml|hz)?\b')
NEGATION_WORDS = {"no", "not", "never", "failed", "neither", 
                  "nor", "without", "lack", "absent", "negative"}
HEDGED_WORDS = {"may", "might", "suggest", "suggests", "suggested",
                "possible", "possibly", "preliminary", "perhaps",
                "could", "uncertain", "unclear", "potential",
                "potentially", "indicate", "indicates", "appears"}


def classify_chunk_rule_based(chunk: str) -> list[str]:
    """
    Rule-based evidence type classification.
    Fast, no model needed.
    Returns list of matched types (multi-label).
    
    Returns empty list if no type matched (ambiguous).
    Returns 3 types if all match (also ambiguous — send to LLM).
    """
    chunk_lower = chunk.lower()
    chunk_words = set(chunk_lower.split())
    labels = []
    
    # Check numerical
    if NUMERICAL_PATTERN.search(chunk):
        labels.append("numerical")
    
    # Check negation
    if chunk_words & NEGATION_WORDS:
        labels.append("negation")
    
    # Check hedged
    if chunk_words & HEDGED_WORDS:
        labels.append("hedged")
    
    return labels


def classify_chunk_llm(chunk: str) -> list[str]:
    """
    LLM fallback classifier for ambiguous chunks.
    Uses zero-shot classification with BART.
    """
    classifier = get_classifier()
    
    result = classifier(
        chunk,
        candidate_labels=EVIDENCE_TYPES,
        multi_label=True
    )
    
    # Keep labels with score > 0.5
    labels = [
        label for label, score 
        in zip(result["labels"], result["scores"]) 
        if score > 0.5
    ]
    
    # If nothing above threshold take top 1
    if not labels:
        labels = [result["labels"][0]]
    
    return labels


def classify_chunk(chunk: str) -> list[str]:
    """
    Full classification pipeline.
    Rule-based first, LLM fallback for ambiguous cases.
    
    Ambiguous = 0 labels or all 3 labels from rule-based.
    """
    rule_labels = classify_chunk_rule_based(chunk)
    
    # Ambiguous cases go to LLM
    if len(rule_labels) == 0 or len(rule_labels) == 3:
        return classify_chunk_llm(chunk)
    
    return rule_labels


class BlindspotMap:
    """
    Corpus-level aggregation of ignored chunks across N queries.
    Builds the Source Blindspot Map after running DRIFT on many queries.
    """
    
    def __init__(self):
        # For each evidence type track:
        # - how many times a chunk of that type was available
        # - how many times it was ignored
        self.available = {t: 0 for t in EVIDENCE_TYPES}
        self.ignored = {t: 0 for t in EVIDENCE_TYPES}
        self.log = []  # Full log for inspection
    
    def record_query(
        self,
        query_id: str,
        span: str,
        di_chunks: list[str],
        is_hallucinated: bool
    ):
        """
        Records one span's evidence chunks into the map.
        
        For each chunk in Di:
        - Always counts as available for its evidence types
        - If span is hallucinated, counts as ignored
        
        Args:
            query_id: identifier for this query
            span: the DriftSpan text
            di_chunks: the Di chunks for this span
            is_hallucinated: whether this span was hallucinated
        """
        for chunk in di_chunks:
            chunk_types = classify_chunk(chunk)
            
            for evidence_type in chunk_types:
                # Always record as available
                self.available[evidence_type] += 1
                
                # If hallucinated, record as ignored
                if is_hallucinated:
                    self.ignored[evidence_type] += 1
            
            # Log entry for inspection
            self.log.append({
                "query_id": query_id,
                "span": span,
                "chunk": chunk,
                "chunk_types": chunk_types,
                "hallucinated": is_hallucinated
            })
    
    def compute_blindspot_scores(self) -> dict:
        """
        Computes blindspot score for each evidence type.
        
        BS(T) = ignored(T) / available(T)
        
        Higher score = LLM more systematically ignores this type.
        """
        scores = {}
        for evidence_type in EVIDENCE_TYPES:
            available = self.available[evidence_type]
            ignored = self.ignored[evidence_type]
            
            if available == 0:
                scores[evidence_type] = 0.0
            else:
                scores[evidence_type] = round(ignored / available, 4)
        
        return scores
    
    def get_summary(self) -> dict:
        """Returns full Blindspot Map summary."""
        scores = self.compute_blindspot_scores()
        
        return {
            "blindspot_scores": scores,
            "available_counts": self.available.copy(),
            "ignored_counts": self.ignored.copy(),
            "total_spans_logged": len(self.log)
        }
    
    def save(self, path: str):
        """Saves the blindspot map to a JSON file."""
        summary = self.get_summary()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Blindspot map saved to {path}")


if __name__ == "__main__":
    # Simulate running DRIFT across multiple queries
    bmap = BlindspotMap()
    
    # Query 1 — hallucinated span with numerical evidence ignored
    bmap.record_query(
        query_id="q1",
        span="The drug showed 90% efficacy",
        di_chunks=[
            "Clinical trials showed 67% efficacy in the treatment group.",
            "The study enrolled 450 patients across three sites.",
            "Results were statistically significant with p < 0.05."
        ],
        is_hallucinated=True
    )
    
    # Query 2 — grounded span, nothing ignored
    bmap.record_query(
        query_id="q2",
        span="The FDA approved the drug in 2022",
        di_chunks=[
            "The FDA granted approval to the medication in late 2022.",
            "Approval was based on Phase 3 trial results.",
            "The drug may require additional post-market surveillance."
        ],
        is_hallucinated=False
    )
    
    # Query 3 — hallucinated span with negation evidence ignored
    bmap.record_query(
        query_id="q3",
        span="The treatment had no side effects",
        di_chunks=[
            "Patients did not report serious adverse events.",
            "No significant differences were found between groups.",
            "The treatment showed promising results in 78% of cases."
        ],
        is_hallucinated=True
    )
    
    print("=" * 50)
    print("SOURCE BLINDSPOT MAP")
    print("=" * 50)
    summary = bmap.get_summary()
    
    print(f"\nTotal spans logged: {summary['total_spans_logged']}")
    print("\nEvidence Type Analysis:")
    print(f"{'Type':<12} {'Available':<12} {'Ignored':<12} {'BS Score':<12}")
    print("-" * 48)
    for etype in EVIDENCE_TYPES:
        avail = summary['available_counts'][etype]
        ign = summary['ignored_counts'][etype]
        score = summary['blindspot_scores'][etype]
        print(f"{etype:<12} {avail:<12} {ign:<12} {score:<12}")