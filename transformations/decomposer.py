import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load spaCy model for sentence splitting
nlp = spacy.load("en_core_web_sm")

# Load flan-t5 for claim refinement
MODEL_NAME = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def split_into_sentences(text: str) -> list[str]:
    """
    Uses spaCy to split text into sentences.
    This is reliable and handles edge cases well.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    sentences = [s for s in sentences if len(s.split()) >= 3]
    return sentences


def sentence_contains_multiple_claims(sentence: str) -> bool:
    """
    Checks if a sentence likely contains more than one factual claim.
    Signals: coordinating conjunctions joining independent clauses.
    Examples: "X was approved in 2022 and had no side effects"
    """
    doc = nlp(sentence)
    
    # Count root verbs — multiple roots = multiple claims
    root_verbs = [token for token in doc if token.dep_ == "ROOT"]
    if len(root_verbs) > 1:
        return True
    
    # Check for coordinating conjunctions between clauses
    conjunctions = [token for token in doc 
                   if token.dep_ == "conj" and token.pos_ == "VERB"]
    if len(conjunctions) > 0:
        return True
        
    return False


def split_sentence_into_claims(sentence: str) -> list[str]:
    """
    Uses spaCy dependency parsing to split complex sentences
    into atomic claims. More reliable than flan-t5 for this task.
    """
    doc = nlp(sentence)
    claims = []
    
    # Find the main subject of the sentence
    subject = None
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject = token
            break
    
    # Find root verb and all conjunct verbs
    root = None
    conjunct_verbs = []
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
        if token.dep_ == "conj" and token.pos_ == "VERB":
            conjunct_verbs.append(token)
    
    if not root or not conjunct_verbs:
        # No conjunction found — sentence is already atomic
        return [sentence]
    
    # Build first claim: subject + root verb + its dependents
    # (everything before the first conjunction)
    # Build first claim: subject + root verb + its dependents
    # (everything before the first conjunction)
    first_claim_tokens = []
    for token in doc:
        if token == conjunct_verbs[0]:
            break
        # Skip coordinating conjunctions at end
        if token.dep_ == "cc":
            continue
        first_claim_tokens.append(token.text)
    
    first_claim = " ".join(first_claim_tokens).strip()
    # Clean up trailing punctuation and whitespace
    first_claim = first_claim.rstrip(" ,;")
    if first_claim:
        claims.append(first_claim)
    
    # Build remaining claims: subject + each conjunct verb + its dependents
    for conj_verb in conjunct_verbs:
        claim_tokens = []
        
        # Add the determiner + subject if exists
        if subject:
            # Include any determiners before subject
            for token in doc:
                if token.dep_ == "det" and token.head == subject:
                    claim_tokens.append(token.text)
                    break
            claim_tokens.append(subject.text)
        
        # Add conjunct verb and all its children
        subtree_tokens = [t.text for t in conj_verb.subtree]
        claim_tokens.extend(subtree_tokens)
        
        claim = " ".join(claim_tokens).strip()
        if claim and len(claim.split()) >= 3:
            claims.append(claim)
    
    return claims if len(claims) > 1 else [sentence]


def verify_span_in_answer(span: str, answer: str) -> bool:
    """
    Checks that a DriftSpan is traceable back to the original answer.
    Guards against decomposition drift.
    """
    # Simple check: at least 50% of span words appear in answer
    span_words = set(span.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(span_words & answer_words)
    ratio = overlap / len(span_words) if span_words else 0
    return ratio >= 0.5


def decompose_into_driftspans(answer: str) -> list[str]:
    """
    Main function. Takes full LLM answer and returns list of DriftSpans.
    
    Pipeline:
    1. spaCy splits answer into sentences
    2. Each sentence checked for multiple claims
    3. Complex sentences split further by flan-t5
    4. Each span verified against original answer
    
    Args:
        answer: full LLM generated answer string
    
    Returns:
        list of DriftSpan strings
    """
    # Step 1 — sentence splitting via spaCy
    sentences = split_into_sentences(answer)
    
    driftspans = []
    
    for sentence in sentences:
        # Step 2 — check if sentence has multiple claims
        if sentence_contains_multiple_claims(sentence):
            # Step 3 — split into atomic claims via flan-t5
            claims = split_sentence_into_claims(sentence)
        else:
            claims = [sentence]
        
        # Step 4 — verify each claim is traceable to original answer
        for claim in claims:
            if verify_span_in_answer(claim, answer):
                driftspans.append(claim)
    
    return driftspans


if __name__ == "__main__":
    test_answer = """The 2021 trial showed 78% fever reduction. 
    The drug was approved by FDA in 2022 and had no serious side effects.
    Researchers concluded that the treatment was effective for adults."""
    
    print("Input answer:")
    print(test_answer)
    print("\nExtracted DriftSpans:")
    spans = decompose_into_driftspans(test_answer)
    for i, span in enumerate(spans):
        print(f"  s{i+1}: {span}")