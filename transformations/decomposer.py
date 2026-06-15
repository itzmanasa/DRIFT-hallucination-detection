import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import spacy

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm")


def split_into_sentences(text: str) -> list[str]:
    """
    Uses spaCy to split text into sentences.
    Reliable across different answer styles and lengths.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    sentences = [s for s in sentences if len(s.split()) >= 3]
    return sentences


def sentence_contains_multiple_claims(sentence: str) -> bool:
    """
    Detects if a sentence contains more than one factual claim.
    Signals: coordinating conjunctions joining verb phrases.
    Example: "X was approved in 2022 and had no side effects"
    → two claims bundled in one sentence.
    """
    doc = nlp(sentence)

    # Multiple root verbs = multiple independent claims
    root_verbs = [token for token in doc if token.dep_ == "ROOT"]
    if len(root_verbs) > 1:
        return True

    # Coordinating conjunction between verb phrases
    conjunctions = [
        token for token in doc
        if token.dep_ == "conj" and token.pos_ == "VERB"
    ]
    if len(conjunctions) > 0:
        return True

    return False


def split_sentence_into_claims(sentence: str) -> list[str]:
    """
    Uses spaCy dependency parsing to split complex sentences
    into atomic claims at conjunction boundaries.

    More reliable than LLM-based splitting for this task
    because spaCy was built for syntactic analysis.
    """
    doc = nlp(sentence)
    claims = []

    # Find subject
    subject = None
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject = token
            break

    # Find root verb and conjunct verbs
    root = None
    conjunct_verbs = []
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
        if token.dep_ == "conj" and token.pos_ == "VERB":
            conjunct_verbs.append(token)

    if not root or not conjunct_verbs:
        return [sentence]

    # Build first claim — everything before first conjunction
    first_claim_tokens = []
    for token in doc:
        if token == conjunct_verbs[0]:
            break
        if token.dep_ == "cc":
            continue
        first_claim_tokens.append(token.text)

    first_claim = " ".join(first_claim_tokens).strip().rstrip(" ,;")
    if first_claim:
        claims.append(first_claim)

    # Build remaining claims — subject + each conjunct verb subtree
    for conj_verb in conjunct_verbs:
        claim_tokens = []

        if subject:
            for token in doc:
                if token.dep_ == "det" and token.head == subject:
                    claim_tokens.append(token.text)
                    break
            claim_tokens.append(subject.text)

        subtree_tokens = [t.text for t in conj_verb.subtree]
        claim_tokens.extend(subtree_tokens)

        claim = " ".join(claim_tokens).strip()
        if claim and len(claim.split()) >= 3:
            claims.append(claim)

    return claims if len(claims) > 1 else [sentence]


def verify_span_in_answer(span: str, answer: str) -> bool:
    """
    Guards against decomposition drift.
    Checks that at least 50% of span words appear in the answer.
    Drops any span the model hallucinated during extraction.
    """
    span_words = set(span.lower().split())
    answer_words = set(answer.lower().split())
    if not span_words:
        return False
    overlap = len(span_words & answer_words)
    return (overlap / len(span_words)) >= 0.5


def decompose_into_driftspans(answer: str) -> list[str]:
    """
    Transformation 1 — Decomposes LLM answer A into DriftSpans.

    Pipeline:
    1. spaCy splits answer into sentences
    2. Each sentence checked for multiple claims
    3. Complex sentences split at conjunction boundaries
    4. Each span verified against original answer

    A DriftSpan is the minimal unit of verifiable factual content —
    a contiguous text segment expressing exactly one checkable claim
    that can be independently assessed as grounded or ungrounded
    against retrieved source documents.

    Args:
        answer: full LLM generated answer string

    Returns:
        list of DriftSpan strings
    """
    if not answer or not answer.strip():
        return []

    sentences = split_into_sentences(answer)
    driftspans = []

    for sentence in sentences:
        if sentence_contains_multiple_claims(sentence):
            claims = split_sentence_into_claims(sentence)
        else:
            claims = [sentence]

        for claim in claims:
            if verify_span_in_answer(claim, answer):
                driftspans.append(claim)

    return driftspans


if __name__ == "__main__":
    test_answer = (
        "The 2021 trial showed 78% fever reduction. "
        "The drug was approved by FDA in 2022 and had no serious side effects. "
        "Researchers concluded that the treatment was effective for adults."
    )

    print("Input answer:")
    print(test_answer)
    print("\nExtracted DriftSpans:")
    spans = decompose_into_driftspans(test_answer)
    for i, span in enumerate(spans):
        print(f"  s{i+1}: {span}")