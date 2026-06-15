import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import json


def load_source_info(path: str = "data/source_info.jsonl") -> dict:
    """
    Loads source_info.jsonl and indexes by source_id.
    """
    sources = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                sources[record["source_id"]] = record
    print(f"Loaded {len(sources)} source records")
    return sources


def load_responses(path: str = "data/response.jsonl") -> list[dict]:
    """
    Loads response.jsonl.
    """
    responses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(json.loads(line))
    print(f"Loaded {len(responses)} response records")
    return responses


def extract_question_from_prompt(prompt: str) -> str:
    """
    Extracts the question from the RAGTruth prompt field.
    """
    if not prompt:
        return ""
    lines = prompt.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("output:") and len(line.split()) > 3:
            return line
    return lines[0] if lines else ""


def chunk_source_text(source_text: str, chunk_size: int = 150) -> list[str]:
    """
    Splits source text into chunks of approximately chunk_size words.
    Handles RAGTruth QA passage format with "passage N:" headers.
    """
    if not source_text:
        return []

    # Handle QA passage format — split on passage markers
    import re
    passage_pattern = re.compile(
        r'passage\s+\d+\s*:', re.IGNORECASE
    )
    
    if passage_pattern.search(source_text):
        # Split on passage markers
        parts = passage_pattern.split(source_text)
        chunks = []
        for part in parts:
            part = part.strip()
            if part and len(part.split()) >= 5:
                # Further split long passages
                if len(part.split()) > chunk_size:
                    sentences = []
                    for sent in part.replace("\n", " ").split(". "):
                        sent = sent.strip()
                        if sent:
                            sentences.append(sent)
                    current = []
                    current_words = 0
                    for sent in sentences:
                        current.append(sent)
                        current_words += len(sent.split())
                        if current_words >= chunk_size:
                            chunks.append(". ".join(current) + ".")
                            current = []
                            current_words = 0
                    if current:
                        chunks.append(". ".join(current) + ".")
                else:
                    chunks.append(part)
        if chunks:
            return chunks

    # Default chunking for non-QA formats
    sentences = []
    for sentence in source_text.replace("\n", " ").split(". "):
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        current.append(sentence)
        current_words += word_count
        if current_words >= chunk_size:
            chunks.append(". ".join(current) + ".")
            current = []
            current_words = 0

    if current:
        chunks.append(". ".join(current) + ".")

    return chunks


def extract_source_text(raw_source) -> str:
    """
    Handles different source_info formats across RAGTruth task types.
    Summary: string
    QA: dict with passage fields
    Data2Text: dict or list
    """
    if isinstance(raw_source, str):
        return raw_source
    elif isinstance(raw_source, dict):
        return " ".join(str(v) for v in raw_source.values())
    elif isinstance(raw_source, list):
        return " ".join(str(item) for item in raw_source)
    else:
        return str(raw_source)


def build_samples(
    responses: list[dict],
    sources: dict,
    split: str = "test",
    max_samples: int = None,
    task_type: str = None
) -> list[dict]:
    """
    Joins responses and sources into DRIFT-ready samples.
    """
    samples = []

    for resp in responses:
        # Filter by split
        if split != "all" and resp.get("split", "") != split:
            continue

        source_id = resp.get("source_id", "")
        source = sources.get(source_id, {})

        if not source:
            continue

        # Filter by task type
        if task_type:
            source_task = source.get("task_type", "").strip().lower()
            filter_task = task_type.strip().lower()
            if source_task != filter_task:
                continue

        # Extract Q
        question = extract_question_from_prompt(source.get("prompt", ""))

        # Extract D
        raw_source = source.get("source_info", "")
        source_text = extract_source_text(raw_source)
        chunks = chunk_source_text(source_text)

        # Extract A
        answer = resp.get("response", "")

        # Extract labels
        labels = resp.get("labels", [])

        # Skip incomplete samples
        # Skip incomplete samples
        if not question or not chunks or not answer:
            continue
        
        # Skip refusal responses — not hallucinations
        refusal_phrases = [
            "unable to answer",
            "cannot answer", 
            "not enough information",
            "no information provided",
            "based on the given passages, i cannot",
            "the passages do not"
        ]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            continue

        sample = {
            "query_id": f"ragtruth_{resp['id']}",
            "question": question,
            "chunks": chunks,
            "answer": answer,
            "labels": labels,
            "model": resp.get("model", "unknown"),
            "task_type": source.get("task_type", "unknown"),
            "is_hallucinated": 1 if len(labels) > 0 else 0
        }

        samples.append(sample)

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def load_ragtruth(
    split: str = "test",
    max_samples: int = None,
    task_type: str = None
) -> list[dict]:
    """
    Main loader. Returns DRIFT-ready RAGTruth samples.

    Args:
        split: "train", "test", or "all"
        max_samples: limit samples (None = all)
        task_type: "QA", "Summary", "Data2Text", or None for all

    Returns:
        list of sample dicts with Q, D, A, labels
    """
    sources = load_source_info()
    responses = load_responses()
    samples = build_samples(responses, sources, split, max_samples, task_type)
    print(f"Built {len(samples)} {split} samples "
          f"({'all tasks' if not task_type else task_type})")
    return samples


if __name__ == "__main__":
    samples = load_ragtruth(split="test", max_samples=10, task_type="QA")
    if samples:
        s = samples[0]
        print(f"\nSample preview:")
        print(f"  Query ID : {s['query_id']}")
        print(f"  Model    : {s['model']}")
        print(f"  Task     : {s['task_type']}")
        print(f"  Chunks   : {len(s['chunks'])}")
        print(f"  Label    : {'HALLUCINATED' if s['is_hallucinated'] else 'CLEAN'}")
        print(f"  Question : {s['question'][:100]}")
        print(f"  Answer   : {s['answer'][:150]}")

        hallucinated = sum(s["is_hallucinated"] for s in samples)
        print(f"\nStats: {len(samples)} samples, "
              f"{hallucinated} hallucinated")