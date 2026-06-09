import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import json


def load_source_info(path: str = "data/source_info.jsonl") -> dict:
    """
    Loads source_info.jsonl and indexes by source_id.
    Returns dict mapping source_id -> source record.
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
    Returns list of response records.
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
    Extracts the question Q from the prompt field.
    RAGTruth prompt contains source text + instruction.
    We extract the instruction part as the question.
    """
    if not prompt:
        return ""
    
    # The prompt ends with "output:" — question is the instruction line
    lines = prompt.strip().split("\n")
    
    # First line usually contains the task instruction
    for line in lines:
        line = line.strip()
        if line and not line.startswith("output:"):
            # Return first meaningful line as question
            if len(line.split()) > 3:
                return line
    
    return lines[0] if lines else ""


def chunk_source_text(source_text: str, chunk_size: int = 200) -> list[str]:
    """
    Splits source_info text into chunks for retrieval.
    Uses sentence boundaries where possible.
    
    Args:
        source_text: full retrieved document text
        chunk_size: approximate words per chunk
    
    Returns:
        list of text chunks
    """
    if not source_text:
        return []
    
    # Split on sentence boundaries
    sentences = []
    current = []
    
    for sentence in source_text.replace("\n", " ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        current.append(sentence)
        
        # Check if current chunk is large enough
        word_count = sum(len(s.split()) for s in current)
        if word_count >= chunk_size:
            chunks_text = ". ".join(current) + "."
            sentences.append(chunks_text)
            current = []
    
    # Add remaining sentences
    if current:
        sentences.append(". ".join(current) + ".")
    
    return sentences


def build_samples(
    responses: list[dict],
    sources: dict,
    split: str = "test",
    max_samples: int = None
) -> list[dict]:
    """
    Joins responses and sources to build DRIFT-ready samples.
    
    Each sample contains:
    - query_id: unique id
    - question: Q
    - chunks: D (list of text chunks)
    - answer: A (LLM response)
    - labels: ground truth hallucination spans
    - model: which LLM
    - task_type: Summary/QA/Data2Text
    - is_hallucinated: response-level binary label
    
    Args:
        responses: loaded response records
        sources: loaded source records indexed by source_id
        split: filter by split (train/test/all)
        max_samples: limit number of samples (None = all)
    
    Returns:
        list of DRIFT-ready sample dicts
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
        
        # Extract Q
        question = extract_question_from_prompt(
            source.get("prompt", "")
        )
        
        # Extract D — chunk the source text
        source_text = source.get("source_info", "")
        chunks = chunk_source_text(source_text)
        
        # Extract A
        answer = resp.get("response", "")
        
        # Extract labels
        labels = resp.get("labels", [])
        
        # Skip samples missing core fields
        if not question or not chunks or not answer:
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
    max_samples: int = None
) -> list[dict]:
    """
    Main loader function. Returns DRIFT-ready samples.
    
    Args:
        split: "train", "test", or "all"
        max_samples: limit samples for quick testing
    
    Returns:
        list of DRIFT-ready sample dicts
    """
    sources = load_source_info()
    responses = load_responses()
    samples = build_samples(responses, sources, split, max_samples)
    
    print(f"\nBuilt {len(samples)} {split} samples")
    return samples


def print_sample(sample: dict):
    """Pretty prints one sample."""
    print("=" * 60)
    print(f"Query ID : {sample['query_id']}")
    print(f"Model    : {sample['model']}")
    print(f"Task     : {sample['task_type']}")
    print(f"Label    : {'HALLUCINATED' if sample['is_hallucinated'] else 'CLEAN'}")
    print(f"\nQuestion : {sample['question'][:150]}...")
    print(f"\nChunks   : {len(sample['chunks'])} chunks")
    for i, chunk in enumerate(sample['chunks'][:2]):
        print(f"  [{i+1}] {chunk[:100]}...")
    print(f"\nAnswer   : {sample['answer'][:200]}...")
    print(f"\nLabels   : {len(sample['labels'])} hallucinated spans")
    print("=" * 60)


if __name__ == "__main__":
    # Load small sample first to verify
    samples = load_ragtruth(split="test", max_samples=50)
    
    if samples:
        print("\nFirst sample:")
        print_sample(samples[0])
        
        # Statistics
        hallucinated = sum(s["is_hallucinated"] for s in samples)
        models = set(s["model"] for s in samples)
        tasks = set(s["task_type"] for s in samples)
        
        print(f"\nSample statistics:")
        print(f"  Total        : {len(samples)}")
        print(f"  Hallucinated : {hallucinated} "
              f"({hallucinated/len(samples)*100:.1f}%)")
        print(f"  Clean        : {len(samples) - hallucinated}")
        print(f"  Models       : {models}")
        print(f"  Task types   : {tasks}")