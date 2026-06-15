from data.ragtruth_loader import load_ragtruth
import os, warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

samples = load_ragtruth(split="test", max_samples=50, task_type="QA")
hallucinated = [s for s in samples if s["is_hallucinated"] == 1]
clean = [s for s in samples if s["is_hallucinated"] == 0]
print(f"QA samples: {len(samples)}")
print(f"Hallucinated: {len(hallucinated)}")
print(f"Clean: {len(clean)}")
if hallucinated:
    print(f"\nFirst hallucinated sample:")
    print(f"Answer: {hallucinated[0]['answer'][:200]}")
    print(f"Labels: {hallucinated[0]['labels'][:2]}")