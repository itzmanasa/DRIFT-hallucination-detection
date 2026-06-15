from data.ragtruth_loader import load_ragtruth
import os, warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

for split in ["train", "test"]:
    samples = load_ragtruth(split=split, task_type="QA")
    hallucinated = sum(s["is_hallucinated"] for s in samples)
    print(f"{split}: {len(samples)} samples, "
          f"{hallucinated} hallucinated "
          f"({hallucinated/len(samples)*100:.1f}%)")