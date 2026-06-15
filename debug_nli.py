from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

chunk = "The FDA granted approval to the medication in late 2022 after reviewing trial data."
span = "The drug was approved by FDA in 2022"

result = nli(f"{chunk} [SEP] {span}", truncation=True, max_length=512)
print("NLI output:")
for item in result:
    print(f"  {item['label']}: {item['score']:.4f}")