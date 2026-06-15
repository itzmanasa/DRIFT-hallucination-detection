import os, warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import json

with open("results/drift_test_QA_20.json") as f:
    results = json.load(f)

print("Results file loaded")
print(f"Response metrics: {results['response_metrics']}")