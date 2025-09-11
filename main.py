import subprocess
import multiprocessing
from itertools import product

# List of models and datasets to run
MODELS = [
    "PatchTST",
    "NHITS",
    "Informer",
    "DeepAR",
    "FEDformer"
]

DATASETS = [
    "etth1",
    "electricity",
    "sunspots",
    "taxi"
]

def run_pipeline(model, dataset):
    cmd = [
        "nohup","python3", "forecasting_pipeline.py",
        "--model", model,
        "--dataset", dataset    
        
    ]
    print(f"Launching: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {model} on {dataset}: {e}")

def main():
    # Create all (model, dataset) pairs
    tasks = list(product(MODELS, DATASETS))

    # Use multiprocessing to run in parallel
    with multiprocessing.Pool(processes=min(len(tasks), multiprocessing.cpu_count())) as pool:
        pool.starmap(run_pipeline, tasks)

if __name__ == "__main__":
    main()
    print("All done!")
