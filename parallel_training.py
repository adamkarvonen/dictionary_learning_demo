#!/usr/bin/env python3
import subprocess
import time
import os

# Configuration for 4x 3090s
# For our current implementation, relative training speed is:
# standard / p_anneal > top_k > batch_top_k > jump_relu > gated
# So, we have jump_relu and gated on their own GPUs


MODEL_NAME = "EleutherAI/pythia-160m-deduped"
MODEL_NAME = "google/gemma-2-2b"
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
if "gemma" in MODEL_NAME:
    layer = 12
elif "pythia-70m" in MODEL_NAME:
    layer = 3
elif "pythia-160m" in MODEL_NAME:
    layer = 8
else:
    raise ValueError("Unknown model name")


configurations = [
    {
        "arch": "jump_relu",
        "layers": layer,
        "device": "cuda:0"
    },
    {
        "arch": "top_k p_anneal",
        "layers": layer,
        "device": "cuda:1"
    },
    {
        "arch": "batch_top_k standard_new",
        "layers": layer,
        "device": "cuda:2"
    },
    {
        "arch": "gated",
        "layers": layer,
        "device": "cuda:3"
    },
]

# config for 2x 3090s
# configurations = [
#     {
#         "arch": "gated top_k p_anneal",
#         "layers": layer,
#         "device": "cuda:0"
#     },
#     {
#         "arch": "jump_relu batch_top_k standard_new",
#         "layers": layer,
#         "device": "cuda:1"
#     },
# ]

SAVE_DIR = "trained_saes/"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Launch jobs
for i, config in enumerate(configurations):
    log_file = f"logs/{(config['arch'].replace(' ', '_'))}_l{config['layers']}_{config['device'].replace(':', '_')}.out"
    
    cmd = [
        "python", "demo.py",
        "--save_dir", SAVE_DIR,
        "--model_name", MODEL_NAME,
        "--architectures", config["arch"],
        "--layers", str(config["layers"]),
        "--device", config["device"],
        "--use_wandb"
    ]
    
    # Launch with nohup
    with open(log_file, "w") as f:
        subprocess.Popen(
            f"nohup {' '.join(cmd)} > {log_file} 2>&1",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    print(f"Started job {i + 1}/{len(configurations)}: {config['arch']} with layers: {config['layers']}")
    time.sleep(2)

print("All jobs submitted!")