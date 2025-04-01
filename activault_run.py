import subprocess

# Commands to run
commands = [
    # mistral_8b
    [
        "python",
        "activault_s3_demo.py",
        "--save_dir",
        "./mistral_8b",
        "--model_name",
        "mistralai/Ministral-8B-Instruct-2410",
        "--layers",
        str(layer),
        "--architectures",
        "batch_top_k",
        "--use_wandb",
        "--hf_repo_id",
        "adamkarvonen/mistral_test",
        "--device",
        f"cuda:{device}",
    ]
    for layer, device in [(9, 0), (18, 1), (27, 2)]
] + [
    # mistral_24b
    [
        "python",
        "activault_s3_demo_24b.py",
        "--save_dir",
        "./mistral_24b",
        "--model_name",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "--layers",
        str(layer),
        "--architectures",
        "batch_top_k",
        "--use_wandb",
        "--hf_repo_id",
        "adamkarvonen/mistral_test",
        "--device",
        f"cuda:{device}",
    ]
    for layer, device in [(10, 3), (20, 4), (30, 5)]
]

# Launch all commands in parallel
processes = []
for i, cmd in enumerate(commands):
    log_file = open(f"run_log_{i}.txt", "w")
    print(f"Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    processes.append((proc, log_file))

# Optional: Wait for all to complete
for proc, log_file in processes:
    proc.wait()
    log_file.close()
