import os

# I believe this environment variable should be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import itertools
import random
import json
import torch.multiprocessing as mp
import time
import huggingface_hub
from datasets import config
from transformers import AutoTokenizer

import demo_config

# Kind of janky double importing dictionary_learning.dictionary_learning, but it works
# This is leftover from when dictionary_learning was a only used as a submodule
from dictionary_learning.dictionary_learning.utils import (
    hf_dataset_to_generator,
    hf_mixed_dataset_to_generator,
    hf_sequence_packing_dataset_to_generator,
)
from dictionary_learning.dictionary_learning.pytorch_buffer import ActivationBuffer
from dictionary_learning.dictionary_learning.evaluation import evaluate
from dictionary_learning.dictionary_learning.training import trainSAE
import dictionary_learning.dictionary_learning.utils as utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=str, required=True, help="where to store sweep"
    )
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument(
        "--save_checkpoints", action="store_true", help="save checkpoints"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="which language model to use",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in demo_config.TrainerType],
        required=True,
        help="which SAE architectures to train",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to train on"
    )
    parser.add_argument(
        "--hf_repo_id", type=str, help="Hugging Face repo ID to push results to"
    )
    parser.add_argument(
        "--mixed_dataset", action="store_true", help="use mixed dataset"
    )
    parser.add_argument(
        "--target_l0s", type=int, nargs="+", help="target L0 values"
    )
    parser.add_argument(
        "--dictionary_widths", type=int, nargs="+", help="dictionary widths"
    )
    parser.add_argument(
        "--num_tokens_m", type=int, default=50, help="number of tokens in millions to train on (e.g. 500 for 500M tokens)"
    )

    args = parser.parse_args()
    return args


def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
    mixed_dataset: bool = False,
):
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    # model and data parameters
    context_length = demo_config.LLM_CONFIG[model_name].context_length

    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    num_buffer_inputs = buffer_tokens // context_length
    print(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {buffer_tokens}")

    log_steps = 100  # Log the training on wandb or print to console every log_steps

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train

    if save_checkpoints:
        # Creates checkpoints at 0.0%, 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    llm_config = demo_config.LLM_CONFIG[model_name]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=dtype,
        use_safetensors=llm_config.use_safetensors,
        trust_remote_code=llm_config.trust_remote_code
    )

    model = utils.truncate_model(model, layer)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    submodule = model.transformer.h[layer].mlp.c_proj if model.config.architectures[0] == "GPT2LMHeadModel" else utils.get_submodule(model, layer)
    submodule_name = f"mlp_out_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    if mixed_dataset:
        qwen_system_prompt_to_remove = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"

        assert "Qwen" in model_name, "Make sure system prompt matches model"

        generator = hf_mixed_dataset_to_generator(
            tokenizer,
            system_prompt_to_remove=qwen_system_prompt_to_remove,
            min_chars=context_length * 4,
        )
    else:
        generator = hf_sequence_packing_dataset_to_generator(
            tokenizer,
            min_chars=context_length * 4,
        )

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=num_buffer_inputs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
        add_special_tokens=False,
    )

    trainer_configs = demo_config.get_trainer_configs(
        architectures,
        learning_rates,
        random_seeds,
        activation_dim,
        dictionary_widths,
        model_name,
        device,
        layer,
        submodule_name,
        steps,
        target_l0s=args.target_l0s,
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            wandb_project=demo_config.get_wandb_project(model_name),
            normalize_activations=True,
            verbose=False,
            autocast_dtype=t.bfloat16,
            backup_steps=1000,
        )


@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
) -> dict:
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    max_layer = 0

    for ae_path in ae_paths:
        config_path = f"{ae_path}/config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        layer = config["trainer"]["layer"]
        max_layer = max(max_layer, layer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=dtype
    )

    model = utils.truncate_model(model, max_layer)

    buffer_size = n_inputs
    io = "out"
    n_batches = n_inputs // loss_recovered_batch_size

    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > n_inputs * 5:
            break

    eval_results = {}

    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue

        dictionary, config = utils.load_dictionary(ae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = config["trainer"]["layer"]
        submodule = utils.get_submodule(model, layer)

        activation_dim = config["trainer"]["activation_dim"]

        activation_buffer = ActivationBuffer(
            iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    # return the final eval_results for testing purposes
    return eval_results


def push_to_huggingface(save_dir: str, repo_id: str):
    api = huggingface_hub.HfApi()

    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=save_dir,
    )


if __name__ == "__main__":
    """python demo.py --save_dir ./run2 --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated --use_wandb
    python demo.py --save_dir ./run3 --model_name google/gemma-2-2b --layers 12 --architectures standard top_k --use_wandb
    python demo.py --save_dir ./jumprelu --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures jump_relu --use_wandb"""
    args = get_args()

    hf_repo_id = args.hf_repo_id

    if hf_repo_id:
        try:
            if not huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model"):
                print(f"Repository {hf_repo_id} does not exist. Creating it...")
                huggingface_hub.create_repo(
                    repo_id=hf_repo_id,
                    repo_type="model",
                    private=False,
                    exist_ok=True
                )
                print(f"Created repository {hf_repo_id}")
        except Exception as e:
            print(f"Error with HuggingFace repository: {e}")
            print("Make sure you are logged in to HuggingFace (`huggingface-cli login`)")
            exit(1)

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # For wandb to work with multiprocessing
    mp.set_start_method("spawn", force=True)

    # Rarely I have internet issues on cloud GPUs and then the streaming read fails
    # Hopefully the outage is shorter than 100 * 20 seconds
    config.STREAMING_READ_MAX_RETRIES = 100
    config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    save_dir = (
        f"{args.save_dir}_{args.model_name}_{'_'.join(args.architectures)}_tokens{args.num_tokens_m}M".replace(
            "/", "_"
        )
    )

    for layer in args.layers:
        run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=save_dir,
            device=args.device,
            architectures=args.architectures,
            num_tokens=args.num_tokens_m * 1_000_000,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=args.dictionary_widths or demo_config.dictionary_widths,
            learning_rates=demo_config.learning_rates,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
            mixed_dataset=args.mixed_dataset,
        )

    ae_paths = utils.get_nested_folders(save_dir)

    eval_saes(
        args.model_name,
        ae_paths,
        demo_config.eval_num_inputs,
        args.device,
        overwrite_prev_results=True,
    )

    print(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)
