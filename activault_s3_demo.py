import torch as t
import argparse
import itertools
from nnsight import LanguageModel
import os
import random
import json
import torch.multiprocessing as mp
import time
import huggingface_hub
from datasets import config
from transformers import AutoConfig
from activault_rcache import S3RCache, create_s3_client, ActivaultS3ActivationBuffer

import demo_config
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from dictionary_learning.training import trainSAE
import dictionary_learning.utils as utils


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
):
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    dtype = demo_config.LLM_CONFIG[model_name].dtype
    submodule_name = f"resid_post_layer_{layer}"

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

    io = "out"
    activation_dim = AutoConfig.from_pretrained(model_name).hidden_size

    BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "main")
    s3_prefix = ["mistral.8b.fineweb/blocks.9.hook_resid_post"]
    cache = S3RCache.from_credentials(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        s3_prefix=s3_prefix,
        bucket_name=BUCKET_NAME,
        device=device,
        buffer_size=2,
        return_ids=True,
        shuffle=True,
        n_workers=2,
    )

    s3_buffer = ActivaultS3ActivationBuffer(
        cache, batch_size=sae_batch_size, device=device, io=io
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
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=s3_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            wandb_project=demo_config.wandb_project,
            normalize_activations=True,
            verbose=False,
            autocast_dtype=t.bfloat16,
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

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)

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
    """python activault_s3_demo.py --save_dir ./mistral_8b --model_name mistralai/Ministral-8B-Instruct-2410 --layers 9 --architectures batch_top_k --use_wandb --hf_repo_id adamkarvonen/mistral_test
    Run pip install boto3 before running this script"""
    args = get_args()

    hf_repo_id = args.hf_repo_id

    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # For wandb to work with multiprocessing
    mp.set_start_method("spawn", force=True)

    start_time = time.time()

    save_dir = (
        f"{args.save_dir}_{args.model_name}_{'_'.join(args.architectures)}".replace(
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
            num_tokens=demo_config.num_tokens,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=demo_config.dictionary_widths,
            learning_rates=demo_config.learning_rates,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
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
