import torch
import argparse
import json
import gym_train
import ray
import os


@ray.remote(num_gpus=1) 
def run_data_generation_and_training(config, gpu_id=None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        assigned_gpu_ids = ray.get_gpu_ids()
        if assigned_gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(assigned_gpu_ids[0]))
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    
    exp_config_str = ", ".join([f"{k}={v}" for k, v in config.items() if k not in ['global_config', 'training_config', 'env_config']])

    def log_message(message):
        with open("main_output.log", "a") as log:
            log.write(f"PID {os.getpid()} - Device {device} - {message}\n")

    log_message(f"Experiment config ({exp_config_str}): Starting training")
    if torch.cuda.is_available():
        log_message(f"Using GPU: {torch.cuda.get_device_name(device)}")
        log_message(f"Allocated GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    else:
        log_message("Running on CPU")

    # Training
    gym_train.train(config)
    
    log_message(f"Experiment config ({exp_config_str}): Completed training")

    return f"Completed run with experiment config: {exp_config_str}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)

    global_config = full_config['global_config']
    training_config = full_config['training_config']
    env_config = full_config['env_config']

    # Ensure theta is a list of numbers

    experiments = full_config['experiments']

    # Initialize Ray
    ray.init()

    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()

    envs = env_config.get("envs") if isinstance(env_config, dict) else None
    if not envs:
        envs = env_config.get("env")
        if not envs:
            envs = []
    if not isinstance(envs, list):
        envs = [envs]
        
    # Submit tasks to Ray
    results = []
    run_index = 0
    for env in envs:
        per_env_config = {k: v for k, v in env_config.items() if k != "envs"}
        per_env_config["env"] = env
        for experiment in experiments:
            # Merge configurations for each experiment
            config = {**global_config, **training_config, **per_env_config, **experiment}

            # Assign GPU in a round-robin fashion if GPUs are available
            gpu_id = run_index % num_gpus if num_gpus > 0 else None
            run_index += 1

            results.append(run_data_generation_and_training.remote(config, gpu_id))

    # Wait for all tasks to complete
    ray.get(results)

    print("All data generation and training processes are complete.")