# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

import wandb
import argparse
import subprocess
import sys
import os
# Global variable to hold arguments passed from Slurm/Bash
MANUAL_ARGS = []
WANDB_PROJECT = None
WANDB_GROUP = None


# Assuming MANUAL_ARGS is defined globally or passed in
# MANUAL_ARGS = ["--vec.backend", "native", ...]

def train_agent():
    """
    1. Starts WandB run.
    2. Combines Manual Args (High Priority) with Sweep Config (Low Priority).
    3. Executes the training command.
    """
    run = wandb.init(
        resume="allow"
    )


    env = os.environ.copy()
    env["WANDB_RUN_ID"] = run.id
    # Base command
    cmd = ["puffer", "train", "puffer_drive"]
    
    # --- 1. Add Manual Arguments (Priority: HIGH) ---
    cmd.extend(MANUAL_ARGS)
    
    # enable wandb logging	
    cmd.extend(["--wandb","--wandb-project",WANDB_PROJECT,"--wandb-group",WANDB_GROUP])
    # Create a set of flags that were manually provided
    manual_flags = {arg for arg in MANUAL_ARGS if arg.startswith("--")}

    # --- 2. Add Sweep Arguments (Priority: LOW) ---
    print(f"--- Configuration ---")
    print(f"Manual Overrides: {manual_flags}")

    def process_param(key_path, value):
        """
        Recursive function to handle nested dictionaries and add flags.
        args:
            key_path: The current dot-notation key (e.g. "train.steps")
            value: The value associated with the key
        """
        # Case A: Value is a dictionary (Nested Config)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # Create new prefix (e.g. "train" + "." + "steps")
                new_path = f"{key_path}.{sub_key}"
                process_param(new_path, sub_value)
            return

        # Case B: Value is a primitive (int, float, string, bool)
        # Format the flag name. We replace top-level underscores if standard 
        # CLI args use dashes (e.g., learning_rate -> learning-rate)
        clean_key = key_path.replace("_", "-") 
        flag_name = f"--{clean_key}"

        # OVERRIDE CHECK:
        if flag_name in manual_flags:
            print(f"[IGNORING] Sweep param '{key_path}' (Value: {value}) "
                  f"because '{flag_name}' was passed manually.")
            return


        cmd.append(flag_name)
        cmd.append(str(value))

    # Iterate over the WandB config and start recursion
    for key, value in wandb.config.items():
        process_param(key, value)

    wandb.config.update({"slurm_job_id":SLURM_JOB_ID})



    run.finish()

    # --- 3. Execute ---
    print(f"Final Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True,env=env)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1) 
    finally:
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_sweep_id", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="puffer-drive")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    
    # 'unknown' captures everything else (e.g. --vec.backend Serial)
    args, unknown = parser.parse_known_args()
    
    MANUAL_ARGS = unknown
    WANDB_PROJECT = args.wandb_project
    WANDB_GROUP = args.wandb_group
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")


    print(f"Starting WandB Agent for Sweep: {args.wandb_sweep_id}")
    
    # count=1 ensures that 1 Slurm Job = 1 Experiment. 
    # When the experiment finishes, the Slurm job ends.
    wandb.agent(
        args.wandb_sweep_id, 
        function=train_agent, 
        count=1, 
        project=args.wandb_project, 
        entity=args.entity
    )
