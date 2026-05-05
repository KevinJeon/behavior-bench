#!/bin/bash

# Get the directory where this script itself is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd )

# --- Defaults ---
num_jobs=1
wandb_project="puffer_drive"
# NEW: Default to current branch if not specified
current_branch=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
target_branch="$current_branch"

# --- Help Function ---
usage() {
    echo "Usage: $0 --partition <PARTITION> --wall_time <WALL_TIME> [options]"
    echo ""
    echo "Required:"
    echo "  -p, --partition      Slurm partition to use"
    echo "  -t, --wall_time      Wall time (e.g., 02:00:00)"
    echo ""
    echo "Options:"
    echo "  -b, --branch         Git branch to use (Default: current branch '$current_branch')"
    echo "  -w, --wandb_project  WandB project name (Default: puffer_drive)"
    echo "  -n, --num_jobs       Number of jobs to submit (Default: 1)"
    echo "  -s, --sweep_id       WandB sweep ID (optional)"
    echo "  -h, --help           Show this help message"
    exit 1
}

# --- Parse Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--partition)     partition="$2"; shift;;
        -t|--wall_time)     wall_time="$2"; shift;;
        -s|--sweep_id)      sweep_id="$2"; shift;; 
        -n|--num_jobs)      num_jobs="$2"; shift;;
        -w|--wandb_project) wandb_project="$2"; shift;; # Added short flag -w
	-b|--branch)        target_branch="$2"; shift;; # NEW branch flag
        -h|--help)          usage ;;
        *) extra_args+=("$1") ;; 
    esac
    shift
done

# --- Validation ---
if [[ -z "$wall_time" || -z "$partition" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# --- Load Secrets ---
SECRETS_FILE="$SCRIPT_DIR/.slurm.env"
if [ -f "$SECRETS_FILE" ]; then
    source "$SECRETS_FILE"
fi

# Validate RESULTS_DIR
if [[ -z "$RESULTS_DIR" ]]; then
    echo "ERROR: RESULTS_DIR is not set in .slurm.env"
    exit 1
fi

# --- Configure Exports ---
export WANDB_API_KEY
export DRIVE_DATA_ROOT
export DRIVE_BINARIES_DATA_ROOT
export RESULTS_DIR
export WANDB_SWEEP_ID="$sweep_id"
export WANDB_PROJECT="$wandb_project" # This will now use the overridden value or default
export WANDB_GROUP
export TARGET_BRANCH="$target_branch"
export REPO_DIR="$PROJECT_ROOT"

EXPORT_LIST="WANDB_API_KEY,DRIVE_DATA_ROOT,DRIVE_BINARIES_DATA_ROOT,DRIVE_BINARIES_DATA_ROOT_TAR,RESULTS_DIR,WANDB_SWEEP_ID,WANDB_PROJECT,WANDB_GROUP,TARGET_BRANCH,REPO_DIR"

echo "Submitting $num_jobs job(s)."
echo "Branch: $target_branch | Partition: $partition | Wall Time: $wall_time"
echo "Sweep: $sweep_id | WandB Project: $wandb_project"
echo "Results will be saved to: $RESULTS_DIR/$wandb_project"

# --- Submission Loop ---
for ((i=1; i<=num_jobs; i++)); do
    sbatch --partition="$partition" \
           --time="$wall_time" \
           --job-name="${sweep_id}_${wandb_project}" \
           --export="$EXPORT_LIST" \
           --chdir="$PROJECT_ROOT" \
           "$SCRIPT_DIR/run_train.sh" \
           "${extra_args[@]}" 
done

echo "Jobs submitted."
