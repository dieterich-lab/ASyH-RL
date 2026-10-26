#!/bin/bash
#SBATCH --job-name=long-asyh-rl
#SBATCH --output=logs/asyh-rl_%j.out
#SBATCH --error=logs/asyh-rl_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --partition=long

source /home/gsergei/asyh/bin/activate

# sbatch runs a spooled copy of this script, so use the submit dir as the project root
PROJECT_DIR="$SLURM_SUBMIT_DIR"
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/asyh_${SLURM_JOB_ID}_XXXXXX")

# copy results back to the project dir even if the job fails or is cancelled
cleanup() {
    cd "$PROJECT_DIR"
    cp -f "$WORKDIR"/*.csv "$WORKDIR"/*.json "$PROJECT_DIR"/ 2>/dev/null
    # rm -rf "$WORKDIR"
    mv -iv "$WORKDIR" /scratch/ghermans/.trash/.
}
trap cleanup EXIT

ln -s "$PROJECT_DIR/puf_wide" "$WORKDIR/puf_wide"
cd "$WORKDIR"

python3 "$PROJECT_DIR/main_rl_optimization.py" --num-model "$@"
