#!/usr/bin/env bash
#SBATCH --account=cvgl
#SBATCH --partition=svl --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=0-03:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -e -o pipefail

GPU_ID=$(nvidia-smi -L | grep -oP '(?<=GPU-)[a-fA-F0-9\-]+' | head -n 1)
ISAAC_CACHE_PATH="/scr-ssd/${SLURM_JOB_USER}/isaac_cache_${GPU_ID}"

# Define env kwargs to pass
declare -A ENVS=(
    [NVIDIA_DRIVER_CAPABILITIES]=all
    [NVIDIA_VISIBLE_DEVICES]=0
    [DISPLAY]=""
    [OMNIGIBSON_HEADLESS]=1
    [CREDENTIALS_FPATH]=/cvgl/group/Gibson/og-data-0-3-0/key.json
    [SAMPLING_SCENE_MODEL]=""
    [SAMPLING_ACTIVITIES]=""
    [SAMPLING_START_AT]=""
    [SAMPLING_RANDOMIZE]=""
    [SAMPLING_OVERWRITE_EXISTING]=""
    [SAMPLING_THREAD_ID]=${SLURM_JOB_ID}
)

# Parse command-line args
# m - scene model
# a - activities
# s - start at
# r - randomize order
# o - overwrite

print_usage() {
  printf "Usage: ..."
}

while getopts 'm:asro' flag; do
  case "${flag}" in
    m) ENVS[SAMPLING_SCENE_MODEL]="${OPTARG}" ;;
    a) ENVS[SAMPLING_ACTIVITIES]="${OPTARG}" ;;
    s) ENVS[SAMPLING_START_AT]="${OPTARG}" ;;
    r) ENVS[SAMPLING_RANDOMIZE]="1" ;;
    o) ENVS[SAMPLING_OVERWRITE_EXISTING]="1" ;;
    *) print_usage
       exit 1 ;;
  esac
done

for env_var in "${!ENVS[@]}"; do
    # Add to env kwargs we'll pass to enroot command later
    ENV_KWARGS="${ENV_KWARGS} --env ${env_var}=${ENVS[${env_var}]}"
done

# Define mounts to create (maps local directory to container directory)
declare -A MOUNTS=(
    [/cvgl/group/Gibson/og-data-0-3-0]=/data
    [${ISAAC_CACHE_PATH}/isaac-sim/kit/cache/Kit]=/isaac-sim/kit/cache/Kit
    [${ISAAC_CACHE_PATH}/isaac-sim/cache/ov]=/root/.cache/ov
    [${ISAAC_CACHE_PATH}/isaac-sim/cache/pip]=/root/.cache/pip
    [${ISAAC_CACHE_PATH}/isaac-sim/cache/glcache]=/root/.cache/nvidia/GLCache
    [${ISAAC_CACHE_PATH}/isaac-sim/cache/computecache]=/root/.nv/ComputeCache
    [${ISAAC_CACHE_PATH}/isaac-sim/logs]=/root/.nvidia-omniverse/logs
    [${ISAAC_CACHE_PATH}/isaac-sim/config]=/root/.nvidia-omniverse/config
    [${ISAAC_CACHE_PATH}/isaac-sim/data]=/root/.local/share/ov/data
    [${ISAAC_CACHE_PATH}/isaac-sim/documents]=/root/Documents
    # Feel free to include lines like the below to mount a workspace or a custom OG version
    # [/cvgl2/u/jdwong/PAIR/omnigibson-enroot]=/omnigibson-src
    [/cvgl]=/cvgl
)

MOUNT_KWARGS=""
for mount in "${!MOUNTS[@]}"; do
    # Verify mount path in local directory exists, otherwise, create it
    if [ ! -e "$mount" ]; then
        mkdir -p ${mount}
    fi
    # Add to mount kwargs we'll pass to enroot command later
    MOUNT_KWARGS="${MOUNT_KWARGS} --mount ${mount}:${MOUNTS[${mount}]}"
done

# Create the image if it doesn't already exist
CONTAINER_NAME=omnigibson_${GPU_ID}
enroot create --force --name ${CONTAINER_NAME} /cvgl/group/Gibson/og-data-0-3-0/omnigibson-dev.sqsh

# Remove leading space in string
ENV_KWARGS="${ENV_KWARGS:1}"
MOUNT_KWARGS="${MOUNT_KWARGS:1}"

# The last line here is the command you want to run inside the container.
# Here I'm running some unit tests.
ENROOT_MOUNT_HOME=no enroot start \
    --root \
    --rw \
    ${ENV_KWARGS} \
    ${MOUNT_KWARGS} \
    ${CONTAINER_NAME} \
    micromamba run -n omnigibson /bin/bash --login -c "cd / git clone https://github.com/StanfordVL/bddl.git --branch develop --single-branch bddl-src && cd bddl-src && pip install -e . && cd / && mv omnigibson-src omnigibson-src-backup && git clone https://github.com/StanfordVL/OmniGibson.git --branch feat/sampling_2024 --single-branch omnigibson-src && cd /omnigibson-src && source /isaac-sim/setup_conda_env.sh && pip install gspread && python omnigibson/sampling/sample_b1k_scenes.py"

# Clean up the image if possible.
enroot remove -f ${CONTAINER_NAME}
