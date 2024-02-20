#!/bin/bash

#SBATCH --job-name=generate_psm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-share
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=hhnode-ib-107

srun bash tools/dist_run.sh generate_psm_for_custom_data.py 8 --video-list ucf101_videos.list