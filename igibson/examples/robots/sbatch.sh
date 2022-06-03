#!/bin/bash
#SBATCH -J Dense_Skill_RL
#SBATCH -N 1
#SBATCH --mem 40G
#SBATCH --gres=gpu:1
#SBATCH -p viscam
#SBATCH -w viscam7
#SBATCH -c 8
srun python stable_baselines3_ppo_skill_example.py
wait