#!/bin/bash
#SBATCH --job-name "Edison"
#SBATCH --ntasks=1
#SBATCH --partition=A100-80GB
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=40G
#SBATCH --time 2-19:00:18
#SBATCH --output output/console.%A_%a.out
#SBATCH --error output/console.%A_%a.error
#SBATCH --mail-type=END
#SBATCH --mail-user=dhruv_aditya.mittal@dfki.de

srun  -K -N1\
	  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.02-py3.sqsh \
	  --container-workdir=`pwd` \
	  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
	  --task-prolog=`pwd`/install.sh \
	  --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
	  python edison_paper_implementation_online_session.py --model='resnet18' --dataset='realworld'
          
# edison_paper_implementation_online_DeepLSTM
# G-Baseline_Contrastive
# G-Baseline_NCE
# G-Baseline
# Online_Finetuning
# Offline
# Offline_NCE
# Online_Finetuning_NCE
# G-Baseline_NCE_WFR