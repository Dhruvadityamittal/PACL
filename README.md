# 📦 Continual Learning for IMU-based Human Activity Recognition

This project implements a continual learning framework for Human Activity Recognition (HAR) using Proxy Anchor Loss and Contrastive Learning loss

---

## 📁 Datasets

Realworld
MHEALTH
Wisdm

---

## 🚀 Setup Instructions


### 1. Clone the Repository
git clone https://github.com/Dhruvadityamittal/PACL.git
cd PACL
### 2. Create a conda environment
conda create -n PACL python=3.8.10
### 3. Activate conda environment
conda activate PACL
### 4. Install all the requirements
pip install -r requiments.txt
### 5. Execute the python code
python CGCD-HAR_GBASELINE_WANDB_ALL_session.py --dataset='realworld' --contrastive_loss_type='G-Baseline_NCE'

