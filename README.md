# 📦 Continual Learning for IMU-based Human Activity Recognition

This project implements a continual learning framework for Human Activity Recognition (HAR) using Proxy Anchor Loss and Contrastive Learning loss. The framework is designed for user-centric class-incremental learning using IMU data, aiming to reduce catastrophic forgetting.

---

## 📚 Datasets Used

- RealWorld
- MHEALTH
- WISDM

Ensure these datasets are downloaded and properly organized before running the experiments.

---

## 🚀 Setup Instructions

### 1. Clone the Repository
git clone https://github.com/Dhruvadityamittal/PACL.git
cd PACL

### 2. Create a Conda Environment
conda create -n PACL python=3.8.10

### 3. Activate the Environment
conda activate PACL

### 4. Install Required Dependencies
pip install -r requirements.txt

✅ Ensure the file is correctly named `requirements.txt`.

### 5. Run the Code
python CGCD-HAR_GBASELINE_WANDB_ALL_session.py --dataset='realworld' --contrastive_loss_type='G-Baseline_NCE'

You can also use `mhealth` or `wisdm` as dataset options and experiment with different contrastive loss types.

---

## ⚙️ Command-line Options

| Argument                  | Description                                    | Example                          |
|---------------------------|------------------------------------------------|----------------------------------|
| --dataset                 | Dataset to use (realworld, mhealth, etc.)     | --dataset='realworld'           |
| --contrastive_loss_type   | Type of contrastive loss to apply              | --contrastive_loss_type='G-Baseline_NCE' |

---

## 📩 Contact

For issues or questions, please reach out via GitHub or email:
📧 dhruvadityamittal@gmail.com

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
