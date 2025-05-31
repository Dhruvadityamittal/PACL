# 📦 Continual Learning for IMU-based Human Activity Recognition

This project implements a continual learning framework for Human Activity Recognition (HAR) using Proxy Anchor Loss and Contrastive Learning loss. The framework is designed for user-centric class-incremental learning using IMU data, aiming to reduce catastrophic forgetting.

---

## 📚 Datasets Used

- RealWorld - https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- MHEALTH - https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- WISDM  - https://www.cis.fordham.edu/wisdm/dataset.php

Ensure these datasets are downloaded and properly organized before running the experiments.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruvadityamittal/PACL.git
cd PACL
```

### 2. Create a Conda Environment

```bash
conda create -n PACL python=3.8.10
```

### 3. Activate the Environment

```bash
conda activate PACL
```

### 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

✅ Ensure the file is correctly named `requirements.txt`.

### 5. Run the Code

```bash
python CGCD-HAR_GBASELINE_WANDB_ALL_session.py --dataset='realworld' --contrastive_loss_type='G-Baseline_NCE'
```

You can also use `mhealth` or `wisdm` as dataset options and experiment with different contrastive loss types.

---

## ⚙️ Command-line Options

## ⚙️ Command-line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--LOG_DIR` | Path to log folder | `./logs` |
| `--dataset` | Dataset name (`realworld`, `mhealth`, `pamap`, `wisdn`) | `realworld` |
| `--embedding-size` | Embedding size | `1024` |
| `--batch-size` | Batch size | `256` |
| `--epochs` | Number of training epochs | `100` |
| `--gpu-id` | GPU ID for training | `0` |
| `--workers` | DataLoader workers | `4` |
| `--model` | Model to use (`resnet18`, `resnet50`, `VIT`) | `resnet18` |
| `--loss` | Loss function (`Proxy_Anchor`, `Contrastive`) | `Proxy_Anchor` |
| `--optimizer` | Optimizer to use | `adamw` |
| `--lr` | Learning rate | `1e-3` |
| `--alpha` | Scaling parameter | `16` |
| `--mrg` | Margin | `0.4` |
| `--warm` | Warm-up epochs | `5` |
| `--bn-freeze` | Freeze batch norm layers | `True` |
| `--l2-norm` | Use L2 normalization | `True` |
| `--use_wandb` | Use Weights & Biases logging | `False` |
| `--contrastive_loss_type` | Type of contrastive loss (e.g., `G-Baseline_NCE`, `Offline`, `EWC`, etc.) | `G-Baseline_NCE` |
| `--only_test_step1` | Test only initial step | `False` |
| `--only_test_step2` | Test only incremental step | `False` |
| `--standarization_prerun` | Data standardization before training | `False` |
| `--standarization_run_time` | Data standardization during training | `False` |
| `--learnable_loss_weights` | Use learnable loss weights | `True` |
| `--log_results` | Log results | `True` |
| `--session_split` | Use session-wise split | `True` |
| `--visualize_proxies` | Visualize learned proxies | `False` |
| `--sampling` | Replay sampling type (`Gaussian`, `kde`) | `Gaussian` |
| `--exp` | Experiment ID | `'0'` |
| `--kd_weight` | Knowledge distillation weight | `10` |
| `--pa_weight` | Proxy anchor loss weight | `1` |
| `--processes` | Number of processes | `1` |
| `--threads` | Number of threads | `32` |

---

---

## 📩 Contact

For issues or questions, please reach out via GitHub or email:  
📧 dhruvadityamittal@gmail.com

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
