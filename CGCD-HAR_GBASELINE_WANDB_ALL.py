import os
os.system('pip install einops')
os.system('pip install pytorch_metric_learning')
os.system('pip install plotly')
os.system('pip install -U kaleido')

import argparse, os, copy, random, sys
import itertools
import pandas as pd
from models.tinyhar import tinyhar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from functools import partial
from tqdm import *
import dataset, utils_CGCD, losses, net
from net.resnet import *
from models.modelgen import ModelGen, ModelGen_new
# from torchsummary import summary
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import json
from infoNCE import InfoNCE
from utils_CGCD import calculate_accuracy
import os
from collections import Counter
import multiprocessing
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description=
                                        'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                        + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`')
        # export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
parser.add_argument('--dataset', default='wisdm', help='Training dataset, e.g. mhealth, realworld, pamap, wisdn') 
parser.add_argument('--embedding-size', default=1024, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
parser.add_argument('--batch-size', default=256, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
parser.add_argument('--epochs', default=1, type=int, dest='nb_epochs', help='Number of training epochs.')
parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')
parser.add_argument('--workers', default=4, type=int, dest='nb_workers', help='Number of workers for dataloader.')
parser.add_argument('--model', default='tinyhar', help='Model for training')  # resnet50 #resnet18  VIT
parser.add_argument('--loss', default='Proxy_Anchor', help='Criterion for training') #Proxy_Anchor #Contrastive
parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate setting')  #1e-4
parser.add_argument('--alpha', default=16, type=float, help='Scaling Parameter setting')   # 32
parser.add_argument('--mrg', default=0.4, type=float, help='Margin parameter setting')    # 0.1
parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs')  # 1
parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
parser.add_argument('--l2-norm', default=True, type=bool, help='L2 normlization') 
parser.add_argument('--use_wandb', default=False, type=bool, help='Use Wanb to upload parameters')
parser.add_argument('--contrastive_loss_type', default='G-Baseline_NCE', help='Type of Contrastive Loss: G-Baseline_NCE, G-Baseline_Contrastive, G-Baseline, Online_Finetuning, Offline, Offline_NCE, Online_Finetuning_NCE, G-Baseline_NCE_WFR' )
parser.add_argument('--only_test_step1', default=True, type=bool, help='Test only Initial Step (No training if set to True)')
parser.add_argument('--only_test_step2', default=True, type=bool, help='Test only Incremental Step (No training if set to True)')
parser.add_argument('--standarization_prerun', default=False, type=bool, help='Data Standarization Preruntime') 
parser.add_argument('--standarization_run_time', default=False, type=bool, help='Data Standarization RunTime')
parser.add_argument('--learnable_loss_weights', default=True, type=bool, help='Use Learnable Loss?')
parser.add_argument('--log_results', default=True, type=bool, help='Do you want to log the results')
parser.add_argument('--exp', type=str, default='0')
parser.add_argument('--kd_weight', default = 10, type=float)
parser.add_argument('--pa_weight', default = 1 , type=float)
parser.add_argument('--processes', default = 1 , type=int)
parser.add_argument('--threads', default = 32 , type=int)

args = parser.parse_args()
# args.nb_workers = multiprocessing.cpu_count()




lrs = [1e-4]
kd_weights = [100] #[1,5,10, 15, 20]
pa_wights = [20] #[1, 5, 10, 15, 20]
contrastive_weights = [20]#[1,10]
folds =[0,1,2,3,4] #[0,1,2,3,4] #[0,1,2,3,4] #[0,1,2,3,4] #[0,1,2,3,4] #[0,1,2,3,4]#[0,1,2,3,4] #[1,2,3,4,5]  #[1,2,3,4,5]

results_df = pd.DataFrame(columns = ['Dataset', 'Iteration', 'Initial Acc', 'Initial F1', 'Incremental Acc Seen1',  'Incremental F1 Seen1', 'Incremental Acc Seen2',  'Incremental F1 Seen2', 'Incremental Acc Unseen', 'Incremental F1 Unseen', 'Incremental Acc All', 'Incremental F1 All', "Forget Acc", 'Forget F1' ])

root_dir = f'{os.getcwd()}' #/continual_learning'


pth_rst_exp_step_1 = f'{root_dir}/Saved_Models/Initial/{args.contrastive_loss_type}/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
pth_rst_exp_log_step_1 = pth_rst_exp_step_1+f'results_{args.dataset}_{args.model}.log'

os.makedirs(pth_rst_exp_step_1, exist_ok=True)

pth_rst_exp_step_2 = f'{root_dir}/Saved_Models/Incremental/{args.contrastive_loss_type}/' + args.dataset + '/'
pth_rst_exp_log_step_2 = pth_rst_exp_step_2  + f"results_{args.dataset}_{args.model}.log"
os.makedirs(pth_rst_exp_step_2, exist_ok=True)

if(args.log_results):
    logger_1 = logging.getLogger('logger_1')
    logger_1.setLevel(logging.INFO)
    file_handler_1 = logging.FileHandler(pth_rst_exp_log_step_1, mode='w')
    # file_handler_1.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger_1.addHandler(file_handler_1)

    # Configure the second logger for 'experiment_results_step_2.log'
    logger_2 = logging.getLogger('logger_2')
    logger_2.setLevel(logging.INFO)
    file_handler_2 = logging.FileHandler(pth_rst_exp_log_step_2, mode='w')
    # file_handler_2.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger_2.addHandler(file_handler_2)

    logger_1.info(f"\n******** Dataset: {args.dataset}, Standardize: True, Model: {args.model} ******** \n")
    logger_2.info(f"\n******** Dataset: {args.dataset}, Standardize: True, Model: {args.model} ******** \n")


os.makedirs(pth_rst_exp_step_2, exist_ok= True)
stage_1_test_accs_seen, stage_1_test_f1s_seen, stage_2_test_accs_seen, stage_2_test_f1s_seen= [], [], [], []
stage_2_val_accs_seen, stage_2_val_accs_unseen, stage_2_val_accs_overall   = [], [], []
test_accs_seen, test_accs_unseen, test_accs_overall   = [], [], []
test_f1s_seen, test_f1s_unseen, test_f1s_overall   = [], [], []

if __name__ == '__main__':
    print("\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n")
    print("Using device", device)
    print("Method Executed :", args.contrastive_loss_type)
    print("Model Architecture :", args.model)
    print("Dataset :", args.dataset)
    print('Training for {} epochs'.format(args.nb_epochs))
    

    for lr, kd_weight, pa_weight,contrastive_weight, fold in itertools.product(lrs, kd_weights, pa_wights,contrastive_weights, folds):
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        fold_style_print_log = f"\n******************Fold: {fold} ******************************\n"
        print(fold_style_print_log)
        
        if(args.log_results):    
            logger_1.info(fold_style_print_log)
            logger_2.info(fold_style_print_log)   
  
        args.lr = lr
        args.weight_decay =  args.lr
        args.kd_weight  = kd_weight
        args.pa_weight = pa_weight
        args.contrastive_weight = contrastive_weight
        

        
        if args.dataset =='wisdm':
            pth_dataset = f'{root_dir}/HAR_data/Wisdm/'
            window_len = 40
            n_channels = 3
        elif args.dataset =='realworld':
            pth_dataset = f'{root_dir}/HAR_data/realworld/'
            nb_classes_now = 8
            window_len = 100
            n_channels = 3
        elif args.dataset =='oppo':
            pth_dataset = f'{root_dir}/HAR_data/oppo/'
        elif args.dataset =='pamap':
            pth_dataset = f'{root_dir}/HAR_data/pamap/'
            nb_classes_now = 12
            window_len = 200
            n_channels = 9
        elif args.dataset =='mhealth':
            pth_dataset = f'{root_dir}/HAR_data/mhealth/'
            nb_classes_now = 6
            window_len = 100
            n_channels = 9
        
        if(args.use_wandb):
            import wandb
            wandb.init(
                # set the wandb project where this run will be logged
                project="CGCD-HAR-Supervised_NCE",
                name='specific-run-name',
                resume='allow',
                # track hyperparameters and run metadata
                config={
                "learning_rate_step_1": args.lr,
                "learning_rate_step_2": args.lr,
                "sz_embedding" : args.sz_embedding,
                "window_len": window_len,
                "batch-size" : args.sz_batch,
                "loss" : args.loss,
                "nb_workers": args.nb_workers,
                "architecture": args.model,
                "dataset": args.dataset,
                "epochs": args.nb_epochs,
                "optimizer" : args.optimizer, 
                "weight-decay" :args.weight_decay,
                "lr-decay-step" : args.lr_decay_step,
                "lr-decay-gamma" : args.lr_decay_gamma,
                "alpha" : args.alpha,
                "mrg" : args.mrg,
                "warm" : args.warm,
                "bn-freeze" : args.bn_freeze,
                "l2-norm" : args.l2_norm,
                "exp" : args.exp,
                "KD_weight" : args.kd_weight,
                "pa_weight" : args.pa_weight,
                'contrastive_weight' : contrastive_weight

                }
            ) 
            wandb.log({"Method": args.contrastive_loss_type})
        
        # Set the dimensionality reduction method to t-SNE for later visualization or processing
        dim_reduction_method = "tsne"         

        # Initializing Dataloaders for training, evaluation, and testing datasets
        print("Setting up Dataloaders..")

        # Load the training dataset with the first fold and standardization parameters
        dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen=window_len, 
                                autoencoderType=None, standardize=args.standarization_prerun, fold=fold)

        # Calculate the mean and standard deviation of the training dataset for Standardization
        mean = np.mean(dset_tr_0.xs, axis=0)
        std = np.std(dset_tr_0.xs, axis=0)



        # Standardize the training dataset (z-score normalization)
        if(args.standarization_run_time):
            dset_tr_0.xs = (dset_tr_0.xs - mean) / (std + 1e-5)  # Adding a small value to prevent division by zero

        # Create the dataloader for the training dataset with batch shuffling
        dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=True, 
                                                num_workers=args.nb_workers, pin_memory=True, 
                                                worker_init_fn=lambda x: np.random.seed(42))

        # Load the evaluation dataset using the same mean and std as the training dataset
        dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', windowlen=window_len, 
                            autoencoderType=None, standardize=args.standarization_prerun, fold=fold)

        # Normalize the evaluation dataset using the training set statistics
        
        if(args.standarization_run_time):dset_ev.xs = (dset_ev.xs - mean) / (std + 1e-5)

        # Create the dataloader for the evaluation dataset (no shuffling)
        dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False, 
                                            num_workers=args.nb_workers, pin_memory=True, 
                                            worker_init_fn=lambda x: np.random.seed(42))

        # Load the test dataset and normalize using the training set statistics
        dset_test_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='test_0', windowlen=window_len, 
                                autoencoderType=None, standardize=args.standarization_prerun, fold=fold)

        # Normalize the test dataset
        if(args.standarization_run_time): dset_test_0.xs = (dset_test_0.xs - mean) / (std + 1e-5)

        # Create the dataloader for the test dataset (no shuffling)
        dlod_test_0 = torch.utils.data.DataLoader(dset_test_0, batch_size=args.sz_batch, shuffle=False, 
                                                num_workers=args.nb_workers, pin_memory=True, 
                                                worker_init_fn=lambda x: np.random.seed(42))

        # Set the number of classes for classification tasks
        print("\n--------------------- Initial Step --------------------- \n")
        nb_classes = dset_tr_0.nb_classes()
        nb_subjects = dset_tr_0.nb_subjects()
        print("\n")
        # Configuration for different model architectures
        if(args.model == 'resnet18'):
            # Setting up configuration for ResNet18 model
            cfg = {'weights_path': 'CGCD-main/src/Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', 
                'use_ssl_weights': False, 'conv_freeze': False, 'load_finetuned_mtl': False,
                'checkpoint_name': '', 'epoch_len': 2, 'output_size': nb_classes, 
                'embedding_dim': args.sz_embedding, 'bottleneck_dim': None,
                'weight_norm_dim': '', 'n_channels': n_channels, 'window_len': window_len}

            # Create the ResNet18 model
            model = ModelGen_new(cfg).create_model().to(device)

        elif(args.model == 'tinyhar'):
            # Create the tinyHAR model
            model = tinyhar(n_channels, window_len, args.sz_embedding).to(device)

        elif(args.model == 'harnet'):
            # Load a pretrained HARNet model from the torch hub and modify its architecture
            repo = 'OxWearables/ssl-wearables'
            model = torch.hub.load(repo, 'harnet5', class_num=nb_classes, pretrained=True).to(device)
            
            # Remove the classifier and add a new embedding layer
            del model.classifier
            model.embedding = nn.Sequential(nn.Linear(model.feature_extractor.layer5[0].out_channels, 
                                                    args.sz_embedding)).to(device)

        # Count and print the total number of parameters in the model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params}")

        # Initialize Proxy Anchor Loss for metric learning
        criterion_pa = losses.Proxy_Anchor(nb_classes=nb_classes, sz_embed=args.sz_embedding, 
                                        mrg=args.mrg, alpha=args.alpha).to(device)

        # GPU settings for training
        gpu_id = 0

        # Define training parameter groups depending on the model
        if(args.model == 'tinyhar'):
            # Freeze batch normalization layers for tinyHAR model
            args.bn_freeze = False
            args.warm = 0
            
            # Set different learning rates for the prediction head and the rest of the model
            param_groups = [
                {'params': list(set(model.model.parameters()).difference(set(model.model.prediction.parameters()))) 
                if gpu_id != -1 else list(set(model.module.parameters()).difference(set(model.module.model.prediction.parameters())))},
                {'params': model.model.prediction.parameters() 
                if gpu_id != -1 else model.model.prediction.parameters(), 'lr': float(args.lr) * 1},
            ]
        else:
            # Define parameter groups for other models with different learning rates for the embedding layer
            param_groups = [
                {'params': list(set(model.parameters()).difference(set(model.embedding.parameters()))) 
                if gpu_id != -1 else list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
                {'params': model.embedding.parameters() 
                if gpu_id != -1 else model.embedding.parameters(), 'lr': float(args.lr) * 1},
            ]

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


        # Add the Proxy Anchor loss parameters with a higher learning rate
        param_groups.append({'params': criterion_pa.parameters(), 'lr': float(args.lr) * 100})

        # Initialize the AdamW optimizer with weight decay
        opt_pa = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

        # Optionally, a learning rate scheduler can be added (commented out)
        # scheduler_pa = torch.optim.lr_scheduler.StepLR(opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)


            # print('Training parameters: {}'.format(vars(args)))
        # Print the training parameters and the total number of epochs for the current training run.
        
        losses_list = []
        best_recall = [0]
        best_epoch = 0
        best_acc, es_count = 0, 0  # Initialize best accuracy and early stopping count.

        version = 1
        step = 1

        # Initialize the InfoNCE loss function for contrastive learning with a specified temperature.
        nce_loss = InfoNCE(temperature=0.07, negative_mode='paired')

        # Define weights for the Proxy Anchor loss and contrastive loss.
        l_pa_weight = torch.tensor(1.0, requires_grad=True)
        l_contrastive_weight = torch.tensor(1.0, requires_grad=True)

        # Initial Validation Lossesi
        if(args.only_test_step1 == False):
            first_validation_losses = losses.get_validation_losses_step_1(model, criterion_pa, nb_classes, args, nce_loss, dlod_ev)
            validation_losses = first_validation_losses
        # Start the training loop for the specified number of epochs.
        for epoch in range(0, args.nb_epochs):
            # Skip training if only testing is required.
            if args.only_test_step1: continue
            
            # Set the model to training mode.
            model.train()

            # If batch normalization freezing is enabled, set all BatchNorm layers to evaluation mode.
            # This prevents the update of batch normalization statistics during training.
            if args.bn_freeze:
                modules = model.modules() if gpu_id != -1 else model.module.model.modules()
                for m in modules:
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

            # Warmup strategy: freeze certain layers in the initial epochs and gradually unfreeze them.
            if args.warm > 0:
                if gpu_id != -1:
                    if args.model == 'tinyhar':
                        # Unfreeze the prediction layers and criterion for the 'tinyhar' model.
                        unfreeze_model_param = list(model.model.prediction.parameters()) + list(criterion_pa.parameters())
                    else:
                        # Unfreeze the embedding layers and criterion for other models.
                        unfreeze_model_param = list(model.embedding.parameters()) + list(criterion_pa.parameters())
                else:
                    if args.model == 'tinyhar':
                        unfreeze_model_param = list(model.model.prediction.parameters()) + list(criterion_pa.parameters())
                    else:
                        unfreeze_model_param = list(model.embedding.parameters()) + list(criterion_pa.parameters())

                # Freeze all layers except the specified parameters during warmup.
                if epoch == 0:
                    for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = False
                
                # Unfreeze the layers after the warmup period is over.
                if epoch == args.warm:
                    for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = True

            total, correct = 0, 0
            total_train_loss = 0
            total_data = 0
            total_pa_loss = 0
            total_contrastive_loss = 0

            # Iterate over the training data (features x, labels y, and other data z).
            for batch_idx, (x, y, z) in enumerate(dlod_tr_0):

                # Forward pass: extract features from the model.
                feats = model(x.squeeze().to(device))

                # Extract positive proxies for the given labels y.
                positive_proxies = criterion_pa.proxies[[torch.tensor(y, dtype=torch.long)]]

                # Collect negative proxies for all labels except the current label.
                negetive_proxies = []
                for y_temp in y:
                    t = [i for i in range(nb_classes) if i != y_temp.item()]  # All other proxies are negative.
                    if len(negetive_proxies) == 0:
                        negetive_proxies = criterion_pa.proxies[t].unsqueeze(0)
                    else:
                        negetive_proxies = torch.concat((negetive_proxies, criterion_pa.proxies[t].unsqueeze(0)))

                # Compute Proxy Anchor loss.
                loss_pa = criterion_pa(feats, y.squeeze().to(device)).to(device)

                # Compute contrastive loss between the extracted features and proxies.
                # contrastive_loss = utils_CGCD.contrastive_loss(feats, y, criterion_pa.proxies, True).to(device)

                # If contrastive loss type is NCE (Noise Contrastive Estimation), compute the NCE loss.
                if args.contrastive_loss_type in ['G-Baseline_NCE', 'Offline_NCE', 'Online_Finetuning_NCE', 'G-Baseline_NCE_WFR']:
                    contrastive_loss = nce_loss(feats, positive_proxies, negetive_proxies)
                # If contrastive loss type is the baseline contrastive loss, compute the corresponding loss.
                elif args.contrastive_loss_type == 'G-Baseline_Contrastive':
                    contrastive_loss = utils_CGCD.contrastive_loss(feats, y, criterion_pa.proxies, True).to(device)
                # If no contrastive loss is required, set contrastive loss to zero.
                else:
                    contrastive_loss = torch.tensor(0).to(device)
                    # model.contrastive_weight_lr.requires_grad = False

                # Combine the Proxy Anchor loss and contrastive loss with their respective weights.
                # print(validation_losses)#
                # Only Offline and Online Finetuning Models donot have contrastive losses.
                if(args.contrastive_loss_type not in ['Offline', 'Online_Finetuning', 'G-Baseline']):
                    training_losses = [loss_pa.item(), contrastive_loss.item()]
                    max_ratio = max([validation_losses[i]/training_losses[i] for i in range(len(validation_losses))])

                    weights = [(validation_losses[i]/training_losses[i])/first_validation_losses[i] for i in range(len(validation_losses))]

                    normalized_weights = [weights[i]/min(weights) for i in range(len(weights))]
                    total_loss =  weights[0]*loss_pa + weights[1]*contrastive_loss
                else:
                    total_loss = loss_pa
                # Zero the gradients before performing backpropagation.
                opt_pa.zero_grad()

                # Perform backpropagation to compute the gradients.
                total_loss.backward()

                # Clip gradients to prevent exploding gradients.
                torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                if args.loss == 'Proxy_Anchor':
                    torch.nn.utils.clip_grad_value_(criterion_pa.parameters(), 10)

                # Perform an optimization step to update the model parameters.
                opt_pa.step()

                # Accumulate the total loss for reporting.
                total_pa_loss += loss_pa.item() * x.size(0)
                total_contrastive_loss += contrastive_loss.item() * x.size(0)
                total_train_loss += total_loss.item() * x.size(0)
                total_data += x.size(0)

                # Print the progress of the training epoch.
                print("Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}".format(epoch, batch_idx + 1, len(dlod_tr_0), 100. * batch_idx / len(dlod_tr_0), total_loss.item(), 0, 0), end="\r")

            # Print the total contrastive loss, Proxy Anchor loss, and overall loss after the epoch.
            print("Contrastive Loss {:.4f}, PA Loss {:.4f}, Total Loss {:.4f}".format(total_contrastive_loss / total_data, total_pa_loss / total_data, total_train_loss / total_data))

            validation_losses = losses.get_validation_losses_step_1(model, criterion_pa, nb_classes, args, nce_loss, dlod_ev)
            # exit()
            # Evaluate the model using cosine similarity with proxies.
            with torch.no_grad():
                feats, _ = utils_CGCD.evaluate_cos_(model, dlod_ev)
                cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
                _, preds_lb = torch.max(cos_sim, dim=1)
                preds = preds_lb.detach().cpu().numpy()
                # acc_0, _ = utils_CGCD._hungarian_match_(np.array(dlod_ev.dataset.ys), preds)
                acc_0 = accuracy_score(np.array(dlod_ev.dataset.ys), preds)
                print('Valid Epoch: {} Acc: {:.4f}'.format(str(-1), acc_0))
                if args.use_wandb:
                    wandb.log({"Step 1: Val Accuracy": acc_0, "custom_step": epoch})
                    wandb.log({"Step 1: F1 Score": f1_score(np.array(dlod_ev.dataset.ys), preds, average='macro'), "custom_step": epoch})

            # Save the model if it achieves a new best accuracy.
            if best_acc < acc_0:
                print("Got better model with from accuracy {:.4f} to {:.4f}".format(best_acc, acc_0))
                best_acc = acc_0
                torch.save({'model_pa_state_dict': model.state_dict(), 'proxies_param': criterion_pa.proxies},
                        '{}{}_{}_best_windowlen_{}_embedding_size_{}_alpha{}_mrg_{}_version_{}_step_{}_fold_{}.pth'.format(
                            pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, args.alpha, args.mrg, version, step, fold))
                es_count = 0
            else:
                # Increase the early stopping counter if no improvement in accuracy is found.
                es_count += 1
                if epoch > args.warm:
                    print("Early stopping count {}".format(es_count))

            # Reset early stopping counter during the warmup period.
            if epoch < args.warm:
                es_count = 0

            # Stop training early if the early stopping counter reaches 10 or if the accuracy is perfect (1.0).
            if es_count == 10 or int(acc_0) == 1:
                print("Early Stopping with Count {} and Best Accuracy {:.4f}".format(es_count, best_acc))
                break

            del total_loss, feats


                #################################################################
                # Proxy Anchor training is now complete.                        #
                #################################################################

        # Load the best model checkpoint for further evaluation or fine-tuning.
        print('==> Resuming from checkpoint..')
        pth_pth = '{}{}_{}_best_windowlen_{}_embedding_size_{}_alpha{}_mrg_{}_version_{}_step_{}_fold_{}.pth'.format(
            pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, args.alpha, args.mrg, version, step, fold)
        checkpoint = torch.load(pth_pth, map_location=torch.device(device))

        # Load the model and proxies from the checkpoint.
        model.load_state_dict(checkpoint['model_pa_state_dict'])
        criterion_pa.proxies = checkpoint['proxies_param']

        # Move the model to the specified device and set it to evaluation mode.
        model = model.to(device)
        model.eval()

        
# Visualization using t-SNE (or another dimensionality reduction method)
        method = 'tsne'
        utils_CGCD.visualize_proxy_anchors(model, dlod_test_0, criterion_pa.proxies, args.dataset, args.sz_embedding, nb_classes, step, dim_reduction_method)
        exit(0)
        ###
        print('==> Evaluation of Stage 1 Dataset on Stage 1 Model\n')
        model.eval()  # Set model to evaluation mode (disables dropout, batchnorm)
        with torch.no_grad():  # Disable gradient calculation (speeds up and reduces memory usage)
            # Extract features and perform evaluation on the test dataset
            feats, _ = utils_CGCD.evaluate_cos_(model, dlod_test_0)
            
            
            # Calculate cosine similarity between normalized features and proxy anchors
            cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))  # Cosine similarity between -1 and 1

            # Get predicted labels based on maximum cosine similarity (class with highest similarity)
            _, preds_lb = torch.max(cos_sim, dim=1)
            preds = preds_lb.detach().cpu().numpy()

            # Compute accuracy and F1 score using Hungarian matching on the predictions
            # stage_1_acc_0, _ = utils_CGCD._hungarian_match_(np.array(dlod_test_0.dataset.ys), preds)
            stage_1_acc_0 = accuracy_score(np.array(dlod_test_0.dataset.ys), preds)
            stage_1_f1_0 = f1_score(np.array(dlod_test_0.dataset.ys), preds, average='macro')


            # Log results to Weights & Biases if enabled
            if args.use_wandb:
                wandb.log({"M1-TA1_Old": stage_1_acc_0})
                wandb.log({"M1-TF1_Old": stage_1_f1_0})

        # Store predictions and print results for Stage 1
        preds1 = preds
        # print(f"Stage 1 {len(preds)}, Unique {np.unique(np.array(dlod_test_0.dataset.ys))}")
        print("Step 1 Test Dataset Accuracy on Stage 1 trained Model", stage_1_acc_0)
        print("Step 1 Test Dataset F1 Score on Stage 1 trained Model", stage_1_f1_0)

        # Write results to log file
        # with open(pth_rst_exp_log_step_1, "a") as file:
        #     file.write(f"Step 1: Test Dataset Acc: {stage_1_acc_0} F1: {stage_1_f1_0}\n")
        if(args.log_results ):    logger_1.info(f"Step 1: Test Dataset Acc: {stage_1_acc_0} F1: {stage_1_f1_0}\n")

        # Append Stage 1 results for future reference
        stage_1_test_accs_seen.append(stage_1_acc_0)
        stage_1_test_f1s_seen.append(stage_1_f1_0)
        ###

        # Prepare data for Step 2 training
        dlod_tr_prv = dlod_tr_0
        dset_tr_now_md = 'train_1'  # Set mode for Step 2 training dataset
        dset_ev_now_md = 'eval_1'  # Set mode for Step 2 evaluation dataset
        nb_classes_prv = nb_classes  # Keep track of number of classes seen in Stage 1


        # exit()
        # Load the new training and evaluation datasets for Step 2
        dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, windowlen=window_len, autoencoderType=None, standardize=args.standarization_prerun, fold=fold)
        

        # If contrastive loss is 'Offline', extend current dataset with Stage 1 dset_tr_now
        if args.contrastive_loss_type in ['Offline', 'Offline_NCE']:
            print("Adding Stage 1 Training Data to the Step 2 Training Data For Offline Model..")
            dset_tr_now.xs.extend(dset_tr_0.xs)
            dset_tr_now.ys.extend(dset_tr_0.ys)
            dset_tr_now.I.extend(dset_tr_0.I)

        # Standardize training data
        mean = np.mean(dset_tr_now.xs, axis=0)
        std = np.std(dset_tr_now.xs, axis=0)
        if(args.standarization_run_time): dset_tr_now.xs = (dset_tr_now.xs - mean) / (std + 1e-5)

        # Standardize evaluation and test datasets using same mean and std from training data
        dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_ev_now_md, windowlen=window_len, autoencoderType=None, standardize=args.standarization_prerun, fold=fold)
        if(args.standarization_run_time): dset_ev_now.xs = (dset_ev_now.xs - mean) / (std + 1e-5)
        dset_test = dataset.load(name=args.dataset, root=pth_dataset, mode='test_1', windowlen=window_len, autoencoderType=None, standardize=args.standarization_prerun, fold=fold)
        if(args.standarization_run_time): dset_test.xs = (dset_test.xs - mean) / (std + 1e-5)

        

        # Create data loaders for the newly loaded datasets
        dlod_tr_now = torch.utils.data.DataLoader(dset_tr_now, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))
        dlod_ev_now = torch.utils.data.DataLoader(dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))
        dlod_test = torch.utils.data.DataLoader(dset_test, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))

        print("\n--------------------- Incremental Step --------------------- \n")
        nb_classes_now = dset_tr_now.nb_classes()
        nb_subjects_now = dset_tr_now.nb_subjects()
        print("\n")

        # Evaluate Step 2 test dataset on Stage 1 trained model
        print('==> Evaluation of Step 2 Dataset on Step 1 Model')
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Extract features from the test data
            feats, _ = utils_CGCD.evaluate_cos_(model, dlod_test)
            
            # Compute cosine similarity between features and proxy anchors
            cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
            _, preds_lb = torch.max(cos_sim, dim=1)
            preds = preds_lb.detach().cpu().numpy()

            # Split predictions into old (Stage 1) and new (Step 2) classes
            y_temp = torch.tensor(dlod_test.dataset.ys)

            old_classes = np.nonzero(torch.where(y_temp < nb_classes, 1, 0))
            new_classes = np.nonzero(torch.where(y_temp >= nb_classes, 1, 0))

            # Compute accuracy and F1 score for old and new classes separately
            # print(old_classes)
            # print(np.array(dlod_test.dataset.ys)[old_classes].shape)
            # print(preds[old_classes].)

            stage_2_acc_0 = accuracy_score(np.array(dlod_test.dataset.ys)[old_classes].reshape(-1), preds[old_classes].reshape(-1))
            stage_2_acc_1 = accuracy_score(np.array(dlod_test.dataset.ys)[new_classes].reshape(-1), preds[new_classes].reshape(-1))
            stage_2_f1_0 = f1_score(np.array(dlod_test.dataset.ys)[old_classes].reshape(-1), preds[old_classes].reshape(-1), average='macro')


        # Print the Step 2 accuracy results
        print("Step 2 Test Dataset on Stage 1 trained model, Seen Accuracy {:.4f}  Unseen Accuracy  {:.4f}".format(stage_2_acc_0, stage_2_acc_1))
    
    
        # Log results to a file
        # with open(pth_rst_exp_log_step_1, "a") as file:
        #     file.write(f"Step 2: Test Dataset Acc Seen Acc: {stage_2_acc_0}, Seen F1: {stage_2_f1_0} Unseen Acc: {stage_2_acc_1} \n")
        if(args.log_results):    logger_1.info(f"Step 2: Test Dataset Acc Seen Acc: {stage_2_acc_0}, Seen F1: {stage_2_f1_0} Unseen Acc: {stage_2_acc_1} \n")
        # Log Step 2 results to Weights & Biases if enabled
        if args.use_wandb:
            wandb.log({"M1-TA2_Old": stage_2_acc_0})  # Step 2 test accuracy (seen classes)
            wandb.log({"M1-TF2_Old": stage_2_f1_0})  # Step 2 test F1 score (seen classes)

        # Append Step 2 results to the lists for future reference
        stage_2_test_accs_seen.append(stage_2_acc_0)
        stage_2_test_f1s_seen.append(stage_2_f1_0)

        ####
        # If we're not just testing on step 2, proceed with training
        if args.only_test_step2 != True:
            load_exempler = False
            if load_exempler:
                # Load saved exemplar data for the model
                print("Loading Saved Exempler ..")
                expler_s = torch.load(pth_rst_exp_step_1 + 'expler_s_tensor_model_{}_{}_windowlen_{}_sz_embedding_{}_alpha{}_mrg_{}_PA_{}_KD_{}_lr_{}_wd_{}_fold_{}.pt'.format(args.dataset, args.model, window_len, args.sz_embedding, args.alpha, args.mrg, args.pa_weight, args.kd_weight, args.lr, args.weight_decay, fold), map_location=torch.device(device))
            else:
                # Calculate exemplar statistics (mean and standard deviation) for the proxies
                print('==> Calc. proxy mean and sigma for exemplar..')
                with torch.no_grad():
                    feats, _ = utils_CGCD.evaluate_cos_(model, dlod_tr_prv)
                    feats = losses.l2_norm(feats)
                    # print(feats.shape)
                    expler_s = feats.std(dim=0).to(device)

                    torch.save(expler_s, pth_rst_exp_step_1 + 'expler_s_tensor_model_{}_{}_windowlen_{}_sz_embedding_{}_alpha{}_mrg_{}_PA_{}_KD_{}_lr_{}_wd_{}_fold_{}.pt'.format(args.dataset, args.model, window_len, args.sz_embedding, args.alpha, args.mrg, args.pa_weight, args.kd_weight, args.lr, args.weight_decay, fold))



                                ################################  Incremental Step ########################################
                            ##########################################        ################################################################

        
        
        

        print("==> Extracting Cluster Information ..")
        if args.contrastive_loss_type not in  ["Online_Finetuning", 'Offline', 'Online_Finetuning_NCE', 'Offline_NCE', 'G-Baseline_NCE_WFR']:
            # feats_initial_train, y_initial_train = utils_CGCD.evaluate_cos_(model, dlod_tr_0)
            
            # cluster_data, fitted_kdes = utils_CGCD.get_cluster_information(feats_initial_train.to('cpu').detach().numpy(), y_initial_train)
            cluster_data_raw, fitted_kdes_raw = utils_CGCD.get_cluster_information_raw(dset_tr_0.xs, dset_tr_0.ys)
            # utils_CGCD.visualize_sampling_raw(dset_tr_0.xs, dset_tr_0.ys, [0], cluster_data_raw, fitted_kdes_raw)
        # utils_CGCD.plot_data(feats_initial_train, y_initial_train, 1, cluster_data, fitted_kdes)
       
       
        
        # Update number of classes for current training phase and initialize the proxy-anchor loss
        
        criterion_pa_now = losses.Proxy_Anchor(nb_classes=nb_classes_now, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).to(device)
        criterion_pa_now.proxies.data[:nb_classes_prv] = copy.deepcopy(criterion_pa.proxies.data)  # Reuse proxies from Stage 1
        criterion_pa.proxies.requires_grad = False
        

        # Initialize best performance metrics for tracking
        bst_acc_a, bst_acc_oo, bst_acc_on, bst_acc_no, bst_acc_nn = 0., 0., 0., 0., 0.
        bst_epoch_a, bst_epoch_o, bst_epoch_n = 0., 0., 0.

        # Create a deep copy of the model for Step 2 training
        
        model_now = copy.deepcopy(model)
        model_now = model_now.to(device)

        # Set up different learning rates for different parts of the model
        if args.model == 'tinyhar':
            param_groups = [
                {'params': list(set(model_now.model.parameters()).difference(set(model_now.model.prediction.parameters()))) if gpu_id != -1 else list(set(model_now.module.parameters()).difference(set(model_now.module.model.prediction.parameters())))},
                {'params': model_now.model.prediction.parameters() if gpu_id != -1 else model_now.model.prediction.parameters(), 'lr': float(args.lr) * 1},
            ]
        else:
            param_groups = [
                {'params': list(set(model_now.parameters()).difference(set(model_now.embedding.parameters()))) if gpu_id != -1 else list(set(model_now.module.parameters()).difference(set(model_now.module.model.embedding.parameters())))},
                {'params': model_now.embedding.parameters() if gpu_id != -1 else model_now.embedding.parameters(), 'lr': float(args.lr) * 1},
            ]


        # Add proxy-anchor parameters to optimizer with a higher learning rate
        param_groups.append({'params': criterion_pa_now.parameters(), 'lr': float(args.lr)})

        # Initialize the optimizer (AdamW) with the specified parameter groups
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, betas=(0.9, 0.999))
        
        if args.use_wandb:
            wandb.log({"M1 Classes": nb_classes})
            wandb.log({"M2 Classes": nb_classes_now})

        # Training the Incremental Step
        # If loss weights are learnable, initialize them as tensors with requires_grad=True.
        if args.learnable_loss_weights:
            args.pa_weight = torch.tensor(1.0, requires_grad=True)
            args.kd_weight = torch.tensor(1.0, requires_grad=True)
            args.contrastive_weight = torch.tensor(1.0, requires_grad=True)

        epoch = 0
        best_weighted_acc = 0
        es_count = 0
        step = 2
        ep = args.nb_epochs

        if(args.only_test_step2 == False):
            first_validation_losses= losses.get_validation_losses_step_2(model, model_now, criterion_pa, criterion_pa_now, expler_s, nb_classes_prv, nb_classes_now, args, nce_loss, dlod_ev_now)
            validation_losses  = first_validation_losses
        
        ########################## Training for incremental step starts here ##############################

        # Training loop for the incremental step
        for epoch in range(0, args.nb_epochs):
            if args.only_test_step2:
                continue  # Skip training if only testing step 2

            model_now.train()  # Set the model to training mode

            # Freeze BatchNorm layers if bn_freeze is enabled
            if args.bn_freeze:
                modules = model_now.modules() if args.gpu_id != -1 else model_now.module.model.modules()
                for m in modules:
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

            # Handle warm-up phase
            if args.warm > 0:
                if args.gpu_id != -1:
                    if args.model == 'tinyhar':
                        unfreeze_model_param = list(model_now.model.prediction.parameters()) + list(criterion_pa_now.parameters())
                    else:
                        unfreeze_model_param = list(model_now.embedding.parameters()) + list(criterion_pa_now.parameters())
                else:
                    if args.model == 'tinyhar':
                        unfreeze_model_param = list(model_now.model.prediction.parameters()) + list(criterion_pa_now.parameters())
                    else:
                        unfreeze_model_param = list(model_now.embedding.parameters()) + list(criterion_pa_now.parameters())

                if epoch == 0:
                    for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = False
                if epoch == args.warm:
                    for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = True

            # Iterate over the training data
            pbar = tqdm(enumerate(dlod_tr_now))
            total_train_loss, total_kd_loss, total_pa_loss, total_contrastive_loss = 0.0, 0.0, 0.0, 0.0
            total_data = 0
            for batch_idx, (x, y, z) in pbar:
                
                if args.contrastive_loss_type not in  ["Online_Finetuning", 'Offline', 'Online_Finetuning_NCE', 'Offline_NCE', 'G-Baseline_NCE_WFR']:
                    x_sampled, y_sampled = utils_CGCD.get_sampled_data_kde_raw(cluster_data_raw, 512//nb_classes, fitted_kdes_raw )

                    
                    x_sampled =  np.array(x_sampled).reshape(-1, x.shape[1],  x.shape[2])
                    x_sampled = torch.tensor(x_sampled).float()
                    y_sampled = torch.tensor(y_sampled)
                    
                    x = torch.cat((x, x_sampled),0)
                    y = torch.cat((y, y_sampled),0)
                    
                feats = model_now(x.squeeze().to(device))
                
                y = y.to(device)
                y_batch = y
                y_n = torch.where(y >= nb_classes_prv, 1, 0)  # Identify new classes
                

                ########################## Feature Replay Code ###########################

                # Feature replay is not used in Online Finetuning and Offline Model, G-Basleline_Without_Feature_Replay

                # if args.contrastive_loss_type not in  ["Online_Finetuning", 'Offline', 'Online_Finetuning_NCE', 'Offline_NCE', 'G-Baseline_NCE_WFR']:
                #     # Example replay: Generate old class examples to mitigate forgetting
                    

                #     ############################################ Previous Version #######################################
                #     y_o = y.size(0) - y_n.sum()  # Count old classes
                #     # if y_o > 0:
                #     #     y_sp = torch.randint(nb_classes_prv, (y_o,))  # Randomly select old class proxies
                #     #     feats_sp = torch.normal(criterion_pa.proxies[y_sp], expler_s).to(device)  # Generate synthetic features for old classes
                #     #     y = torch.cat((y, y_sp), dim=0)
                #     #     feats = torch.cat((feats, feats_sp), dim=0)

                #     #####################################################################################################

                    
                #     # new_class_ratio = y_n.sum()/(nb_classes_now - nb_classes_prv)
                #     # old_class_ratio = y_o/(nb_classes_prv)
                #     # print(f"New Class Ratio: {new_class_ratio}, Old Class Ratio: {old_class_ratio}")

                #     # We sample a old data points equal to the batch size and then distribute the seen classes equally.
                #     number_of_samples_each_class = args.sz_batch//nb_classes_prv
                #     sampled_x, sampled_y = utils_CGCD.get_sampled_data_kde(cluster_data, number_of_samples_each_class, fitted_kdes)

                #     feats_sp = torch.tensor(sampled_x).float().to(device)
                #     y_sp = torch.tensor(sampled_y).to(device)
                    
                #     y = torch.cat((y, y_sp), dim=0)
                #     feats = torch.cat((feats, feats_sp), dim=0)

                    
                #     # y_n = torch.where(y >= nb_classes_prv, 1, 0)
                #     # y_o = y.size(0) - y_n.sum()  # Count old classes
                
                #     # new_class_ratio = y_n.sum()/(nb_classes_now - nb_classes_prv)
                #     # old_class_ratio = y_o/(nb_classes_prv)
                #     # print(f"After Sampling: New Class Ratio: {new_class_ratio}, Old Class Ratio: {old_class_ratio}")
                #     # exit()
                
                # Compute Proxy-Attention loss
                loss_pa = criterion_pa_now(feats, y.squeeze().to(device))

                
                # # Compute Knowledge Distillation (KD) loss
                # feats_new = model_now(x.squeeze().to(device))
                # logits_new = F.linear(losses.l2_norm(feats_new), losses.l2_norm(criterion_pa_now.proxies))[:, :nb_classes_prv]
                # with torch.no_grad():
                #     feats_old = model(x.squeeze().to(device))
                #     logits_old = F.linear(losses.l2_norm(feats_old), losses.l2_norm(criterion_pa.proxies))
                # T = 2  # Temperature for softening the probability distributions
                # p = F.log_softmax(logits_new / T, dim=1)
                # q = F.softmax(logits_old / T, dim=1)
                # loss_kd = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]

                ### KD
                # We perform no knowledge distillattion for Online and Offline Models.
                if args.contrastive_loss_type not in  ["Online_Finetuning", 'Offline', 'Online_Finetuning_NCE', 'Offline_NCE']:
                    # y_n = torch.where(y_batch >= nb_classes_prv, 0, 1) # Remove it
                    y_o_msk = torch.nonzero(y_n)  # Provides index of all new classes

                    # if y_o_msk.size(0) > 1:
                    #     y_o_msk = torch.nonzero(y_n).squeeze()
                    #     x_o = torch.unsqueeze(x[y_o_msk[0]], dim=0)  # Just first element

                    #     feats_n = torch.unsqueeze(feats[y_o_msk[0]], dim=0)
                    #     for kd_idx in range(1, y_o_msk.size(0)): # After 1st Index
                    #         try:    
                    #             x_o_ = torch.unsqueeze(x[y_o_msk[kd_idx]], dim=0)
                    #         except:
                    #             raise ValueError("A value error occurred",kd_idx , y_o_msk[kd_idx])

                    #         x_o = torch.cat((x_o, x_o_), dim=0)
                    #         feats_n_ = torch.unsqueeze(feats[y_o_msk[kd_idx]], dim=0)
                    #         feats_n = torch.cat((feats_n, feats_n_), dim=0)
                    #     with torch.no_grad():
                    #         feats_o = model(x_o.squeeze().to(device))
                    #     feats_n = feats_n.to(device)
                    #     # FRoST
                    #     loss_kd = torch.dist(F.normalize(feats_o.view(feats_o.size(0) * feats_o.size(1), 1), dim=0).detach(), F.normalize(feats_n.view(feats_o.size(0) * feats_o.size(1), 1), dim=0))
                    # else:
                    #     loss_kd = torch.tensor(0.).to(device)
                    loss_kd = (criterion_pa.proxies - criterion_pa_now.proxies[0:nb_classes]).pow(2).sum(1).sqrt().sum().to(device)
                else:
                    loss_kd = torch.tensor(0.).to(device)
                    # model_now.kd_weight_lr.requires_grad  = False
                
                

                # Compute INFO_NCE Contrastive Loss
                # Offline_NCE and Online_Finetuning_NCE have contrastive loss
                if args.contrastive_loss_type in ['G-Baseline_NCE', 'Offline_NCE', 'Online_Finetuning_NCE', 'G-Baseline_NCE_WFR']:
                    positive_proxies = criterion_pa_now.proxies[[torch.tensor(y, dtype=torch.long)]]
                    negative_proxies = []
                    for y_temp in y:
                        t = [i for i in range(nb_classes_now) if i != y_temp.item()]

                        #If else is just to append the data int the array
                        if len(negative_proxies) == 0:
                            negative_proxies = criterion_pa_now.proxies[t].unsqueeze(0)
                        else:
                            negative_proxies = torch.concat((negative_proxies, criterion_pa_now.proxies[t].unsqueeze(0)))

                    contrastive_loss = nce_loss(feats, positive_proxies, negative_proxies)

                # Computer Simple Contrastive Loss
                elif args.contrastive_loss_type == 'G-Baseline_Contrastive':
                    contrastive_loss = utils_CGCD.contrastive_loss(feats, y, criterion_pa.proxies, True).to(device)
                else:
                    contrastive_loss = torch.tensor(0).to(device)
                    # model_now.contrastive_weight_lr.requires_grad  = False

                training_losses = [loss_pa, loss_kd, contrastive_loss]
                losses_ind = []
                # Getting Losses indices which are not zero
                for i in range(3):
                    if(validation_losses[i]!=0 and training_losses[i].item()!=0):      losses_ind.append(i)

                # Getting the selected training and validation losses
                training_losses = [training_losses[i] for i in losses_ind]
                validation_losses_t = [validation_losses[i] for i in losses_ind]

                # Getting the initial validation losses calculated before training on selected indices
                first_validation_losses_t = [first_validation_losses[i] for i in losses_ind]

                # Calculating Max Ration
                max_ratio = max([validation_losses_t[i] / training_losses[i].item() for i in range(len(validation_losses_t))])
                
                # Calculating Weights for each valid loss
                weights = [(validation_losses_t[i] / training_losses[i].item()) / first_validation_losses_t[i] for i in range(len(validation_losses_t))]

                # Normalizing the Weights
                normalized_weights = [weights[i] / min(weights) for i in range(len(weights))]
                # Compute the total losspint

                # Adding Weighted loss in the final loss.
                loss = 0
                for i in range(len(validation_losses_t)): loss += weights[i]*training_losses[i]




                total_train_loss += loss.item() * x.size(0)
                total_kd_loss += loss_kd.item() * x.size(0)
                total_pa_loss += loss_pa.item() * x.size(0)
                total_contrastive_loss += contrastive_loss.item() * x.size(0)
                total_data += x.size(0)

                opt.zero_grad()
                loss.backward()
                opt.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}/{:.6f}/{:.6f}'.format(
                    epoch, batch_idx + 1, len(dlod_tr_now), 100. * batch_idx / len(dlod_tr_now),
                    loss.item(), loss_pa.item(), loss_kd.item()
                ))

            if total_data > 0:
                print("Epoch {} Train Total Loss {:.4f}, PA Loss {:.4f}, KD Loss {:.4f} Contrastive Loss {:.4f}".format(
                    epoch, total_train_loss / total_data, total_pa_loss / total_data, 
                    total_kd_loss / total_data, total_contrastive_loss / total_data
                ))

            if args.use_wandb:
                wandb.log({"Step 2: Train Loss": total_train_loss / total_data, "Step 2: Train PA Loss": total_pa_loss / total_data, 
                        "Step 2: Train KD Loss": total_kd_loss / total_data, "Step 2: Train Contrastive Loss": total_contrastive_loss / total_data, 
                        "custom_step": epoch})

            # The scheduler step is commented out. If used, it would adjust the learning rate based on the scheduler's policy.
            # scheduler.step()


            ####
            # Begin the evaluation phase
            print('==> Evaluation..')
            model_now.eval()  # Set the model to evaluation mode
            validation_losses  = losses.get_validation_losses_step_2(model, model_now, criterion_pa, criterion_pa_now, expler_s,
                                                nb_classes_prv, nb_classes_now, args, nce_loss, dlod_ev)
            with torch.no_grad():  # Disable gradient calculation for evaluation
                # Evaluate the model on the validation dataset
                feats, _ = utils_CGCD.evaluate_cos_(model_now, dlod_ev_now)
                cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa_now.proxies))
                _, preds_lb = torch.max(cos_sim, dim=1)  # Get predicted labels

                y = torch.tensor(dlod_ev_now.dataset.ys).type(torch.LongTensor)  # Ground truth labels

                # Identify seen and unseen classes
                seen_classes = torch.where(y < nb_classes, 1, 0)
                seen_classes_idx = torch.nonzero(seen_classes)
                unseen_classes = torch.where(y >= nb_classes, 1, 0)
                unseen_classes_idx = torch.nonzero(unseen_classes)

                # Calculate accuracy for seen classes
                if seen_classes.sum().item() > 0:
                    acc_o = calculate_accuracy(y[seen_classes_idx].to(device), preds_lb[seen_classes_idx].to(device))
                else:
                    acc_o = 0

                # Calculate accuracy for unseen classes
                if unseen_classes.sum().item() > 0:
                    acc_n = calculate_accuracy(y[unseen_classes_idx].to(device), preds_lb[unseen_classes_idx].to(device))
                else:
                    acc_n = 0

                # Calculate overall accuracy
                acc_a = calculate_accuracy(y.to(device), preds_lb.to(device))

                # Logging validation results
                print("Valid Accuracies Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".format(acc_o, acc_n, acc_a))
                if args.use_wandb:
                    wandb.log({"Step 2: Validation Seen Acc": acc_o, "custom_step": epoch})
                    wandb.log({"Step 2: Validation Unseen Acc": acc_n, "custom_step": epoch})
                    wandb.log({"Step 2: Validation Overall Acc": acc_a, "custom_step": epoch})

            # Update best metrics and save the model if the current model performs better
            if (acc_a > best_weighted_acc):
                best_seen = acc_o
                best_unseen = acc_n
                best_overall = acc_a
                best_epoch = epoch
                best_weighted_acc = acc_a
                es_count = 0
                print("Got Better Model with Seen {:.4f}, Unseen Accuracies {:.4f}, Overall {:.4f} and Average as {:.4f}".format(best_seen, best_unseen, best_overall, best_weighted_acc))
                torch.save({'model_pa_state_dict': model_now.state_dict(), 'proxies_param': criterion_pa_now.proxies},
                            '{}{}_{}_model_best_windowlen_{}_sz_embedding_{}_alpha_{}_mrg_{}_lr_{}_wd_{}_step_{}_fold_{}.pth'.format(
                                pth_rst_exp_step_2, args.dataset, args.model, window_len, args.sz_embedding,
                                args.alpha, args.mrg,  args.lr, args.weight_decay,
                                str(step), fold))
            else:
                es_count += 1
                print("Early Stopping Count ", es_count)

            # Handle early stopping based on count or epoch
            if epoch < args.warm:
                es_count = 0  # Reset early stopping counter during warm-up

            if es_count == 10 or epoch == args.nb_epochs - 1:
                # Save best results and stop if early stopping criteria are met
                # with open(pth_rst_exp_log_step_2, "a+") as fval:
                #     fval.write('Best Valid Accuracies Seen {:.4f}, Unseen {:.4f}, Overall {:.4f}, Average {:.4f}\n'.format(
                #         best_seen, best_unseen, best_overall, best_weighted_acc))
                if(args.log_results):
                    logger_2.info('Best Valid Accuracies Seen {:.4f}, Unseen {:.4f}, Overall {:.4f}, Average {:.4f}\n'.format(
                            best_seen, best_unseen, best_overall, best_weighted_acc))
                stage_2_val_accs_seen.append(best_seen)
                stage_2_val_accs_unseen.append(best_unseen)
                stage_2_val_accs_overall.append(best_overall)

                print("Best Valid Accuracies Seen {:.4f}, and Unseen {:.4f}, Overall {:.4f}, Average {:.4f}".format(
                    best_seen, best_unseen, best_overall, best_weighted_acc))
                print("Early Stopping")
                if args.use_wandb:
                    wandb.log({"Step 2: Validation Best Seen Acc": best_seen})
                    wandb.log({"Step 2: Validation Best Unseen Acc": best_unseen})
                    wandb.log({"Step 2: Validation Best Overall Acc": best_overall})
                break
                
        
        # Prepare for testing the final model
        pth_pth_test = '{}{}_{}_model_best_windowlen_{}_sz_embedding_{}_alpha_{}_mrg_{}_lr_{}_wd_{}_step_{}_fold_{}.pth'.format(
            pth_rst_exp_step_2, args.dataset, args.model, window_len, args.sz_embedding,
            args.alpha, args.mrg, args.lr, args.weight_decay, str(step), fold)

        # Define the configuration for testing

        # Initialize the criterion for testing
        criterion_pa_test = losses.Proxy_Anchor(nb_classes=nb_classes_now, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).to(device)

        # Load the model for testing based on the configuration
        if args.model == 'resnet18':
            cfg_test = {'weights_path': 'CGCD-main/src/Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights": False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name': '', 'epoch_len': 2, 'output_size': '', 'embedding_dim': args.sz_embedding, 'bottleneck_dim': None,
            'output_size': nb_classes_now, 'weight_norm_dim': '', 'n_channels': n_channels, 'window_len': window_len}

            model_test = ModelGen_new(cfg_test).create_model().to(device)
        elif args.model == 'tinyhar':
            model_test = tinyhar(n_channels, window_len, args.sz_embedding).to(device)
        elif args.model == 'harnet':
            repo = 'OxWearables/ssl-wearables'
            model_test = torch.hub.load(repo, 'harnet5', class_num=nb_classes, pretrained=True).to(device)
            del model.classifier
            model.embedding = nn.Sequential(nn.Linear(model.feature_extractor.layer5[0].out_channels, args.sz_embedding)).to(device)

        # Load the best model weights and proxies for testing
        checkpoint_test = torch.load(pth_pth_test, map_location=torch.device(device))
        model_test.load_state_dict(checkpoint_test['model_pa_state_dict'])
        criterion_pa_test.proxies = checkpoint_test['proxies_param']

        print("\n--------------------- Testing Step --------------------- \n")
        dset_test.nb_classes()
        nb_subjects = dset_test.nb_subjects()
        print("\n")

        # Perform testing
        model_test.eval()
        with torch.no_grad():
            feats, _ = utils_CGCD.evaluate_cos_(model_test, dlod_test)
            cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa_test.proxies))
            _, preds_lb = torch.max(cos_sim, dim=1)  # Get predicted labels

            y = torch.tensor(dlod_test.dataset.ys).type(torch.LongTensor)  # Ground truth labels

            
            # Identify seen and unseen classes
            seen_classes = torch.where(y < nb_classes, 1, 0)
            seen_classes_idx = torch.nonzero(seen_classes)
            unseen_classes = torch.where(y >= nb_classes, 1, 0)
            unseen_classes_idx = torch.nonzero(unseen_classes)

            # Calculate accuracy and F1 score for seen classes
            if seen_classes.sum().item() > 0:
                acc_o = calculate_accuracy(y[seen_classes_idx].to(device), preds_lb[seen_classes_idx].to(device))
                # f1_o = f1_score(y[seen_classes_idx].to('cpu').detach().numpy(), preds_lb[seen_classes_idx].to('cpu').detach().numpy(), average='macro')
                # f1_o = np.mean(f1_score(y[seen_classes_idx].to('cpu').detach().numpy(), preds_lb[seen_classes_idx].to('cpu').detach().numpy(), average=None)[:nb_classes_prv])

            else:
                acc_o, f1_o = 0, 0

            # Calculate accuracy and F1 score for unseen classes

            if unseen_classes.sum().item() > 0:
                acc_n = calculate_accuracy(y[unseen_classes_idx].to(device), preds_lb[unseen_classes_idx].to(device))
                # f1_n = f1_score(y[unseen_classes_idx].to('cpu').detach().numpy(), preds_lb[unseen_classes_idx].to('cpu').detach().numpy(), average='macro')
                # Taking the last n classes in the confusion matrix.
                # f1_n  = np.mean(f1_score(y[unseen_classes_idx].to('cpu').detach().numpy(),
                #                  preds_lb[unseen_classes_idx].to('cpu').detach().numpy(), average=None)[-(nb_classes_now -nb_classes_prv):])


            else:
                acc_n, f1_n = 0, 0

            # Calculate overall accuracy and F1 scoreprint(f1_score(y.to('cpu').detach().numpy(), preds_lb.to('cpu').detach().numpy(), average=None))
            f1_o = np.mean(f1_score(y.to('cpu').detach().numpy(), preds_lb.to('cpu').detach().numpy(), average=None)[:nb_classes_prv])
            f1_n = np.mean(f1_score(y.to('cpu').detach().numpy(), preds_lb.to('cpu').detach().numpy(), average=None)[nb_classes_prv:])

            acc_all = calculate_accuracy(y.to(device), preds_lb.to(device))

            f1_all = f1_score(y.to('cpu').detach().numpy(), preds_lb.to('cpu').detach().numpy(), average='macro')

            # Logging test results
            print("Test Accuracies Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".format(acc_o, acc_n, acc_all))
            print("Test F1 Scores Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".format(f1_o, f1_n, f1_all))

            # print(f1_score(y.to('cpu').detach().numpy(), preds_lb.to('cpu').detach().numpy(), average = None))
            # # from sklearn.metrics import confusion_matrix
            # cm = confusion_matrix(y.to('cpu').detach().numpy(), preds_lb.to('cpu').detach().numpy())
            # # print(cm)
            # cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print(np.array2string(cmn.astype('float'), formatter={'float_kind': lambda x: f"{x:.2f}"}))
            # # print(cmn)
            # # exit()
           

            test_accs_seen.append(acc_o)
            test_accs_unseen.append(acc_n)
            test_accs_overall.append(acc_all)

            test_f1s_seen.append(f1_o)
            test_f1s_unseen.append(f1_n)
            test_f1s_overall.append(f1_all)

            if args.use_wandb:
                wandb.log({"Test Seen Acc": acc_o})
                wandb.log({"Test Unseen Acc": acc_n})
                wandb.log({"Test Overall Acc": acc_all})
                wandb.log({"Test Seen F1": f1_o})
                wandb.log({"Test Unseen F1": f1_n})
                wandb.log({"Test Overall F1": f1_all})

            # Save test results to file
            # with open(pth_rst_exp_log_step_2, "a+") as fval:
            #     fval.write('Test Accuracies Seen {:.4f}, Unseen {:.4f}, Overall {:.4f}\n'.format(acc_o, acc_n, acc_all))
            #     fval.write('Test F1 Scores Seen {:.4f}, Unseen {:.4f}, Overall {:.4f}\n'.format(f1_o, f1_n, f1_all))
            if(args.log_results):
                logger_2.info('Test Accuracies Seen {:.4f}, Unseen {:.4f}, Overall {:.4f}\n'.format(acc_o, acc_n, acc_all))
                logger_2.info('Test F1 Scores Seen {:.4f}, Unseen {:.4f}, Overall {:.4f}\n'.format(f1_o, f1_n, f1_all))

        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model_now
        del model
        del opt
        del opt_pa
        del criterion_pa
        del criterion_pa_now
        

    # with open(pth_rst_exp_log_step_1, "a") as file1,  open(pth_rst_exp_log_step_2, "a") as file2:
   
    #     file1.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXX Results Summary XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")
    #     file1.write(f"Initial Stage ALL Acc Mean: {np.mean(stage_1_test_accs_seen)} Std: {np.std(stage_1_test_accs_seen)} F1 Mean: {np.mean(stage_1_test_f1s_seen)} Std: {np.std(stage_1_test_f1s_seen)}\n")
    #     file1.write(f"Before CL Seen Acc Mean: {np.mean(stage_2_test_accs_seen)} Std: {np.std(stage_2_test_accs_seen)} F1 Mean: {np.mean(stage_2_test_f1s_seen)} Std: {np.std(stage_2_test_f1s_seen)}\n")
        
    #     file1.write(f"After CL Seen Acc Mean: {np.mean(test_accs_seen)} Std: {np.std(test_accs_seen)} F1 Mean: {np.mean(test_f1s_seen)} Std: {np.std(test_f1s_seen)}\n")
    #     file1.write(f"After CL Unseen Acc Mean: {np.mean(test_accs_unseen)} Std: {np.std(test_accs_unseen)} F1 Mean: {np.mean(test_f1s_unseen)} Std: {np.std(test_f1s_unseen)}\n")
    #     file1.write(f"After CL Overall Acc Mean: {np.mean(test_accs_overall)} Std: {np.std(test_accs_overall)} F1 Mean: {np.mean(test_f1s_overall)} Std: {np.std(test_f1s_overall)}\n")
    #     file1.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
    #     # Writing the same data into file2
    #     file2.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXX Results Summary XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")
    #     file2.write(f"Initial Stage ALL Acc Mean: {np.mean(stage_1_test_accs_seen)} Std: {np.std(stage_1_test_accs_seen)} F1 Mean: {np.mean(stage_1_test_f1s_seen)} Std: {np.std(stage_1_test_f1s_seen)}\n")
    #     file2.write(f"Before CL Seen Acc Mean: {np.mean(stage_2_test_accs_seen)} Std: {np.std(stage_2_test_accs_seen)} F1 Mean: {np.mean(stage_2_test_f1s_seen)} Std: {np.std(stage_2_test_f1s_seen)}\n")
        
    #     file2.write(f"After CL Seen Acc Mean: {np.mean(test_accs_seen)} Std: {np.std(test_accs_seen)} F1 Mean: {np.mean(test_f1s_seen)} Std: {np.std(test_f1s_seen)}\n")
    #     file2.write(f"After CL Unseen Acc Mean: {np.mean(test_accs_unseen)} Std: {np.std(test_accs_unseen)} F1 Mean: {np.mean(test_f1s_unseen)} Std: {np.std(test_f1s_unseen)}\n")
    #     file2.write(f"After CL Overall Acc Mean: {np.mean(test_accs_overall)} Std: {np.std(test_accs_overall)} F1 Mean: {np.mean(test_f1s_overall)} Std: {np.std(test_f1s_overall)}\n")
    #     file2.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    log_content = (
    "XXXXXXXXXXXXXXXXXXXXXXXXXX Results Summary XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    f"Initial Stage ALL Acc Mean: {np.mean(stage_1_test_accs_seen)} Std: {np.std(stage_1_test_accs_seen)} "
    f"F1 Mean: {np.mean(stage_1_test_f1s_seen)} Std: {np.std(stage_1_test_f1s_seen)}\n"
    
    f"Before CL Seen Acc Mean: {np.mean(stage_2_test_accs_seen)} Std: {np.std(stage_2_test_accs_seen)} "
    f"F1 Mean: {np.mean(stage_2_test_f1s_seen)} Std: {np.std(stage_2_test_f1s_seen)}\n"
    
    f"After CL Seen Acc Mean: {np.mean(test_accs_seen)} Std: {np.std(test_accs_seen)} "
    f"F1 Mean: {np.mean(test_f1s_seen)} Std: {np.std(test_f1s_seen)}\n"
    
    f"After CL Unseen Acc Mean: {np.mean(test_accs_unseen)} Std: {np.std(test_accs_unseen)} "
    f"F1 Mean: {np.mean(test_f1s_unseen)} Std: {np.std(test_f1s_unseen)}\n"
    
    f"After CL Overall Acc Mean: {np.mean(test_accs_overall)} Std: {np.std(test_accs_overall)} "
    f"F1 Mean: {np.mean(test_f1s_overall)} Std: {np.std(test_f1s_overall)}\n"
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )
    if(args.log_results):
        logger_1.info(log_content)
        logger_2.info(log_content)

    os.system('exit')