import os
os.system('pip install einops')
os.system('pip install pytorch_metric_learning')
import argparse, os, copy, random, sys
import numpy as np
import argparse, os, copy, random, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from functools import partial
from tqdm import *
import dataset, utils_CGCD, losses, net
from net.resnet import *
from models.modelgen import ModelGen, ModelGen_new
# from torchsummary import summary
from sklearn.metrics import accuracy_score , f1_score
import json
import pandas as pd
import itertools
import multiprocessing
from models.tinyhar import tinyhar
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


torch.manual_seed(1)
np.set_printoptions(threshold=sys.maxsize)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("BASELINE MODEL")

print("Using Device", device)
parser = argparse.ArgumentParser(description=
                                'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`')
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
parser.add_argument('--dataset', default='mhealth', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
parser.add_argument('--bottleneck_dim', default=1024, type=int, dest='bottleneck_dim', help='Size of embedding that is appended to backbone model.')
parser.add_argument('--embedding-size', default=1024, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
parser.add_argument('--batch-size', default=256, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
parser.add_argument('--epochs', default=100, type=int, dest='nb_epochs', help='Number of training epochs.')
parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')
parser.add_argument('--workers', default=8, type=int, dest='nb_workers', help='Number of workers for dataloader.')
parser.add_argument('--model', default='resnet18', help='Model for training')  # resnet50 #resnet18  VIT
parser.add_argument('--loss', default='Proxy_Anchor', help='Criterion for training') #Proxy_Anchor #Contrastive
parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate setting')  #1e-4
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay setting')
parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs')  # 1
parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
parser.add_argument('--l2-norm', default=True, type=bool, help='L2 normlization')
parser.add_argument('--remark', default='', help='Any reamrk')
parser.add_argument('--exp', type=str, default='0')
parser.add_argument('--standarization_prerun', default=False, type=bool, help='Data Standarization Preruntime') 
parser.add_argument('--standarization_run_time', default=True, type=bool, help='Data Standarization RunTime')
parser.add_argument('--processes', default = 1 , type=int)
parser.add_argument('--threads', default = 32 , type=int)
parser.add_argument('--use_wandb', default=False, type=bool, help='Use Wanb to upload parameters')
parser.add_argument('--only_test_step1', default=False, type=bool, help='Test only Initial Step (No training if set to True)')
parser.add_argument('--only_test_step2', default=False, type=bool, help='Test only Incremental Step (No training if set to True)')

args = parser.parse_args()
# args.nb_workers = multiprocessing.cpu_count()



folds = [0,1,2,3,4] #[0,1,2,3,4]
results_df = pd.DataFrame(columns = ['Dataset', 'Iteration', 'Initial Acc', 'Initial F1', 'Incremental Acc Seen1',  'Incremental F1 Seen1', 'Incremental Acc Seen2',  'Incremental F1 Seen2', 'Incremental Acc Unseen', 'Incremental F1 Unseen', 'Incremental Acc All', 'Incremental F1 All', "Forget Acc", 'Forget F1' ])

pth_rst_exp_step_1 = f'{os.getcwd()}/Saved_Models/Initial/Online_Finetuning/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
pth_rst_exp_log_step_1 = pth_rst_exp_step_1+f'results_{args.dataset}_{args.model}.txt'
os.makedirs(pth_rst_exp_step_1, exist_ok=True)

with open(pth_rst_exp_log_step_1, "w") as file:
            file.write(f"\n******** Dataset: {args.dataset}, Standardize: True, Model: {args.model}\n")

pth_rst_exp_step_2 = f'{os.getcwd()}/Saved_Models/Incremental/Online_Finetuning/' + args.dataset + '/'
pth_rst_exp_log_step_2 = pth_rst_exp_step_2  + f"results_{args.dataset}_{args.model}.txt"
os.makedirs(pth_rst_exp_step_2, exist_ok=True)
with open(pth_rst_exp_log_step_2, "w") as file:
            file.write(f"\n******** Dataset: {args.dataset}, Standardize: True, Model: {args.model}\n")

stage_1_test_accs_seen, stage_1_test_f1s_seen, stage_2_test_accs_seen, stage_2_test_f1s_seen= [], [], [], []
stage_2_val_accs_seen, stage_2_val_accs_unseen, stage_2_val_accs_overall   = [], [], []
test_accs_seen, test_accs_unseen, test_accs_overall   = [], [], []
test_f1s_seen, test_f1s_unseen, test_f1s_overall   = [], [], []

if __name__ == '__main__':
    for fold in folds:

        print(f"\n******************Fold: {fold} ******************************\n")
        with open(pth_rst_exp_log_step_1, "a") as file:
            file.write(f"\n******************Fold: {fold} ******************************\n")

        with open(pth_rst_exp_log_step_2, "a") as file:
            file.write(f"\n******************Fold: {fold} ******************************\n")
        
        print("Dataset :", args.dataset)
        if args.dataset =='wisdm':
            pth_dataset = f'{os.getcwd()}/HAR_data/Wisdm/'
            window_len = 40
            n_channels = 3
        elif args.dataset =='realworld':
            pth_dataset = f'{os.getcwd()}/HAR_data/realworld/'
            nb_classes_now = 8
            window_len = 100
            n_channels = 3
    
        elif args.dataset =='oppo':
            pth_dataset = f'{os.getcwd()}/HAR_data/oppo/'
        elif args.dataset =='pamap':
            pth_dataset = f'{os.getcwd()}/HAR_data/pamap/'
            nb_classes_now = 12
            window_len = 200
            n_channels = 9

        elif args.dataset =='mhealth':
            pth_dataset = f'{os.getcwd()}/HAR_data/mhealth/'
            nb_classes_now = 6
            window_len = 100
            n_channels = 9
    

        if(args.use_wandb):
            import wandb
            wandb.init(
                # set the wandb project where this run will be logged
                project="CGCD-HAR-Supervised",
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
                "warm" : args.warm,
                "bn-freeze" : args.bn_freeze,
                }
                )
            wandb.log({"Method": "Baseline"})

        # Initial Step Dataloader ..............
        dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold )
        
        mean = np.mean(dset_tr_0.xs, axis=0)
        std = np.std(dset_tr_0.xs, axis=0)
        dset_tr_0.xs = (dset_tr_0.xs - mean) / (std + 1e-5)
         
        dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers, pin_memory =True, worker_init_fn=lambda x: np.random.seed(42))
        dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', windowlen=window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold)
        dset_ev.xs = (dset_ev.xs - mean) / (std + 1e-5)
        dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory =True, worker_init_fn=lambda x: np.random.seed(42))

        dset_test_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='test_0', windowlen=window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold)
        dset_test_0.xs = (dset_test_0.xs - mean) / (std + 1e-5)
        dlod_test_0 = torch.utils.data.DataLoader(dset_test_0, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))


        nb_classes = dset_tr_0.nb_classes()
        print("Number of Classes :",nb_classes)

        # Configuration for the Model  -> Here We have Botleneck later
        if(args.model == 'resnet18'):
            cfg = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
                'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': args.sz_embedding, 'bottleneck_dim': None,
                    'output_size':nb_classes,'weight_norm_dim': 0 ,  'n_channels' :n_channels, 'window_len': window_len}
            model = ModelGen_new(cfg).create_model().to(device)
        
        if(args.model == 'tinyhar'):
            args.bn_freeze= False
            args.warm = 0
            model = tinyhar(n_channels,window_len, args.sz_embedding).to(device)
            # model.fc = nn.Linear(args.bottleneck_dim,nb_classes).to(device)
        
        criterion_pa = losses.Proxy_Anchor(nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).to(device)

        #  Starting to just train on Seen Dataset. 
        param_groups = [{'params':model.parameters}]
        param_groups.append({'params': criterion_pa.parameters(), 'lr': float(args.lr)*100 })

        best_eval = 9999999
        # Training Parameters
        opt_pa = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
        loss_baseline = nn.CrossEntropyLoss()
        loss_baseline = criterion_pa
        
        # scheduler_pa = torch.optim.lr_scheduler.StepLR(opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
        
        es_count = 1          # Early Stopping Count
        best_acc = -1
        step = 1 
        version = 1
        for epoch in range(0,args.nb_epochs):
            if(args.only_test_step1): continue
            model.train()
            losses_per_epoch = []
            pbar = tqdm(enumerate(dlod_tr_0))

            for batch_idx, (x, y, z) in pbar:
                # if(batch_idx>2):break
                feats = model(x.squeeze().to(device))
                y = y.type(torch.LongTensor)
        
                loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
                opt_pa.zero_grad()
                loss_pa.backward()

                # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                
                losses_per_epoch.append(loss_pa.item())
                opt_pa.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
                    epoch, batch_idx + 1, len(dlod_tr_0), 100. * batch_idx / len(dlod_tr_0), loss_pa.item(), 0, 0))
                 
            # scheduler_pa.step()
            pbar_eval = tqdm(enumerate(dlod_ev))
            eval_loss = 0

            print("Evaluating ------------------->")
            total_batch_size = 0
            acc_total = 0
            model.eval()
            with torch.no_grad():
                feats, _ = utils_CGCD.evaluate_cos_(model, dlod_ev)
                cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
                _, preds_lb = torch.max(cos_sim, dim=1)
                preds = preds_lb.detach().cpu().numpy()
                acc_0, _ = utils_CGCD._hungarian_match_(np.array(dlod_ev.dataset.ys), preds)
                print('Valid Epoch: {} Acc: {:.4f}'.format(str(-1), acc_0)) 
            
            # for batch_idx, (x, y, z) in pbar_eval:
                
            #     feats = model(x.squeeze().to(device))
            #     y = y.type(torch.LongTensor).to(device)
            #     pred_out = torch.argmax(feats, dim=1)
            #     acc = accuracy_score(y.to('cpu').detach().numpy(),pred_out.to('cpu').detach().numpy())
            #     acc_total +=acc*x.shape[0]  # tOTAL CORRECT PREDICTIONS
            #     total_batch_size += x.shape[0]
                
            #     loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
            #     eval_loss = eval_loss+ loss_pa.item()

            #     pbar_eval.set_description('Eval Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
            #         epoch, batch_idx + 1, len(dlod_ev), 100. * batch_idx / len(dlod_ev), loss_pa.item(), 0, 0))

            # print(acc_total/total_batch_size, best_acc)
            
            if(acc_total/total_batch_size > acc_0):  #  eval_loss.item()/(batch_idx+1) < best_eval
                print("Saving New Model")
                print("Intitial Acc {}, Current Acc: {}".format(best_acc,acc_total/total_batch_size))
                best_eval = eval_loss/(batch_idx+1)
                # Just saving the model without botteleneck layer
                torch.save(model.state_dict(), '{}{}_{}_best_windowlen_{}_bottleneck_size_{}_version_{}_step_{}_fold_{}.pth'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.bottleneck_dim, version, step, fold)) 
                print("Best Accuracy changed from {} to {}".format(best_acc,acc_total/total_batch_size) )
                best_acc = acc_0
                es_count = 1
            else:
                es_count +=1
                print("Early Stopping ", es_count)
            
            if(es_count==10):
                print("Performing Early Stopping")
                break

            # if (epoch >= 0):
            #     with torch.no_grad():
            #         print('Evaluating..')
            #         Recalls = utils_CGCD.evaluate_cos(model, dlod_ev, epoch)
            #     #### Best model save
            #     if best_recall[0] < Recalls[0]:
            #         best_recall = Recalls
            #         best_epoch = epoch
            #         # torch.save({'model_pa_state_dict': model.state_dict(), 'proxies_param': criterion_pa.proxies}, '{}/{}_{}_best_step_0.pth'.format(pth_rst_exp, args.dataset, args.model))
            #         # with open('{}/{}_{}_best_results.txt'.format(pth_rst_exp, args.dataset, model_name), 'w') as f:
            #         #     f.write('Best Epoch: {}\tBest Recall@{}: {:.4f}\n'.format(best_epoch, 1, best_recall[0] * 100))

        

    ######################################################################################################
    # Incremental Step
        print("\n Incremental Step \n")

        dset_tr_1 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_1', windowlen= window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold)
        mean = np.mean(dset_tr_1.xs, axis=0)
        std = np.std(dset_tr_1.xs, axis=0)
        dset_tr_1.xs = (dset_tr_1.xs - mean) / (std + 1e-5)

        dlod_tr_1 = torch.utils.data.DataLoader(dset_tr_1, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))
        
        dset_ev_1 = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_1', windowlen= window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold)
        dset_ev_1.xs = (dset_ev_1.xs - mean) / (std + 1e-5)
        dlod_ev_1 = torch.utils.data.DataLoader(dset_ev_1, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))

        dset_test = dataset.load(name=args.dataset, root=pth_dataset, mode='test_1', windowlen= window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold)
        dset_test.xs = (dset_test.xs - mean) / (std + 1e-5)
        dlod_test = torch.utils.data.DataLoader(dset_test, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))

        
        nb_classes_1 = dset_tr_1.nb_classes()
        print("Classes in the initial Step: {}, Incremental Step: {}".format(nb_classes,nb_classes_1))

        # cfg_new = {'weights_path': pth_rst_exp +'/Baseline_Initial_resnet_'+args.dataset+'.mdl' , "use_ssl_weights" : True, 'conv_freeze': False, 'load_finetuned_mtl': False,
        #      'checkpoint_name' :'', 'epoch_len': 10, 'output_size': '', 'embedding_dim': 0, 'bottleneck_dim': bottleneck_dim,
        #         'output_size':nb_classes,'weight_norm_dim': 0}
        

        cfg = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': None, 'bottleneck_dim': args.bottleneck_dim,
                'output_size':nb_classes,'weight_norm_dim': 0, 'n_channels' :n_channels, 'window_len': window_len}

        exit
        # Just Testing the Initital Model
        print("Loading Saved Model")
        pth_pth = '{}{}_{}_best_windowlen_{}_bottleneck_size_{}_version_{}_step_{}_fold_{}.pth'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.bottleneck_dim, version, step, fold)

        checkpoint = torch.load(pth_pth, map_location=torch.device(device))
        model.load_state_dict(checkpoint)

        print('==>Evaluation of Stage 1 Dataset on Stage 1 Model')
        test_loss,total_batche_size, acc_total, epoch = 0, 0, 0, 0
        model.eval()

        predicted_ys = []
        actual_ys = []
        for x, y, z in dlod_test_0:
            feats = model(x.squeeze().to(device))
            y = y.type(torch.LongTensor).to('cpu').detach().numpy()
            pred_out = torch.argmax(feats, dim=1).to('cpu').detach().numpy()

            if(len(predicted_ys) == 0):
                predicted_ys = pred_out 
            else:
                predicted_ys = np.concatenate((predicted_ys, pred_out))
            
            if(len(actual_ys) == 0):
                actual_ys = y 
            else:
                actual_ys = np.concatenate((actual_ys, y))
            

        stage_1_acc_0 = accuracy_score(predicted_ys,actual_ys)
        stage_1_f1_0 = f1_score(predicted_ys, actual_ys, average= 'macro')

        if(args.use_wandb):
            wandb.log({"M1-TA1_Old": stage_1_acc_0})   # Logging Initial Step Accuracy.
            wandb.log({"M1-TF1_Old": stage_1_f1_0})  # Logging Initial Step F1 Score.

        print("Stage 1 Test Dataset on Stage 1 trained model ",stage_1_acc_0)
        print("Step 1 Test Dataset F1 Score on Stage 1 trained Model",stage_1_f1_0)

        with open(pth_rst_exp_log_step_1, "a") as file:
            file.write(f"Step 1: Test Dataset Acc: {stage_1_acc_0} F1: {stage_1_f1_0}\n")

        stage_1_test_accs_seen.append(stage_1_acc_0)
        stage_1_test_f1s_seen.append(stage_1_f1_0)

        
        # We only have seen and unseen classes, but model is only trained on seen classes to unseen accuracy should be zero
        print('==>Evaluation of Stage 2 Dataset on Stage 1 Model')

        model.eval()
        predicted_ys = []
        actual_ys = []
        for x, y, z in dlod_test:
            feats = model(x.squeeze().to(device))
            y = y.type(torch.LongTensor).to('cpu').detach().numpy()
            pred_out = torch.argmax(feats, dim=1).to('cpu').detach().numpy()

            if(len(predicted_ys) == 0):
                predicted_ys = pred_out 
            else:
                predicted_ys = np.concatenate((predicted_ys, pred_out))
            
            if(len(actual_ys) == 0):
                actual_ys = y 
            else:
                actual_ys = np.concatenate((actual_ys, y))
    
        stage_2_acc_0 = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
        stage_2_acc_1 = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])

        stage_2_f1_0 = f1_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes], average='macro')
        stage_2_f1_1 = f1_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes], average= 'macro')

        print("Stage 2 Test Dataset on Stage 1 trained model Accuracy, Seen {}  Unseen {}".
            format(stage_2_acc_0, stage_2_acc_1))
        print("Stage 2 Test Dataset on Stage 1 trained model F1 Score , Seen  {} Unseen {}".
            format(stage_2_f1_0, stage_2_f1_1))
        
        with open(pth_rst_exp_log_step_1, "a") as file:
            file.write(f"Stage 2: Test Dataset Acc Seen Acc:{stage_2_acc_0}, Seen F1: {stage_2_f1_0} Unseen Acc: {stage_2_acc_1} \n")
        if(args.use_wandb):
            wandb.log({"M1-TA2_Old": stage_2_acc_0})   # Logging Initial Step Accuracy.
            wandb.log({"M1-TF2_Old": stage_2_f1_0})  # Logging Initial Step F1 Score.

        stage_2_test_accs_seen.append(stage_2_acc_0)
        stage_2_test_f1s_seen.append(stage_2_f1_0)
        

        opt_pa_new = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)
        # scheduler_pa_new = torch.optim.lr_scheduler.StepLR(opt_pa_new, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
        loss_baseline = nn.CrossEntropyLoss()
        
        
        last_layers = model.fc.weight[0:nb_classes]  # Taking initial model weights
        model.fc = nn.Linear(args.bottleneck_dim, nb_classes_1).to(device)    # Changing Classifier outputs

        # with torch.no_grad(): # Copying the learned seen classes layers to the extended model.
        #     model.fc.weight[:nb_classes].copy_(last_layers)


        best_weighted_acc =-1 
        es_count =1
        step = 2

        load_pretrained = False
        
        if(load_pretrained):
            print("Loading Pre-Trained Model ...")
            step = 1
            pth_pth_pretrained = '{}{}_{}_model_last_windowlen_{}_sz_bottleneck_{}_step_{}_fold_{}.pth'.format(pth_rst_exp_step_2, args.dataset, args.model,window_len,args.bottleneck_dim , str(step), fold)
            checkpoint_pretrained = torch.load(pth_pth_pretrained,map_location=torch.device(device))
            model.load_state_dict(checkpoint_pretrained['model_pa_state_dict'])
        
        
        for epoch in range(0,args.nb_epochs):
            if(args.only_test_step2): continue 
            losses_per_epoch = []
            pbar = tqdm(enumerate(dlod_tr_1))
            total_correct_seen = 0
            total_seen = 0
            total_correct_unseen = 0
            total_unseen = 0
            model.train()
            for batch_idx, (x, y, z) in pbar:
                ####
                # if(batch_idx>2):break
                feats = model(x.squeeze().to(device))
                y = y.type(torch.LongTensor)

                loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
                opt_pa_new.zero_grad()
                loss_pa.backward()

                losses_per_epoch.append(loss_pa.item())
                opt_pa_new.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: Overall {:.4f}'.format(
                    epoch, batch_idx + 1, len(dlod_tr_1), 100. * batch_idx / len(dlod_tr_1), loss_pa.item(), 0,0))
            
            
            # scheduler_pa_new.step()

            eval_loss, total_correct_seen, total_seen, total_correct_unseen, total_unseen, total_data, overall_correct  = 0, 0, 0, 0, 0, 0, 0

            print("Evaluating ------------------->")
            model.eval()

            predicted_ys = []
            actual_ys = []
            for x, y, z in dlod_ev_1:
                            
                feats = model(x.squeeze().to(device))
                y = y.type(torch.LongTensor).to('cpu').detach().numpy()
                pred_out = torch.argmax(feats, dim=1).to('cpu').detach().numpy()

                if(len(predicted_ys) == 0):
                    predicted_ys = pred_out 
                else:
                    predicted_ys = np.concatenate((predicted_ys, pred_out))
                
                if(len(actual_ys) == 0):
                    actual_ys = y 
                else:
                    actual_ys = np.concatenate((actual_ys, y))
        
            acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
            acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])
            acc_a = accuracy_score(predicted_ys, actual_ys)
                    
                

            print("Valid Accuracies Seen {}, Unseen {}, Overall {}".format(acc_o, acc_n, acc_a))
            
            if((acc_o + acc_n + acc_a)/3 > best_weighted_acc):
                best_seen = acc_o
                best_unseen = acc_n
                best_overall = acc_a
                best_weighted_acc = (best_seen + best_unseen + best_overall)/3
                best_epoch = epoch
                es_count = 0
                print("Got Better Model with  Seen {},  Unseen Accuracies {},  Overalll {} and  Average as {} ".format( best_seen,best_unseen, best_overall,best_weighted_acc))
                torch.save({'model_pa_state_dict': model.state_dict()}, '{}{}_{}_model_last_windowlen_{}_sz_bottleneck_{}_step_{}_fold_{}.pth'.format(pth_rst_exp_step_2, args.dataset, args.model,window_len, args.bottleneck_dim, str(step), fold))
            else:
                es_count +=1
                print("Early Stopping Count ", es_count)
                
            if(epoch< args.warm):
                es_count =0    # If we are having warmup then no early stopping
            
            if(es_count ==10 or epoch == args.nb_epochs-1):
                with open(pth_rst_exp_log_step_2, "a+") as fval:
                    fval.write('Best Valid Accuracies Seen {}, Unseen {}, Overall {}, Average {}\n'.format( best_seen, best_unseen, best_overall, best_weighted_acc))
                
                stage_2_val_accs_seen.append(best_seen)
                stage_2_val_accs_unseen.append(best_unseen)
                stage_2_val_accs_overall.append(best_overall)
                
                print("Best  Valid Accuracies Seen {}, and Unseen {},  Overall {}, Average {} ".format( best_seen, best_unseen, best_overall, best_weighted_acc))
                print("Early Stopping")


                break

        
        
        ###########################################  Testing On Test Dataset ...  #####################################################
        
        print("Testing On Test Dataset ...")
        step =2
        
        pth_pth_test = '{}{}_{}_model_last_windowlen_{}_sz_bottleneck_{}_step_{}_fold_{}.pth'.format(pth_rst_exp_step_2, args.dataset, args.model,window_len, args.bottleneck_dim, str(step), fold)

        if(args.model == 'resnet18'):
            cfg_test = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': None, 'bottleneck_dim': args.bottleneck_dim,
                'output_size':nb_classes_1,'weight_norm_dim': 0, 'n_channels' :n_channels, 'window_len': window_len}
            
            model_test = ModelGen_new(cfg_test).create_model().to(device)
        if(args.model == 'tinyhar'):
            model_test = tinyhar(n_channels,window_len, args.bottleneck_dim).to(device)

        if(args.model == 'harnet'):
            repo = 'OxWearables/ssl-wearables'
            model_test = torch.hub.load(repo, 'harnet5', class_num=nb_classes, pretrained=True).to(device)
            del model.classifier
            model_test.embedding = nn.Sequential(nn.Linear(model.feature_extractor.layer5[0].out_channels,args.sz_embedding)).to(device)
        
        

        
        model_test.fc = nn.Linear(args.bottleneck_dim, nb_classes_1).to(device)
        model_test = model_test.to(device)

        checkpoint_test = torch.load(pth_pth_test,map_location=torch.device(device))
        model_test.load_state_dict(checkpoint_test['model_pa_state_dict'])
        
        model_test.eval()
        predicted_ys = []
        actual_ys = []
        for x, y, z in dlod_test:
            feats = model_test(x.squeeze().to(device))
            y = y.type(torch.LongTensor).to('cpu').detach().numpy()
            pred_out = torch.argmax(feats, dim=1).to('cpu').detach().numpy()

            if(len(predicted_ys) == 0):
                predicted_ys = pred_out 
            else:
                predicted_ys = np.concatenate((predicted_ys, pred_out))
            
            if(len(actual_ys) == 0):
                actual_ys = y 
            else:
                actual_ys = np.concatenate((actual_ys, y))

        acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
        acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])
        acc_all = accuracy_score(predicted_ys, actual_ys)

        f1_o = f1_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes], average='macro')
        f1_n = f1_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes], average= 'macro')
        f1_all = f1_score(predicted_ys, actual_ys, average= 'macro')

        print("Testing Accuracies : Seen {} Unseen {} Overall {}".format(acc_o, acc_n, acc_all))
        print("Testing F1 Scores  : Seen {} Unseen {} Overall {}".format(f1_o, f1_n, f1_all))
        with open(pth_rst_exp_log_step_2, "a+") as fval:
                    fval.write(f'\nBest Test Accuracies Seen: {acc_o}, Unseen: {acc_n}, Overall: {acc_all}\n')
                    fval.write(f'Best Test F1 Scores Seen: {f1_o}, Unseen: {f1_n}, Overall: {f1_all}\n') 
        test_accs_seen.append(acc_o)
        test_accs_unseen.append(acc_n)
        test_accs_overall.append(acc_all)

        test_f1s_seen.append(f1_o)
        test_f1s_unseen.append(f1_n)
        test_f1s_overall.append(f1_all)

        if(args.use_wandb):
            wandb.log({"M2-TA2_Old": acc_o})   
            wandb.log({"M2-TF2_Old": f1_o})  
            wandb.log({"M2-TA2_New": acc_n})   
            wandb.log({"M2-TF2_New": f1_n})  
            wandb.log({"M2-TA2_All": acc_all})   
            wandb.log({"M2-TF2_All": f1_all})  
            wandb.finish()
    
        new_row = {'Dataset':args.dataset, 'Iteration':iter, 
        'Initial Acc': stage_1_acc_0, 'Initial F1': stage_1_f1_0, 'Incremental Acc Seen1':stage_2_acc_0,
            'Incremental F1 Seen1':stage_2_f1_0, 'Incremental Acc Seen2' :acc_o,
            'Incremental F1 Seen2':f1_o, 'Incremental Acc Unseen' :acc_n,
                'Incremental F1 Unseen': f1_n, 'Incremental Acc All':acc_all, 
                'Incremental F1 All' :f1_all, "Forget Acc": stage_2_acc_0 - acc_o, "Forget F1": stage_2_f1_0 - f1_n}
        
        
        results_df = results_df._append(new_row, ignore_index = True)

    with open(pth_rst_exp_log_step_1, "a+") as fval:
        fval.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXX Results Summary XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")
        fval.write(f"Initial Stage ALL Acc Mean: {np.mean(stage_1_test_accs_seen)} Std: {np.std(stage_1_test_accs_seen)} F1 Mean: {np.mean(stage_1_test_f1s_seen)} Std: {np.std(stage_1_test_f1s_seen)}\n")
        fval.write(f"Before CL Seen Acc Mean: {np.mean(stage_2_test_accs_seen)} Std: {np.std(stage_2_test_accs_seen)} F1 Mean: {np.mean(stage_2_test_f1s_seen)} Std: {np.std(stage_2_test_f1s_seen)}\n")
        
        fval.write(f"After CL Seen Acc Mean: {np.mean(test_accs_seen)} Std: {np.std(test_accs_seen)} F1 Mean: {np.mean(test_f1s_seen)} Std: {np.std(test_f1s_seen)}\n")
        fval.write(f"After CL Unseen Acc Mean: {np.mean(test_accs_unseen)} Std: {np.std(test_accs_unseen)} F1 Mean: {np.mean(test_f1s_unseen)} Std: {np.std(test_f1s_unseen)}\n")
        fval.write(f"After CL Overall Acc Mean: {np.mean(test_accs_overall)} Std: {np.std(test_accs_overall)} F1 Mean: {np.mean(test_f1s_overall)} Std: {np.std(test_f1s_overall)}\n")
        fval.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    with open(pth_rst_exp_log_step_2, "a+") as fval:
        fval.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXX Results Summary XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")
        fval.write(f"Initial Stage ALL Acc Mean: {np.mean(stage_1_test_accs_seen)} Std: {np.std(stage_1_test_accs_seen)} F1 Mean: {np.mean(stage_1_test_f1s_seen)} Std: {np.std(stage_1_test_f1s_seen)}\n")
        fval.write(f"Before CL Seen Acc Mean: {np.mean(stage_2_test_accs_seen)} Std: {np.std(stage_2_test_accs_seen)} F1 Mean: {np.mean(stage_2_test_f1s_seen)} Std: {np.std(stage_2_test_f1s_seen)}\n")
        
        fval.write(f"After CL Seen Acc Mean: {np.mean(test_accs_seen)} Std: {np.std(test_accs_seen)} F1 Mean: {np.mean(test_f1s_seen)} Std: {np.std(test_f1s_seen)}\n")
        fval.write(f"After CL Unseen Acc Mean: {np.mean(test_accs_unseen)} Std: {np.std(test_accs_unseen)} F1 Mean: {np.mean(test_f1s_unseen)} Std: {np.std(test_f1s_unseen)}\n")
        fval.write(f"After CL Overall Acc Mean: {np.mean(test_accs_overall)} Std: {np.std(test_accs_overall)} F1 Mean: {np.mean(test_f1s_overall)} Std: {np.std(test_f1s_overall)}\n")
        fval.write("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")   
            

results_df.to_csv('results_online_finetuing_{}.csv'.format('_'.join(args.dataset)), index=False)