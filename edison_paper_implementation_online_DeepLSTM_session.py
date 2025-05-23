import os
os.system('pip install einops')
os.system('pip install pytorch_metric_learning')
os.system('pip install tensorflow')
os.system('pip install plotly')
import argparse, os, copy, random, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
np.seterr(all="ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn import utils
from functools import partial
import json
from tqdm import *

import dataset, utils_CGCD, losses, net
from net.resnet import *

from models.modelgen import ModelGen, ModelGen_new
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import tensorflow as tf
from covmet import ConvNet
from edison_functions import compute_euclidean, contrastive_loss, extract_sample, get_accuracy_old_new, get_accuracy_old_new_LSTM
from replay_memory import *
import torch
from edison_models import *
import itertools
import pandas as pd
import warnings
import multiprocessing
warnings.filterwarnings('ignore')
torch.manual_seed(1)
np.set_printoptions(threshold=sys.maxsize)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description=
                                        'Implementaion of Edison Paper using Orignal Model')
# export directory, training and val datasets, test datasets

parser.add_argument('--dataset', default='realworld', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
parser.add_argument('--embedding-size', default=1024, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
parser.add_argument('--batch-size', default=256, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
parser.add_argument('--epochs', default=100, type=int, dest='nb_epochs', help='Number of training epochs.')
parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')
parser.add_argument('--workers', default=8, type=int, dest='nb_workers', help='Number of workers for dataloader.')
parser.add_argument('--model', default='DeepLSTM', help='Model for training')  # resnet50 #resnet18  VIT
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate setting')  #1e-4
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay setting')
parser.add_argument('--processes', default = 1 , type=int)
parser.add_argument('--threads', default = 32 , type=int)
parser.add_argument('--loss', default='Prototypes', help='Criterion for training')
parser.add_argument('--standarization_prerun', default=False, type=bool, help='Data Standarization Preruntime') 
parser.add_argument('--standarization_run_time', default=True, type=bool, help='Data Standarization RunTime')
parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')
parser.add_argument('--use_wandb', default=False, type=bool, help='Use Wandb to upload parameters')
parser.add_argument('--only_test_step1', default=False, type=bool, help='Test only Initial Step (No training if set to True)')
parser.add_argument('--only_test_step2', default=False, type=bool, help='Test only Incremental Step (No training if set to True)')
parser.add_argument('--session_split', default=True, type=bool, help='Session Split')
args = parser.parse_args()
# args.nb_workers = multiprocessing.cpu_count()
# args.nb_workers = 4
# device = 'cpu'

folds = [0,1,2,3,4]# [0,1,2,3,4]


results_df = pd.DataFrame(columns = ['Dataset', 'Iteration', 'Initial Acc', 'Initial F1', 'Incremental Acc Seen1',  'Incremental F1 Seen1', 'Incremental Acc Seen2',  'Incremental F1 Seen2', 'Incremental Acc Unseen', 'Incremental F1 Unseen', 'Incremental Acc All', 'Incremental F1 All', "Forget Acc", 'Forget F1' ])


pth_rst_exp_step_1 = f'{os.getcwd()}/Saved_Models/Initial/Edison_DeepLSTM/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
pth_rst_exp_log_step_1 = pth_rst_exp_step_1+f'results_{args.dataset}_{args.model}.txt'
os.makedirs(pth_rst_exp_step_1, exist_ok=True)

with open(pth_rst_exp_log_step_1, "w") as file:
            file.write(f"\n******** Dataset: {args.dataset}, Standardize: True, Model: {args.model}\n")

pth_rst_exp_step_2 = f'{os.getcwd()}/Saved_Models/Incremental/Edison_DeepLSTM/' + args.dataset + '/'
pth_rst_exp_log_step_2 = pth_rst_exp_step_2  + f"results_{args.dataset}_{args.model}.txt"
os.makedirs(pth_rst_exp_step_2, exist_ok=True)
with open(pth_rst_exp_log_step_2, "w") as file:
            file.write(f"\n******** Dataset: {args.dataset}, Standardize: True, Model: {args.model}\n")

stage_1_test_accs_seen, stage_1_test_f1s_seen, stage_2_test_accs_seen, stage_2_test_f1s_seen= [], [], [], []
stage_2_val_accs_seen, stage_2_val_accs_unseen, stage_2_val_accs_overall   = [], [], []
test_accs_seen, test_accs_unseen, test_accs_overall   = [], [], []
test_f1s_seen, test_f1s_unseen, test_f1s_overall   = [], [], []

args = parser.parse_args()
if __name__ == '__main__':
    for fold in folds:
        
        print(f"\n******************Fold: {fold} ******************************\n")
        with open(pth_rst_exp_log_step_1, "a") as file:
            file.write(f"\n******************Fold: {fold} ******************************\n")

        with open(pth_rst_exp_log_step_2, "a") as file:
            file.write(f"\n******************Fold: {fold} ******************************\n")

        print("Dataset :", args.dataset)
        ####

        
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
                }
                ) 

            wandb.log({"Method": "Edison_LSTM"})
        gpu_id = 0
        # ****************************************************STEP 1 : Prototype Training******************************************************** #

        dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold, session_split = args.session_split  )
        
        mean = np.mean(dset_tr_0.xs, axis=0)
        std = np.std(dset_tr_0.xs, axis=0)
        if(args.standarization_run_time): dset_tr_0.xs = (dset_tr_0.xs - mean) / (std + 1e-5)
         
        dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers, pin_memory =True, worker_init_fn=lambda x: np.random.seed(42))
        dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', windowlen=window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold, session_split = args.session_split)
        if(args.standarization_run_time): dset_ev.xs = (dset_ev.xs - mean) / (std + 1e-5)
        dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory =True, worker_init_fn=lambda x: np.random.seed(42))

        dset_test_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='test_0', windowlen=window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold, session_split = args.session_split)
        if(args.standarization_run_time): dset_test_0.xs = (dset_test_0.xs - mean) / (std + 1e-5)
        dlod_test_0 = torch.utils.data.DataLoader(dset_test_0, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))
        # nb_classes = dset_test_0.nb_classes()
        # x_train_tensor = torch.from_numpy(np.array(dset_tr_0.xs)).float()
        # y_train_tensor = torch.from_numpy(np.array(dset_tr_0.ys)).float()
        # x_train_tensor  = torch.Tensor(np.transpose(x_train_tensor, (0,2,1)))

        nb_classes = dset_tr_0.nb_classes()

        model = DeepConvLSTM(n_classes=nb_classes, NB_SENSOR_CHANNELS = n_channels, SLIDING_WINDOW_LENGTH = window_len).to(device)
    

        # opt_pa = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)
        opt_pa = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=True)
        print("Training on Model ->",model.name)
        # print('Training parameters: {}'.format(vars(args)))
        print('Training for {} epochs'.format(args.nb_epochs))
        losses_list = []
        best_recall = [0]
        best_epoch = 0
        best_acc, es_count = 0,0

        version = 1
        step = 0

        loss_1 = nn.CrossEntropyLoss()
        sttmax = nn.Softmax(1)

        prototypes = dict()
        counters = dict()

        for epoch in range(0,args.nb_epochs):  #args.nb_epochs
            if(args.only_test_step1):
                continue
            model.train()

            losses_per_epoch = []

            total, correct = 0, 0
            pbar = tqdm(enumerate(dlod_tr_0)) 
            n_support = 10
            total_loss_per_epoch = 0
            nsteps = 0
            
            total_ce_loss = 0
            total_contrastive_losses = 0 
            total_accuracy = 0
            x_train_tensor_embeddings = []
            y_train_tensor = []

            for batch_idx, (x, y, z) in pbar:
                

                x_h = model.init_hidden(len(x))
                x_h = tuple([each.data for each in x_h])

                if(len(x_train_tensor_embeddings)==0):
                    _, _, x_train_tensor_embeddings = model(x.to(device),  x_h, len(x))
                    y_train_tensor = y.to(device)
                    
                    # print("Here",x_train_tensor_embeddings.shape, len(x_h))
                else:
                    x_train_tensor_embeddings = torch.concat((x_train_tensor_embeddings, model(x.to(device), x_h, len(x))[2]), dim = 0)
                    y_train_tensor = torch.concat((y_train_tensor, y.to(device)))
                

                # if(batch_idx>1):break
                ####
                # Extracting Support and Query points.
                x_support, y_support, x_query, y_query = extract_sample(len(np.unique(y)), n_support, n_support, x, y, seed = 0,shuffle=True)
                
                #y_query = tf.keras.utils.to_categorical(y_query, num_classes=baseClassesNb, dtype='int32')
                y_query = torch.from_numpy(y_query).long().to(device)
                y_query_temp = y_query
                
                h_support = model.init_hidden(len(x_support))
                h_support = tuple([each.data for each in h_support])

                h_query = model.init_hidden(len(x_query))
                h_query = tuple([each.data for each in h_query])


                _, _, x_support_embeddings = model(x_support.to(device), h_support, len(x_support))
                _, _, x_query_embeddings = model(x_query.to(device), h_query, len(x_query))

                # Get prototypes

                for clas in sorted(np.unique(y_support)):
                    
                    # p_mean = np.mean(x_support_embeddings.data.cpu().numpy()[y_support==clas],axis=0)
                    p_mean = torch.mean(x_support_embeddings.data[[y_support==clas]], 0)
                    
                    prototypes[clas] = p_mean
                    counters[clas] = len(x_support_embeddings[y_support==clas])
                
                # Merging all the prototypes.
                z_proto = torch.stack(list(prototypes.values())).to(device)
                # z_proto = torch.from_numpy(np.array(list(prototypes.values()))).float().to(device)

                # z_proto = prototypes.values()
                dists = compute_euclidean(x_query_embeddings,z_proto)
                # log_p = F.softmax(-dists,dim=1)
                log_p = sttmax(-dists)

                # Change it to torch...
                # for i in range(nb_classes):
                #     if(np.array_equal(np.array(prototypes[0]), np.array(z_proto[0])) == False):
                #         print("Key order and prototype order not matching")
                
                key2idx = torch.empty(nb_classes,dtype=torch.long).to(device)
                proto_keys = list(prototypes.keys())
                
                key2idx[proto_keys] = torch.arange(len(proto_keys)).to(device)

                y_query = key2idx[y_query].view(-1,1)
                y_query = tf.keras.utils.to_categorical(y_query.cpu().numpy(), num_classes=len(proto_keys))
                y_query = torch.from_numpy(y_query).float().to(device)
                

                # ce_loss = loss_1(dists, y_query_temp)
                # ce_loss = nn.BCELoss(log_p, y_query)
                # ce_loss = loss_1(log_p, y_query)
                ce_loss = F.binary_cross_entropy(log_p, y_query)
                contrastive_losses  = contrastive_loss(x_query_embeddings, y_query,prototypes, balance=True)
                

                loss = ce_loss + contrastive_losses
                
                
                opt_pa.zero_grad()
                loss.backward()
                opt_pa.step()

                total_ce_loss += ce_loss.item()
                total_contrastive_losses += contrastive_losses.item()        
                accuracy = accuracy_score(torch.argmax(log_p, dim = 1).to('cpu').detach().numpy(), y_query_temp.to('cpu').detach().numpy())
                total_accuracy += accuracy
                total_loss_per_epoch +=  loss.item()
                
                nsteps += 1

                # pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss {}: {:.4f}/{:.4f} '.format(
                #         epoch, batch_idx + 1, len(dlod_tr_0), 100. * batch_idx / len(dlod_tr_0), loss.item(), 0))
                    
                # print("X Shape {}, Y Shape {}".format(x.shape, y.shape))
                
            print("Epoch {} Total  {} Contrastive  {} CE  {} Acc {:.4f}".format(epoch,total_loss_per_epoch/nsteps, total_contrastive_losses/nsteps, total_ce_loss/nsteps, total_accuracy/nsteps))
            
            model.eval()
            total_accuracy_val = 0
            n_steps = 0
            val_prototypes = dict()
            val_counter = dict()
            with torch.no_grad():
                for x, y, z in dlod_ev:
                    # h_support = model.init_hidden(len(x_train_tensor))
                    # h_support = tuple([each.data for each in h_support])
                    
                    # x = torch.Tensor(np.transpose(x, (1,0)))
                    # _, _, x_train_tensor_embeddings = model(x_train_tensor.to(device), h_support, len(x_train_tensor))
                    # print(x_train_tensor_embeddings.shape,y_train_tensor.shape)
                    for clas in sorted(np.unique(y_train_tensor.cpu().numpy())):
                        p_mean = np.mean(x_train_tensor_embeddings.data.cpu().numpy()[y_train_tensor.cpu().numpy()==clas],axis=0)
                        val_prototypes[clas] = p_mean
                        val_counter[clas] = counters[clas]
                        
                    z_proto_val = torch.from_numpy(np.array(list(val_prototypes.values()))).float().to(device)

                    h = model.init_hidden(len(x))
                    h = tuple([each.data for each in h])

                    _,_,embeddings_val = model(x.to(device), h, len(x))

                    dists_val = compute_euclidean(embeddings_val,z_proto_val)
                    log_p_val = F.softmax(-dists_val,dim=1)
                    val_accuracy = accuracy_score(torch.argmax(log_p_val, dim = 1).to('cpu').detach().numpy(), y.to('cpu').detach().numpy())
                    total_accuracy_val += val_accuracy
                    n_steps +=1

            print("Epoch {} Val Accuracy {:.4f}".format(epoch, total_accuracy_val/n_steps))
            val_accuracy = total_accuracy_val/n_steps



            if(best_acc<val_accuracy):
                best_acc = val_accuracy
                best_prototypes = val_prototypes
                z_proto_best = z_proto_val
                best_model = copy.deepcopy(model)
                
                print("Got Better Model with Accuracy {:.4f}".format(best_acc))
                tuple_list = [(key, value) for key, value in best_prototypes.items()]
                
                np.save('{}{}_{}_prototypes_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.npy'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split), tuple_list)
                tuple_list_counter = [(key, value) for key, value in val_counter.items()]
                np.save('{}{}_{}_counter_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.npy'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split), tuple_list_counter)
                
                torch.save({'model_pa_state_dict': best_model.state_dict()}, '{}{}_{}_best_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.pth'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split))
                es_count = 0
            else:
                es_count +=1
            
            if(es_count==10):
                print("Early Stopping.")
                print("Best Accuracy {}".format(best_acc))
                break
            
            torch.cuda.empty_cache()
            
        # **************************************************Testing on Saved Model**************************************************************
        print("Testing Model Accuracy on Test Data..")

        # Loading the saved model.
        pth_pth = '{}{}_{}_best_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.pth'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split)
        checkpoint = torch.load(pth_pth, map_location = torch.device(device))
        model.load_state_dict(checkpoint['model_pa_state_dict'])  

        # Loading the saved prototypes
        loaded_prototypes = np.load('{}{}_{}_prototypes_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.npy'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split), allow_pickle= True)
        trained_prototypes_dict = dict()
        for loaded_prototype in loaded_prototypes:
            # trained_prototypes_dict[int(loaded_prototype[0])] = loaded_prototype[1] 
            trained_prototypes_dict[int(loaded_prototype[0])] = torch.tensor(loaded_prototype[1])


        loaded_counters = np.load('{}{}_{}_counter_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.npy'.format(pth_rst_exp_step_1, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split), allow_pickle= True)
        trained_loaded_counter = dict()
        for loaded_counter in loaded_counters:
            trained_loaded_counter[int(loaded_counter[0])] = loaded_counter[1] 

        # 
        print("Results of Stage 1 Test dataset on Model 1") 
        stage_1_acc_0, stage_1_f1_0, _, _, _, _ = get_accuracy_old_new_LSTM(model, dlod_test_0, nb_classes, nb_classes, trained_prototypes_dict)
        if(args.use_wandb):
            wandb.log({"M1-TA1_Old": stage_1_acc_0})   # Logging Initial Step Accuracy.
            wandb.log({"M1-TF1_Old": stage_1_f1_0})  # Logging Initial Step F1 Score.
       
        with open(pth_rst_exp_log_step_1, "a") as file:
            file.write(f"Step 1: Test Dataset Acc: {stage_1_acc_0} F1: {stage_1_f1_0}\n")

        stage_1_test_accs_seen.append(stage_1_acc_0)
        stage_1_test_f1s_seen.append(stage_1_f1_0)
        
        

        dset_tr_now_md = 'train_1' # 'train_2'
        dset_ev_now_md = 'eval_1' # 'eval_2'

        dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md,windowlen= window_len, autoencoderType= None,  standardize = args.standarization_prerun, fold=fold, session_split= args.session_split)
        mean = np.mean(dset_tr_0.xs, axis=0)
        std = np.std(dset_tr_0.xs, axis=0)
        if(args.standarization_run_time): dset_tr_0.xs = (dset_tr_0.xs - mean) / (std + 1e-5)

        dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_ev_now_md,windowlen= window_len, autoencoderType= None,  standardize = args.standarization_prerun, fold=fold, session_split= args.session_split)
        if(args.standarization_run_time): dset_ev_now.xs = (dset_ev_now.xs - mean) / (std + 1e-5)
        dset_test = dataset.load(name=args.dataset, root=pth_dataset, mode='test_1', windowlen= window_len, autoencoderType= None, standardize = args.standarization_prerun, fold=fold, session_split= args.session_split)
        if(args.standarization_run_time): dset_test.xs = (dset_test.xs - mean) / (std + 1e-5)

        dlod_tr_now = torch.utils.data.DataLoader(dset_tr_now, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))
        dlod_ev_now = torch.utils.data.DataLoader(dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))
        dlod_test = torch.utils.data.DataLoader(dset_test, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers, pin_memory=True, worker_init_fn=lambda x: np.random.seed(42))

        nb_classes_now = dset_tr_now.nb_classes()

        
        

        
        # Incremental Step .............................................................

        # Removing all the old indices from the dataloader.
        old_indices = []
        for idx,y in enumerate(dset_tr_now.ys):
            if(y < nb_classes):
                old_indices.append(idx)


        dset_tr_only_new = utils_CGCD.generate_dataset(dset_tr_now, old_indices)
        dlod_tr_only_new = torch.utils.data.DataLoader(dset_tr_only_new, batch_size=20, shuffle=True, num_workers=args.nb_workers)
        # np.unique(dset_tr_o.ys)


        replay_size = 20
        replay_buffer = ReplayMemory(replay_size)
        replay_buffer.update((np.array(dset_tr_0.xs), np.array(dset_tr_0.ys)))

        model_now = copy.deepcopy(model).to(device)
        opt_pa_now = torch.optim.AdamW(model_now.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)


        step1_prototypes_dict = copy.deepcopy(trained_prototypes_dict)

        counters = dict()
        best_weighted_acc = 0
        args.gpu_id = 0

        loss_1 = nn.CrossEntropyLoss()
        sftmax = nn.Softmax(1)

        

        for epoch in range(0,args.nb_epochs):  #args.nb_epochs
            if(args.only_test_step2):
                continue
            
            model_now.train()
            ####
            if(model.name=='resnet18'):
                bn_freeze = args.bn_freeze
                if bn_freeze:
                    modules = model_now.modules() if args.gpu_id != -1 else model_now.module.model.modules()
                    for m in modules:
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()
                if args.warm > 0:
                    # Early Stopping to remain one during the warnmp
                    if args.gpu_id != -1:
                        unfreeze_model_param = list(model_now.embedding.parameters()) 
                    else:
                        unfreeze_model_param = list(model_now.module.model.embedding.parameters()) 

                    if epoch == 0:
                        for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                            param.requires_grad = False
                    if epoch == args.warm:
                        for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                            param.requires_grad = True

            
            total_contrastive_loss = 0
            total_ce_loss = 0
            nsteps = 0

            actual_train_ys = []
            predicted_train_ys = []
            
            for x, y, z in dlod_tr_now:
                # if(nsteps>1): break
                y_map_labels = tf.keras.utils.to_categorical(y, num_classes=nb_classes_now)

                # Getting Replay Set
                excluded_classes = []
                replay_set, replay_map_labels = replay_buffer.exemplar_train(excluded_classes)

                replay_set = torch.from_numpy(np.array(replay_set)).float()
                replay_set = torch.Tensor(np.transpose(replay_set, (0,2,1)))
                
                replay_map_labels_temp = torch.Tensor(replay_map_labels)  # Storing the non-binarized form
                replay_map_labels = torch.Tensor(replay_map_labels)
                replay_map_labels = tf.keras.utils.to_categorical(replay_map_labels, num_classes=nb_classes_now)

                query_set_x = torch.cat((x,replay_set),0)  # Merging old and new class elements
                query_set_labels = torch.cat((torch.Tensor(y_map_labels),torch.Tensor(replay_map_labels)),0).to(device)

                h_support = model_now.init_hidden(len(x))
                h_support = tuple([each.data for each in h_support])

                h_query = model_now.init_hidden(len(query_set_x))
                h_query = tuple([each.data for each in h_query])

                _, _, x_support_embeddings = model_now(x.to(device), h_support, len(x) )
                _, _,  x_query_embeddings = model_now(query_set_x.to(device), h_query,len(query_set_x))

                # Update/Create Prototypes of New Classes, here y only contains new classes.
                classes = np.sort(np.unique(y))
                for c in classes:
                    if c in step1_prototypes_dict.keys():
                        # p_mean_old = copy.deepcopy(np.array(step1_prototypes_dict[c]).astype(np.float64))
                        p_mean_old = copy.deepcopy(step1_prototypes_dict[c]).to(device)

                        # print(c, np.shape(p_mean_old), p_mean_old)
                        new_count = len(x_support_embeddings.data[y==c])
                        #print(np.shape(p_mean_old), np.shape(np.array(X)))
                    
                        # p_mean = float((trained_loaded_counter[c]/(1.*(trained_loaded_counter[c]+new_count))))*p_mean_old + np.sum(np.array(x_support_embeddings.data.cpu().numpy())[y==c],axis=0)/(trained_loaded_counter[c]+new_count)
                        p_mean = float((trained_loaded_counter[c]/(1.*(trained_loaded_counter[c]+new_count))))*p_mean_old + torch.sum(x_support_embeddings.data[y==c],axis=0)/(trained_loaded_counter[c]+new_count)
                        
                        #print(p_mean, p_mean_old)
                        #sys.exit()
                        # step1_prototypes_dict[c] = copy.deepcopy(p_mean.flatten().astype(np.float64))
                        step1_prototypes_dict[c] = copy.deepcopy(p_mean.flatten())
                        
                        trained_loaded_counter[c] += new_count
                        #print('old: ',c, p_mean_old, self.prototypes[c])

                    else:
                        # print('new: ', self.prototypes.keys(), c)
                        # print(x_support_embeddings.data.shape)
                        # p_mean = np.mean(x_support_embeddings.data.cpu().numpy()[y==c],axis=0)
                        p_mean = torch.mean(x_support_embeddings.data[y==c],axis=0)

                        #print(np.shape(X[y==c]))
                        step1_prototypes_dict[c] = copy.deepcopy(p_mean.flatten())
                        trained_loaded_counter[c] = len(x_support_embeddings.data.cpu().numpy()[y==c])

                # Calculate Distances.
                dists = torch.ones((len(x_query_embeddings),nb_classes_now))*float('inf')
                dists = dists.float().to(device)
                #print("CURRENT CLASSES IN PROTOTYPE MEMORY: ", list(self.memory.prototypes.keys()))
                for c in step1_prototypes_dict.keys():

                    # z_proto = torch.from_numpy(step1_prototypes_dict[c][None,:]).float().to(device)# Adding None just increases the shape from (128,) -> (1,128)
                    z_proto = torch.tensor(step1_prototypes_dict[c][None,:]).to(device)
                    
                    dist = compute_euclidean(x_query_embeddings,z_proto)
                    
                    #print(np.shape(dist))
                    dists[:,c] = torch.squeeze(dist)

                # Calculate the Loss.
                # log_p = F.softmax(-dists,dim=1)
                # ce_loss = F.binary_cross_entropy(log_p, query_set_labels)
                

                log_p = sftmax(-dists)
                ce_loss = F.binary_cross_entropy(log_p, query_set_labels).to(device)
                # ce_loss = loss_1(log_p, query_set_labels).to(device)
                
                contrastive_losses  = contrastive_loss(x_query_embeddings, query_set_labels, step1_prototypes_dict, balance=True)

                loss = ce_loss  + contrastive_losses
                opt_pa_now.zero_grad()
                loss.backward()
                opt_pa_now.step()
                total_contrastive_loss += contrastive_losses.item()
                total_ce_loss += ce_loss.item() 

                query_set_labels_temp = copy.deepcopy(query_set_labels)
                query_set_labels_temp = torch.argmax(query_set_labels, dim = 1).to('cpu').detach().numpy()

                if(len(predicted_train_ys) ==0):
                        predicted_train_ys =  torch.argmax(log_p, dim = 1).to('cpu').detach().numpy()
                else:
                    predicted_train_ys = np.concatenate((predicted_train_ys, torch.argmax(log_p, dim = 1).to('cpu').detach().numpy()),axis=0)
                    
                if(len(actual_train_ys) ==0):
                    actual_train_ys =  query_set_labels_temp
                else:
                    actual_train_ys = np.concatenate((actual_train_ys,query_set_labels_temp),axis=0)
                nsteps +=1
                

                
                # Online Training.
                online_epochs = 1
                total_online_training_ce_loss = 0
                total_online_training_contrastive_loss = 0
                for i in range(1,online_epochs):
                    query_set1, query_map_labels1 = utils.shuffle(query_set_x,query_set_labels, random_state=i)
                    x_query_embeddings_1 = model_now(query_set1.to(device))

                    dists_1 = torch.ones((len(x_query_embeddings_1),nb_classes_now))*float('inf')
                    
                    dists_1 = dists_1.float().to(device)
                    #print("CURRENT CLASSES IN PROTOTYPE MEMORY: ", list(self.memory.prototypes.keys()))
                    for c in step1_prototypes_dict.keys():

                        z_proto_1 = torch.from_numpy(step1_prototypes_dict[c][None,:]).float().to(device)# Adding None just increases the shape from (128,) -> (1,128)
                        dist_1 = compute_euclidean(x_query_embeddings_1,z_proto_1)
                        #print(np.shape(dist))
                        dists_1[:,c] = torch.squeeze(dist_1)
                        
                    log_p_1 = F.softmax(-dists_1,dim=1)
                    
                    ce_loss1 = F.binary_cross_entropy(log_p_1, query_map_labels1)


                    contrastive_losses = contrastive_loss(x_query_embeddings_1, query_map_labels1, step1_prototypes_dict, balance=True)  
                    loss = ce_loss1 + 0.1*contrastive_losses
                    opt_pa_now.zero_grad()
                    loss.backward()
                    opt_pa_now.step()

                
                
                # Updating the prototypes of the seen/base classes
                model_now.eval()
                if len(replay_set) > 0:
                    h = model_now.init_hidden(len(replay_set))
                    h = tuple([each.data for each in h])
                    base_new_prototypes = dict()
                    base_new_counters = dict()
                    #print(classes)
                    _ ,_ , replay_set_embeddings = model_now(replay_set.to(device) , h, len(replay_set))
                    old_classes = np.unique(replay_map_labels_temp.data.cpu())
                    for c in old_classes:
                        p_mean = replay_set_embeddings[replay_map_labels_temp.data.cpu()==c].mean(0)
                        #print(np.shape(X[y==c]))
                        # base_new_prototypes[int(c)] = copy.deepcopy(list(p_mean.data.cpu().numpy().flatten()))
                        base_new_prototypes[int(c)] = copy.deepcopy(p_mean.data)


                        base_new_counters[int(c)] = len(replay_set_embeddings[replay_map_labels_temp.data.cpu()==c])

                    momentum = 0.9

                    
                    for c in base_new_prototypes.keys():
                        # step1_prototypes_dict[c] = copy.deepcopy((momentum*np.array(step1_prototypes_dict[c]).astype(np.float64) + (1.-momentum)*np.array(base_new_prototypes[c]).astype(np.float64)).flatten().astype(np.float64))

                        step1_prototypes_dict[c] = copy.deepcopy((momentum*step1_prototypes_dict[c].to(device)) + (1.-momentum)*base_new_prototypes[c]).to(device).flatten()

                model_now.train()
            print("Epoch {} CE Loss {:.5f} Contrastive Loss {:.5f} Accuracy ".format(epoch, total_ce_loss/nsteps, total_contrastive_loss/nsteps ))
            
            total_train_accuracy_seen = accuracy_score(predicted_train_ys[actual_train_ys<nb_classes], actual_train_ys[actual_train_ys<nb_classes])
            total_train_accuracy_unseen = accuracy_score(predicted_train_ys[actual_train_ys>=nb_classes], actual_train_ys[actual_train_ys>=nb_classes])
            total_train_accuracy_overall  = accuracy_score(predicted_train_ys, actual_train_ys)
            print("Train Accuracies Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".format(total_train_accuracy_seen, total_train_accuracy_unseen, total_train_accuracy_overall))


            print('==> Evaluation..')
            model_now.eval()

            predicted_ys = []
            actual_ys = []
            with torch.no_grad():
                n_steps_val = 0

                for x_val, y_val, z_val in dlod_ev_now:
                    h = model_now.init_hidden(len(x_val))
                    h = tuple([each.data for each in h])

                    _ ,_ ,embeddings_val = model_now(x_val.to(device), h, len(x_val) )

                    dists_val = torch.ones((len(embeddings_val),nb_classes_now))*float('inf')
                    dists_val = dists_val.float().to(device)        
                    
                    for c in step1_prototypes_dict.keys():

                        # z_proto_val = torch.from_numpy(step1_prototypes_dict[c][None,:]).float().to(device)# Adding None just increases the shape from (128,) -> (1,128)
                        z_proto_val = step1_prototypes_dict[c][None,:].to(device)# Adding None just increases the shape from (128,) -> (1,128)
                        
                        dist_val = compute_euclidean(embeddings_val,z_proto_val)
                        #print(np.shape(dist))
                        dists_val[:,c] = torch.squeeze(dist_val)
                    log_p_val = sftmax(-dists_val)
                    

                    if(len(predicted_ys) ==0):
                        predicted_ys =  torch.argmax(log_p_val, dim = 1).to('cpu').detach().numpy()
                    else:
                        predicted_ys = np.concatenate((predicted_ys, torch.argmax(log_p_val, dim = 1).to('cpu').detach().numpy()),axis=0)
                    
                    if(len(actual_ys) ==0):
                        actual_ys =  y_val
                    else:
                        actual_ys = np.concatenate((actual_ys,y_val),axis=0)

                acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
                acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])
                acc_a = accuracy_score(predicted_ys, actual_ys)
                print("Valid Accuracies Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".format(acc_o, acc_n, acc_a))


                step = 2
                if((acc_o + acc_n + acc_a)/3 > best_weighted_acc):
                    best_seen = acc_o
                    best_unseen = acc_n
                    best_overall = acc_a
                    best_weighted_acc = (best_seen + best_unseen + best_overall)/3
                    best_epoch = epoch
                    es_count = 0
                    print("Got Better Model with  Seen {:.2f} ,  Unseen Accuracies {:.2f},  Overalll {:.2f} and  Average as {:.2f} ".format( best_seen,best_unseen, best_overall,best_weighted_acc))
                    torch.save({'model_pa_state_dict': model_now.state_dict()}, '{}{}_{}_model_last_windowlen_{}_sz_embedding_{}_step_{}_fold_{}_session_split_{}.pth'.format(pth_rst_exp_step_2, args.dataset, args.model,window_len, args.sz_embedding, str(step), fold, args.session_split))
                    
                    tuple_list = [(key, value.cpu()) for key, value in step1_prototypes_dict.items()]
                    np.save('{}{}_{}_prototypes_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.npy'.format(pth_rst_exp_step_2, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split), tuple_list)
                else:                                                                                                     
                    es_count +=1 
                    print("Early Stopping Count ", es_count)
                    
                # if(epoch< args.warm):
                #     es_count =0    # If we are having warmup then no early stopping
                
                if(es_count ==10 or epoch == args.nb_epochs-1):
                    with open(pth_rst_exp_log_step_2, "a+") as fval:
                        fval.write('Best Valid Accuracies Seen {}, Unseen {}, Overall {}, Average {}\n'.format( best_seen, best_unseen, best_overall, best_weighted_acc))
                    
                    stage_2_val_accs_seen.append(best_seen)
                    stage_2_val_accs_unseen.append(best_unseen)
                    stage_2_val_accs_overall.append(best_overall)
                    
                    print("Best  Valid Accuracies Seen {}, and Unseen {},  Overall {}, Average {} ".format( best_seen, best_unseen, best_overall, best_weighted_acc))
                    print("Early Stopping")


                    break
            
            
        del model_now
        del model
        del opt_pa  

        version = 1
        step =2
        print("Testing On Test Dataset ...")
        
        step = 2
        
    
        print("Best Epoch :", best_epoch)
        pth_pth_test = '{}{}_{}_model_last_windowlen_{}_sz_embedding_{}_step_{}_fold_{}_session_split_{}.pth'.format(pth_rst_exp_step_2, args.dataset, args.model,window_len, args.sz_embedding, str(step), fold, args.session_split)

        cfg_test = {'weights_path': 'CGCD-main/src/Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': args.sz_embedding, 'bottleneck_dim': None,
                'output_size':nb_classes_now,'weight_norm_dim': '', 'n_channels' :n_channels, 'window_len': window_len}

        
        model_test =  DeepConvLSTM(n_classes=nb_classes, NB_SENSOR_CHANNELS = n_channels, SLIDING_WINDOW_LENGTH = window_len).to(device)
        model_test = model_test.to(device)
        
        checkpoint_test = torch.load(pth_pth_test, map_location=torch.device(device))
        model_test.load_state_dict(checkpoint_test['model_pa_state_dict'])
        model_test.eval()

        loaded_prototypes = np.load('{}{}_{}_prototypes_windowlen_{}_embedding_size_{}_version_{}_step_{}_fold_{}_session_split_{}.npy'.format(pth_rst_exp_step_2, args.dataset, args.model, window_len, args.sz_embedding, version, step, fold, args.session_split), allow_pickle= True)
        trained_prototypes_dict = dict()
        for loaded_prototype in loaded_prototypes:
            trained_prototypes_dict[int(loaded_prototype[0])] = loaded_prototype[1] 

        # Prediction accuracy on Stage 2 test dataset on Stage 1 Model.
        print("Results of Stage 1 Test dataset on Model 2") 
        stage_2_acc_0, stage_2_f1_0, stage_2_acc_1, stage_2_f1_1, _, _ = get_accuracy_old_new_LSTM(model_test, dlod_test_0, nb_classes, nb_classes_now, trained_prototypes_dict)
        
        with open(pth_rst_exp_log_step_1, "a") as file:
            file.write(f"Stage 2: Test Dataset Acc Seen Acc:{stage_2_acc_0}, Seen F1: {stage_2_f1_0} Unseen Acc: {stage_2_acc_1} \n")
        
        if(args.use_wandb):
            wandb.log({"M1-TA2_Old": stage_2_acc_0})   # Logging Initial Step Accuracy.
            wandb.log({"M1-TF2_Old": stage_2_f1_0})  # Logging Initial Step F1 Score.
        
        stage_2_test_accs_seen.append(stage_2_acc_0)
        stage_2_test_f1s_seen.append(stage_2_f1_0)

        print("Results of Stage 2 Test dataset on Model 2") 
        acc_o, f1_o, acc_n, f1_n, acc_all, f1_all = get_accuracy_old_new_LSTM(model_test, dlod_test, nb_classes, nb_classes_now, trained_prototypes_dict)
        
        new_row = {'Dataset':args.dataset, 'Iteration':iter, 
        'Initial Acc': stage_1_acc_0, 'Initial F1': stage_1_f1_0, 'Incremental Acc Seen1':stage_2_acc_0,
            'Incremental F1 Seen1':stage_2_f1_0, 'Incremental Acc Seen2' :acc_o,
            'Incremental F1 Seen2':f1_o, 'Incremental Acc Unseen' :acc_n,
                'Incremental F1 Unseen': f1_n, 'Incremental Acc All':acc_all, 
                'Incremental F1 All' :f1_all, "Forget Acc": stage_2_acc_0 - acc_o, "Forget F1": stage_2_f1_0 - f1_o}
        results_df = results_df.append(new_row, ignore_index = True)
        
        print("Testing Accuracies : Seen {} Unseen {} Overall {}".format(acc_o, acc_n, acc_all))
        print("Testing F1 Scores  : Seen {} Unseen {} Overall {}".format(f1_o, f1_n, f1_all))
        with open(pth_rst_exp_log_step_2, "a+") as fval:
                    fval.write(f'\nBest Test Accuracies Seen: {acc_o}, Unseen: {acc_n}, Overall: {f1_all}\n')
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
            wandb.log({"M2-TFA":  stage_2_acc_0 - acc_o})  
            wandb.log({"M2-TFF1": stage_2_f1_0 - f1_o})  
            wandb.finish()
    # results_df.to_csv('results_edison_LSTM_{}.csv'.format('_'.join(dstes), index=False))
    
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
        # print("Test Accuracies Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".format(acc_o, total_test_accuracy_unseen, total_test_accuracy_overall))