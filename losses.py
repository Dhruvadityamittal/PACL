import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
import utils_CGCD

import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.manual_seed(42)
torch.manual_seed(42)

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(T, classes = range(0, nb_classes))
    T = torch.FloatTensor(T).to(device)
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).to(device))
       
        # print("pROXIES Shape =", self.proxies.shape)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity

        
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes) # converting [0,1] -> [[1,0],[0,1]]
        N_one_hot = 1 - P_one_hot   # Converting [[1,0],[0,1]]  -> [[0,1],[1,0]]
        
        

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        
        # Positive proxy means if the class belonging to that proxy is availalbe in the batch
    
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies  in the batch
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        
        if num_valid_proxies == 0:
            num_valid_proxies = 1
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).to(device)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

def get_validation_losses_step_1(model, criterion_pa, nb_classes, args, nce_loss,dlod_ev):
    total, correct = 0, 0
    total_train_loss = 0
    total_data = 0
    total_pa_loss = 0
    total_contrastive_loss = 0

    with torch.no_grad():
        # Iterate over the training data (features x, labels y, and other data z).
        for batch_idx, (x, y, z) in enumerate(dlod_ev):

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
            if args.contrastive_loss_type in ['G-Baseline_NCE', 'Offline_NCE', 'Online_Finetuning_NCE','G-Baseline_NCE_WFR']:
                contrastive_loss = nce_loss(feats, positive_proxies, negetive_proxies)
            # If contrastive loss type is the baseline contrastive loss, compute the corresponding loss.
            elif args.contrastive_loss_type == 'G-Baseline_Contrastive':
                contrastive_loss = utils_CGCD.contrastive_loss(feats, y, criterion_pa.proxies, True).to(device)
            # If no contrastive loss is required, set contrastive loss to zero.
            else:
                contrastive_loss = torch.tensor(0).to(device)
                # model.contrastive_weight_lr.requires_grad = False

            # Combine the Proxy Anchor loss and contrastive loss with their respective weights.

            # total_loss = torch.exp(-model.pa_weight_lr) * loss_pa + torch.exp(
            #     -model.contrastive_weight_lr) * contrastive_loss - model.pa_weight_lr - model.contrastive_weight_lr


            # Accumulate the total loss for reporting.
            total_pa_loss += loss_pa.item() * x.size(0)
            total_contrastive_loss += contrastive_loss.item() * x.size(0)
            # total_train_loss += total_loss.item() * x.size(0)
            total_data += x.size(0)

            # Print the progress of the training epoch.
            # print("Validation Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}".format(epoch, batch_idx + 1,
            #                                                                                  len(dlod_tr_0),
            #                                                                                  100. * batch_idx / len(
            #                                                                                      dlod_tr_0),
            #                                                                                  total_loss.item(), 0, 0),
            #       end="\r")

        return total_pa_loss/total_data, total_contrastive_loss/total_data #, total_train_loss/total_data

from tqdm import *
def get_validation_losses_step_2(model, model_now, criterion_pa, criterion_pa_now, expler_s, nb_classes_prv, nb_classes_now, args, nce_loss,dlod_ev):

    with torch.no_grad():
        # pbar = tqdm(enumerate(dlod_ev))
        total_train_loss, total_kd_loss, total_pa_loss, total_contrastive_loss = 0.0, 0.0, 0.0, 0.0
        total_data = 0
        for batch_idx, (x, y, z) in enumerate(dlod_ev):
            feats = model_now(x.squeeze().to(device))
            y_batch = y
            y_n = torch.where(y >= nb_classes_prv, 1, 0)  # Identify new classes
            # Feature replay is not used in Online Finetuning and Offline Model
            if args.contrastive_loss_type not in ["Online_Finetuning", 'Offline', 'Online_Finetuning_NCE', 'Offline_NCE', 'G-Baseline_NCE_WFR']:
                # Example replay: Generate old class examples to mitigate forgetting

                y_o = y.size(0) - y_n.sum()  # Count old classes
                if y_o > 0:
                    y_sp = torch.randint(nb_classes_prv, (y_o,))  # Randomly select old class proxies
                    feats_sp = torch.normal(criterion_pa.proxies[y_sp], expler_s).to(
                        device)  # Generate synthetic features for old classes
                    y = torch.cat((y, y_sp), dim=0)
                    feats = torch.cat((feats, feats_sp), dim=0)

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
            if args.contrastive_loss_type not in ["Online_Finetuning", 'Offline', 'Online_Finetuning_NCE', 'Offline_NCE']:
                # y_n = torch.where(y_batch >= nb_classes_prv, 0, 1) # Remove it
                y_o_msk = torch.nonzero(y_n)  # Provides index of all new classes

                if y_o_msk.size(0) > 1:
                    y_o_msk = torch.nonzero(y_n).squeeze()
                    x_o = torch.unsqueeze(x[y_o_msk[0]], dim=0)  # Just first element

                    feats_n = torch.unsqueeze(feats[y_o_msk[0]], dim=0)
                    for kd_idx in range(1, y_o_msk.size(0)):  # After 1st Index
                        try:
                            x_o_ = torch.unsqueeze(x[y_o_msk[kd_idx]], dim=0)
                        except:
                            raise ValueError("A value error occurred", kd_idx, y_o_msk[kd_idx])

                        x_o = torch.cat((x_o, x_o_), dim=0)
                        feats_n_ = torch.unsqueeze(feats[y_o_msk[kd_idx]], dim=0)
                        feats_n = torch.cat((feats_n, feats_n_), dim=0)
                    with torch.no_grad():
                        feats_o = model(x_o.squeeze().to(device))
                    feats_n = feats_n.to(device)
                    # FRoST
                    loss_kd = torch.dist(F.normalize(feats_o.view(feats_o.size(0) * feats_o.size(1), 1), dim=0).detach(),
                                         F.normalize(feats_n.view(feats_o.size(0) * feats_o.size(1), 1), dim=0))
                    loss_kd = (criterion_pa.proxies - criterion_pa_now.proxies[0:nb_classes_prv]).pow(2).sum(1).sqrt().sum().to(device)
                else:
                    loss_kd = torch.tensor(0.).to(device)
            else:
                loss_kd = torch.tensor(0.).to(device)
                # model_now.kd_weight_lr.requires_grad = False

            # Compute INFO_NCE Contrastive Loss
            # Offline_NCE and Online_Finetuning_NCE have contrastive loss
            if args.contrastive_loss_type in ['G-Baseline_NCE', 'Offline_NCE', 'Online_Finetuning_NCE', 'G-Baseline_NCE_WFR']:
                positive_proxies = criterion_pa_now.proxies[[torch.tensor(y, dtype=torch.long)]]
                negative_proxies = []
                for y_temp in y:
                    t = [i for i in range(nb_classes_now) if i != y_temp.item()]

                    # If else is just to append the data int the array
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
                # model_now.contrastive_weight_lr.requires_grad = False

            # If we have Online Finetuning and Offline model we dont use contrastive loss and knowledge distillation

            # Compute the total loss
            # loss = model_now.pa_weight_lr * loss_pa + model_now.kd_weight_lr * loss_kd + model_now.contrastive_weight_lr * contrastive_loss

            # total_train_loss += loss.item() * x.size(0)
            total_kd_loss += loss_kd.item() * x.size(0)
            total_pa_loss += loss_pa.item() * x.size(0)
            total_contrastive_loss += contrastive_loss.item() * x.size(0)
            total_data += x.size(0)


    return  total_pa_loss / total_data, total_kd_loss / total_data, total_contrastive_loss / total_data

