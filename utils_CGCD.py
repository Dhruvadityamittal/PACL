import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import logging
import losses
import json
from tqdm import tqdm

import math
import os
import sys
from sklearn.cluster import KMeans
from collections import Counter

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, Birch, AffinityPropagation, MeanShift, OPTICS, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.neighbors import KernelDensity
# import hdbscan
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import copy
import plotly.graph_objects as go  


def cluster_pred_2_gt(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size  
    D = max(y_pred.max(), y_true.max()) + 1   
    w = np.zeros((D, D), dtype=np.int64) 
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1    ## Similar to a Confusion matrix
    _, col_idx = linear_sum_assignment(w.max() - w)
    return col_idx

def pred_2_gt_proj_acc(proj, y_true, y_pred):
    proj_pred = proj[y_pred]
    return accuracy_score(y_true, proj_pred)

def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return

def _hungarian_match_(y_pred, y_true):
    # y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # ind = linear_sum_assignment(w.max() - w)
    # acc = 0.
    # for i in range(D):
    #     acc += w[ind[0][i], ind[1][i]]
    # acc = acc * 1. / y_pred.size
    # return acc

    ind_arr, jnd_arr = linear_sum_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output




def predict_batchwise(model, dataloader):
    # device = "cuda"
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]

    outputs,ys, zs = [], [], []

    with torch.no_grad(): 
        for batch in dataloader:
            # Unpack the batch
            x, y, z = batch
            
            # Forward pass: Compute predicted output by passing x to the model
            output = model(x.to(device))  # Model only takes x as input
            
            # Store the outputs and the original y, z
            outputs.append(output)  # Assuming you want to store results in a numpy array
            ys.append(y)
            zs.append(z)

        # Concatenate all outputs, y, and z into single numpy arrays
        outputs = torch.cat(outputs, dim=0)
        ys = torch.cat(ys, dim=0)
        zs = torch.cat(zs, dim=0)
    
    model.train()
    model.train(model_is_training)
    return outputs, ys, zs

    # with torch.no_grad():
    #     # extract batches (A becomes list of samples)
    #     for batch in dataloader:  # Each batch has 3 tensors of size equal to batch_size
    #         for i, J in enumerate(batch):

    #             # i = 0: sz_batch * images
    #             # i = 1: sz_batch * labels
    #             # i = 2: sz_batch * indices
    #             if i == 0:
    #                 # move images to device of model (approximate device)
    #                 J = model(J.to(device)) 
    #                 # J, _ = model(J.cuda())

    #             for j in J:    # For each example in the Batch
    #                 A[i].append(j)
            
        
    # model.train()
    # model.train(model_is_training) # revert to previous training state
    
    # return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, _ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean



def evaluate_cos_ev(model, dataloader, proxies_new):
    nb_classes = dataloader.dataset.nb_classes()

    # acc, _ = _hungarian_match_(clustering.labels_, np.array(dlod_tr_n.dataset.ys)) #pred, true

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()

    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_(model, dataloader):
    # nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)
    # X = l2_norm(X)
    # X = torch.softmax(X, dim=1)

    # cos_sim = F.linear(X, X)  # 2158x2158
    # v, i = cos_sim.topk(1 + 5)
    # T1 = T[i[:, 1]]
    # V = v[:, 1].float().cpu()

    # return X[i[:, 1]], T, T1
    # return X, T, T1
    return X, T

    # clustering = AffinityPropagation(damping=0.5).fit(X.cpu().numpy())  ###
    # u, c = np.unique(clustering.labels_, return_counts=True)
    # print(u, c)

    # get predictions by assigning nearest 8 neighbors with cosine

    xs = []

    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall






def saveImage(strPath, input):
    normalize_un = transforms.Compose([transforms.Normalize(mean=[-0.4914/0.2471, -0.4824/0.2435, -0.4413/0.2616], std=[1/0.2471, 1/0.2435, 1/0.2616])])

    sqinput = input.squeeze()
    unnorminput = normalize_un(sqinput)
    npinput = unnorminput.cpu().numpy()
    npinput = np.transpose(npinput, (1,2,0))
    npinput = np.clip(npinput, 0.0, 1.0)

    plt.imsave(strPath, npinput)


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def evaluate_cos(model, dataloader, epoch):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)

    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X).to(device)
    T = T.to(device)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float()

    recall = []
    r_at_k = calc_recall_at_k(T, Y, 1)
    recall.append(r_at_k)
    print("R@{} : {:.3f}".format(1, 100 * r_at_k))
    return recall

# m - model, y - true labels, v - cosine similarity
def show_OnN(m, y, v, nb_classes, pth_result, thres=0., is_hist=False, iter=0):
    oo_i, on_i, no_i, nn_i = 0, 0, 0, 0
    o, n = [], []

    for j in range(m.size(0)):
        
        if y[j] < nb_classes:
            o.append(v[j].cpu().numpy())   # Appending Cosine Similarity
            if v[j] >= thres:  # if cosine similarity is greater than the threshold
                oo_i += 1      # Old data close to old proxies
            else:
                on_i += 1     # Old data not close to old proxies
        else:
            n.append(v[j].cpu().numpy())
            if v[j] >= thres:
                no_i += 1    # New data close to old proxies
            else:
                nn_i += 1   # New data away from old proxies

    if is_hist is True:

        plt.hist((o, n), histtype='bar', bins=100)
        plt.legend(['Old','New'])
        plt.savefig(pth_result + '/' + 'Init_Split_' + str(iter) + '.png')
        
        plt.close()
        # plt.clf()

    print('Init. Split result(0.)\t oo: {}\t on: {}\t no: {}\t nn: {}'.format(oo_i, on_i, no_i, nn_i))


def visualize_proxy_anchors(model, dloader, proxies, dataset, embedding_size, classes, step, method):

    model.eval()
    with torch.no_grad():
        feats, ys = evaluate_cos_(model, dloader)
    
    if(method == 'pca'):
        pca = PCA(n_components=2, random_state=42)

        pca.fit(feats.to('cpu').detach().numpy())
        feats_transformed = pca.transform(feats.to('cpu').detach().numpy())

        embedded_data = pca.transform(losses.l2_norm(proxies).to('cpu').detach().numpy())
    elif(method =='tsne'):
        tsne = TSNE(n_components=2, random_state=42)
        
        # fits_transformed = tsne.transform(feats)
        orignal_len_feats = len(feats)
    
        feats = np.concatenate((losses.l2_norm(feats).to('cpu').detach().numpy(),losses.l2_norm(proxies.to('cpu')).detach().numpy() ))
        
        feats_transformed = tsne.fit_transform(feats)
        embedded_data = feats_transformed[orignal_len_feats:]  # Proxy Anchors
        feats_transformed = feats_transformed[:orignal_len_feats]

    # clst_a = AffinityPropagation(damping =0.75).fit(feats_transformed) # 0.75
    # p, c = np.unique(clst_a.labels_, return_counts=True)  
    # nb_classes_k = len(p)   # Number of Determined Unique Clusters
    # print("Number of Clusters determined for New Classes.", nb_classes_k)
    # print(p,c)


    kmeans = KMeans(n_clusters=classes, random_state=0).fit(feats_transformed)
    # ari = adjusted_rand_score(true_labels, clusters)
    # nmi = normalized_mutual_info_score(true_labels, clusters)
    predicted_outputs = kmeans.predict(feats_transformed)

    
    
    
    def mapped_clusters(clusters, ys):
        mapped_cl = [-1]*len(clusters)
        for clas in np.unique(ys):
            specific_class_indices = np.where(ys == clas)
        
            most_matching_clusters = Counter(clusters[specific_class_indices])
            most_matching_cluster_label = most_matching_clusters.most_common(1)[0][0]
            
            for j in range(len(ys)):
                if(clusters[j]==most_matching_cluster_label):
                    mapped_cl[j] = clas
            
        return mapped_cl
    

    # print(ys[0:50], mapped_clusters(clusters, ys)[0:50])
    # print("Clustering Accuracy Score: {:.2f} %".format(accuracy_score(ys, mapped_clusters(clusters, ys))))

    plt.figure(figsize=(15, 15))
    colors = plt.cm.tab20(np.linspace(0, 1, classes))
    Mapping = {}

    if dataset == 'realworld':
        Mapping[0], Mapping[1], Mapping[2], Mapping[3] = 'Climbing Down', 'Climbing Up', 'Jumping', 'Lying'
        Mapping[4], Mapping[5], Mapping[6], Mapping[7] = 'Running', 'Sitting', 'Standing', 'Walking'
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'magenta', 'teal']

    handles = []
    labels = []

    for class_idx in range(classes):
        class_data = feats_transformed[ys == class_idx]
        # cluster_center = np.mean(class_data, axis = 0)
        plt.scatter(class_data[:, 0], class_data[:, 1], alpha=0.3, color=colors[class_idx], s=30)
        # plt.scatter(embedded_data[class_idx][0], embedded_data[class_idx][1], s=800,  color = colors[class_idx], marker = 'X', edgecolors='black', linewidth=6)
        # plt.scatter(cluster_center[0], cluster_center[1],marker = 'H', s = 200, facecolor = colors[class_idx], edgecolor='black', linewidth=3)
        
        if len(list(Mapping.keys())) == 0:
            label = f'Class {class_idx}'
            
        else:
            label = Mapping[int(class_idx)]
            
        # plt.annotate(label, (embedded_data[class_idx][0], embedded_data[class_idx][1]), textcoords="offset points", xytext=(10, 10), ha='center', fontsize=20)
        # Collect handles and labels for legend
        handles.append(plt.scatter([], [], color=colors[class_idx], label=label))
        labels.append(label)

    for class_idx in range(classes):
        class_data = feats_transformed[ys == class_idx]
        # cluster_center = np.mean(class_data, axis = 0)
        # plt.scatter(class_data[:, 0], class_data[:, 1], alpha=0.3, color=colors[class_idx], s=30)
        plt.scatter(embedded_data[class_idx][0], embedded_data[class_idx][1], s=800,  color = colors[class_idx], marker = 'X', edgecolors='black', linewidth=6)
        # plt.scatter(cluster_center[0], cluster_center[1],marker = 'H', s = 200, facecolor = colors[class_idx], edgecolor='black', linewidth=3)
        
        # if len(list(Mapping.keys())) == 0:
        #     label = f'Class {class_idx}'
            
        # else:
        #     label = Mapping[int(class_idx)]

    # Plot legend
    handles.append(plt.scatter([], [], color='black', marker ='X', label='Anchor', s= 50))
    handles.append(plt.scatter([], [], color='black', marker ='H', label='Center', s= 50))
    labels.append("Anchor")
    # labels.append("Center")

    plt.legend(handles, labels, fontsize=25, markerscale=2.5, ncols =len(labels)//2, loc="upper center")

    # plt.xlabel('Dimension 1', fontsize=20)
    # plt.ylabel('Dimension 2', fontsize=20)
    # plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(False)  # Remove grid lines
    plt.show()
        # Save the figure
    plt.savefig('/netscratch/dmittal/continual_learning/Visualizations/proxy_visualization_{}_embedding_size_{}_nb_classes_{}_method_{}_step_{}.pdf'.format(dataset,embedding_size,classes, method, step), format='pdf', dpi=300)
    plt.savefig('/netscratch/dmittal/continual_learning/Visualizations/proxy_visualization_{}_embedding_size_{}_nb_classes_{}_method_{}_step_{}.png'.format(dataset,embedding_size,classes, method, step))
    


def contrastive_loss(embeddings, labels, proxies, balance):
    # labels = torch.argmax(labels, dim=1)
    # prototypes_values = torch.from_numpy(np.array(list(prototypes.values()))).float().to(device)
    # prototypes_keys = torch.from_numpy(np.array(list(prototypes.keys()))).float().to(device)
    
    proxy_labels = torch.from_numpy(np.arange(0,len(proxies)))
    
    embeddings = torch.cat((embeddings,proxies),0)
    labels = torch.cat((labels.to(device),proxy_labels.to(device)),0)

    labels_numpy = labels.cpu().data.numpy()
    all_pairs = np.array(list(combinations(range(len(labels_numpy)),2)))
    all_pairs = torch.LongTensor(all_pairs)

    # print(all_pairs.shape)

    positive_pairs = all_pairs[(labels_numpy[all_pairs[:,0]] == labels_numpy[all_pairs[:,1]]).nonzero()]
    negative_pairs = all_pairs[(labels_numpy[all_pairs[:,0]] != labels_numpy[all_pairs[:,1]]).nonzero()]
		#print(np.shape(positive_pairs), np.shape(negative_pairs))
    if balance:
        negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

    positive_loss = (embeddings[positive_pairs[:,0]] - embeddings[positive_pairs[:,1]]).pow(2).sum(1)
    margin = 1.0
    negative_loss = F.relu(margin - (embeddings[negative_pairs[:,0]] - embeddings[negative_pairs[:,1]]).pow(2).sum(1).sqrt()).pow(2)
    
    loss = torch.cat([positive_loss, negative_loss], dim = 0)
    return loss.mean()

def calculate_accuracy(predicted_classes, true_labels):

    # Ensure both tensors have the same shape
    
    if predicted_classes.shape != true_labels.shape:
        raise ValueError("Shapes of predicted classes and true labels do not match.")
    
    # Calculate accuracy
    correct_predictions = (predicted_classes == true_labels).sum().item()
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions

    return accuracy

def generate_dataset(dataset, index, index_target=None, target=None):
    dataset_ = copy.deepcopy(dataset)

    if target is not None:
        for i, v in enumerate(index_target):
            dataset_.ys[v] = target[i]

    for i, v in enumerate(index):
        # print("Index ",index, "Dataset.I",dataset_.I)
        j = v - i    # We seperate i because as we pop a element outside the array moves towards left and its size decreases
        dataset_.I.pop(j)
        dataset_.ys.pop(j)
        # dataset_.im_paths.pop(j)
    return dataset_


def merge_dataset(dataset_o, dataset_n):
    dataset_ = copy.deepcopy(dataset_o)
    # if len(dataset_n.classes) > len(dataset_.classes):
    #     dataset_.classes = dataset_n.classes
    dataset_.I.extend(dataset_n.I)
    # dataset_.I  = list(set(dataset_n.I))
    # dataset_.im_paths.extend(dataset_n.im_paths)
    dataset_.ys.extend(dataset_n.ys)

    return dataset_


def get_cluster_information(feats, ys_, fit_kde = True):
    
    ys_ = np.array([int(t.item()) for t in ys_])
    classes = set(ys_)
    

    all_class_clusters = dict()
    all_class_clusters_kde_fits = dict()
    for class_idx in classes:

        # Filter data for the current class
        class_data = feats[ys_ == class_idx]
        
        # Perform KMeans clustering for a particular Class
        kmeans = KMeans(n_clusters=5, random_state=0).fit(class_data)
        
        # kmeans = KMedoids(n_clusters=3, random_state=42).fit(class_data)

        # Getting Cluster Predictions
        predicted_clusters = kmeans.predict(class_data)
        

        class_clusters_data = []
        class_fitted_kdes = []
        # Getting Each Cluster Data for each class
        for cluster_idx in range(len(kmeans.cluster_centers_)):
            cluster_data = class_data[predicted_clusters == cluster_idx]
            class_clusters_data.append(cluster_data)
            if(fit_kde):
                class_fitted_kdes.append(KernelDensity(kernel='gaussian', bandwidth=0.1).fit(cluster_data))

        all_class_clusters[class_idx] = class_clusters_data
        all_class_clusters_kde_fits[class_idx] = class_fitted_kdes

    return all_class_clusters, all_class_clusters_kde_fits

def get_sampled_data_kde(cluster_data, samples_each_class, fitted_kdes):
    x_sampled = []
    y_sampled = []
    for (class_name,class_clusters), (_, fitted_kde)  in zip(cluster_data.items(), fitted_kdes.items()):
        # print(class_name, np.array(class_clusters).shape)
        sampled_x = []
        sampled_y = []
        
        # Calculating the total number instance of each class
        class_data_len = sum(len(cluster) for cluster in class_clusters)
        for class_cluster, kde in zip(class_clusters, fitted_kde):

            n_samples = (int(len(class_cluster)/class_data_len*samples_each_class))
            sampled_points = kde.sample(n_samples=n_samples)

            sampled_x.extend(sampled_points)
            sampled_y.extend([class_name]*n_samples)
        x_sampled.extend(sampled_x)
        y_sampled.extend(sampled_y)

    return x_sampled, y_sampled

def plot_data(feats, y_initial_train, nb_classes, cluster_data, fitted_kdes):
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'magenta', 'teal']
    symbols  = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']  
    
    feats = feats.to('cpu').detach().numpy()
    fig = go.Figure()
    pca = PCA(n_components=2, random_state=42)
    feats_transformed = pca.fit_transform(feats)
    for class_idx in range(0,nb_classes):

        transformed_class_data = feats_transformed[y_initial_train == class_idx]

        sampled_x, sampled_y = get_sampled_data_kde(cluster_data, 256//nb_classes, fitted_kdes)
        sampled_transformed = pca.transform(np.array(sampled_x))
        samples_class_data = sampled_transformed[np.array(sampled_y) == class_idx]
        # Plotting the actual class
        fig.add_trace(go.Scatter(
                x=transformed_class_data[:, 0],
                y=transformed_class_data[:, 1],
                mode='markers',
                marker=dict(
                    symbol=symbols[class_idx],
                    size=5,
                    color= colors[class_idx]  # Same color for all clusters in a class
                ),
                name=f'Class {class_idx}'
            ))

        # Plotting the actual class
        fig.add_trace(go.Scatter(
                x=samples_class_data[:, 0],
                y=samples_class_data[:, 1],
                mode='markers',
                marker=dict(
                    symbol=symbols[class_idx],
                    size=5,
                    color= colors[class_idx],
                    line=dict(color='black', width=2)  
                ),
                name=f'Sampled Class {class_idx}'
            ))
    

    # Layout adjustments for labels and aesthetics
    fig.update_layout(
        title="Interactive Visualization of Proxies and Classes with PCA",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title="Classes",
        width=800,
        height=800,
        
    )

    # Display the plot
    fig.show()
    fig.write_image("/netscratch/dmittal/continual_learning/Visualizations/interactive_pca_visualization.png")


def get_sampled_data(cluster_data, samples_each_class):
    x_sampled = []
    y_sampled = []
    for class_name,class_clusters in cluster_data.items() :
        # print(class_name, np.array(class_clusters).shape)
        sampled_x = []
        sampled_y = []
        
        # Calculating the total number instance of each class
        class_data_len = sum(len(cluster) for cluster in class_clusters)
        for class_cluster in class_clusters:

            n_samples = (int(len(class_cluster)/class_data_len*samples_each_class))
            # sampled_points = kde.sample(n_samples=n_samples)

            sampled_x.extend(sampled_points)
            sampled_y.extend([class_name]*n_samples)
        x_sampled.extend(sampled_x)
        y_sampled.extend(sampled_y)

    return x_sampled, y_sampled



def get_cluster_information_raw(x, ys_, sampling):
    x = np.array(x)
    ys_ = np.array(ys_)
    # ys_ = np.array([int(t.item()) for t in ys_])
    classes = set(ys_)
    

    all_class_clusters = dict()
    all_class_clusters_kde_fits = dict()
    for class_idx in classes:

        # Filter data for the current class
        class_data = x[ys_ == class_idx]

        flattened_class_data = class_data.reshape(class_data.shape[0], -1)  

        k = 10  # Number of clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        predicted_clusters = kmeans.fit_predict(flattened_class_data)

    
        class_clusters_data = []
        class_fitted_kdes = []
        # Getting Each Cluster Data for each class
        for cluster_idx in range(len(kmeans.cluster_centers_)):
            cluster_data = flattened_class_data[predicted_clusters == cluster_idx]
            class_clusters_data.append(cluster_data)
            if(sampling == 'kde'):  # USing KDE for sampling
                class_fitted_kdes.append(KernelDensity(kernel='gaussian', bandwidth=0.1).fit(cluster_data))
            else:   # Using Gaussian Distribution
                mean = np.mean(cluster_data, axis=0)  # Shape: (100, 3)
                std_dev = np.std(cluster_data, axis=0)  # Shape: (100, 3)
                class_fitted_kdes.append([mean, 2*std_dev])

        all_class_clusters[class_idx] = class_clusters_data
        all_class_clusters_kde_fits[class_idx] = class_fitted_kdes

    return all_class_clusters, all_class_clusters_kde_fits

def get_sampled_data_kde_raw(cluster_data, samples_each_class, fitted_kdes):

    x_sampled = []
    y_sampled = []
    for (class_name,class_clusters), (_, fitted_kde)  in zip(cluster_data.items(), fitted_kdes.items()):
        # print(class_name, np.array(class_clusters).shape)
        sampled_x = []
        sampled_y = []
        
        # Calculating the total number instance of each class
        class_data_len = sum(len(cluster) for cluster in class_clusters)
        for class_cluster, kde in zip(class_clusters, fitted_kde):

            n_samples = (int(len(class_cluster)/class_data_len*samples_each_class))
            try: # Uses KDE Distribution
                sampled_points = kde.sample(n_samples=n_samples)
            except:# Used Gaussian Distribution
                #kde[0] contains mean and kde[1] contains standard deviation
                sampled_points = np.random.normal(loc=kde[0], scale=kde[1], size=(n_samples, *kde[0].shape))

            sampled_x.extend(sampled_points)
            sampled_y.extend([class_name]*n_samples)
        x_sampled.extend(sampled_x)
        y_sampled.extend(sampled_y)
    
    
    

    return np.array(x_sampled), np.array(y_sampled)

def visualize_sampling_raw(actual_data, y_initial_train, classes, cluster_data, fitted_kdes, method = 'kde'):

    y_initial_train = np.array(y_initial_train)
    actual_data = np.array(actual_data)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'magenta', 'teal']
    symbols  = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']  

    flattened_actual_data = np.array(actual_data).reshape(actual_data.shape[0],-1)

    pca = PCA(n_components=2)  # Reduce to 2D for visualization
    flattened_actual_data_transformed = pca.fit_transform(flattened_actual_data)  # Fit and transform original data
   

    
    fig = go.Figure()
    for class_idx in classes:
        
    
        transformed_class_data = flattened_actual_data_transformed[y_initial_train == class_idx]

        sampled_x, sampled_y = get_sampled_data_kde_raw(cluster_data, 256//len(classes), fitted_kdes)

        flattened_sampled_x = np.array(sampled_x).reshape(sampled_x.shape[0],-1)

        sampled_transformed = pca.transform(np.array(flattened_sampled_x))
        samples_class_data = sampled_transformed[np.array(sampled_y) == class_idx]
        # Plotting the actual class
        fig.add_trace(go.Scatter(
                x=transformed_class_data[:, 0],
                y=transformed_class_data[:, 1],
                mode='markers',
                marker=dict(
                    symbol=symbols[class_idx],
                    size=5,
                    color= colors[class_idx]  # Same color for all clusters in a class
                ),
                # name=f'Actual Class Data'
            ))

        # Plotting the actual class
        fig.add_trace(go.Scatter(
                x=samples_class_data[:, 0],
                y=samples_class_data[:, 1],
                mode='markers',
                marker=dict(
                    symbol=symbols[class_idx],
                    size=5,
                    color= colors[class_idx],
                    line=dict(color='black', width=2),
                      
                ),
                # name=f'Sampled Class Data',
                
            ))
    
    
    # Layout adjustments for labels and aesthetics
    fig.update_layout(
        # title="Interactive Visualization of Proxies and Classes with PCA",
        # xaxis_title="Dimension 1",
        # yaxis_title="Dimension 2",
        # legend_title="Classes",
        showlegend=False,
        width=800,
        height=800,
        # legend=dict(
        #     x=0.2,           # Horizontal position (0 = left, 1 = right)
        #     y=0.95,           # Vertical position (0 = bottom, 1 = top)
        #     xanchor="center", # Anchor point for x position
        #     yanchor="top",    # Anchor point for y position
        #     bgcolor="rgba(255,255,255,0.7)",  # Background color with transparency
        #     bordercolor="black",              # Border color of the legend box
        #     borderwidth=1, # Border width of the legend box
        #     font=dict(size=16),                                      
        # )
    )
        
    

    # Display the plot
    fig.show()
    fig.write_image(f"/netscratch/dmittal/continual_learning/Visualizations/interactive_pca_visualization_raw_method_{method}.png")
    fig.write_image(f"/netscratch/dmittal/continual_learning/Visualizations/interactive_pca_visualization_raw_method_{method}.pdf", format='pdf', engine="kaleido")


def estimate_fisher(model, dataset, n_samples, criterion_pa, ewc_gamma=1.):

    est_fisher_info = {}
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        est_fisher_info[n] = p.detach().clone().zero_()


    mode = model.training
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for index,(x,y,z) in enumerate(data_loader):
        if n_samples is not None:
            if index > n_samples:
                break
        # output = model(x)
        output = model(x.to(device))

        with torch.no_grad():
            # label_weights = F.softmax(output, dim=1)
            label_weights = F.linear(losses.l2_norm(output), losses.l2_norm(criterion_pa.proxies))

        for label_index in range(label_weights.shape[1]):
            # label = torch.LongTensor([label_index])
            # negloglikelihood = F.cross_entropy(output, label)
            negloglikelihood = criterion_pa(output, y.to(device)).to(device)
            model.zero_grad()
            try:
                negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
            except:
                print("",end="\r")
            for n, p in model.named_parameters():
                n = n.replace('.', '__')
                if p.grad is not None:
                    est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

    est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        model.register_buffer('{}_EWC_param_values'.format(n,), p.detach().clone(), persistent = False)
        if hasattr(model, '{}_EWC_estimated_fisher'.format(n)):
            existing_values = getattr(model, '{}_EWC_estimated_fisher'.format(n))
            est_fisher_info[n] += ewc_gamma * existing_values
        model.register_buffer('{}_EWC_estimated_fisher'.format(n), est_fisher_info[n], persistent = False)

    model.train(mode=mode)
    return model