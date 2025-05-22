import torch
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset,ConcatDataset

class train_ewc_opt(torch.nn.Module):

    def __init__(self):
        super(train_ewc_opt, self).__init__()

    def train_ewc(self, model, dataset, iters, lr, batch_size, current_context, ewc_lambda=100.):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        model.train()
        iters_left = 1
        progress_bar = tqdm.tqdm(range(1, iters+1))

        for batch_index in range(1, iters+1):
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                           shuffle=True, drop_last=True))
                iters_left = len(data_loader)
            x, y = next(data_loader)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = torch.nn.functional.cross_entropy(input=y_hat, target=y, reduction='mean')

            if current_context>1:
                ewc_losses = []
                for n, p in model.named_parameters():
                    n = n.replace('.', '__')
                    mean = getattr(model, '{}_EWC_param_values'.format(n))
                    fisher = getattr(model, '{}_EWC_estimated_fisher'.format(n))
                    ewc_losses.append((fisher * (p-mean)**2).sum())
                ewc_loss = (1./2)*sum(ewc_losses)
                total_loss = loss + ewc_lambda*ewc_loss
            else:
                total_loss = loss

            accuracy = (y == y_hat.max(1)[1]).sum().item()*100 / x.size(0)
            total_loss.backward()
            optimizer.step()
            progress_bar.set_description(
            '<CLASSIFIER> | training loss: {loss:.3} | training accuracy: {prec:.3}% |'
                .format(loss=total_loss.item(), prec=accuracy)
            )
            progress_bar.update(1)
        progress_bar.close()
        return model
    
    def train_ewc_incremental_model(self,train_input, train_label, test_input, test_label, model,ewc_lambda=100,old_dataset=None):
        # net = ChannelFusionModelSearch(n_filters=self.n_filter, filter_size=self.kernel_size, n_class=self.n_classes, n_dense=128, net_config=self.net_config)

        dataset = {}
        dataset['train'] = TensorDataset(torch.from_numpy(train_input).float(),torch.from_numpy(train_label).long())
        dataset['test'] = TensorDataset(torch.from_numpy(test_input).float(),torch.from_numpy(test_label).long())

        if old_dataset == None:
            updated_model = self.train_ewc(model, dataset['train'], iters=400, lr=0.1, batch_size=512, current_context=1, ewc_lambda=ewc_lambda)
        else:
            ewc_model = self.estimate_fisher(model, old_dataset, n_samples=5000)
            updated_model = self.train_ewc(ewc_model, dataset['train'], iters=100, lr=0.1, batch_size=512, current_context=2, ewc_lambda=ewc_lambda)
    
        return updated_model, dataset
    
    def estimate_fisher(self, model, dataset, n_samples, ewc_gamma=1.):

        est_fisher_info = {}
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            est_fisher_info[n] = p.detach().clone().zero_()


        mode = model.training
        model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        for index,(x,y) in enumerate(data_loader):
            if n_samples is not None:
                if index > n_samples:
                    break
            output = model(x)
            with torch.no_grad():
                label_weights = F.softmax(output, dim=1)
            for label_index in range(output.shape[1]):
                label = torch.LongTensor([label_index])
                negloglikelihood = F.cross_entropy(output, label)
                model.zero_grad()
                negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
                for n, p in model.named_parameters():
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_EWC_param_values'.format(n,), p.detach().clone())
            if hasattr(model, '{}_EWC_estimated_fisher'.format(n)):
                existing_values = getattr(model, '{}_EWC_estimated_fisher'.format(n))
                est_fisher_info[n] += ewc_gamma * existing_values
            model.register_buffer('{}_EWC_estimated_fisher'.format(n), est_fisher_info[n])

        model.train(mode=mode)
        return model


    
    