import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
#from torch_geometric.loader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.model_selection import StratifiedKFold

def separate_data(dataset_len, seed=0):
    # Use same splitting/10-fold as GIN paper
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    idx_list = []
    for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
        idx_list.append(idx)
    return idx_list

def train_loop(model, epoch, loader, optimizer, device):
    model.train()
    loss_all = 0
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logs, aux_logs = model(data)
        loss = F.nll_loss(logs, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        n += len(data.y)
        optimizer.step()
    return loss_all / n

def val(loader, device):
    model.eval()
    with torch.no_grad():
        loss_all = 0
        for data in loader:
            data = data.to(device)
            logs, aux_logs = model(data)
            loss_all += F.nll_loss(logs, data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data in loader:
            data = data.to(device)
            logs, aux_logs = model(data)
            pred = logs.max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def train(dataset, model, batch_size=32):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    acc = []
    splits = separate_data(len(dataset), seed=0)
    for i, (train_idx, test_idx) in enumerate(splits):
        model.reset_parameters()
        lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        test_dataset = dataset[test_idx.tolist()]
        train_dataset = dataset[train_idx.tolist()]

        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset)*50/(len(train_dataset)/batch_size))), batch_size=batch_size, drop_last=False) #, collate_fn=Collater(follow_batch=[],exclude_keys=[]))	# GIN like epochs/batches - they do 50 radom batches per epoch

        print('---------------- Split {} ----------------'.format(i), flush=True)

        test_acc = 0
        acc_temp = []
        for epoch in range(1, 350+1):
            if epoch == 350:
                start = time.time()
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train_loop(model, epoch, train_loader, optimizer, device)
            scheduler.step()
            test_acc = test(model, test_loader, device)
            if epoch == 350:
                print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:7f}'.format(
                        epoch, lr, train_loss, 0, test_acc, time.time() - start), flush=True)
            acc_temp.append(test_acc)
        acc.append(torch.tensor(acc_temp))
    acc = torch.stack(acc, dim=0)
    acc_mean = acc.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    print('---------------- Final Epoch Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,-1].mean(), acc[:,-1].std()))
    print(f'---------------- Best Epoch: {best_epoch} ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,best_epoch].mean(), acc[:,best_epoch].std()), flush=True)

