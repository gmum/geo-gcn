import argparse
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MNISTSuperpixels

from architectures import SGCN

parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str, default='MNISTSuperpixels')
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--train_augmentation', type=bool, default=False)
parser.add_argument('--layers_num', type=int, default=3)
parser.add_argument('--model_dim', type=int, default=16)
parser.add_argument('--out_channels_1', type=int, default=64)
parser.add_argument('--use_cluster_pooling', type=bool, default=True)
parser.add_argument('--dim_coor', type=int, default=2)
parser.add_argument('--label_dim', type=int, default=1)
parser.add_argument('--out_dim', type=int, default=10)

args = parser.parse_args()

print(args.use_cluster_pooling)

if args.dataset_name == 'MNISTSuperpixels':
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'MNIST')
    train_dataset = MNISTSuperpixels(path, True, transform=T.Cartesian())
    test_dataset = MNISTSuperpixels(path, False, transform=T.Cartesian())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SGCN(dim_coor=args.dim_coor,
             out_dim=args.out_dim,
             input_features=args.label_dim,
             layers_num=args.layers_num,
             model_dim=args.model_dim,
             out_channels_1=args.out_channels_1,
             dropout=args.dropout,
             use_cluster_pooling=args.use_cluster_pooling).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

rotation_0 = T.RandomRotate(degrees=180, axis=0)
rotation_1 = T.RandomRotate(degrees=180, axis=1)
rotation_2 = T.RandomRotate(degrees=180, axis=2)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if args.train_augmentation:
            data = rotation_0(data)
            data = rotation_1(data)
            data = rotation_2(data)
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


train_acc_array = []
test_acc_array = []

for epoch in range(1, args.num_epoch):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    train_acc_array.append(train_acc)
    test_acc_array.append(test_acc)

    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, train_acc, test_acc))
