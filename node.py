import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import norm, ResBlock, conv1x1, Flatten, conv3x3, ConcatConv2d
from dataloaders import get_cifar_loaders, get_mnist_loaders
from utils import get_logger, makedirs, count_parameters, inf_generator, learning_rate_with_decay, RunningAverageMeter, accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='cifar')
parser.add_argument('--save', type=str, default='./experiment_node1')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# class ConcatConv2d(nn.Module):

#     def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
#         super(ConcatConv2d, self).__init__()
#         module = nn.ConvTranspose2d if transpose else nn.Conv2d
#         self._layer = module(
#             dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
#             bias=bias
#         )

#     def forward(self, t, x):
#         tt = torch.ones_like(x[:, :1, :, :]) * t
#         ttx = torch.cat([tt, x], 1)
#         return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'
    in_channel = 3
    if args.dataset == 'mnist':
        in_channel = 1

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(in_channel, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(in_channel, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    if args.dataset == 'mnist':
        train_loader, test_loader, train_eval_loader = get_mnist_loaders(
            args.data_aug, args.batch_size, args.test_batch_size
        )
    elif args.dataset == 'cifar':
        train_loader, test_loader, train_eval_loader = get_cifar_loaders(
            args.data_aug, args.batch_size, args.test_batch_size
        )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001], lr=args.lr
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
    
    epoch_arr = []
    time_val_arr = []
    time_avg_arr = []
    nfe_f_arr = []
    nfe_b_arr = []
    train_acc_arr = []
    test_acc_arr = []

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader, device)
                val_acc = accuracy(model, test_loader, device)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )
                epoch_arr += [itr // batches_per_epoch]
                time_val_arr += [batch_time_meter.val]
                time_avg_arr += [batch_time_meter.avg]
                nfe_f_arr += [f_nfe_meter.avg]
                nfe_b_arr += [b_nfe_meter.avg]
                train_acc_arr += [train_acc]
                test_acc_arr += [val_acc]
                    
    epoch_arr = np.asarray(epoch_arr)
    time_val_arr = np.asarray(time_val_arr)
    time_avg_arr = np.asarray(time_avg_arr)
    nfe_f_arr = np.asarray(nfe_f_arr)
    nfe_b_arr = np.asarray(nfe_b_arr)
    train_acc_arr = np.asarray(train_acc_arr)
    test_acc_arr = np.asarray(test_acc_arr)
    
    np.save(os.path.join(args.save, 'epoch_arr.npy'), epoch_arr)
    np.save(os.path.join(args.save, 'time_val_arr.npy'), time_val_arr)
    np.save(os.path.join(args.save, 'time_avg_arr.npy'), time_avg_arr)
    np.save(os.path.join(args.save, 'nfe_f_arr.npy'), nfe_f_arr)
    np.save(os.path.join(args.save, 'nfe_b_arr.npy'), nfe_b_arr)
    np.save(os.path.join(args.save, 'train_acc_arr.npy'), train_acc_arr)
    np.save(os.path.join(args.save, 'test_acc_arr.npy'), test_acc_arr)
