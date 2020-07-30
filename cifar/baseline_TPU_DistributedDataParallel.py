

"""
Baseline XLA CIFAR-10 training script. 
Can use multiple TPU Devices.

Simply running "CUDA_VISIBLE_DEVICES=X python3 baseline.py" should get you the following results for the first few epochs:

(base) sauravkadavath@shadowfax:~/nnets-playground/cifar:[master !?]$ CUDA_VISIBLE_DEVICES=0 python3 baseline.py
{'batch_size': 128, 'dataset': 'cifar10', 'decay': 0.0005, 'droprate': 0.0, 'epochs': 100, 'layers': 40, 'learning_rate': 0.1, 'load': '', 'model': 'wrn', 'momentum': 0.9, 'ngpu': 1, 'prefetch': 4, 'save': './snapshots/rot_five', 'test': False, 'test_bs': 200, 'widen_factor': 2}
Beginning Training

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:20<00:00, 19.36it/s]
Epoch   1 | Time    22 | Train Loss 1.0872 | Test Loss 1.371 | Test Error 46.04
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:19<00:00, 20.58it/s]
Epoch   2 | Time    20 | Train Loss 0.7776 | Test Loss 1.045 | Test Error 35.85
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:18<00:00, 21.26it/s]
Epoch   3 | Time    20 | Train Loss 0.6481 | Test Loss 1.138 | Test Error 35.28
"""

# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet

import torch_xla
import torch_xla.debug.metrics as xmetrics
import torch_xla.distributed.data_parallel as xdp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['wrn'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/TEMP', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--nprocs', type=int, default=1, help='Number of TPUs to distribute over')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

torch.manual_seed(1)
np.random.seed(1)

writer = SummaryWriter(os.path.join(args.save, "tensorboard_dir"))

# # mean and standard deviation of channels of CIFAR-10 images
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4), trn.ToTensor()])
test_transform = trn.Compose([trn.ToTensor()])

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + args.model +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')


# /////////////// Training ///////////////


def train(train_loader, net, optimizer, scheduler, xla_device, args):

    parallel_loader = pl.ParallelLoader(train_loader, [xla_device]).per_device_loader(xla_device)

    net = net.train().to(xla_device) # enter train mode

    loss_avg = 0.0
    for i, (bx, by) in enumerate(parallel_loader):
        print("Doing minibatch", i)
        curr_batch_size = bx.size(0)

        optimizer.zero_grad()

        # forward
        logits = net(bx * 2 - 1)
        loss = F.cross_entropy(logits, by)
        # backward
        loss.backward()
        xm.optimizer_step(optimizer)

        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1

    # state['train_loss'] = loss_avg
    return loss_avg

def test(test_loader, net, xla_device, args):
    loss_total = 0.0
    correct = 0
    guesses = 0

    parallel_loader = pl.ParallelLoader(test_loader, [xla_device]).per_device_loader(xla_device)
    net = net.eval().to(xla_device)

    with torch.no_grad():
        for data, target in test_loader:
            batch_size = data.shape[0]
            data, target = data.to(xla_device), target.to(xla_device)

            # forward
            output = net(data * 2 - 1)
            loss = F.cross_entropy(output, target, reduction='sum')

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_total += float(loss.data)

            guesses += batch_size

    return loss_total, correct, guesses


def main(index, args):
    if xm.is_master_ordinal():
        print(state)

    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    xla_device = xm.xla_device()  

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10('~/cifarpy/', train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR10('~/cifarpy/', train=False, download=True, transform=test_transform)
        num_classes = 10
    else:
        train_data = dset.CIFAR100('~/cifarpy/', train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR100('~/cifarpy/', train=False, download=True, transform=test_transform)
        num_classes = 100
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size,
        num_workers=args.prefetch, 
        drop_last=True, 
        sampler=train_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.test_bs,
        num_workers=args.prefetch, 
        drop_last=True, 
        sampler=test_sampler
    )

    # Create model
    if args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).train().to(xla_device)
    else:
        raise NotImplementedError()

    start_epoch = 0

    optimizer = torch.optim.SGD(
        net.parameters(), 
        args.learning_rate, 
        momentum=args.momentum,
        weight_decay=args.decay, 
        nesterov=True
    )

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    print('Beginning Training')

    # Main loop
    for epoch in range(start_epoch, args.epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()

        # Spawn a bunch of processes, one for each TPU core.
        train_loss = train(train_loader, net, optimizer, scheduler, xla_device, args)

        # Calculate test loss
        test_results = test(test_loader, net, xla_device, args)

        # TODO: Make this work.
        all_train_losses = xm.rendezvous("calc_train_loss", payload=str(train_loss))
        all_test_results = xm.rendezvous("calc_test_results", payload=str(test_results))
        all_test_results = parse_test_results(all_test_results)

        if xm.is_master_ordinal():
            child_train_losses = [float(L) for L in all_train_losses]
            train_loss = sum(child_train_losses) / float(len(train_losses))
            state['train_loss'] = train_loss

            test_loss = sum([r[0] for r in all_test_results]) / sum([r[2] for r in all_test_results])
            test_acc = sum([r[1] for r in all_test_results]) / sum([r[2] for r in all_test_results])
            state['test_loss'] = test_loss
            state['test_accuracy'] = test_acc

            # Save model
            xm.save(
                net.state_dict(),
                os.path.join(
                    args.save, 
                    args.dataset + args.model + '_baseline_epoch_' + str(epoch) + '.pt'
                )
            )

            # Let us not waste space and delete the previous model
            prev_path = os.path.join(
                args.save, args.dataset + args.model +
                '_baseline_epoch_' + str(epoch - 1) + '.pt'
            )
            if os.path.exists(prev_path): os.remove(prev_path)

            # Show results
            with open(os.path.join(args.save, args.dataset + args.model +
                                            '_baseline_training_results.csv'), 'a') as f:
                f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                    (epoch + 1),
                    time.time() - begin_epoch,
                    state['train_loss'],
                    state['test_loss'],
                    100 - 100. * state['test_accuracy'],
                ))

            xm.print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'])
            )

            writer.add_scalar("test_loss", state["test_loss"], epoch + 1)
            writer.add_scalar("test_accuracy", state["test_accuracy"], epoch + 1)
        
        # Wait for master to finish Disk I/O above
        xm.rendezvous("epoch_finish")


def parse_test_results(results):
    parsed = []
    for result in results:
        result = str(result)
        result = result.split(',')
        result[0] = result[0][1:]
        result[-1] = result[-1][:-1]
        parsed.append([float(r) for r in results])
    
    return parsed

if __name__ == "__main__":
    xmp.spawn(main, args=(args,), nprocs=args.nprocs)