"""
Also, we should train a ResNet-50 + DeepAugment (EDSR) and ResNet-50 + DeepAugment (CAE) for an ablation on ImageNet-1K.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np
from PIL import ImageOps, Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

import torch_xla
import torch_xla.debug.metrics as xmetrics
import torch_xla.distributed.data_parallel as xdp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='DeepAugment ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--save', type=str, required=True)
parser.add_argument('--extra-dataset', action='append', choices=['edsr', 'cae', 'noise2net'], default=[], required=False)
parser.add_argument('--noisenet-max-eps', default=0.75, type=float)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', choices=['cpu', 'gpu', 'tpu'])
parser.add_argument('--nprocs', type=int, default=1, help='Number of TPUs or GPUs to distribute over')

args = parser.parse_args()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


best_acc1 = 0


def main():
    if os.path.exists(args.save):
        resp = "None"
        while resp.lower() not in {'y', 'n'}:
            resp = input("Save directory {0} exits. Continue? [Y/n]: ".format(args.save))
            if resp.lower() == 'y':
                break
            elif resp.lower() == 'n':
                exit(1)
            else:
                pass
    else:
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if not os.path.isdir(args.save):
            raise Exception('%s is not a dir' % args.save)
        else:
            print("Made save directory", args.save)

    xmp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

def main_worker(index, ngpus_per_node, args):
    global best_acc1

    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    DEVICE = xm.xla_device()
    
    print("DEVICE = ", DEVICE)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating NON pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False)
    
    model = model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if False:#args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            print('Start epoch:', args.start_epoch)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_datasets = []

    train_datasets.append(datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    )

    if 'edsr' in args.extra_dataset:
        print("Adding EDSR Data")
        train_datasets.append(datasets.ImageFolder(
            '/data/hendrycks/deepaugment/EDSR',
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        )
    
    if 'cae' in args.extra_dataset:
        print("Adding CAE Data")
        train_datasets.append(datasets.ImageFolder(
            '/data/hendrycks/deepaugment/CAE',
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        )

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=ngpus_per_node,
        rank=index,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / (args.lr * args.batch_size / 256.)))
    if args.start_epoch != 0:
        scheduler.step(args.start_epoch * len(train_loader))

    if args.evaluate:
        validate(val_loader, model, args)
        return
    
    if not args.resume:
        with open(os.path.join(args.save, 'training_log.csv'), 'w') as f:
            f.write('epoch,train_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5,imagenetR_acc1\n')

        with open(os.path.join(args.save, 'command.txt'), 'w') as f:
            import pprint
            to_print = vars(args)
            to_print['FILENAME'] = __file__
            pprint.pprint(to_print, stream=f)
    else:
        with open(os.path.join(args.save, 'command_resume.txt'), 'w') as f:
            import pprint
            to_print = vars(args)
            to_print['FILENAME'] = __file__
            pprint.pprint(to_print, stream=f)

    print("BEGINNING TRAINING")

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, args, DEVICE)

        # evaluate on validation set
        acc1 = validate(val_loader, model, args, DEVICE)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, optimizer, scheduler, epoch, args, DEVICE):    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model = model.train().to(DEVICE)

    loader = pl.ParallelLoader(train_loader, [DEVICE]).per_device_loader(DEVICE)
    # noise2net = Res2Net(epsilon=0.50, hidden_planes=16, batch_size=args.batch_size).train().to(DEVICE)

    end = time.time()
    for i, (images, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        bx = images
        by = target

        print("Zero grad")
        optimizer.zero_grad()

        # with torch.no_grad():
        #     if random.random() < 0.5:
        #         batch_size = bx.shape[0]
        #         noise2net.reload_parameters()
        #         noise2net.set_epsilon(random.uniform(args.noisenet_max_eps / 2.0, args.noisenet_max_eps))
        #         bx = bx.reshape((1, batch_size * 3, 224, 224))
        #         bx = noise2net(bx)
        #         bx = bx.reshape((batch_size, 3, 224, 224))

        print("Forward")
        logits = model(bx)

        print("Cross Entropy")
        loss = F.cross_entropy(logits, by)

        # measure accuracy and record loss
        output, target = logits, by
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        print("Backward")
        loss.backward()

        print("Step")
        xm.optimizer_step(optimizer)
        
        print("Scheduler step")
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and xm.is_master_ordinal():
            progress.display(i)


def validate(val_loader, model, args, DEVICE):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model = model.eval().to(DEVICE)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(DEVICE)
            target = target.to(DEVICE)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename=os.path.join(args.save, 'checkpoint.pth.tar')):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, './model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

########################################################################################################
### Noise2Net
########################################################################################################

import sys
import os
import numpy as np
import os
import shutil
import tempfile
from PIL import Image
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as trn
from torchvision import datasets
import torchvision.transforms.functional as trnF 
from torch.nn.functional import gelu, conv2d
import torch.nn.functional as F
import random
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class GELU(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, hidden_planes=9, scale = 4, batch_size=5):
        """ Constructor
        Args:
            inplanes: input channel dimensionality (multiply by batch_size)
            planes: output channel dimensionality (multiply by batch_size)
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = hidden_planes * batch_size
        self.conv1 = nn.Conv2d(inplanes * batch_size, width*scale, kernel_size=1, bias=False, groups=batch_size)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        
        convs = []
        bns = []
        for i in range(self.nums):
            K = random.choice([1, 3])
            D = random.choice([1, 2, 3])
            P = int(((K - 1) / 2) * D)

            convs.append(nn.Conv2d(width, width, kernel_size=K, stride = stride, padding=P, dilation=D, bias=True, groups=batch_size))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * batch_size, kernel_size=1, bias=False, groups=batch_size)
        self.bn3 = nn.BatchNorm2d(planes * batch_size)

        self.act = nn.ReLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.hidden_planes = hidden_planes
        self.batch_size = batch_size

    def forward(self, x):
        _, _, H, W = x.shape
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out) # [1, hidden_planes*batch_size*scale, H, W]
        
        # Hack to make different scales work with the hacky batches
        out = out.view(1, self.batch_size, self.scale, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)
        
        spx = torch.split(out, self.width, 1) # [ ... (1, hidden_planes*batch_size, H, W) ... ]
        
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.act(self.bns[i](sp))
          
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]),1)
        
        # Undo hack to make different scales work with the hacky batches
        out = out.view(1, self.scale, self.batch_size, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class Res2Net(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=16, batch_size=5):
        super(Res2Net, self).__init__()
        
        self.epsilon = epsilon
        self.hidden_planes = hidden_planes
                
        self.block1 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block2 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block3 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block4 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)

    def reload_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()
 
    def set_epsilon(self, new_eps):
        self.epsilon = new_eps

    def forward_original(self, x):                
        x = (self.block1(x) * self.epsilon) + x
        x = (self.block2(x) * self.epsilon) + x
        x = (self.block3(x) * self.epsilon) + x
        x = (self.block4(x) * self.epsilon) + x
        return x

    def forward(self, x):
        return self.forward_original(x)


if __name__ == '__main__':
    main()
