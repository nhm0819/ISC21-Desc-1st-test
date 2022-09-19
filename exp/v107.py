import argparse
import builtins
import os
import pickle
import random
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from augly.image.functional import overlay_emoji, overlay_image, overlay_text
from augly.image.transforms import BaseTransform
from augly.utils import pathmgr
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR, SMILEY_EMOJI_DIR
from PIL import Image, ImageFilter
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist
from tqdm import tqdm

warnings.simplefilter('ignore', UserWarning)
ver = __file__.replace('.py', '')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=os.cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--gem-p', default=3.0, type=float)
parser.add_argument('--gem-eval-p', default=4.0, type=float)

parser.add_argument('--mode', default='train', type=str, help='train or extract')
parser.add_argument('--target-set', default='qr', type=str, help='q: query, r: reference, t: training')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--pos-margin', default=0.0, type=float)
parser.add_argument('--neg-margin', default=0.7, type=float)
parser.add_argument('--ncrops', default=2, type=int)
parser.add_argument('--input-size', default=224, type=int)
parser.add_argument('--sample-size', default=100000, type=int)
parser.add_argument('--weight', type=str)
parser.add_argument('--eval-subset', action='store_true')
parser.add_argument('--memory-size', default=1024, type=int)
parser.add_argument('--tta', action='store_true')


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class ISCNet(nn.Module):

    def __init__(self, backbone, fc_dim=256, p=3.0, eval_p=4.0):
        super().__init__()

        self.backbone = backbone

        self.fc = nn.Linear(self.backbone.feature_info.info[-1]['num_chs'], fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.eval_p = eval_p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)[-1]
        p = self.p if self.training else self.eval_p
        x = gem(x, p).view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x)
        return x


class ISCTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        ref_paths,
        ref_paths2,
        transforms,
        num_negatives,
    ):
        self.paths = paths
        self.ref_paths = ref_paths
        self.ref_paths2 = ref_paths2
        self.transforms = transforms
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        ref_path = self.ref_paths[i]
        image = Image.open(path)
        image = self.transforms(image)
        ref_image = Image.open(ref_path)
        ref_image = self.transforms(ref_image)

        js = [random.choice(range(len(self.ref_paths2))) for _ in range(self.num_negatives)]

        ret = [
            image,
            ref_image,
            *[self.transforms(Image.open(self.ref_paths2[j])) for j in js],
        ]

        return i, ret, [j + 1000000 for j in js]


class ISCTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        transforms,
    ):
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i])
        image = self.transforms(image)
        return i, image


def train(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier(device_ids=[args.gpu])

    backbone = timm.create_model(args.arch, features_only=True, pretrained=True)
    model = ISCNet(backbone, p=args.gem_p, eval_p=args.gem_eval_p)

    if args.weight is not None:
        state_dict = torch.load(args.weight, map_location='cpu')['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = state_dict[k]
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    # infer learning rate before changing batch size
    init_lr = args.lr  # * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    loss_fn = losses.ContrastiveLoss(pos_margin=args.pos_margin, neg_margin=args.neg_margin)
    loss_fn = losses.CrossBatchMemory(loss_fn, embedding_size=256, memory_size=args.memory_size)
    loss_fn = pml_dist.DistributedLossWrapper(loss=loss_fn, device_ids=[args.rank])

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or "gain" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optim_params = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': args.weight_decay}
    ]

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum)
    scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    gt = pd.read_csv('../input/public_ground_truth.csv')
    gt_ = gt[gt['reference_id'].notna()]
    query_paths = [f'../input/query_images/{qid}.jpg' for qid in gt_['query_id']]
    reference_paths = [f'../input/reference_images/{rid}.jpg' for rid in gt_['reference_id']]
    all_ref_ids = set([f'R{i:06d}' for i in range(1000000)])
    diff_ref_ids = all_ref_ids - set(gt_['reference_id'])
    diff_reference_paths = [f'../input/reference_images/{rid}.jpg' for rid in diff_ref_ids]

    preprocess = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=backbone.default_cfg['mean'], std=backbone.default_cfg['std'])
    ])

    train_dataset = ISCTrainDataset(
        query_paths,
        reference_paths,
        diff_reference_paths,
        preprocess,
        num_negatives=torch.cuda.device_count() * 2 - 2,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        train_one_epoch(train_loader, model, loss_fn, optimizer, scaler, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, is_best=False, filename=f'{ver}/train/checkpoint_{epoch:04d}.pth.tar')


def train_one_epoch(train_loader, model, loss_fn, optimizer, scaler, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    progress = tqdm(train_loader, desc=f'epoch {epoch}', leave=False, total=len(train_loader))

    model.train()

    for i, images, j in progress:
        optimizer.zero_grad()

        labels = torch.cat([torch.tile(i, dims=(2,)), torch.tensor(j)])
        labels = labels.cuda(args.gpu, non_blocking=True)
        images = torch.cat([
            image for image in images
        ], dim=0).cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

        losses.update(loss.item(), images.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress.set_postfix(loss=losses.avg)

    print(f'epoch={epoch}, loss={losses.avg}')


def extract(args):

    if 'q' in args.target_set:
        query_paths = sorted(Path(args.data).glob('query_images_phase2/**/*.jpg'))
        query_ids = np.array([p.stem for p in query_paths], dtype='S6')
    else:
        query_paths = None
        query_ids = None
    if 'r' in args.target_set:
        reference_paths = sorted(Path(args.data).glob('reference_images/**/*.jpg'))
        reference_ids = np.array([p.stem for p in reference_paths], dtype='S7')
    else:
        reference_paths = None
        reference_ids = None
    if 't' in args.target_set:
        train_paths = sorted(Path(args.data).glob('training_images/**/*.jpg'))
    else:
        train_paths = None

    if args.eval_subset:
        with open('../input/rids_subset.pickle', 'rb') as f:
            rids_subset = pickle.load(f)
        isin_subset = np.isin(reference_ids, rids_subset)
        reference_paths = np.array(reference_paths)[isin_subset]
        reference_ids = np.array(reference_ids)[isin_subset]
        assert len(reference_paths) == len(reference_paths) == len(rids_subset)

    if args.dryrun:
        query_paths = query_paths[:100]
        reference_paths = reference_paths[:100]

    backbone = timm.create_model(args.arch, features_only=True, pretrained=True)
    model = ISCNet(backbone, p=args.gem_p, eval_p=args.gem_eval_p)
    model = nn.DataParallel(model)

    state_dict = torch.load(args.weight, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=False)

    model.eval().cuda()

    cudnn.benchmark = True

    if args.tta:
        preprocesses = [
            transforms.Resize((int(args.input_size * 1.4142135623730951), int(args.input_size * 1.4142135623730951))),
            transforms.ToTensor(),
            transforms.Normalize(mean=backbone.default_cfg['mean'], std=backbone.default_cfg['std'])
        ]
    else:
        preprocesses = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=backbone.default_cfg['mean'], std=backbone.default_cfg['std'])
        ]

    datasets = {
        'query': ISCTestDataset(query_paths, transforms.Compose(preprocesses)),
        'reference': ISCTestDataset(reference_paths, transforms.Compose(preprocesses)),
        'train': ISCTestDataset(train_paths, transforms.Compose(preprocesses)),
    }
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    data_loaders = {
        'query': torch.utils.data.DataLoader(datasets['query'], **loader_kwargs),
        'reference': torch.utils.data.DataLoader(datasets['reference'], **loader_kwargs),
        'train': torch.utils.data.DataLoader(datasets['train'], **loader_kwargs),
    }

    def calc_feats(loader):
        feats = []
        for _, image in tqdm(loader, total=len(loader)):
            image = image.cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                if args.tta:
                    image_big = image
                    image = F.interpolate(image_big, size=args.input_size, mode='bilinear', align_corners=False)
                    image_small = F.interpolate(image, scale_factor=0.7071067811865475, mode='bilinear', align_corners=False)
                    f = (
                        model(image) + model(image_small) + model(image_big)
                        + model(transforms.functional.hflip(image))
                        + model(transforms.functional.hflip(image_small))
                        + model(transforms.functional.hflip(image_big))
                    )
                    f /= torch.linalg.norm(f, dim=1, keepdim=True)
                else:
                    f = model(image)
            feats.append(f.cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        return feats.astype(np.float32)

    if 'q' in args.target_set:
        query_feats = calc_feats(data_loaders['query'])
        reference_feats = calc_feats(data_loaders['reference'])

        out = f'{ver}/extract/fb-isc-submission-phase2.h5'
        with h5py.File(out, 'w') as f:
            f.create_dataset('query', data=query_feats)
            f.create_dataset('reference', data=reference_feats)
            f.create_dataset('query_ids', data=query_ids)
            f.create_dataset('reference_ids', data=reference_ids)

        ngpu = -1 if 'A100' not in torch.cuda.get_device_name() else 0
        subprocess.run(
            f'python ../scripts/eval_metrics.py {ver}/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv --ngpu {ngpu}', shell=True)

    if 't' in args.target_set:
        train_feats = calc_feats(data_loaders['train'])
        np.save(f'{ver}/extract/train_feats_phase2.npy', train_feats)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * (1 - (epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    if not Path(f'{ver}/train').exists():
        Path(f'{ver}/train').mkdir(parents=True)
    if not Path(f'{ver}/extract').exists():
        Path(f'{ver}/extract').mkdir(parents=True)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'extract':
        extract(args)
