import argparse
import builtins
import os
import glob
import csv
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
from PIL import Image, ImageFilter, ImageFile
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist
from tqdm import tqdm

warnings.simplefilter('ignore', UserWarning)
ver = __file__.replace('.py', '')

### Ignore trunc image
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        image = Image.open(self.paths[i]).convert("RGB")
        image = self.transforms(image)
        return i, image



def extract(args):

    if 'q' in args.target_set:
        query_paths = sorted(Path(args.data).glob('**/*_thumbnail.png'))
        query_ids = np.array([p.stem for p in query_paths], dtype='S6')
    else:
        query_paths = None
        query_ids = None
    if 'r' in args.target_set:
        all_images = sorted(Path(args.data).glob('**/*.png'))
        reference_paths = [
            fn for fn in all_images \
            if not os.path.basename(fn).endswith("thumbnail.png")
        ]
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
        sdtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(out, 'w') as f:
            f.create_dataset('query', data=query_feats)
            f.create_dataset('reference', data=reference_feats)
            f.create_dataset('query_ids', data=query_ids)
            f.create_dataset('reference_ids', data=reference_ids)
            f.create_dataset('query_paths', data=["/".join(p._parts[-2:]) for p in query_paths],
                             dtype=sdtype)
            f.create_dataset('reference_paths', data=["/".join(p._parts[-2:]) for p in reference_paths],
                             dtype=sdtype)

        # ngpu = -1 if 'A100' not in torch.cuda.get_device_name() else 0
        # subprocess.run(
        #     f'python ../scripts/eval_metrics.py {ver}/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv --ngpu {ngpu}', shell=True)

    if 't' in args.target_set:
        train_feats = calc_feats(data_loaders['train'])
        np.save(f'{ver}/extract/train_feats_phase2.npy', train_feats)



if __name__ == '__main__':
    if not Path(f'{ver}/extract').exists():
        Path(f'{ver}/extract').mkdir(parents=True)

    args = parser.parse_args()

    extract(args)
