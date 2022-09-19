import argparse
import os
import glob
import csv
import pickle
import warnings
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from PIL import Image, ImageFilter, ImageFile

from tqdm import tqdm

warnings.simplefilter('ignore', UserWarning)
ver = __file__.replace('.py', '')

### Ignore trunc image
# ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-j', '--workers', default=os.cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--target-set', default='qr', type=str, help='q: query, r: reference, t: training')


class ISCTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths
    ):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            image = Image.open(self.paths[i]).convert("RGB")
        except:
            print(self.paths[i])
            return i, self.paths[i]
        return i, ""



def check(args):

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
            if not os.path.basename(fn).endswith("_thumbnail.png") or \
            not os.path.basename(fn).endswith("_thumbnail.jpg")
        ]
        reference_ids = np.array([p.stem for p in reference_paths], dtype='S7')
    else:
        reference_paths = None
        reference_ids = None

    datasets = {
        'query': ISCTestDataset(query_paths),
        'reference': ISCTestDataset(reference_paths),
    }
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    data_loaders = {
        'query': torch.utils.data.DataLoader(datasets['query'], **loader_kwargs),
        'reference': torch.utils.data.DataLoader(datasets['reference'], **loader_kwargs)
    }

    def check_image(loader):
        paths = []
        for _, path in tqdm(loader, total=len(loader)):
            if path != "":
                paths.append(path)
        return paths

    if 'q' in args.target_set:
        query_paths = check_image(data_loaders['query'])
        reference_paths = check_image(data_loaders['reference'])

    out = f'trunc_list.h5'
    with h5py.File(out, 'w') as f:
        f.create_dataset('query', data=query_paths)
        f.create_dataset('reference', data=reference_paths)
        f.create_dataset('query_ids', data=query_ids)
        f.create_dataset('reference_ids', data=reference_ids)




if __name__ == '__main__':

    args = parser.parse_args()

    check(args)
