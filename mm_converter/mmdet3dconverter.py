import copy
import os
import sys
from pathlib import Path
import argparse
from mmcv import Config
from mmdet.datasets import build_dataset, replace_ImageToTensor
import pickle
from collections.abc import Sequence
from ffrecord.utils import dump as ffdump


class MM2FFR(Sequence):
    def __init__(self, dataset, out_dir, nfiles, dump_key="img_filename"):
        self.dataset = dataset
        self.out_dir = out_dir
        self.nfiles = nfiles

        self.filename2idx = {}
        self.all_filenames = []
        fileid = 0
        # build id to file name mapping, dump img to ffr file
        for i in range(len(dataset)):
            # print(dataset[i])
            filenames = dataset[i][dump_key]
            if isinstance(filenames, str):
                filenames = [filenames]
            for name in filenames:
                self.filename2idx[os.path.basename(name)] = fileid
                self.all_filenames.append(name)
                fileid += 1
        self.img_len = len(self.all_filenames)

    def __len__(self):
        return self.img_len

    def __getitem__(self, idx):
        if idx < self.img_len:
            fname = self.all_filenames[idx]
            return self.read_img_bytes(fname)
        else:
            raise ValueError(f"{idx} is max than {self.img_num}")

    def read_img_bytes(self, filepath):
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf

    def dump_to_ffr(self):
        print(f'Convert dataset to {self.out_dir}, nfiles {self.nfiles}')
        ffdump(dataset=self, fname=self.out_dir, nfiles=self.nfiles, verbose=True)
        with open(self.out_dir / "filename2idx.pkl", "wb") as f:
            pickle.dump(self.filename2idx, f)


def main(args):
    datasets = create_datasets(args)
    MM2FFR(datasets[0], Path(args.out_dir) / "train", nfiles=50, dump_key=args.dump_key).dump_to_ffr()
    MM2FFR(datasets[1], Path(args.out_dir) / "val", nfiles=15, dump_key=args.dump_key).dump_to_ffr()


def create_datasets(args):
    config_file = args.config_file
    out_dir = args.out_dir

    cfg = Config.fromfile(config_file)
    out_dir = Path(out_dir)

    cfg.data.train.test_mode = True
    cfg.data.train.pipeline = []
    cfg.data.val.pipeline = []
    datasets = [build_dataset(cfg.data.train, dict(test_mode=True))]
    # Support batch_size > 1 in validation
    val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
    if val_samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
    datasets.append(build_dataset(cfg.data.val, dict(test_mode=True)))
    return datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert mmdet3d datasets to ffr')
    parser.add_argument('config_file', help='train config file path')
    parser.add_argument('out_dir', help='the dir to save ffr data')
    parser.add_argument('dump_key', help='dump key, default is img_filename')
    args = parser.parse_args()
    main(args)
