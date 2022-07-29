import logging
import shutil
import sys
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from ffrecord import FileWriter
from pathlib import Path
from ffrecord.utils import dump as ffdump
from torch.utils.data import Dataset

from general_utils import get_data_dir


class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        """
        Wrap dataset obj with serializing sample to bytes

        Args:
            dataset (torch.utils.data.Dataset): dataset needs to convert

        """
        self.dataset = dataset
        self.serialize_sample = getattr(dataset, 'serialize_sample', None)
        # use pickle.dumps as serialize method, input sample, output bytes
        if not callable(self.serialize_sample):
            self.serialize_sample = lambda x: pickle.dumps(x)

    def __getitem__(self, idx):
        """
        Get serialized sample

        Args:
            idx (int): sample index

        Returns:
            serialized_sample (bytes): serialized sample

        """
        serialized_sample = self.serialize_sample(self.dataset[idx])
        assert isinstance(serialized_sample,
                          bytes), f'Your serialize_sample function should return bytes rather than {type(serialized_sample)}'
        return serialized_sample

    def __len__(self):
        """
        Get dataset length

        Returns:
            len (int): length of dataset

        """
        return len(self.dataset)


class GeneralConverter:
    def __init__(self, dataset: Dataset, ori_data_dir, cvt_name, split, mini_num=0):
        """
        Automatic ffrecord converter for torch.utils.data.Dataset

        Args:
            dataset (torch.utils.data.Dataset): dataset needs to convert
            ori_data_dir (Path): directory for original dataset
            cvt_name (str): converted directory name for saving ffrecord dataset
            split (str): subdirectory name of ffrecord dataset
            mini_num (str): sample number of mini set, 0 means using mini set, else full set

        """
        self.dataset = DatasetWrapper(dataset)
        self.ori_data_dir = ori_data_dir
        data_dir = get_data_dir()
        if mini_num:
            data_dir = data_dir / "mini"
        self.cvt_data_dir = data_dir / cvt_name
        self.split = split
        self.mini_num = mini_num
        self.MB = 1024.0 * 1024.0
        if split:
            self.cvt_split_dir = self.cvt_data_dir / split
        else:
            self.cvt_split_dir = self.cvt_data_dir
        self.ffr_dir = self.cvt_split_dir / 'ffrdata'
        self.meta_path = self.cvt_split_dir / 'meta.pkl'
        if self.cvt_split_dir.exists():
            print(f'Remove target dir: {self.cvt_data_dir} ...')
            shutil.rmtree(self.cvt_data_dir)
        self.ffr_dir.mkdir(parents=True)

    def dump_samples(self):
        """

        Dump serialized samples to ffrecord format.

        """
        if self.mini_num != 0:
            n_data = self.mini_num
        else:
            n_data = len(self.dataset)
        check_size_sample_num = min(n_data, 100)
        sizes = []
        for i in range(check_size_sample_num):
            sample = self.dataset[i]
            sizes.append(sys.getsizeof(sample))
        sample_bytes = np.mean(sizes)
        # Each ffrecord file should be larger than 256MB.
        sample_num_per_file = int(256 * self.MB / sample_bytes) + 1
        # Number of ffrecord files should be more than 100, and less than 200
        n_files = (n_data + sample_num_per_file - 1) // sample_num_per_file
        n_files = min(200, n_files)
        print(
            f'1. Dump {n_data} samples to {self.ffr_dir} (origin {len(self.dataset)} samples), '
            f'{sample_bytes} bytes per sample, '
            f'{sample_num_per_file} samples per file, '
            f'dump to {n_files} files, '
            f'miniset {self.mini_num is not None} ...')
        if n_files == 1:
            self.ffdump_one_file(n_data)
        else:
            ffdump(self.dataset, fname=self.ffr_dir, nfiles=n_files, verbose=True)

    def ffdump_one_file(self, n_data):
        """
        Dump dataset to one ffrecord file

        Args:
            n_data (int): dataset length

        """
        with FileWriter(self.ffr_dir / "PART_00000.ffr", n_data) as w:
            for i in tqdm(range(n_data)):
                item = self.dataset[i]
                w.write_one(item)

    def dump_meta(self):
        """

        Dump dataset meta

        """
        get_meta = getattr(self.dataset.dataset, 'get_meta', None)
        if callable(get_meta):
            meta = get_meta()
            print(f'2. Dump meta to {self.meta_path}')
            with open(self.meta_path, "wb") as fp:
                pickle.dump(meta, fp)
        else:
            print('2. Dump meta. No meta.')


def copy_path(ori_data_dir, cvt_name, foo, mini_num):
    """
    Copy path from dataset directory to output directory

    Args:
        ori_data_dir (Path): directory for original dataset
        cvt_name (str): converted directory name for saving ffrecord dataset
        foo (types.ModuleType): dataset module need to be converted

    """
    data_dir = get_data_dir()
    if mini_num:
        data_dir = data_dir / "mini"
    cvt_data_dir = data_dir / cvt_name / 'files'
    get_files_or_dirs = getattr(foo, 'get_files_or_dirs', None)
    if callable(get_files_or_dirs):
        files_or_directories = get_files_or_dirs()
        print(f'\nCopy files {files_or_directories}.')
        assert isinstance(files_or_directories, list)
        for data_rel_path, out_rel_path in files_or_directories:
            data_path = ori_data_dir / data_rel_path
            if not Path(data_path).exists():
                logging.warning(f'file {data_path} does not exists, ignore it!')
                continue
            if out_rel_path is None:
                out_rel_path = data_rel_path
            out_path = cvt_data_dir / out_rel_path
            print(f'Copy from {data_path} to {out_path}, make directory {out_path.parent}')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if data_path.is_dir():
                shutil.copytree(data_path, out_path)
            else:
                shutil.copy(data_path, out_path)
    else:
        print('\nCopy files. No file needs to be copy.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help='dataset.py path')
    parser.add_argument("ori_data_dir", type=str, help='input original dataset directory')
    parser.add_argument("cvt_name", type=str, help='output ffrecord dataset directory name')
    parser.add_argument("--mini_num", type=int, default=0, help='0 means full set, else miniset')
    args = parser.parse_args()
    assert Path(args.dataset).exists(), f'{args.dataset} not exists'
    import importlib.util

    args.ori_data_dir = Path(args.ori_data_dir)

    spec = importlib.util.spec_from_file_location("my_dataset", args.dataset)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["my_dataset"] = foo
    spec.loader.exec_module(foo)

    dataset_tuples = foo.get_datasets(args.ori_data_dir)
    print(f'Convert datasets: {args.dataset} {[dataset_tuple[1] for dataset_tuple in dataset_tuples]}')
    for dataset, split in dataset_tuples:
        print(f'\nConvert {args.dataset} {split} to ffrecord ...')
        converter = GeneralConverter(
            dataset,
            args.ori_data_dir,
            args.cvt_name,
            split,
            args.mini_num
        )
        converter.dump_samples()
        converter.dump_meta()

    copy_path(args.ori_data_dir, args.cvt_name, foo, args.mini_num)
    print(f'\nConvert {args.dataset} done.')
