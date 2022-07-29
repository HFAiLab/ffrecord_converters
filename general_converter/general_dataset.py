import pickle
import os
from pathlib import Path
from typing import Callable, Optional
from ffrecord import FileReader
from torch.utils.data import Dataset
from general_utils import get_data_dir

"""
Expected file organization:

    [cvt_data_dir]
        split1
            meta.pkl
            ffrdata
                PART_00000.ffr
                PART_00001.ffr
                ...
        split2
            meta.pkl
            ffrdata
                PART_00000.ffr
                PART_00001.ffr
                ...
        files
            ...
"""


class GeneralDataset(Dataset):
    """
   这是一个通用数据集

   Args:
       cvt_name (str): 转换后的数据集目录名称，例如：（``ImageNet``） 或者　（``COCO``）
       split (str): 数据集划分形式，包括：训练集（``train``）、验证集（``val``）或者其他自定义集
       transform (Callable): transform 函数，对样本进行 transfrom，接受样本 id、样本、meta 信息作为输入，输出 transform 之后的样本
       check_data (bool): 是否对每一条样本检验校验和（默认为 ``True``）
       miniset (bool): 是否使用 mini 集合（默认为 ``False``）
       deserialize_sample (Callable): 解序列函数，接受二进制样本作为输入，输出解序列后的样本，默认使用 pickle.loads 进行解序列

   Returns:
       sample (tuple): 返回的每个样本是一个元组，包括一个样本数据

   Examples:

   .. code-block:: python

       from torchvision import transforms

       def transform(id, sample, meta):
           transformed_img = transforms.Compose([
                   transforms.Resize(224),
                   transforms.ToTensor(),
                   transforms.Normalize(mean=mean, std=std),
               ])(sample)
           return (transformed_img, meta["targets"][id])

       dataset = GeneralDataset('ImageNet', split, transform)
       loader = dataset.loader(batch_size=64, num_workers=4)

       for sample in loader:
           # training model

   """

    def __init__(
            self,
            cvt_name: str,
            split: str,
            transform: Optional[Callable] = None,
            check_data: bool = True,
            miniset: bool = False,
            deserialize_sample: Optional[Callable] = None
    ):
        super(GeneralDataset, self).__init__()
        self.split = split
        self.transform = transform
        data_dir = get_data_dir()
        if miniset:
            data_dir = data_dir / 'mini'
        self.cvt_data_dir = data_dir / cvt_name
        if split:
            self.cvt_split_dir = self.cvt_data_dir / split
        else:
            self.cvt_split_dir = self.cvt_data_dir
        self.ffr_dir = self.cvt_split_dir / 'ffrdata'
        self.meta_path = self.cvt_split_dir / 'meta.pkl'
        self.reader = FileReader(self.ffr_dir, check_data)
        if self.meta_path.exists():
            with open(self.meta_path, 'rb') as fp:
                self.meta = pickle.load(fp)
        else:
            self.meta = None
        self.deserialize_sample = deserialize_sample if deserialize_sample else pickle.loads

    def __len__(self):
        """
        Get dataset length

        Returns:
            len (int): length of dataset

        """
        return self.reader.n

    def __getitem__(self, indices):
        """
        Get the samples with given indices

        Args:
            indices (list): sample indices

        Returns:
            samples (list): list of sample with given indices

        """
        bytes = self.reader.read(indices)

        samples = []
        for i, bytes_ in enumerate(bytes):
            samples.append(self.deserialize_sample(bytes_))

        transformed_samples = []
        for i, sample in enumerate(samples):
            if self.transform:
                idx = indices[i]
                sample = self.transform(idx, sample, self.meta)
            transformed_samples.append(sample)

        return transformed_samples

    def list_files(self):
        """
        Get file list in files directory

        Returns:
            file_list (list): list of file in files directory

        """
        files = []
        files_path = self.cvt_data_dir / 'files'
        if not (files_path).exists():
            print(f'file directory {files_path} not exists.')
            return []
        for fpathe, dirs, fs in os.walk(files_path):
            for f in fs:
                files.append(Path(fpathe) / f)
        return files
