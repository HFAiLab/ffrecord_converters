import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, ori_data_dir, split):
        """
        ImageNet dataset

        Args:
            ori_data_dir (Path): directory for original dataset
            split (str): split of dataset

        """
        super(ImageNetDataset, self).__init__()
        self.dataset = torchvision.datasets.ImageNet(ori_data_dir, split=split)
        self.imgs = self.dataset.imgs
        self.img_num = len(self.imgs)

    def __len__(self):
        """
        Get dataset length

        Returns:
            len (int): length of dataset

        """
        return self.img_num

    def __getitem__(self, idx):
        """
        Get sample

        Args:
            idx (int): sample index

        Returns:
            sample: the id-th sample of the dataset

        """
        if idx < self.img_num:
            fname, _ = self.imgs[idx]
            sample = Image.open(fname)
            return sample
        else:
            raise ValueError(f"{idx} is max than {self.img_num}")

    def get_meta(self):
        """
        Get meta

        Returns:
            meta: meta of dataset
        """
        dataset = self.dataset
        meta = {}
        meta["classes"] = dataset.classes
        meta["class_to_idx"] = dataset.class_to_idx
        meta["targets"] = np.array(dataset.targets, dtype=np.int32)
        return meta


def get_datasets(ori_data_dir):
    """
    Get dataset and name tuple list of COCO

    Args:
        ori_data_dir (Path): directory for original dataset

    Returns:
        dataset_pairs (list): dataset and split tuple list

    """
    train_data = ImageNetDataset(ori_data_dir, 'train')
    val_data = ImageNetDataset(ori_data_dir, 'val')
    return [
        (train_data, 'train'),
        (val_data, 'val')
    ]
