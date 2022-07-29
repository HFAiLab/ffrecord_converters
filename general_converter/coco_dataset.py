import json
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection


class CocoPanoptic(COCO):
    def __init__(self, img_dir, ann_file):
        """
        Coco panoptic dataset

        Args:
            img_dir (Path): directory for original dataset
            ann_file (Path): annotation file path for original dataset
            
        """
        with open(ann_file, "r") as f:
            self.anns = json.load(f)
        imgs = {}
        if "images" in self.anns:
            for img in self.anns["images"]:
                img["file_name"] = img["file_name"].replace(".jpg", ".png")
                imgs[img["id"]] = img
        self.imgs = imgs
        self.ids = list(sorted(self.imgs.keys()))
        self.coco = self
        self.img_dir = img_dir

    def __len__(self):
        """
        Get dataset length

        Returns:
            len (int): length of dataset

        """
        return len(self.anns["images"])


class CocoDataset(Dataset):
    def __init__(self, coco, img_dir, img_ids):
        self.coco = coco
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_num = len(img_ids)
        super().__init__()

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
            img_id = self.img_ids[idx]
            image = self.read_img(img_id)
            return image
        else:
            raise ValueError(f"{idx} is max than {self.img_num}")

    def read_img(self, img_id):
        """
        Read image given image id

        Args:
            img_id (int): image id

        Returns:
            image (Image): image object

        """
        fname = self.coco.loadImgs(img_id)[0]["file_name"]
        # read img
        fname = self.img_dir / fname
        # Image read
        image = Image.open(fname)
        return image


def get_datasets(ori_data_dir):
    """
    Get dataset and name tuple list of COCO

    Args:
        ori_data_dir (Path): directory for original dataset

    Returns:
        dataset_pairs (list): dataset and split tuple list

    """
    train_coco_obj = CocoDetection(
        str(ori_data_dir / 'train2017'),
        str(ori_data_dir / 'annotations' / 'instances_train2017.json')
    )
    train_data = CocoDataset(train_coco_obj.coco, ori_data_dir / 'train2017', train_coco_obj.ids)

    val_coco_obj = CocoDetection(
        str(ori_data_dir / 'val2017'),
        str(ori_data_dir / 'annotations' / 'instances_val2017.json')
    )
    val_data = CocoDataset(val_coco_obj.coco, ori_data_dir / 'val2017', val_coco_obj.ids)

    panoptic_train_obj = CocoPanoptic(
        ori_data_dir / 'panoptic_train2017',
        ori_data_dir / 'annotations' / 'panoptic_train2017.json'
    )
    panoptic_train_data = CocoDataset(panoptic_train_obj.coco, ori_data_dir / 'panoptic_train2017', panoptic_train_obj.ids)

    panoptic_val_obj = CocoPanoptic(
        ori_data_dir / 'panoptic_val2017',
        ori_data_dir / 'annotations' / 'panoptic_val2017.json'
    )
    panoptic_val_data = CocoDataset(panoptic_val_obj.coco, ori_data_dir / 'panoptic_val2017', panoptic_val_obj.ids)

    return [
        (train_data, 'train'),
        (val_data, 'val'),
        (panoptic_train_data, 'panoptic_train'),
        (panoptic_val_data, 'panoptic_val')
    ]


def get_files_or_dirs():
    """
    Get copy files or dirs

    Returns:
        rel_path_tuples (list): list of relative path tuple [(src, tgt), ...]
    """
    return [('annotations', None)]
