import sys
import os
import shutil
import json
from pathlib import Path
import pickle
from PIL import Image
from torchvision.datasets.coco import CocoDetection

from pycocotools.coco import COCO
from ffrecord.utils import dump as ffdump
from collections.abc import Sequence

"""
[origin coco_data_dir]
    train2017
        00000000000.jpg
        ...
    val2017
        00000000000.jpg
        ...
    panoptic_train2017
        00000000000.png
        ...
    panoptic_val2017
        00000000000.png
        ...
    annotations
        captions_train2017.json
        captions_val2017.json
        instances_train2017.json
        instances_val2017.json
        person_keypoints_train2017.json
        person_keypoints_val2017.json
        panoptic_train2017.json
        panoptic_val2017.json

[ffrecord coco_data_dir]
    train2017.ffr
        PART_00000.ffr
        PART_00001.ffr
        ...
    val2017.ffr
        PART_00000.ffr
        PART_00001.ffr
        ...
    panoptic_train2017.ffr
        PART_00000.ffr
        PART_00001.ffr
        ...
    panoptic_val2017.ffr
        PART_00000.ffr
        PART_00001.ffr
        ...
    annotations/
        captions_train2017.json
        captions_val2017.json
        instances_train2017.json
        instances_val2017.json
        person_keypoints_train2017.json
        person_keypoints_val2017.json
        panoptic_train2017.json
        panoptic_val2017.json
"""


class CocoPanopticBase(COCO):
    def __init__(self, img_dir, annFile):
        """
        Constructor of Microsoft COCO Panoptic helper class for reading and visualizing annotations.
        :param img_dir (str): location of img dir
        :param annFile (str): location of panoptic annotation file
        :return:
        """
        with open(annFile, "r") as f:
            self.anns = json.load(f)
        imgs = {}
        if "images" in self.anns:
            for img in self.anns["images"]:
                img["file_name"] = img["file_name"].replace(".jpg", ".png")
                imgs[img["id"]] = img
        self.imgs = imgs
        self.ids = list(sorted(self.imgs.keys()))
        self.coco = self

    def __len__(self):
        return len(self.anns["images"])


class COCOImgReader(Sequence):
    def __init__(self, coco, img_dir, img_ids):
        self.coco = coco
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_num = len(img_ids)
        super().__init__()

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        if idx < self.img_num:
            img_id = self.img_ids[idx]
            data = self.read_one_pickle_img(img_id)
            return data
        else:
            raise ValueError(f"{idx} is max than {self.img_num}")

    def read_one_pickle_img(self, img_id):
        fname = self.coco.loadImgs(img_id)[0]["file_name"]
        # read img
        fname = self.img_dir / fname
        # Image read
        image = Image.open(fname)
        # pickle dump
        data = pickle.dumps(image)
        return data


def convert_part(coco_dir: os.PathLike, split: str, out_dir: os.PathLike, nfiles: int, coco_obj, prefix: str = ""):
    assert prefix in [
        "",
        "panoptic_",
    ], f"{prefix} not support! current support sets are {{'base': '', 'panoptic': 'panoptic_'}}"
    assert nfiles > 0
    is_panoptic_part = "panoptic" in prefix
    # output
    out_dir = out_dir / f"{prefix}{split}2017.ffr"
    if not nfiles == 1:
        Path(out_dir).mkdir(exist_ok=True, parents=True)
    # input
    img_dir = coco_dir / f"{prefix}{split}2017"
    if is_panoptic_part:
        ann_prefix = "panoptic_"
    else:
        ann_prefix = "instances_"
    annFile = coco_dir / "annotations" / f"{ann_prefix}{split}2017.json"

    dataset = coco_obj(img_dir, annFile=annFile)
    coco = dataset.coco
    img_ids = dataset.ids
    n = len(dataset)

    coco_img_reader = COCOImgReader(coco, img_dir, img_ids)

    print(f"Dump {n} samples to {out_dir}")
    ffdump(dataset=coco_img_reader, fname=out_dir, nfiles=nfiles, verbose=True)


def convert(coco_dir: os.PathLike, split: str, out_dir: os.PathLike, nfiles: (int, dict)):
    if isinstance(nfiles, dict):
        nfiles_normal, nfiles_panoptic = nfiles["normal"], nfiles["panoptic"]
    else:
        nfiles_normal = nfiles_panoptic = nfiles
    print(f"Start convert. nfiles_normal: {nfiles_normal}, nfiles_panoptic {nfiles_panoptic}")
    if Path(coco_dir / f"panoptic_{split}2017").exists():
        print("Start convert panoptic COCO datasets...")
        convert_part(coco_dir, split, out_dir, nfiles_panoptic, coco_obj=CocoPanopticBase, prefix="panoptic_")
    print("Start convert normal COCO datasets...")
    convert_part(coco_dir, split, out_dir, nfiles_normal, coco_obj=CocoDetection, prefix="")


def copy_annotations(coco_dir, out_dir):
    # copy annotations to out_dir
    if out_dir != coco_dir:
        shutil.copytree(coco_dir / "annotations", out_dir / "annotations")
        print(f'copy annotations to { out_dir / "annotations"}')


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [coco_dir] [out_dir]"
    coco_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    convert(coco_dir, "train", out_dir, nfiles={"normal": 150, "panoptic": 150})
    convert(coco_dir, "val", out_dir, nfiles={"normal": 12, "panoptic": 12})
    copy_annotations(coco_dir, out_dir)
