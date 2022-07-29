import sys
from pathlib import Path
import pickle
import numpy as np
import torchvision
from PIL import Image

from ffrecord.utils import dump as ffdump
from collections.abc import Sequence


class ImageNetReader(Sequence):
    def __init__(self, data_dir, split):
        self.dataset = torchvision.datasets.ImageNet(data_dir, split=split)
        self.imgs = self.dataset.imgs
        self.img_num = len(self.imgs)
        super().__init__()

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        if idx < self.img_num:
            fname, _ = self.imgs[idx]
            data = self.read_one_pickle_img(fname)
            return data
        else:
            raise ValueError(f"{idx} is max than {self.img_num}")

    def read_one_pickle_img(self, fname):
        # Image read
        image = Image.open(fname)
        # pickle dump
        data = pickle.dumps(image)
        return data


def dump_imagenet(split, data_dir, out_dir, nfiles):
    # we recommend users to split data into >= 50 files under the
    # premise of file size greater than 256 MiB
    if not nfiles == 1:
        out_dir.mkdir(exist_ok=True, parents=True)

    imgnet_reader = ImageNetReader(data_dir, split=split)
    dataset = imgnet_reader.dataset

    meta = {}
    meta["classes"] = dataset.classes
    meta["class_to_idx"] = dataset.class_to_idx
    meta["targets"] = np.array(dataset.targets, dtype=np.int32)
    with open(out_dir / "meta.pkl", "wb") as fp:
        pickle.dump(meta, fp)

    print(f"Dump {len(dataset)} samples to {out_dir}")
    ffdump(dataset=imgnet_reader, fname=out_dir, nfiles=nfiles, verbose=True)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [input_directory]  [output_directory]"
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    val_dir = Path(out_dir) / "val.ffr"
    train_dir = Path(out_dir) / "train.ffr"
    dump_imagenet("val", data_dir, val_dir, nfiles=10)
    dump_imagenet("train", data_dir, train_dir, nfiles=150)
