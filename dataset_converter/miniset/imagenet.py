from pathlib import Path
import pickle
import numpy as np
import argparse
import torchvision
from PIL import Image
from tqdm import tqdm
from ffrecord import FileWriter

from ffrecord.utils import dump as ffdump
from collections.abc import Sequence


class ImageNetReader(Sequence):
    def __init__(self, data_dir, split, mini_num):
        self.dataset = torchvision.datasets.ImageNet(data_dir, split=split)
        self.imgs = self.dataset.imgs
        self.img_num = len(self.imgs)
        if mini_num is not None:
            self.img_num = mini_num
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


def dump_imagenet(split, data_dir, out_dir, nfiles, mini_num):
    # we recommend users to split data into >= 50 files under the
    # premise of file size greater than 256 MiB
    out_dir.mkdir(exist_ok=True, parents=True)

    imgnet_reader = ImageNetReader(data_dir, split=split, mini_num=mini_num)
    dataset = imgnet_reader.dataset

    meta = {}
    meta["classes"] = dataset.classes
    meta["class_to_idx"] = dataset.class_to_idx
    meta["targets"] = np.array(dataset.targets, dtype=np.int32)
    if mini_num is not None:
        meta["targets"] = meta["targets"][:mini_num]
    with open(out_dir / "meta.pkl", "wb") as fp:
        pickle.dump(meta, fp)

    print(f"Dump {len(imgnet_reader)} samples to {out_dir}")
    if nfiles == 1:
        with FileWriter(out_dir / "PART_00000.ffr", mini_num) as w:
            for i in tqdm(range(mini_num)):
                item = imgnet_reader[i]
                w.write_one(item)
    else:
        ffdump(dataset=imgnet_reader, fname=out_dir, nfiles=nfiles, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default='/3fs-jd/prod/platform_team/dataset/ImageNet/')
    parser.add_argument("--mini-dir", type=str, default='/weka-jd/prod/platform_team/bx/oss/dataset/mini')
    args = parser.parse_args()
    data_dir = args.input_dir
    out_dir = Path(args.mini_dir) / 'ImageNet'
    val_dir = out_dir / "val.ffr"
    train_dir = out_dir / "train.ffr"
    mini_num = 8
    val_nfiles = 1
    train_nfiles = 1

    dump_imagenet("val", data_dir, val_dir, nfiles=val_nfiles, mini_num=mini_num)
    dump_imagenet("train", data_dir, train_dir, nfiles=train_nfiles, mini_num=mini_num)
