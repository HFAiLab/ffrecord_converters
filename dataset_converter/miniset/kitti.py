import sys
from pathlib import Path
import pickle

import csv
from tqdm import tqdm
from ffrecord import FileWriter
from utils import get_suit_size, get_dir_size

"""

It corresponds to the "left color images of object" dataset, for object detection.

KITTI/
    Object/
        training/
            image_2/
                000000.png
                000001.png
                ...
            label_2/
                000000.txt
                000001.txt
                ...
        testing/
            image_2/
                000000.png
                000001.png
                ...
            label_2/
                000000.txt
                000001.txt
                ...

"""

TYPES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]


def write_samples(chunk_id, split, imgs, labels, out_file):
    assert len(imgs) == len(labels)
    n = len(imgs)
    writer = FileWriter(out_file, n)

    for img_file, label_file in tqdm(zip(imgs, labels)):
        with open(img_file, "rb") as fp:
            img_bytes = fp.read()

        if split == "train":
            target = []
            with open(label_file) as inp:
                content = csv.reader(inp, delimiter=" ")
                for line in content:
                    assert line[0] in TYPES
                    target.append(
                        {
                            "type": line[0],
                            "truncated": float(line[1]),
                            "occluded": int(line[2]),
                            "alpha": float(line[3]),
                            "bbox": [float(x) for x in line[4:8]],
                            "dimensions": [float(x) for x in line[8:11]],
                            "location": [float(x) for x in line[11:14]],
                            "rotation_y": float(line[14]),
                        }
                    )
            bytes_ = pickle.dumps((img_bytes, target))
        else:
            bytes_ = img_bytes

        writer.write_one(bytes_)

    writer.close()


def dump_kitti_object2d(split, data_dir, out_dir, mini_num):
    # we recommend users to split data into >= 50 files under the
    # premise of file size greater than 256 MiB
    out_dir.mkdir(exist_ok=True, parents=True)

    data_dir = data_dir / "Object" / (split + "ing")
    img_dir = data_dir / "image_2"
    imgs = list(img_dir.glob("*.png"))
    imgs.sort()

    if split == "train":
        label_dir = data_dir / "label_2"
        labels = list(label_dir.glob("*.txt"))
        labels.sort()
    else:
        labels = ["."] * len(imgs)

    print(len(imgs), len(labels))
    assert len(imgs) == len(labels)

    # split data int into multiple files
    n = mini_num
    chunk_id = 0
    sub_imgs = imgs[:mini_num]
    sub_labels = labels[:mini_num]
    out_file = str(out_dir / f"PART_{chunk_id:05d}.ffr")
    write_samples(chunk_id, split, sub_imgs, sub_labels, out_file)
    size, size_name = get_suit_size(get_dir_size(out_dir))
    print(split, f'file size: {size:.2f} {size_name}')


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [input_directory]  [output_directory]"
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    test_dir = out_dir / "Object2D" / "test.ffr"
    train_dir = out_dir / "Object2D" / "train.ffr"
    dump_kitti_object2d("test", data_dir, test_dir, 16)
    dump_kitti_object2d("train", data_dir, train_dir, 16)
