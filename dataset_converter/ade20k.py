import sys
from pathlib import Path
import pickle
import shutil
from multiprocessing import Pool

from tqdm import tqdm
from ffrecord import FileWriter


"""
ADEChallengeData2016
    images/
        training/
            ADE_train_00000001.jpg
            ADE_train_00000002.jpg
            ...
        validation/
            ADE_val_00000001.jpg
            ADE_val_00000002.jpg
            ...
    annotations/
        training/
            ADE_train_00000001.png
            ADE_train_00000002.png
            ...
        validation/
            ADE_val_00000001.png
            ADE_val_00000002.png
            ...

    objectInfo150.txt
    sceneCategories.txt

"""


def write_imgs(chunk_id, imgs, segs, out_file):
    assert len(imgs) == len(segs)
    n = len(imgs)
    writer = FileWriter(out_file, n)

    for img_file, seg_file in tqdm(zip(imgs, segs)):
        with open(img_file, "rb") as fp:
            img_bytes = fp.read()
        with open(seg_file, "rb") as fp:
            seg_bytes = fp.read()

        bytes_ = pickle.dumps((img_bytes, seg_bytes))
        writer.write_one(bytes_)

    writer.close()


def dump_ade20k(split, data_dir, out_dir, nfiles):
    # we recommend users to split data into >= 50 files under the
    # premise of file size greater than 256 MiB
    out_dir.mkdir(exist_ok=True, parents=True)
    if split == "train":
        split_name = "training"
    elif split == "val":
        split_name = "validation"
    else:
        assert False

    img_dir = data_dir / "images" / split_name
    imgs = list(img_dir.glob("*.jpg"))
    imgs.sort()

    seg_dir = data_dir / "annotations" / split_name
    segs = list(seg_dir.glob("*.png"))
    segs.sort()

    print(len(imgs), len(segs))
    assert len(imgs) == len(segs)

    # split data int into multiple files
    n = len(imgs)
    chunk_size = (n + nfiles - 1) // nfiles

    chunk_id = 0
    tasks = []
    for i0 in range(0, n, chunk_size):
        ni = min(n - i0, chunk_size)
        sub_imgs = imgs[i0 : (i0 + ni)]
        sub_segs = segs[i0 : (i0 + ni)]
        out_file = str(out_dir / f"PART_{chunk_id:05d}.ffr")
        tasks.append((chunk_id, sub_imgs, sub_segs, out_file))
        chunk_id += 1

    with Pool(8) as pool:
        pool.starmap(write_imgs, tasks)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [input_directory]  [output_directory]"
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    val_dir = out_dir / "val.ffr"
    train_dir = out_dir / "train.ffr"
    dump_ade20k("val", data_dir, val_dir, nfiles=1)
    dump_ade20k("train", data_dir, train_dir, nfiles=2)

    shutil.copy(data_dir / "objectInfo150.txt", out_dir / "objectInfo150.txt")
    shutil.copy(data_dir / "sceneCategories.txt", out_dir / "sceneCategories.txt")
