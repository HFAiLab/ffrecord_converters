import sys
import shutil
from pathlib import Path

from tqdm import tqdm
from ffrecord import FileWriter


def write_imgs(imgs, out_file):
    n = len(imgs)
    writer = FileWriter(out_file, n)

    for fname in tqdm(imgs):
        with open(fname, "rb") as fp:
            bytes_ = fp.read()
        writer.write_one(bytes_)
    writer.close()


def dump_isc2021(split, data_dir, out_dir, mini_num):
    # extract miniset
    data_dir = data_dir / f"{split}_images"
    out_dir = out_dir / f"{split}.ffr"
    out_dir.mkdir(exist_ok=True, parents=True)

    imgs = list(data_dir.glob("*.jpg"))
    imgs.sort()

    # split data int into multiple files
    chunk_id = 0
    write_imgs(imgs[:mini_num], str(out_dir / f"PART_{chunk_id:05d}.ffr"))


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [input_directory]  [output_directory]"
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    dump_isc2021("training", data_dir, out_dir, 16)
    dump_isc2021("reference", data_dir, out_dir, 16)
    dump_isc2021("query", data_dir, out_dir, 16)
    shutil.copy(data_dir / "public_ground_truth_50K.csv", out_dir / "public_ground_truth_50K.csv")
