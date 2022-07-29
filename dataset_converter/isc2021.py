import sys
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
from ffrecord import FileWriter


def write_imgs(chunk_id, imgs, out_file):
    n = len(imgs)
    writer = FileWriter(out_file, n)

    for fname in tqdm(imgs):
        with open(fname, "rb") as fp:
            bytes_ = fp.read()
        writer.write_one(bytes_)
    writer.close()


def dump_isc2021(split, data_dir, out_dir, nfiles):
    # we recommend users to split data into >= 50 files under the
    # premise of file size greater than 256 MiB
    data_dir = data_dir / f"{split}_images"
    out_dir = out_dir / f"{split}.ffr"
    out_dir.mkdir(exist_ok=True, parents=True)

    imgs = list(data_dir.glob("*.jpg"))
    imgs.sort()

    # split data int into multiple files
    n = len(imgs)
    chunk_size = (n + nfiles - 1) // nfiles

    chunk_id = 0
    tasks = []
    for i0 in range(0, n, chunk_size):
        ni = min(n - i0, chunk_size)
        sub_imgs = imgs[i0 : (i0 + ni)]
        out_file = str(out_dir / f"PART_{chunk_id:05d}.ffr")
        tasks.append((chunk_id, sub_imgs, out_file))
        chunk_id += 1

    with Pool(8) as pool:
        pool.starmap(write_imgs, tasks)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [input_directory]  [output_directory]"
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    dump_isc2021("training", data_dir, out_dir, nfiles=50)
    dump_isc2021("reference", data_dir, out_dir, nfiles=50)
    dump_isc2021("query", data_dir, out_dir, nfiles=20)
