import os
import pickle
import argparse

from ffrecord import FileWriter
from hfai import datasets
from tqdm import tqdm
from pathlib import Path
from utils import get_suit_size, get_dir_size


def extract_nuscenes_miniset(split, mini_num, mini_dir):
    ds = datasets.NuScenes(split)
    ffrecord_dir = mini_dir / 'NuScenes' / f'{split}.ffr'
    ffrecord_file = ffrecord_dir / f'PART_{0:05d}.ffr'
    os.makedirs(ffrecord_dir, exist_ok=True)
    print(f"write to {ffrecord_file}, {mini_num} samples")
    data_dict_writer = FileWriter(ffrecord_file, mini_num)

    for idx in tqdm(range(mini_num)):
        data_dict = ds[[idx]][0]
        data_dict_writer.write_one(pickle.dumps(data_dict))
        # print({k:type(v) for k,v in data_dict.items()})
    size, size_name = get_suit_size(get_dir_size(ffrecord_dir))
    print(split, f'file size: {size:.2f} {size_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-dir", type=str, default='/weka-jd/prod/platform_team/bx/oss/dataset/mini')
    args = parser.parse_args()
    mini_dir = Path(args.mini_dir)

    extract_nuscenes_miniset('train', 4, mini_dir)
    extract_nuscenes_miniset('val', 4, mini_dir)
