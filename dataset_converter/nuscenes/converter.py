import argparse
import yaml
from easydict import EasyDict
import os
import tqdm
import pickle
import shutil
from data import compile_train_data, compile_val_data
from multiprocessing import Pool
from ffrecord import FileWriter

"""
[origin nuscenes]
    maps
        36092f0b03a857c6a3403e25b4b7aab3.png
        37819e65e09e5547b8a3ceaefba56bb2.png
        53992ee3023e5494b90c316c183be829.png
        93406b464a165eaba6d9de76ca09f5da.png
        basemap
        expansion
        prediction
    samples
        n015-2018-11-21-19-58-31+0800__CAM_BACK__1542801732437525.jpg
        n015-2018-11-21-19-58-31+0800__CAM_BACK__1542801732937525.jpg
        ...
    sweeps
        CAM_BACK
        CAM_BACK_LEFT
        LIDAR_TOP
        RADAR_BACK_LEFT
        ...

[ffrecord nuscenes]
    train.ffr
        PART_00000.ffr
        PART_00001.ffr
        ...
    val.ffr
        PART_00000.ffr
        PART_00001.ffr
        ...
"""


def main(data_conf):
    train_data = compile_train_data(data_conf)
    val_data = compile_val_data(data_conf)
    data_conf.ffrecord_dir = os.path.join(data_conf.dataroot, 'samples', f'ffrecord-{data_conf.version}-converter')
    if not os.path.exists(data_conf.ffrecord_dir):
        os.makedirs(data_conf.ffrecord_dir)
    else:
        shutil.rmtree(data_conf.ffrecord_dir)
        os.makedirs(data_conf.ffrecord_dir)
    print('mkdir', data_conf.ffrecord_dir)
    process(train_data, 'train', data_conf)
    process(val_data, 'val', data_conf)


def process(dataset, state, data_conf):
    n = len(dataset)
    chunk_size = (n + data_conf.nprocess - 1) // data_conf.nprocess
    print(f'{state}: using {data_conf.nprocess} processes, {chunk_size} samples per process ')
    chunk_id = 0
    tasks = []
    sample_ids = range(n)
    for i0 in range(0, n, chunk_size):
        ni = min(n - i0, chunk_size)
        sub_sample_ids = sample_ids[i0: (i0 + ni)]
        tasks.append((chunk_id, sub_sample_ids, state, data_conf))
        chunk_id += 1

    with Pool(data_conf.nprocess) as pool:
        pool.starmap(extract_labels, tasks)

    print(f"writing {n} samples to {data_conf.ffrecord_dir} done")


def extract_labels(chunk_id, sample_ids, state, data_conf):
    print(f'[{chunk_id}] start loading hdmapdataset for version {data_conf.version}, state {state}')
    dataset = eval(f'compile_{state}_data')(data_conf)
    print(f'[{chunk_id}] loading done')
    ffrecord_dir = os.path.join(data_conf.ffrecord_dir, f'{state}.ffr')
    ffrecord_file = os.path.join(ffrecord_dir, f'PART_{chunk_id:05d}.ffr')
    os.makedirs(ffrecord_dir, exist_ok=True)
    print(f"write to {ffrecord_file, len(sample_ids)}")
    data_dict_writer = FileWriter(ffrecord_file, len(sample_ids))
    for idx in tqdm.tqdm(sample_ids):
        data_dict = dataset.__getitem__(idx)
        data_dict_writer.write_one(pickle.dumps(data_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument('--nprocess', type=int, default=30)
    args = parser.parse_args()

    cfg_file = 'default.yaml'
    with open(cfg_file, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config.data.nprocess = args.nprocess
    if args.version:
        config.data.version = args.version
    print(config.data)
    main(config.data)
