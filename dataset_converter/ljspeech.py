import sys
from pathlib import Path
import pickle
from multiprocessing import Pool

from scipy.io import wavfile
from tqdm import tqdm
from ffrecord import FileWriter


def write_samples(chunk_id, wavs_dir, transcriptions, out_file):
    n = len(transcriptions)
    writer = FileWriter(out_file, n)

    for wav_id, text, normalized_text in tqdm(transcriptions):
        fname = wavs_dir / (wav_id + ".wav")
        sr, data = wavfile.read(fname)
        assert sr == 22050

        sample = (wav_id, sr, data, text, normalized_text)
        bytes_ = pickle.dumps(sample)
        writer.write_one(bytes_)
    writer.close()


def dump_ljspeech(data_dir, out_dir, nfiles):
    # we recommend users to split data into >= 50 files under the
    # premise of file size greater than 256 MiB
    out_dir = out_dir / "LJSpeech.ffr"
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(data_dir / "metadata.csv", "r") as fp:
        lines = fp.readlines()
        transcriptions = [x.strip().split("|") for x in lines]

    wavs_dir = data_dir / "wavs"

    # split data int into multiple files
    n = len(transcriptions)
    chunk_size = (n + nfiles - 1) // nfiles

    chunk_id = 0
    tasks = []
    for i0 in range(0, n, chunk_size):
        ni = min(n - i0, chunk_size)
        sub_transcriptions = transcriptions[i0 : (i0 + ni)]
        out_file = str(out_dir / f"PART_{chunk_id:05d}.ffr")
        tasks.append((chunk_id, wavs_dir, sub_transcriptions, out_file))
        chunk_id += 1

    with Pool(8) as pool:
        pool.starmap(write_samples, tasks)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [input_directory]  [output_directory]"
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    dump_ljspeech(data_dir, out_dir, nfiles=10)
