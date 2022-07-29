import sys
from pathlib import Path
import pickle

from scipy.io import wavfile
from tqdm import tqdm
from ffrecord import FileWriter


def write_samples(wavs_dir, transcriptions, out_file):
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


def dump_ljspeech(data_dir, out_dir, nfiles, mini_num):
    # extract miniset of ljspeech
    out_dir = out_dir / "LJSpeech.ffr"
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(data_dir / "metadata.csv", "r") as fp:
        lines = fp.readlines()
        transcriptions = [x.strip().split("|") for x in lines]

    wavs_dir = data_dir / "wavs"

    chunk_id = 0
    out_file = str(out_dir / f"PART_{chunk_id:05d}.ffr")
    sub_transcriptions = transcriptions[0:mini_num]
    write_samples(wavs_dir, sub_transcriptions, out_file)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python convert.py [input_directory] [output_directory]"
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    dump_ljspeech(data_dir, out_dir, nfiles=1, mini_num=16)
