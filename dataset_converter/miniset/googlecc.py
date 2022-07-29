import os.path

from ffrecord import FileWriter
import pandas as pd
from PIL import Image
import pickle
import io


def ffhandle(data_dir, out_dir, filelist_csv_path, mini_num):
    print(f'convert from {data_dir} to {out_dir}...')
    df = pd.read_csv(os.path.join(data_dir, filelist_csv_path), sep="\t")
    images = df["filepath"]
    captions = df["title"]
    convert_data = pd.DataFrame([], columns=["title", "filepath"])
    print("[ffhandle] total images len:", len(images))

    round = 0
    out_file = filelist_csv_path.replace(".csv", f"_{round}.ffr")
    out_path = os.path.join(out_dir, out_file.replace('cc_data_1_proc/', ''))
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
        print(f'create dir {os.path.dirname(out_path)}')
    writer = FileWriter(out_path, mini_num)
    print(f"[ffhandle] write mini_num {mini_num}")
    for ind in range(mini_num):
        image_path = images[ind].replace('/3fs-jd/prod/private/nxt/clip', data_dir)
        caption = captions[ind]
        image = Image.open(image_path)
        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format=image.format)
        image_bytes = img_bytes_io.getvalue()
        unit = {"caption": caption, "image_bytes": image_bytes}
        data = pickle.dumps(unit)
        writer.write_one(data)
        convert_data = convert_data.append(
            {"filepath": f"{round}_{ind}", "title": caption}, ignore_index=True
        )
    writer.close()


data_dir = "/3fs-jd/prod/platform_team/bx/oss/dataset/ori/googlecc"
out_dir = "/weka-jd/prod/platform_team/bx/oss/dataset/mini/googlecc"

ffhandle(
    data_dir, out_dir,
    "cc_data_1_proc/train/Train_GCC-training_output_washed.csv", 16)
ffhandle(
    data_dir, out_dir,
    "cc_data_1_proc/val/Validation_GCC-1.1.0-Validation_output_washed.csv", 16)
