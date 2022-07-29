from ffrecord import FileWriter
import pandas as pd
from PIL import Image
import pickle
import io
import math


def ffhandle(filelist_csv_path, per_size):
    df = pd.read_csv(filelist_csv_path, sep="\t")
    images = df["filepath"]
    captions = df["title"]
    convert_data = pd.DataFrame([], columns=["title", "filepath"])
    print("[ffhandle] total images len:", len(images))

    bundles_size = math.ceil(len(images) / per_size)
    print("[ffhandle] bundle_size:", bundles_size)

    for round in range(0, bundles_size):
        print("[ffhandle] round:", round)
        begin = round * per_size
        end = min((round + 1) * per_size, len(images))
        writer = FileWriter(filelist_csv_path.replace(".csv", f"_{round}.ffr"), end - begin)
        print(f"[ffhandle] write begin：{begin}, end: {end}")
        for ind in range(begin, end):
            image_path = images[ind]
            caption = captions[ind]
            image = Image.open(image_path)
            img_bytes_io = io.BytesIO()
            image.save(img_bytes_io, format=image.format)
            image_bytes = img_bytes_io.getvalue()
            unit = {"caption": caption, "image_bytes": image_bytes}
            data = pickle.dumps(unit)
            writer.write_one(data)
            convert_data = convert_data.append(
                {"filepath": f"{round}_{ind - begin}", "title": caption}, ignore_index=True
            )
        writer.close()

    convert_data.to_csv(filelist_csv_path.replace(".csv", f"_ffr.csv"), index=False, sep="\t")


ffhandle("/3fs-jd/prod/private/nxt/clip/cc_data_1_proc/train/Train_GCC-training_output_washed.csv", 10000)
ffhandle("/3fs-jd/prod/private/nxt/clip/cc_data_1_proc/val/Validation_GCC-1.1.0-Validation_output_washed.csv", 10000)
