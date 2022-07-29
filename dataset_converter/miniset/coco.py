import os
import hfai
import json
import argparse

from pathlib import Path
from hfai import datasets
from utils import get_suit_size


def extract_coco_detection_anno_miniset(split, mini_num, mini_dir, anno_type):
    coco_dir = hfai.datasets.get_data_dir() / "COCO"
    mini_coco_dir = mini_dir / "COCO"
    annotation_file = coco_dir / f"annotations/{anno_type}_{split}2017.json"
    mini_annotation_file = mini_coco_dir / f"annotations/{anno_type}_{split}2017.json"
    os.makedirs(os.path.dirname(mini_annotation_file), exist_ok=True)
    size, size_name = get_suit_size(os.path.getsize(annotation_file))
    print(f'\nextract {split} {anno_type} set, read anno from {annotation_file}, file size: {size:.2f} {size_name}')
    with open(annotation_file, "r") as fp:
        anno_obj = json.load(fp)
        anno_obj['images'] = [anno_obj['images'][i] for i in range(mini_num)]
        image_ids = [anno_item['id'] for anno_item in anno_obj['images']]
        anno_obj['annotations'] = [anno_item for anno_item in anno_obj['annotations'] if
                                   anno_item['image_id'] in image_ids]
        with open(mini_annotation_file, "w") as fw:
            json.dump(anno_obj, fw)
    size, size_name = get_suit_size(os.path.getsize(mini_annotation_file))
    print(f'write to {mini_annotation_file}, file size: {size:.2f} {size_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-dir", type=str, default='/weka-jd/prod/platform_team/bx/oss/dataset/mini')
    args = parser.parse_args()
    mini_dir = Path(args.mini_dir)

    for anno_type in ['instances', 'person_keypoints', 'captions', 'panoptic']:
        extract_coco_detection_anno_miniset('train', 4, mini_dir, anno_type)
        extract_coco_detection_anno_miniset('val', 4, mini_dir, anno_type)
