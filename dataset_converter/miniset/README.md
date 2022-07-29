# hfai.datasets miniset converter

Converter 用于提取数据集的 miniset。

下面是目前支持 miniset 提取的数据集。

## NuScenes

```shell
python nuscenes.py --mini-dir $mini_dir
```

该脚本会将 `NuScenes` 数据集的 `miniset` 提取保存到 `$mini_dir/NuScenes`。


## ImageNet

```shell
python imagenet.py --input-dir $input_dir --mini-dir $mini_dir
```
该脚本会将 `ImageNet` 数据集的 `miniset` 提取保存到 `$mini_dir/ImageNet`。

## COCO

```shell
python coco.py
```
该脚本会将 `COCO` 数据集的 `miniset` 提取保存到 `$mini_dir/COCO`。