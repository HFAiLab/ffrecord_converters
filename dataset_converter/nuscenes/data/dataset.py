import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from ffrecord import FileReader
import pickle
from torch.utils.data import Dataset
from .rasterize import preprocess_map
from .const import NUM_CLASSES
from .vector_map import VectorizedLocalMap
from .lidar import get_lidar_data
from .utils import label_onehot_encoding
from .utils import pad_or_trim_to_np


class HDMapNetDataset(Dataset):
    def __init__(self, data_conf, is_train):
        super(HDMapNetDataset, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
        canvas_h = int(patch_h / data_conf['ybound'][2])
        canvas_w = int(patch_w / data_conf['xbound'][2])
        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        version = data_conf['version']
        dataroot = data_conf['dataroot']
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)
        self.scenes = self.get_scenes(version, is_train)
        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]
        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_lidar(self, rec):
        lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
        lidar_data = lidar_data.transpose(1, 0)
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        return lidar_data, lidar_mask

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']
        pos_rotation = Quaternion(ego_pose['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def get_imgs(self, rec):
        imgs = []
        trans = []
        rots = []
        intrins = []

        for cam in self.data_conf['cams']:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            imgs.append(img)
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
        return imgs, torch.stack(trans), torch.stack(rots), torch.stack(intrins)

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        return vectors


class HDMapNetSemanticDataset(HDMapNetDataset):
    def __init__(self, data_conf, is_train):
        super(HDMapNetSemanticDataset, self).__init__(data_conf, is_train)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']
        self.ffrecord = data_conf['ffrecord']

    def get_semantic_map(self, rec):
        vectors = self.get_vectors(rec)
        instance_gt, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size,
                                                                    NUM_CLASSES, self.thickness, self.angle_class)
        semantic_gt = instance_gt != 0
        semantic_gt = torch.cat([(~torch.any(semantic_gt, axis=0)).unsqueeze(0), semantic_gt])
        instance_gt = instance_gt.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class + 1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class + 1)
        direction_gt = forward_oh_masks + backward_oh_masks
        direction_gt = direction_gt / direction_gt.sum(0)
        return semantic_gt, instance_gt, forward_masks, backward_masks, direction_gt

    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        semantic_gt, instance_gt, _, _, direction_gt = self.get_semantic_map(rec)
        return {
            'imgs': imgs, 'trans': trans, 'rots': rots,
            'intrins': intrins, 'lidar_data': lidar_data,
            'lidar_mask': lidar_mask, 'car_trans': car_trans,
            'yaw_pitch_roll': yaw_pitch_roll, 'semantic_gt': semantic_gt,
            'instance_gt': instance_gt, 'direction_gt': direction_gt
        }
