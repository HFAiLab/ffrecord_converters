import hfai_env
hfai_env.set_env("weather")

import xarray as xr
import numpy as np

np.random.seed(2020)

import pandas as pd
import pickle
from pathlib import Path
import torch
import math

from ffrecord import FileWriter


class Sphere:
    def __init__(self, dim=2):
        self.dim = dim

    def _is_in_unit_sphere(self, x):
        norm_2 = torch.norm(x, dim=-1)
        return ~(torch.abs(norm_2 - 1) > 1e-7).prod().bool()

    def _ensure_in_unit_sphere(self, x):
        assert self._is_in_unit_sphere(x), "One of the given vector is not on the unit sphere"

    def _is_in_tangent_space(self, center, v):
        """
        inputs:
            center: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            if_in_tangence: bool
        """
        self._ensure_in_unit_sphere(center)
        product = torch.matmul(v, center[:, :, None])
        product[torch.isnan(product)] = 0.0
        return (torch.abs(torch.matmul(v, center[:, :, None])) <= 1e-7).prod().bool()

    def _ensure_in_tangent_space(self, center, v):
        assert self._is_in_tangent_space(center, v), "One of the given vector is not on the tangent space"

    def _is_in_ctangent_space(self, center, v):
        """
        inputs:
            center: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            if_in_tangence: bool
        """
        self._ensure_in_unit_sphere(center)
        v_minus = v[:, :, :-1]
        center_minus = center[:, :-1]
        product = torch.matmul(v_minus, center_minus[:, :, None])
        product[torch.isnan(product)] = 0.0
        return (torch.abs(torch.matmul(v_minus, center_minus[:, :, None])) <= 1e-7).prod().bool()

    def _ensure_in_ctangent_space(self, center, v):
        assert self._is_in_ctangent_space(center, v), "One of the given vector is not on the cylindrical-tangent space"

    def geo_distance(self, u, v):
        """
        inputs:
            u: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            induced_distance(u,v): (N, M)
        """
        assert u.shape[1] == v.shape[2] == self.dim + 1, "Dimension is not identical."
        self._ensure_in_unit_sphere(u)
        self._ensure_in_unit_sphere(v)
        return torch.arccos(torch.matmul(v, u[:, :, None]))

    def tangent_space_projector(self, x, v):
        """
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            project_x(v): (N, M, self.dim + 1)
        """
        assert x.shape[1] == v.shape[2], "Dimension is not identical."

        x_normalized = torch.divide(x, torch.norm(x, dim=-1, keepdim=True))
        v_normalized = torch.divide(v, torch.norm(v, dim=-1, keepdim=True))
        v_on_x_norm = torch.matmul(v_normalized, x_normalized[:, :, None])  # N, M, 1
        v_on_x = v_on_x_norm * x_normalized[:, None, :]  # N,M,dim
        p_x = v_normalized - v_on_x  # N,M,dim
        return p_x

    def exp_map(self, x, v):
        """
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) which is on the tangent space of x
        outputs:
            exp_x(v): (N, M, self.dim + 1)
        """
        assert x.shape[1] == v.shape[2] == self.dim + 1, "Dimension is not identical."
        self._ensure_in_unit_sphere(x)
        self._ensure_in_tangent_space(x, v)

        v_norm = torch.norm(v, dim=-1)[:, :, None]  # N,M, 1
        return torch.cos(v_norm) * x[:, None, :] + torch.sin(v_norm) * torch.divide(v, v_norm)

    def log_map(self, x, v):
        """
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) # v is on the sphere
        outputs:
            log_x(v): (N, M, self.dim + 1)
        """
        assert x.shape[1] == v.shape[2] == self.dim + 1, "Dimension is not identical."
        self._ensure_in_unit_sphere(x)
        self._ensure_in_unit_sphere(v)

        p_x = self.tangent_space_projector(x, v - x[:, None, :])  # N,M,d
        p_x_norm = torch.norm(p_x, dim=-1)[:, :, None]  # N,M,1
        distance = self.geo_distance(x, v)  # N,M,1
        log_xv = torch.divide(distance * p_x, p_x_norm)
        log_xv[torch.isnan(log_xv)] = 0.0  # map itself to the origin

        return log_xv

    def horizon_map(self, x, v):
        """
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) # v is on the sphere
        outputs:
            H_x(v): (N, M, self.dim + 1)
        """
        assert x.shape[1] == v.shape[2] == self.dim + 1, "Dimension is not identical."
        self._ensure_in_unit_sphere(x)
        self._ensure_in_unit_sphere(v)

        x_minus = x[:, :-1]
        v_minus = v[:, :, :-1]
        p_x_minus = self.tangent_space_projector(x_minus, v_minus - x_minus[:, None, :])
        p_x = torch.cat([p_x_minus, v[:, :, [-1]] - x[:, None, [-1]]], dim=-1)
        p_x_norm = torch.norm(p_x, dim=-1)[:, :, None]
        distance = self.geo_distance(x, v)
        H_xv = torch.divide(distance * p_x, p_x_norm)
        H_xv[torch.isnan(H_xv)] = 0.0  # map itself to the origin

        return H_xv

    def cart3d_to_ctangent_local2d(self, x, v):
        """
        inputs:
            x: (N, 3)
            v: (N, M, 3) # v is on the ctangent space of x
        outputs:
            \Pi_x(v): (N, M, 2)
        """
        assert x.shape[1] == v.shape[2] == 3, "the method can only used for 2d sphere, so the input should be in R^3."
        self._ensure_in_ctangent_space(x, v)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        lat, lon = self.xyz2latlon(x1, x2, x3)

        v_temp = v.sum(dim=-1, keepdim=True)
        idx_zero = v_temp == 0

        e_phi = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
        v_phi = torch.matmul(v, e_phi[:, :, None])
        v_phi[idx_zero] = 0
        v_z = v[:, :, [-1]]
        v_z[idx_zero] = 0
        return torch.cat([v_phi, v_z], dim=-1)

    def cart3d_to_tangent_local2d(self, x, v):
        """
        inputs:
            x: (N, 3)
            v: (N, M, 3) # v is on the tangent space of x
        outputs:
            \Pi_x(v): (N, M, 2)
        """
        assert x.shape[1] == v.shape[2] == 3, "the method can only used for 2d sphere, so the input should be in R^3."
        self._ensure_in_tangent_space(x, v)

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        lat, lon = self.xyz2latlon(x1, x2, x3)
        e_theta = torch.stack(
            [torch.sin(lat) * torch.cos(lon), torch.sin(lat) * torch.sin(lon), torch.cos(lat)], dim=-1
        )  # N,3
        e_phi = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)  # N,3

        v_temp = v.sum(dim=-1, keepdim=True)
        idx_zero = v_temp == 0

        v_theta = torch.matmul(v - x[:, None, :], e_theta[:, :, None])  # N,M,1
        v_theta[idx_zero] = 0
        v_phi = torch.matmul(v - x[:, None, :], e_phi[:, :, None])  # N,M,1
        v_phi[idx_zero] = 0
        return torch.cat([v_theta, v_phi], dim=-1)

    @classmethod
    def latlon2xyz(self, lat, lon, is_input_degree=True):
        if is_input_degree == True:
            lat = lat * math.pi / 180
            lon = lon * math.pi / 180
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return x, y, z

    @classmethod
    def xyz2latlon(self, x, y, z):
        lat = torch.atan2(z, torch.norm(torch.stack([x, y], dim=-1), dim=-1))
        lon = torch.atan2(y, x)
        return lat, lon


def latlon2xyz(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


class KernelGenerator:
    def __init__(self, lonlat, k_neighbors=25, local_map="fast") -> None:
        self.lonlat = lonlat
        self.k_neighbors = k_neighbors
        self.local_map = local_map

        self.nbhd_idx, col, row, self.geodesic = self.get_geo_knn_graph(self.lonlat, self.k_neighbors)
        self.sparse_idx = np.array([row, col])
        self.MLP_inputs, self.centers, self.points = self.X2KerInput(
            self.lonlat, sparse_idx=self.sparse_idx, k_neighbors=self.k_neighbors, local_map=self.local_map
        )
        _, self.ratio_lists = self.XY2Ratio(self.MLP_inputs[:, -2:], k_neighbors=self.k_neighbors)

    def get_geo_knn_graph(self, X, k=25):
        # X: num_node, dim
        lon = X[:, 0]
        lat = X[:, 1]
        x, y, z = latlon2xyz(lat, lon)
        coordinate = np.stack([x, y, z])
        product = np.matmul(coordinate.T, coordinate).clip(min=-1.0, max=1.0)
        geodesic = np.arccos(product)
        nbhd_idx = np.argsort(geodesic, axis=-1)[:, :k]
        col = nbhd_idx.flatten()
        row = np.expand_dims(np.arange(geodesic.shape[0]), axis=-1).repeat(k, axis=-1).flatten()
        return nbhd_idx, col, row, np.sort(geodesic, axis=-1)[:, :k]

    def X2KerInput(self, x, sparse_idx, k_neighbors, local_map="fast"):
        """
        x: the location list of each point
        sparse_idx: the sparsity matrix of 2*num_nonzero
        """
        sample_num = x.shape[0]
        loc_feature_num = x.shape[1]
        centers = x[sparse_idx[0]]
        points = x[sparse_idx[1]]
        if local_map == "fast":
            delta_x = points - centers
            delta_x[delta_x > 180] = delta_x[delta_x > 180] - 360
            delta_x[delta_x < -180] = delta_x[delta_x < -180] + 360
            inputs = np.concatenate((centers, delta_x), axis=-1).reshape(-1, loc_feature_num * 2)
            inputs = inputs / 180 * np.pi

        elif local_map == "log":
            centers = torch.from_numpy(centers.reshape(-1, k_neighbors, loc_feature_num))
            points = torch.from_numpy(points.reshape(-1, k_neighbors, loc_feature_num))
            sphere_2d = Sphere(2)
            centers_x = torch.stack(Sphere.latlon2xyz(centers[:, 0, 1], centers[:, 0, 0]), dim=-1)
            points = torch.stack(Sphere.latlon2xyz(points[:, :, 1], points[:, :, 0]), dim=-1)
            log_cp = sphere_2d.log_map(centers_x, points)
            local_coor = sphere_2d.cart3d_to_tangent_local2d(centers_x, log_cp)

            centers = centers.reshape(-1, loc_feature_num).numpy()
            local_coor = local_coor.reshape(-1, loc_feature_num).numpy()
            inputs = np.concatenate((centers / 180 * np.pi, local_coor), axis=-1).reshape(-1, loc_feature_num * 2)

        elif local_map == "horizon":
            centers = torch.from_numpy(centers.reshape(-1, k_neighbors, loc_feature_num))
            points = torch.from_numpy(points.reshape(-1, k_neighbors, loc_feature_num))
            sphere_2d = Sphere(2)
            centers_x = torch.stack(Sphere.latlon2xyz(centers[:, 0, 1], centers[:, 0, 0]), dim=-1)
            points = torch.stack(Sphere.latlon2xyz(points[:, :, 1], points[:, :, 0]), dim=-1)
            h_cp = sphere_2d.horizon_map(centers_x, points)
            local_coor = sphere_2d.cart3d_to_ctangent_local2d(centers_x, h_cp)

            centers = centers.reshape(-1, loc_feature_num).numpy()
            local_coor = local_coor.reshape(-1, loc_feature_num).numpy()
            inputs = np.concatenate((centers / 180 * np.pi, local_coor), axis=-1).reshape(-1, loc_feature_num * 2)
        else:
            raise NotImplementedError("The mapping is not provided.")

        return inputs, centers, points

    def XY2Ratio(self, X, k_neighbors=25):
        x = X[:, 0]
        y = X[:, 1]
        thetas = np.arctan2(y, x)
        thetas = thetas.reshape(-1, k_neighbors)
        ratio_lists = []
        multiples = []
        for theta in thetas:
            theta_unique, counts = np.unique(theta, return_counts=True)
            multiple_list = np.array([theta_unique, counts]).T
            idx = np.argsort(theta_unique)
            multiple_list = multiple_list[idx]
            ratios = []
            ratios_theta = np.zeros_like(theta)
            for i in range(multiple_list.shape[0]):
                if i < multiple_list.shape[0] - 1:
                    ratio = (
                        np.abs(multiple_list[i + 1][0] - multiple_list[i][0])
                        + np.abs(multiple_list[i - 1][0] - multiple_list[i][0])
                    ) / (2 * 2 * np.pi)
                else:
                    ratio = (
                        np.abs(multiple_list[0][0] - multiple_list[i][0])
                        + np.abs(multiple_list[i - 1][0] - multiple_list[i][0])
                    ) / (2 * 2 * np.pi)
                ratio = ratio / multiple_list[i][1]
                ratios.append(ratio)
                ratios_theta[theta == multiple_list[i][0]] = ratio
            ratio_lists.append(ratios_theta)
            multiple_list = np.concatenate([multiple_list, np.array([ratios]).T], axis=-1)
            multiples.append(multiple_list)
        return thetas, np.array(ratio_lists)


def date_to_inputseq(time_data, mask_dataset, output_horizon_len, input_seq_len, step_size):
    tmpdata = time_data[mask_dataset]
    L = tmpdata.shape[0]
    dataset_year = []
    dataset_month = []
    dataset_day = []
    dataset_hour = []
    for i in range(input_seq_len):
        dataset_year.append(pd.DatetimeIndex(np.array(tmpdata[i : L - output_horizon_len - input_seq_len + i])).year)
        dataset_month.append(pd.DatetimeIndex(np.array(tmpdata[i : L - output_horizon_len - input_seq_len + i])).month)
        dataset_day.append(pd.DatetimeIndex(np.array(tmpdata[i : L - output_horizon_len - input_seq_len + i])).day)
        dataset_hour.append(pd.DatetimeIndex(np.array(tmpdata[i : L - output_horizon_len - input_seq_len + i])).hour)

    dataset_year = np.stack(dataset_year, axis=1)
    dataset_month = np.stack(dataset_month, axis=1)
    dataset_day = np.stack(dataset_day, axis=1)
    dataset_hour = np.stack(dataset_hour, axis=1)

    num_samples, input_len = dataset_year.shape[0], dataset_year.shape[1]

    idx = [i * step_size for i in range(num_samples // step_size)]

    dataset_year, dataset_month, dataset_day, dataset_hour = (
        dataset_year[idx],
        dataset_month[idx],
        dataset_day[idx],
        dataset_hour[idx],
    )

    dataset_time = np.stack([dataset_year, dataset_month, dataset_day, dataset_hour], axis=-1)

    return dataset_time


def dataset_to_seq2seq(raw_data, mask_dataset, output_horizon_len, input_seq_len, step_size):
    tmpdata = raw_data[mask_dataset]
    mean, max, min, std = tmpdata.mean().values, tmpdata.max().values, tmpdata.min().values, tmpdata.std().values
    print("Mean:{:.4f}, Max:{:.4f}, Min:{:.4f}, Std:{:.4f}".format(mean, max, min, std))
    L = tmpdata.shape[0]
    dataset_x = []
    dataset_y = []
    for i in range(input_seq_len):
        dataset_x.append(tmpdata[i : L - output_horizon_len - input_seq_len + i])
    for j in range(output_horizon_len):
        dataset_y.append(tmpdata[j + input_seq_len : L - output_horizon_len + j])
    dataset_x = np.stack(dataset_x, axis=1)
    dataset_y = np.stack(dataset_y, axis=1)
    num_samples, input_len = dataset_x.shape[0], dataset_x.shape[1]
    dataset_x = dataset_x.reshape(num_samples, input_len, -1)
    dataset_y = dataset_y.reshape(num_samples, output_horizon_len, -1)
    idx = [i * step_size for i in range(num_samples // step_size)]
    dataset_x = dataset_x[idx]
    dataset_y = dataset_y[idx]
    return (dataset_x - mean) / std, dataset_y, mean, std


def write_dataset(data_x, data_y, data_context, out_file):
    n = data_x.shape[0]
    writer = FileWriter(out_file, n)

    for item in zip(data_x, data_y, data_context):
        bytes_ = pickle.dumps(item)
        writer.write_one(bytes_)
    writer.close()


def dump_weatherbench(
    data_dir, out_dir, data_name, attr_name, step_size, input_seq_len, output_horizon_len, start_date, end_date, shuffle
):

    constants = xr.open_mfdataset(str(data_dir / "constants" / "*.nc"), combine="by_coords")
    lsm = np.array(constants.lsm).flatten()
    height = np.array(constants.orography).flatten()
    latitude = np.array(constants.lat2d).flatten()
    longitude = np.array(constants.lon2d).flatten()
    geo_context = np.stack([lsm, height, latitude, longitude], axis=-1)

    if data_name != "component_of_wind":
        data = xr.open_mfdataset(str(data_dir / data_name / "*.nc"), combine="by_coords")
        time = data.time.values
        mask_dataset = np.bitwise_and(np.datetime64(start_date) <= time, time <= np.datetime64(end_date))
        lon, lat = np.meshgrid(data.lon - 180, data.lat)
        lonlat = np.array([lon, lat])
        lonlat = lonlat.reshape(2, 32 * 64).T
        raw_data = data.__getattr__(attr_name)

        if len(raw_data.shape) == 4:  # when there are different level, we choose the 13-th level which is sea level
            raw_data = raw_data[:, -1, ...]
        seq2seq_data, seq2seq_label, seq_mean, seq_std = dataset_to_seq2seq(
            raw_data, mask_dataset, output_horizon_len, input_seq_len, step_size
        )

        seq_scaler = {
            "mean": np.asarray([seq_mean]),
            "std": np.asarray([seq_std]),
        }

        time_data = data.time
        time_context = date_to_inputseq(time_data, mask_dataset, output_horizon_len, input_seq_len, step_size)

        num_samples = seq2seq_data.shape[0]
        node_num = geo_context.shape[0]
        time_len = time_context.shape[1]

        time_context = np.repeat(time_context[:, :, None, :], node_num, axis=2)
        geo_context = np.repeat(geo_context[None, :, :], time_len * num_samples, axis=0).reshape(
            num_samples, time_len, node_num, -1
        )
        context = np.concatenate([time_context, geo_context], axis=-1)

        num_test = round(num_samples * 0.1)
        num_train = round(num_samples * 0.85)
        num_val = num_samples - num_test - num_train
        print(
            "Number of training samples: {}, validation samples:{}, test samples:{}".format(
                num_train, num_val, num_test
            )
        )

        if shuffle:
            idx = np.random.permutation(np.arange(num_samples))
            seq2seq_data = seq2seq_data[idx]
            context = context[idx]
            seq2seq_label = seq2seq_label[idx]

        train_x = seq2seq_data[:num_train][:, :, :, None]
        train_context = context[:num_train]
        train_y = seq2seq_label[:num_train][:, :, :, None]

        val_x = seq2seq_data[num_train : num_train + num_val][:, :, :, None]
        val_context = context[num_train : num_train + num_val]
        val_y = seq2seq_label[num_train : num_train + num_val][:, :, :, None]

        test_x = seq2seq_data[num_train + num_val :][:, :, :, None]
        test_context = context[num_train + num_val :]
        test_y = seq2seq_label[num_train + num_val :][:, :, :, None]

    else:
        data_u = xr.open_mfdataset(str(data_dir / "10m_u_component_of_wind/*.nc"), combine="by_coords")
        data_v = xr.open_mfdataset(str(data_dir / "10m_v_component_of_wind/*.nc"), combine="by_coords")

        time = data_u.time.values
        mask_dataset = np.bitwise_and(np.datetime64(start_date) <= time, time <= np.datetime64(end_date))
        lon, lat = np.meshgrid(data_u.lon - 180, data_u.lat)
        lonlat = np.array([lon, lat])
        lonlat = lonlat.reshape(2, 32 * 64).T

        raw_data_u = data_u.u10
        raw_data_v = data_v.v10

        seq2seq_data_u, seq2seq_label_u, seq_mean_u, seq_std_u = dataset_to_seq2seq(
            raw_data_u, mask_dataset, output_horizon_len, input_seq_len, step_size
        )
        seq2seq_data_v, seq2seq_label_v, seq_mean_v, seq_std_v = dataset_to_seq2seq(
            raw_data_v, mask_dataset, output_horizon_len, input_seq_len, step_size
        )

        seq_scaler = {
            "mean": np.asarray([seq_mean_u, seq_mean_v]),
            "std": np.asarray([seq_std_u, seq_std_v]),
        }

        time_data = data_u.time
        time_context = date_to_inputseq(time_data, mask_dataset, output_horizon_len, input_seq_len, step_size)

        num_samples = seq2seq_data_u.shape[0]
        node_num = geo_context.shape[0]
        time_len = time_context.shape[1]

        time_context = np.repeat(time_context[:, :, None, :], node_num, axis=2)
        geo_context = np.repeat(geo_context[None, :, :], time_len * num_samples, axis=0).reshape(
            num_samples, time_len, node_num, -1
        )
        context = np.concatenate([time_context, geo_context], axis=-1)

        num_test = round(num_samples * 0.1)
        num_train = round(num_samples * 0.85)
        num_val = num_samples - num_test - num_train
        print(
            "Number of training samples: {}, validation samples:{}, test samples:{}".format(
                num_train, num_val, num_test
            )
        )

        if shuffle:
            idx = np.random.permutation(np.arange(num_samples))
            seq2seq_data_u = seq2seq_data_u[idx]
            seq2seq_label_u = seq2seq_label_u[idx]
            seq2seq_data_v = seq2seq_data_v[idx]
            seq2seq_label_v = seq2seq_label_v[idx]
            context = context[idx]

        train_x = np.stack([seq2seq_data_u[:num_train], seq2seq_data_v[:num_train]], axis=-1)
        train_context = context[:num_train]
        train_y = np.stack([seq2seq_label_u[:num_train], seq2seq_label_v[:num_train]], axis=-1)

        val_x = np.stack(
            [seq2seq_data_u[num_train : num_train + num_val], seq2seq_data_v[num_train : num_train + num_val]], axis=-1
        )
        val_context = context[num_train : num_train + num_val]
        val_y = np.stack(
            [seq2seq_label_u[num_train : num_train + num_val], seq2seq_label_v[num_train : num_train + num_val]],
            axis=-1,
        )

        test_x = np.stack([seq2seq_data_u[num_train + num_val :], seq2seq_data_v[num_train + num_val :]], axis=-1)
        test_context = context[num_train + num_val :]
        test_y = np.stack([seq2seq_label_u[num_train + num_val :], seq2seq_label_v[num_train + num_val :]], axis=-1)

    print(f"train_x: {train_x.shape}, train_y: {train_y.shape}, train_context: {train_context.shape}")
    print(f"val_x: {val_x.shape}, val_y: {val_y.shape}, val_context: {val_context.shape}")
    print(f"test_x: {test_x.shape}, test_y: {test_y.shape}, test_context: {test_context.shape}")

    out_path = out_dir / data_name
    out_path.mkdir(exist_ok=True, parents=True)

    write_dataset(train_x, train_y, train_context, out_path / "train.ffr")
    write_dataset(val_x, val_y, val_context, out_path / "val.ffr")
    write_dataset(test_x, test_y, test_context, out_path / "test.ffr")

    print("Generate Kernel...")
    kernel_generator = KernelGenerator(lonlat)
    kernel_info = {
        "sparse_idx": kernel_generator.sparse_idx,
        "MLP_inputs": kernel_generator.MLP_inputs,
        "geodesic": kernel_generator.geodesic.flatten(),
        "angle_ratio": kernel_generator.ratio_lists.flatten(),
    }

    with open(out_path / "kernel.pkl", "wb") as f:
        pickle.dump(kernel_info, f)

    with open(out_path / "scaler.pkl", "wb") as f:
        pickle.dump(seq_scaler, f)

    print("Done.")


if __name__ == "__main__":
    data_dir = Path("***/all_5.625deg")
    out_dir = Path("/3fs-jd/prod/platform_team/dchq/ffdataset/WeatherBench")

    data_names = ["2m_temperature", "relative_humidity", "component_of_wind", "total_cloud_cover"]
    attr_names = ["t2m", "r", "uv10", "tcc"]

    step_size = 24
    input_seq_len = 12
    output_horizon_len = 12
    start_date = "1980-01-01"
    end_date = "2019-01-01"
    shuffle = False

    for data_name, attr_name in zip(data_names, attr_names):
        print(data_name, attr_name)
        dump_weatherbench(
            data_dir,
            out_dir,
            data_name,
            attr_name,
            step_size,
            input_seq_len,
            output_horizon_len,
            start_date,
            end_date,
            shuffle,
        )
