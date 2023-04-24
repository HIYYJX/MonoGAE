import os
import numpy as np
import json
from tqdm import tqdm
import csv

def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])
def get_denorm(rotation_matrix, translation):
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = rotation_matrix
    lidar2cam[:3, 3] = translation.flatten()
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T

    denorm = -1 * equation_plane(ground_points_cam)
    return denorm
splitpath = "/DAIR-V2X/data/split_datas/single-infrastructure-split-data.json" ## /DAIR-V2X/是根据https://github.com/AIR-THU/DAIR-V2X/下载得到
dair_root = "/DAIR-V2X-I/single-infrastructure-side/" # 根据数据集官网https://thudair.baai.ac.cn/roadtest下载得到
with open(splitpath, "r") as f:
        infos  = json.load(f)
split='train'
split_list = infos[split]
kittifile = "/training/denorm/" # 自定义的路径
if not os.path.exists(kittifile):
        os.makedirs(kittifile)
for sample_id in tqdm(split_list):
    virtuallidar_to_camera_path = os.path.join(dair_root, "calib", "virtuallidar_to_camera", sample_id + ".json")
    with open(virtuallidar_to_camera_path, "r") as load_f:
        my_json = json.load(load_f)
    t_velo2cam = np.array(my_json["translation"])
    r_velo2cam = np.array(my_json["rotation"])
    denorm = get_denorm(r_velo2cam, t_velo2cam) # np.array([a, b, c, d])
    denorm_file = os.path.join(kittifile, sample_id+".txt") 
    with open(denorm_file, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        w.writerow(denorm)