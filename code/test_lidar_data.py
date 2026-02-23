import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
import open3d as o3d
import copy
from scipy.spatial import KDTree
from load_data import *
import matplotlib
matplotlib.use("TkAgg")  # 或 Qt5Agg
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_dataset

# parameters 
test1 = 534
test2 = 537


data = load_dataset(dataset=21, data_dir="data")
lidar_angle_min = data["lidar_angle_min"]
lidar_angle_max = data["lidar_angle_max"]
lidar_angle_increment = data["lidar_angle_increment"]
lidar_range_min = data["lidar_range_min"]
lidar_range_max = data["lidar_range_max"]
lidar_ranges = data["lidar_ranges"]
lidar_stamps = data["lidar_stamps"]
angle_min = float(lidar_angle_min)
angle_inc = float(lidar_angle_increment)
print("lidar shape",lidar_angle_increment.shape)
#print all data keys and their shapes
for key, value in data.items():
    print(f"{key}: shape {value.shape}")
print("lidar_range_max:", lidar_range_max)
print("lidar_range_min:", lidar_range_min)
print("lidar_angle_min:", lidar_angle_min)
print("lidar_angle_max:", lidar_angle_max)
print("lidar_angle_increment:", lidar_angle_increment)
n_beams = lidar_ranges.shape[0]

angles = angle_min + angle_inc * np.arange(n_beams)
cos = np.cos(angles)[:, None]   # (1081, 1)
sin = np.sin(angles)[:, None]   # (1081, 1)

point_clouds_x = lidar_ranges * cos   # (1081, 4962)
point_clouds_y = lidar_ranges * sin   # (1081, 4962)

point_clouds = np.stack((point_clouds_x, point_clouds_y), axis=-1)
# shape: (1081, 4962, 2)
print("point_clouds shape:", point_clouds.shape)




plt.figure(figsize=(10,5))
# plt.plot(lidar_ranges[:,0], label="lidar scan")
plt.scatter(point_clouds_x[:,test1],point_clouds_y[:,test1], label="before")
plt.scatter(point_clouds_x[:,test2],point_clouds_y[:,test2], label="after")
plt.legend()
plt.show()
def best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == B.shape and A.shape[1] == 2
    print ("best_fit_transform: A shape:", A.shape, "B shape:", B.shape)
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection if det(R) < 0
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = R
    T[:2, 2] = t
    return T

def icp_one_iteration_scipy(src_pts, dst_pts, T_init, max_corr_dist, tree):
    # 1. 初始化與座標變換
    if T_init is None: T_init = np.eye(3)
    
    # 將 source 點雲套用目前的變換矩陣 (Ns, 3)
    src_tf = (src_pts @ T_init[:2, :2].T) + T_init[:2, 2]

    # 2. 建立 KDTree (對目標點雲 dst_pts)
    # 注意：在正式的 loop 中，建議把這行移到 loop 外面只做一次
    

    # 3. 核心查詢：一次查完所有點的最近鄰
    # distances: 每個點到最近鄰的距離
    # indices: 最近鄰在 dst_pts 中的索引
    distances, indices = tree.query(src_tf, k=1, workers=-1)

    # 4. 篩選 Inliers (向量化操作，取代 for 迴圈與 if)
    mask = distances <= max_corr_dist
    
    # 這裡保證了兩者的 shape 一定會完全相同，解決你之前的 assert 報錯
    matched_src = src_tf[mask]         # 變換後的 source 點 (用於計算增量 dT)
    matched_dst = dst_pts[indices[mask]] # 對應的 target 點

    n_in = len(matched_src)
    print(n_in, "inliers found with max_corr_dist =", max_corr_dist)
    if n_in < 3:
        return T_init, float("inf"), n_in

    # 5. 計算增量變換 dT (matched_src -> matched_dst)
    # 這裡傳入 matched_src (已經在 T_init 位置的點)
    dT = best_fit_transform(matched_src, matched_dst)
    T_new = dT @ T_init
    
    rmse = np.sqrt(np.mean(distances[mask]**2))
    #calculate varianence of inlier distances
    inlier_distances = distances[mask]
    variance = np.var(inlier_distances)
    return T_new, rmse, n_in, variance

trans_init = np.asarray([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0.0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
ranges_A = lidar_ranges[:, test1]
ranges_B = lidar_ranges[:, test2]

# 2. 建立布林遮罩 (Boolean Mask)，只保留 1m 到 20m 之間的有效點
mask_A = (ranges_A > 1.5) & (ranges_A < 10)
mask_B = (ranges_B > 1.5) & (ranges_B < 10)

# 3. 用遮罩篩選出對應的 X, Y 座標
A = point_clouds[:, test1, :][mask_A]  # 形狀會從 (1081, 2) 變成類似 (850, 2)
B = point_clouds[:, test2, :][mask_B]  # 形狀會從 (1081, 2) 變成類似 (862, 2)
tree = KDTree(A)
# for i in range(num_pc):
#     target_pc = to_o3d_pcd(load_pc(obj_name, i))
T_init = np.eye(3)
best_rmse = float("inf")
rmse = 5
variance = 0.5
size= 1*2
iteration_size = 50   
for it in range(iteration_size):
    T_new, rmse, n_in, variance = icp_one_iteration_scipy(B, A, T_init, max_corr_dist=20/(it**1.2+1),tree=tree)
    print(f"iter {it:02d}: rmse={rmse:.4f}, inliers={n_in}, variance={variance:.4f}")
    if n_in < 30:
        print("Too few correspondences, stop.")
        break
    if it > 0 and np.linalg.norm(rmse - rmse_prev)/rmse_prev < 1e-10 and rmse < 0.01:
        print("Converged.")
        T = T_new
        break
    rmse_prev = rmse
    T = T_new
    T_init = T_new
    # recode the minimum rmse and corresponding T
    if rmse < best_rmse:
        best_rmse = rmse
        best_T = T
# print("Best RMSE:", best_rmse)
# print("rmse:", rmse, "inliers:", n_in)
print("T:\n", T)
BtoA = (best_T[:2, :2] @ B.T).T + best_T[:2, 2]
# visualize_icp_result(target_pc_data, target_pc_data, T)
R = best_T[:2, :2]
print("Estimated rotation:\n", R)
t = best_T[:2, 2]
yaw_angle = np.arctan2(R[1, 0], R[0, 0])
print("Estimated yaw (degrees):", np.degrees(yaw_angle))
plt.figure(figsize=(10,5))
plt.scatter(A[:,0], A[:,1], label="A (source) / T")
plt.scatter(B[:,0], B[:,1], label="B (target) / T+1")
plt.plot(BtoA[:,0], BtoA[:,1], label="Transformed B (final) / T+1 transformed to T")
plt.legend()
plt.title("ICP Result")
plt.show()








