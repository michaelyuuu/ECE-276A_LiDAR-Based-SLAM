import os

import gtsam
import math as m
import numpy as np
import cv2
from scipy.spatial.transform import Rotation 
from load_data import *
from pr2_utils import *
from bresenham2D import bresenham2D, test_bresenham2D
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("TkAgg")  # 或 Qt5Agg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from load_data import load_dataset

dataset = 20
data = load_dataset(dataset, data_dir="data")
imu_angular_velocity = data["imu_angular_velocity"]
print(type(imu_angular_velocity), len(imu_angular_velocity) if hasattr(imu_angular_velocity, "__len__") else "no len")
encoder_counts = data["encoder_counts"]
encoder_time_stamps = data["encoder_stamps"]
imu_time_stamps = data["imu_stamps"]
lidar_angle_min = data["lidar_angle_min"]
lidar_angle_max = data["lidar_angle_max"]
lidar_angle_increment = data["lidar_angle_increment"]
lidar_range_min = data["lidar_range_min"]
lidar_range_max = data["lidar_range_max"]
lidar_ranges = data["lidar_ranges"]
lidar_stamps = data["lidar_stamps"]
print(type(encoder_counts), len(encoder_counts) if hasattr(encoder_counts, "__len__") else "no len")
print("encoder_counts keys:", encoder_counts.keys() if hasattr(encoder_counts, "keys") else "no keys")
freq = 40
FR = encoder_counts[0,:]
FL = encoder_counts[1,:]
RR = encoder_counts[2,:]
RL = encoder_counts[3,:]
Te = encoder_time_stamps
trackwidth = 0.3937

roll = imu_angular_velocity[0,:]
pitch = imu_angular_velocity[1,:]
yaw = imu_angular_velocity[2,:]
Ti = imu_time_stamps
Vr = (FR + RR) / 2 *0.0022 *freq
Vl = (FL + RL) / 2 *0.0022 *freq
x = np.zeros(len(Vr))
y = np.zeros(len(Vr))
phi = np.zeros(len(Vr))
v = np.zeros(len(Vr))

v = (Vr + Vl) / 2
angle_min = float(lidar_angle_min)
angle_inc = float(lidar_angle_increment)
print("lidar shape",lidar_angle_increment.shape)

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

# downsizing lidar points for faster processing
ds = 3
cos_ds = cos[::ds]
sin_ds = sin[::ds]
point_clouds_x = lidar_ranges[::ds,:] * cos_ds
point_clouds_y = lidar_ranges[::ds,:] * sin_ds

point_clouds = np.stack((point_clouds_x, point_clouds_y), axis=-1)
idx_lidar = align_closest(lidar_stamps, data["rgb_stamps"])
idx_disp = align_closest(data["disp_stamps"], data["rgb_stamps"])
def draw_car(ax, x, y, yaw, size=0.5, color='red'):
    """
    在指定的座標軸 (ax) 上畫一個代表車體方向的三角形
    size: 控制車子的大小 (單位與你的 x, y 相同)
    """
    pt_front = [x + size * np.cos(yaw), 
                y + size * np.sin(yaw)]
    pt_left  = [x - (size/2) * np.cos(yaw) - (size/2.5) * np.sin(yaw), 
                y - (size/2) * np.sin(yaw) + (size/2.5) * np.cos(yaw)]
    pt_right = [x - (size/2) * np.cos(yaw) + (size/2.5) * np.sin(yaw), 
                y - (size/2) * np.sin(yaw) - (size/2.5) * np.cos(yaw)]
    
    triangle = np.array([pt_front, pt_left, pt_right, pt_front])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black', linewidth=1, zorder=10)
    ax.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.9, zorder=10)
def printimage(i):
 
    try:
        i_rgb = idx_lidar[i] 
    except:
        i_rgb = i

    img = cv2.imread(f"data/dataRGBD/RGB{dataset}/rgb{dataset}_{i_rgb}.png")
    if img is None:
        img = cv2.imread(f"data/dataRGBD/RGB{dataset}/rgb{dataset}_{i}.png")
        if img is None:
            print(f"[Warning] Missing RGB image for scan {i}. Skipping color.")
            img = np.zeros((480, 640, 3), dtype=np.uint8) # 建立虛擬畫布，防止後續 shape 錯誤
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_rgb, W_rgb = img.shape[:2]

    i_disp = idx_disp[i]
    disparity = cv2.imread(f"data/dataRGBD/Disparity{dataset}/disparity{dataset}_{i_disp}.png", cv2.IMREAD_UNCHANGED)
    if disparity is None:
        print(f"[Warning] Missing Disparity for scan {i}. Skipping color.")
        disparity = np.zeros((480, 640, 3), dtype=np.uint8) # 建立虛擬畫布，防止後續 shape 錯誤
        return None, None, None, None
        
    disp = disparity.astype(np.float32)
    disparity = disparity[::ds, ::ds]  # 3. 降採樣 Disparity
    H, W = disp.shape

    # 3. 【關鍵降採樣】：只降採樣「座標網格」與「深度值」
    v, u = np.mgrid[0:H:ds, 0:W:ds].astype(np.float32) 
    disp_ds = disp[0:H:ds, 0:W:ds] 
    H_ds, W_ds = u.shape

    # 4. 開始計算深度與座標 
    dd = -0.00304 * disp_ds + 3.31            
    depth = 1.03 / dd
    u_rgb = (526.37 * u + 19276.0 - 7877.07 * dd) / 585.051 
    v_rgb = (526.37 * v + 16662.0) / 585.051   
    xc = depth *(u - 315.84) / 585.051 
    yc = depth *(v - 242.94) / 585.051 
    zc = depth

    depth_aligned_to_rgb = np.zeros((H_rgb, W_rgb), dtype=np.float32)
    u_rgb_int = np.round(u_rgb).astype(np.int32)
    v_rgb_int = np.round(v_rgb).astype(np.int32)

    valid_mask = (u_rgb_int >= 0) & (u_rgb_int < W_rgb) & \
                 (v_rgb_int >= 0) & (v_rgb_int < H_rgb) & \
                 (depth > 0) & (disp_ds > 0)

    depth_aligned_to_rgb[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] = depth[valid_mask]
    
    R_depth_to_robot = np.eye(3) 
    roll = 0
    pitch = 0.36
    yaw_cam = 0.021 
    t = [0.18,0.005,0.36]
    R_rot_robot = Rotation.from_euler('xyz', [roll, pitch, yaw_cam], degrees=False).as_matrix()
    R_depth_to_robot = R_rot_robot @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    
    t_depth_to_robot = np.array(t).reshape(3, 1)
    points_depth = np.stack((xc.flatten(), yc.flatten(), zc.flatten()), axis=0) 
    points_robot = R_depth_to_robot @ points_depth + t_depth_to_robot  
    xc = points_robot[0, :].reshape(H_ds, W_ds)
    yc = points_robot[1, :].reshape(H_ds, W_ds)
    zc = points_robot[2, :].reshape(H_ds, W_ds)
    
    tolerance = 0.1
    valid_mask = valid_mask & (zc < tolerance) 
    valid_xc = xc[valid_mask]
    valid_yc = yc[valid_mask]
    valid_zc = zc[valid_mask]
    colors = img[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] / 255.0
    return valid_xc, valid_yc, valid_zc, colors

def best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == B.shape and A.shape[1] == 2
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = R
    T[:2, 2] = t
    return T

def icp_one_iteration_scipy(src_pts, dst_pts, T_init, max_corr_dist, tree):
    if T_init is None: T_init = np.eye(3)
    
    src_tf = (src_pts @ T_init[:2, :2].T) + T_init[:2, 2]
    distances, indices = tree.query(src_tf, k=1, workers=-1)

    mask = distances <= max_corr_dist
    matched_src = src_tf[mask]         
    matched_dst = dst_pts[indices[mask]] 

    n_in = len(matched_src)
    if n_in < 3:
        return T_init, float("inf"), n_in
    dT = best_fit_transform(matched_src, matched_dst)
    T_new = dT @ T_init   
    rmse = np.sqrt(np.mean(distances[mask]**2))
    
    inlier_distances = distances[mask]
    variance = np.var(inlier_distances)
    return T_new, rmse, n_in
def icp_loop(tree, A, B, T_init, max_corr):
    best_rmse = float("inf")
    rmse = 5
    iteration_size = 100  
    best_T = T_init.copy()         
    icp_ok = False
    rmse_prev = None
    for it in range(iteration_size):
        T_new, rmse, n_in = icp_one_iteration_scipy(B, A, T_init, max_corr, tree=tree)
        if n_in < 100:
            print("Too few correspondences, stop.")
            break
        if it > 0 and np.linalg.norm(rmse - rmse_prev) < 1e-5 and rmse < 0.1:
            # print("Converged.")
            T = T_new
            break
        rmse_prev = rmse
        T = T_new
        T_init = T_new
        if rmse < best_rmse:
            best_rmse = rmse
            best_T = T_new
            icp_ok = True
    # print("not converged, best rmse:", best_rmse)
        
    if not icp_ok:
        best_T = T_init.copy()
        best_rmse = rmse
    return best_T, icp_ok, best_rmse

grid_res = 0.05
map_max = 50 
map_dim = int(map_max / grid_res) 
grid_map = np.zeros((map_dim, map_dim))
offset_in_m = 20 
offset_in_px = int(offset_in_m / grid_res)  
offset = offset_in_px
observation_prior = np.log(80/20) 
maps_prior = 1 
log_odds = observation_prior*maps_prior
yaw_angle = np.zeros(len(lidar_stamps)-1)
current_accumulated_yaw = 0
T_temp = np.eye(3)
T_robot_lidar = np.eye(3)
T_robot_lidar[:2, 2] = np.array([0.13323, 0])

step = 10
close_step = 5 # Moved up for scope availability
frame_num = 20
total_frames = len(lidar_stamps) - step*2
# total_frames = 200
for l in range(frame_num):
    end_scan = (total_frames * (l + 1)) // frame_num
    print (f"Processing frame {l+1}/{frame_num}, total scans: {end_scan}")
    node_idx = 1
    yaw_measured = np.zeros(len(lidar_stamps))
    x_true = np.zeros(len(lidar_stamps))
    y_true = np.zeros(len(lidar_stamps))
    yaw_estimate = np.zeros(len(lidar_stamps))
    color_map = np.zeros((map_dim, map_dim, 3), dtype=np.float32)
    color_count = np.zeros((map_dim, map_dim), dtype=np.int32)
    graph = gtsam.NonlinearFactorGraph()
    prior_mean = gtsam.Pose2(0.0, 0.0, 0.0) 
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.00001, 0.00001, 0.000001])) 
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.08])) 
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001])) 
    ICP_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
    graph.add(gtsam.PriorFactorPose2(1, prior_mean, prior_noise))
    initial = gtsam.Values()
    initial.insert(1, gtsam.Pose2(0.0, 0.0, 0.0))
    color_count = np.zeros((map_dim, map_dim), dtype=np.int32)
    current_accumulated_yaw = 0
    grid_map = np.zeros((map_dim, map_dim))
    x[:] = 0
    y[:] = 0
    count_success = 0
    T_temp = np.eye(3)
    for i in range(0, end_scan, step):
    # for i in range(0, 100, step):
        print(f"Processing scan {i} to {i+step}...")
        ranges_A = lidar_ranges[::ds, i]
        ranges_B = lidar_ranges[::ds, i+step]
        mask_A = (ranges_A > 0.5) & (ranges_A < 20)
        mask_B = (ranges_B > 0.5) & (ranges_B < 20)

        A = point_clouds[:, i, :][mask_A]  
        B = point_clouds[:, i+step, :][mask_B]  
        tree = KDTree(A)
        
        T_init = np.eye(3)  
        dt = Te[i+step] - Te[i]  
        delta_yaw_enc = yaw[i] * dt
        dx_enc = v[i] * dt * np.sinc(delta_yaw_enc/2) * np.cos(delta_yaw_enc/2)
        dy_enc = v[i] * dt * np.sinc(delta_yaw_enc/2) * np.sin(delta_yaw_enc/2)
        
        T_robot_local = np.eye(3)
        T_robot_local[0, 2] = dx_enc
        T_robot_local[1, 2] = dy_enc
        T_robot_local[:2, :2] = np.array([
            [np.cos(delta_yaw_enc), -np.sin(delta_yaw_enc)], 
            [np.sin(delta_yaw_enc),  np.cos(delta_yaw_enc)]
        ])
        
        T_init = np.linalg.inv(T_robot_lidar) @ T_robot_local @ T_robot_lidar
        best_T= icp_loop(tree, A, B, T_init, max_corr = 1)[0] 
        
        R_dot = best_T[:2, :2]
        t_dot = best_T[:2, 2]    
        yaw_dot = np.arctan2(R_dot[1, 0], R_dot[0, 0])   
        if yaw_dot > np.radians(90) or yaw_dot < np.radians(-90):
            print("Unrealistic yaw change detected, ignoring ICP result.")
            best_T = T_init.copy()
            yaw_dot = 0

        T_robot_oldtonew = T_robot_lidar @ best_T @ np.linalg.inv(T_robot_lidar) 
        T_robot_new = T_temp @ T_robot_oldtonew 
        T_temp = T_robot_new

        R = T_robot_new[:2, :2]
        t = T_robot_new[:2, 2]  
        R_dot = T_robot_oldtonew[:2, :2]
        t_dot = T_robot_oldtonew[:2, 2]
        yaw_dot_rob = np.arctan2(R_dot[1, 0], R_dot[0, 0]) 
        global_yaw = np.arctan2(R[1, 0], R[0, 0])
        yaw_estimate[i+step] = yaw_estimate[i] + yaw_dot_rob

        x_true[i+step] = x_true[i] + dt * v[i] * np.sinc(yaw[i]*dt/2) * np.cos(yaw_measured[i]+yaw[i]*dt/2)
        y_true[i+step] = y_true[i] + dt * v[i] * np.sinc(yaw[i]*dt/2) * np.sin(yaw_measured[i]+yaw[i]*dt/2)
        yaw_measured[i+step] = yaw_measured[i] + yaw[i] * dt
        x_true_dot = x_true[i+step] - x_true[i]
        y_true_dot = y_true[i+step] - y_true[i]
        x[i+step] = T_robot_new[0, 2]
        y[i+step] = T_robot_new[1, 2] 
        
        node_idx += 1
        initial.insert(node_idx, gtsam.Pose2(T_robot_new[0, 2], T_robot_new[1, 2], global_yaw))
        graph.add(gtsam.BetweenFactorPose2(node_idx-1, node_idx, gtsam.Pose2(T_robot_oldtonew[0, 2], T_robot_oldtonew[1, 2], yaw_dot_rob), ICP_noise))
        # graph.add(gtsam.BetweenFactorPose2(node_idx-1, node_idx, gtsam.Pose2(x_true_dot, y_true_dot, yaw[i] * dt), odometry_noise))

        # Loop Closure Detection for close poses
        SEARCH_RADIUS = 1.5
        MIN_LOOP_AGE = 20    
        for prev_idx in range(1, node_idx - MIN_LOOP_AGE):
            prev_pose = initial.atPose2(prev_idx)
            
            dist = np.sqrt((T_robot_new[0, 2] - prev_pose.x())**2 + 
                        (T_robot_new[1, 2] - prev_pose.y())**2)
            if dist < SEARCH_RADIUS and node_idx > 60: # only start looking for loop closures after we have enough nodes
                print(f"find candidate {node_idx} near previous node {prev_idx}")
                dx_enc = T_robot_new[0, 2] - prev_pose.x()
                dy_enc = T_robot_new[1, 2] - prev_pose.y()
                delta_yaw_enc = np.arctan2(np.sin(global_yaw - prev_pose.theta()), 
                                        np.cos(global_yaw - prev_pose.theta())) 
                T_robot_local = np.eye(3)
                theta_curr = prev_pose.theta()
                dx_local = dx_enc * np.cos(theta_curr) + dy_enc * np.sin(theta_curr)
                dy_local = -dx_enc * np.sin(theta_curr) + dy_enc * np.cos(theta_curr)

                T_robot_local[0, 2] = dx_local
                T_robot_local[1, 2] = dy_local
                T_robot_local[:2, :2] = np.array([
                [np.cos(delta_yaw_enc), -np.sin(delta_yaw_enc)], 
                [np.sin(delta_yaw_enc),  np.cos(delta_yaw_enc)]
                ])
                T_init = np.linalg.inv(T_robot_lidar) @ T_robot_local @ T_robot_lidar
                
                idex_prev_lidar = (prev_idx - 1) * step            
                ranges_A = lidar_ranges[::ds, idex_prev_lidar]
                mask_A = (ranges_A > 0.1) & (ranges_A < 20)
                A_loop = point_clouds[:, idex_prev_lidar, :][mask_A] 
                tree_loop = KDTree(A_loop)
                T_best_near, icp_ok, rmse = icp_loop(tree_loop, A_loop, B, T_init, max_corr = 2)     
                BtoA = (T_best_near[:2, :2] @ B.T).T + T_best_near[:2, 2]

                if icp_ok and rmse < 0.15:
                    graph.add(gtsam.BetweenFactorPose2(prev_idx, node_idx, gtsam.Pose2(T_best_near[0, 2], T_best_near[1, 2], np.arctan2(T_best_near[1, 0], T_best_near[0, 0])), loop_noise))
                    print(f"Loop closure added between node {node_idx} and node {prev_idx} with rmse {rmse:.4f}")
                    # print("loop closure Success at global position", T_robot_new[0, 2], T_robot_new[1, 2])
                    # plot A and BtoA for debugging
                    # plt.figure(figsize=(8, 8))
                    # plt.scatter(A_loop[:, 0], A_loop[:, 1], s=5, label="A (Previous Scan)", alpha=0.5)
                    # plt.scatter(BtoA[:, 0], BtoA[:, 1], s=5, label="B transformed to A (Current Scan)", alpha=0.5)      
                    # plt.title(f"Loop Closure ICP between Node {prev_idx} and Node {node_idx}\nRMSE: {rmse:.4f}")
                    # plt.legend()
                    # plt.axis("equal")
                    # plt.show(block=False)
                    count_success += 1
                else:
                    # plot A and BtoA for debugging
                    # plt.figure(figsize=(8, 8))
                    # plt.scatter(A_loop[:, 0], A_loop[:, 1], s=5, label="A (Previous Scan)", alpha=0.5)
                    # plt.scatter(BtoA[:, 0], BtoA[:, 1], s=5, label="B transformed to A (Current Scan)", alpha=0.5)      
                    # plt.title(f"Loop Closure ICP between Node {prev_idx} and Node {node_idx}\nRMSE: {rmse:.4f}")
                    # plt.legend()
                    # plt.axis("equal")
                    # plt.show(block=True)
                    print(f"Loop closure ICP failed between node {node_idx} and node {prev_idx}. rmse: {rmse:.4f}")
        #add loop closure with fixed step to close previous loop           
        if node_idx % close_step == 1: 
            print("Adding loop closure factor between node", node_idx, "and node", node_idx-close_step)
            T_init = np.eye(3)    
            local_prev_idx = node_idx - close_step
            idex_local_prev = (local_prev_idx - 1) * step
            idex_curr = i + step
            dx_enc = x_true[idex_curr] - x_true[idex_local_prev]
            dy_enc = y_true[idex_curr] - y_true[idex_local_prev]
            delta_yaw_enc = np.arctan2(np.sin(yaw_measured[idex_curr] - yaw_measured[idex_local_prev]), 
                            np.cos(yaw_measured[idex_curr] - yaw_measured[idex_local_prev]))

            T_robot_local = np.eye(3)
            theta_curr = yaw_measured[idex_local_prev]
            dx_local = dx_enc * np.cos(theta_curr) + dy_enc * np.sin(theta_curr)
            dy_local = -dx_enc * np.sin(theta_curr) + dy_enc * np.cos(theta_curr)

            T_robot_local[0, 2] = dx_local
            T_robot_local[1, 2] = dy_local
            T_robot_local[:2, :2] = np.array([
            [np.cos(delta_yaw_enc), -np.sin(delta_yaw_enc)], 
            [np.sin(delta_yaw_enc),  np.cos(delta_yaw_enc)]
            ])
            T_init = np.linalg.inv(T_robot_lidar) @ T_robot_local @ T_robot_lidar
            
            ranges_A = lidar_ranges[::ds, idex_local_prev]
            ranges_B = lidar_ranges[::ds, idex_curr]
            mask_A = (ranges_A > 0.2) & (ranges_A < 20)
            mask_B = (ranges_B > 0.2) & (ranges_B < 20)
            A_local = point_clouds[:, idex_local_prev, :][mask_A] 
            B_local = point_clouds[:, idex_curr, :][mask_B] 
            tree_local = KDTree(A_local)
            T_best_10 , icp_ok, rmse = icp_loop(tree_local, A_local, B_local, T_init, max_corr = 0.5)    
            if not icp_ok or rmse > 0.15:
                print("ICP failed for loop closure for fixed timestep.")
            else:
                T_robot_oldtnew_10 = T_robot_lidar @ T_best_10 @ np.linalg.inv(T_robot_lidar) 
                graph.add(gtsam.BetweenFactorPose2(node_idx-close_step, node_idx, gtsam.Pose2(T_robot_oldtnew_10[0, 2], T_robot_oldtnew_10[1, 2], np.arctan2(T_robot_oldtnew_10[1, 0], T_robot_oldtnew_10[0, 0])), loop_noise))
                print("Loop closure success")

        T_world_lidar = T_robot_new @ T_robot_lidar 
        world_scan = (T_world_lidar[:2, :2] @ point_clouds[:,i+step,:].T).T + T_world_lidar[:2, 2]
        # xc, yc, zc, colors = printimage(i+step)
        # posc_world = T_robot_new[:2, :2] @ np.stack((xc, yc), axis=0) + T_robot_new[:2, 2].reshape(2, 1)
        
        # xc_world = posc_world[0, :] 
        # yc_world = posc_world[1, :] 
        world_scan_x = world_scan[:, 0] 
        world_scan_y = world_scan[:, 1]
        car_px_x = T_world_lidar[0, 2] / grid_res + offset
        car_px_y = T_world_lidar[1, 2] / grid_res + offset
        points_px = np.vstack((world_scan_x / grid_res + offset, 
                        world_scan_y / grid_res + offset)).T.astype(np.int32)
        # color_points_px = np.vstack((xc_world / grid_res + offset,
        #                             yc_world / grid_res + offset)).T.astype(np.int32)
        all_free = []  
        
        for grid_x, grid_y in points_px.astype(np.int32):
            ray = bresenham2D(car_px_x, car_px_y, int(grid_x), int(grid_y)).T.astype(np.int32)  
            if ray.shape[0] <= 1:
                continue
            all_free.append(ray[:-1])  
        if len(all_free) > 0:
            free_points_px = np.vstack(all_free)  
        else:
            free_points_px = np.zeros((0, 2), dtype=np.int32)
        for grid_x, grid_y in free_points_px:
            if 0 <= grid_x < grid_map.shape[0] and 0 <= grid_y < grid_map.shape[1]:
                grid_map[grid_y, grid_x] -= log_odds
        for grid_x, grid_y in points_px:
            if 0 <= grid_x < grid_map.shape[0] and 0 <= grid_y < grid_map.shape[1]:
                grid_map[grid_y, grid_x] += log_odds
        grid_map = np.clip(grid_map, -10, 10)
        # for (px, py), c in zip(color_points_px, colors):
        #     if 0 <= px < map_dim and 0 <= py < map_dim:
        #         color_map[py, px] = c
        #         color_count[py, px] += 1
    os.makedirs("result", exist_ok=True)
    os.makedirs(f"animation/dataset{dataset}", exist_ok=True)
    grid_map_pmf = 1.0 / (1.0 + np.exp(-grid_map)) 
    # ==========================================================
    # 1. 繪製「重建前 (Unoptimized)」的 SLAM 地圖與軌跡
    # ==========================================================
    print("Plotting UNOPTIMIZED maps...")
    grid_map_pmf_unopt = 1.0 / (1.0 + np.exp(-grid_map))

    occupied_unopt = grid_map_pmf_unopt > 0.7      
    free_unopt     = grid_map_pmf_unopt < 0.3
    color_map_unopt = color_map.copy()
    color_map_unopt[~free_unopt] = 0.0 # 把沒有走過的地方設為黑色

    # --- 圖 A: 重建前 (網格 + 軌跡 + 彩色地板) ---
    plt.figure(figsize=(10, 10))
    grid_cmap = LinearSegmentedColormap.from_list("occ_grid", [(0.0, "white"), (0.5, "lightgray"), (1.0, "black")])
    plt.imshow(grid_map_pmf_unopt, cmap=grid_cmap, vmin=0, vmax=1, origin='lower')
    ax = plt.gca() # 取得目前的座標軸
    x_px = x / grid_res + offset
    y_px = y / grid_res + offset
    mask_x = (x != 0) | (y != 0)
    plt.plot(x_px[mask_x], y_px[mask_x], label="Odometry + ICP Trajectory", color="blue", linewidth=1.5)
    h, w = grid_map_pmf_unopt.shape
    rgba_map_unopt = np.zeros((h, w, 4), dtype=np.float32)
    rgba_map_unopt[..., :3] = color_map_unopt
    rgba_map_unopt[..., 3] = ((color_count > 0) & free_unopt).astype(np.float32) * 0.9
    plt.imshow(rgba_map_unopt, origin='lower')
    plt.title("UNOPTIMIZED SLAM Map + RGB Floor")
    plt.legend()
    plt.axis("off")
    #save figure to result folder

    plt.savefig(f"animation/dataset{dataset}/UNOPTIMIZED_slam_map_dataset{dataset}_{l}.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)

    # --- 圖 B: 重建前 (純 Texture Map) ---
    plt.figure(figsize=(10, 10))
    # 直接畫出彩色地圖，origin='lower' 保持座標系一致
    plt.imshow(color_map_unopt, origin='lower') 
    plt.title("UNOPTIMIZED Pure Texture Map")
    plt.axis("off")
    plt.savefig(f"animation/dataset{dataset}/UNOPTIMIZED_pure_texture_dataset{dataset}_{l}.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()


    # ==========================================================
    # 2. 執行 GTSAM 優化
    # ==========================================================
    print("Running GTSAM Optimization...")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()

    estimated_poses = []
    for k in range(1, node_idx + 1):
        if result.exists(k):
            pose = result.atPose2(k)
            estimated_poses.append((pose.x(), pose.y(), pose.theta()))
    estimated_poses = np.array(estimated_poses)
    estimated_poses_px = estimated_poses[:, :2] / grid_res + offset

    # ==========================================================
    # 3. 清空舊地圖，準備重建 (Rebuild Map)
    # ==========================================================
    print("Rebuilding maps using optimized poses...")
    grid_map = np.zeros((map_dim, map_dim))
    color_map = np.zeros((map_dim, map_dim, 3), dtype=np.float32)
    color_count = np.zeros((map_dim, map_dim), dtype=np.int32)

    for k in range(1, node_idx + 1):
        if not result.exists(k):
            continue
            
        pose = result.atPose2(k)
        opt_x, opt_y, opt_yaw = pose.x(), pose.y(), pose.theta()
        
        T_robot_opt = np.eye(3)
        T_robot_opt[0, 2] = opt_x
        T_robot_opt[1, 2] = opt_y
        T_robot_opt[:2, :2] = np.array([
            [np.cos(opt_yaw), -np.sin(opt_yaw)], 
            [np.sin(opt_yaw),  np.cos(opt_yaw)]
        ])
        
        scan_idx = (k - 1) * step
        T_world_lidar_opt = T_robot_opt @ T_robot_lidar 
        world_scan_opt = (T_world_lidar_opt[:2, :2] @ point_clouds[:, scan_idx, :].T).T + T_world_lidar_opt[:2, 2]
        
        car_px_x = T_world_lidar_opt[0, 2] / grid_res + offset
        car_px_y = T_world_lidar_opt[1, 2] / grid_res + offset
        
        points_px = np.vstack((world_scan_opt[:, 0] / grid_res + offset, 
                            world_scan_opt[:, 1] / grid_res + offset)).T.astype(np.int32)
        
        all_free = []
        for grid_x, grid_y in points_px:
            ray = bresenham2D(car_px_x, car_px_y, int(grid_x), int(grid_y)).T.astype(np.int32)
            if ray.shape[0] > 1:
                all_free.append(ray[:-1])
                
        if len(all_free) > 0:
            free_points_px = np.vstack(all_free)
            for grid_x, grid_y in free_points_px:
                if 0 <= grid_x < map_dim and 0 <= grid_y < map_dim:
                    grid_map[grid_y, grid_x] -= log_odds
                    
        for grid_x, grid_y in points_px:
            if 0 <= grid_x < map_dim and 0 <= grid_y < map_dim:
                grid_map[grid_y, grid_x] += log_odds
                
        # 取 RGB-D
        ret = printimage(scan_idx)
        if ret[0] is not None:
            xc, yc, zc, colors = ret
            posc_world_opt = T_robot_opt[:2, :2] @ np.stack((xc, yc), axis=0) + T_robot_opt[:2, 2].reshape(2, 1)
            
            color_points_px = np.vstack((posc_world_opt[0, :] / grid_res + offset,
                                        posc_world_opt[1, :] / grid_res + offset)).T.astype(np.int32)
            
            for (px, py), c in zip(color_points_px, colors):
                if 0 <= px < map_dim and 0 <= py < map_dim:
                    color_map[py, px] = c
                    color_count[py, px] += 1

    grid_map = np.clip(grid_map, -10, 10)
    grid_map_pmf_opt = 1.0 / (1.0 + np.exp(-grid_map)) 
    print("Map Rebuilding Complete!")

    # ==========================================================
    # 4. 繪製「重建後 (Optimized)」的 SLAM 地圖與軌跡
    # ==========================================================
    occupied_opt = grid_map_pmf_opt > 0.7      
    free_opt     = grid_map_pmf_opt < 0.3
    color_map[~free_opt] = 0.0
    # --- 圖 C: 重建後 (網格 + 軌跡 + 彩色地板) ---
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_map_pmf_opt, cmap=grid_cmap, vmin=0, vmax=1, origin='lower')
    plt.plot(estimated_poses_px[:, 0], estimated_poses_px[:, 1], label="GTSAM Optimized Trajectory", color="green", linewidth=2)
    ax = plt.gca()
    final_x = estimated_poses_px[-1, 0]
    final_y = estimated_poses_px[-1, 1]
    final_yaw = estimated_poses[-1, 2]
    draw_car(ax, final_x, final_y, final_yaw, size=12, color='red')
    rgba_map_opt = np.zeros((h, w, 4), dtype=np.float32)
    rgba_map_opt[..., :3] = color_map 
    rgba_map_opt[..., 3] = ((color_count > 0) & free_opt).astype(np.float32) * 0.9
    plt.imshow(rgba_map_opt, origin='lower')

    plt.title("OPTIMIZED SLAM Map + RGB Floor")
    plt.legend()
    plt.axis("off")
    plt.savefig(f"animation/dataset{dataset}/FINAL_optimized_slam_map_dataset{dataset}_{l}.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    # --- 圖 D: 重建後 (純 Texture Map) ---
    plt.figure(figsize=(10, 10))
    plt.imshow(color_map, origin='lower') 
    plt.title("OPTIMIZED Pure Texture Map")
    plt.axis("off")
    plt.savefig(f"animation/dataset{dataset}/FINAL_optimized_pure_texture_dataset{dataset}_{l}.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()
    # ==========================================================
    # 5. 繪製軌跡比較圖 (Trajectory Comparison)
    # ==========================================================
    print("Plotting trajectory comparison...")
    mask = (x_true != 0) & (y_true != 0)

    # --- 圖 E: 軌跡比較圖 ---
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.plot(estimated_poses[:, 0], estimated_poses[:, 1], label="GTSAM Optimized", color="green", linewidth=1.5)
    plt.plot(x[mask_x], y[mask_x], label="Odometry + ICP", color="blue", linewidth=1.5)
    plt.plot(x_true[mask], y_true[mask], label="Dead Reckoning (Encoder)", color="red", linewidth=1.5)

    # 抓出目前 X 軸的總寬度 (公尺)
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]

    # 讓車子的大小永遠保持在當前畫面寬度的 3%
    dynamic_size = x_span * 0.03 
    
    # 【關鍵修正】：使用公尺座標 (estimated_poses) 而不是像素座標 (estimated_poses_px)
    metric_final_x = estimated_poses[-1, 0]
    metric_final_y = estimated_poses[-1, 1]

    draw_car(ax, metric_final_x, metric_final_y, final_yaw, size=dynamic_size, color='green')
    metric_final_x = x[-1]
    metric_final_y = y[-1]
    final_yaw = yaw_estimate[-1]

    draw_car(ax, metric_final_x, metric_final_y, final_yaw, size=dynamic_size, color='green')
    metric_final_x = x_true[-1]
    metric_final_y = y_true[-1]
    final_yaw = yaw_measured[-1]

    draw_car(ax, metric_final_x, metric_final_y, final_yaw, size=dynamic_size, color='green')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.title("Trajectory Comparison")
    plt.grid(True)
    plt.axis("equal") 
    plt.savefig(f"animation/dataset{dataset}/FINAL_trajectory_comparison_dataset{dataset}_{l}.png", dpi=300, bbox_inches='tight')
    plt.show(block=False) # 最後這個設為 True，讓所有圖片視窗可以一起停留
    plt.close('all')