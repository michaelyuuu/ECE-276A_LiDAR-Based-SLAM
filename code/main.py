
import math as m
import numpy as np
import time
from load_data import *
from pr2_utils import *
from bresenham2D import bresenham2D, test_bresenham2D
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("TkAgg")  # 或 Qt5Agg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from load_data import load_dataset
    # "encoder_counts": encoder_counts,
    # "encoder_stamps": encoder_stamps,
    # "lidar_angle_min": lidar_angle_min,
    # "lidar_angle_max": lidar_angle_max,
    # "lidar_angle_increment": lidar_angle_increment,
    # "lidar_range_min": lidar_range_min,
    # "lidar_range_max": lidar_range_max,
    # "lidar_ranges": lidar_ranges,
    # "lidar_stamps": lidar_stamps,
    # "imu_angular_velocity": imu_angular_velocity,
    # "imu_linear_acceleration": imu_linear_acceleration,
    # "imu_stamps": imu_stamps,
    # "disp_stamps": disp_stamps,
    # "rgb_stamps": rgb_stamps,
data = load_dataset(dataset=20, data_dir="data")
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
Vr = (FR + RR) / 2 *0.0022 *freq
Vl = (FL + RL) / 2 *0.0022 *freq
roll = imu_angular_velocity[0,:]
pitch = imu_angular_velocity[1,:]
yaw = imu_angular_velocity[2,:]
Ti = imu_time_stamps
x = np.zeros(len(Vr))
y = np.zeros(len(Vr))
phi = np.zeros(len(Vr))
v = np.zeros(len(Vr))
tau = 1/40
v = (Vr + Vl) / 2
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
# downsizing lidar points for faster processing
ds = 3
cos_ds = cos[::ds]
sin_ds = sin[::ds]
point_clouds_x = lidar_ranges[::ds,:] * cos_ds
point_clouds_y = lidar_ranges[::ds,:] * sin_ds

point_clouds = np.stack((point_clouds_x, point_clouds_y), axis=-1)
# shape: (1081, 4962, 2)
# print("point_clouds shape:", point_clouds.shape)


def best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == B.shape and A.shape[1] == 2
    # print ("best_fit_transform: A shape:", A.shape, "B shape:", B.shape)
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
    # print(n_in, "inliers found with max_corr_dist =", max_corr_dist)
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
    return T_new, rmse, n_in
grid_res = 0.05
map_max = 50 #25m max
map_dim = int(map_max / grid_res) # 放大一點，避免機器人走出邊界 (50m 範圍)
grid_map = np.zeros((map_dim, map_dim))
offset_in_m = 20 #offset the origin - meter from left upper corner
offset_in_px = int(offset_in_m / grid_res)  # 對應的像素偏移
offset = offset_in_px
observation_prior = np.log(80/20) # inverse of observation model
maps_prior = 1 # map prior odd
log_odds = observation_prior*maps_prior
yaw_angle = np.zeros(len(lidar_stamps)-1)
step = 10
current_accumulated_yaw = 0
T_temp = np.eye(3)
T_robot_lidar = np.eye(3)
T_robot_lidar[:2, 2] = np.array([0.13323, 0])
# for i in range(len(v)-1):
phi_encoder = np.zeros(len(lidar_stamps))
x_true = np.zeros(len(lidar_stamps))
y_true = np.zeros(len(lidar_stamps))
yaw_estimate = np.zeros(len(lidar_stamps))
for i in range(0, len(lidar_stamps)-step*2, step):
# for i in range(500, 1000, step):
    # print(f"Processing scan {i} to {i+step}...")
    ranges_A = lidar_ranges[::ds, i]
    ranges_B = lidar_ranges[::ds, i+step]

    # 2. 建立布林遮罩 (Boolean Mask)，只保留 1.5m 到 10m 之間的有效點
    mask_A = (ranges_A > 0.5) & (ranges_A < 20)
    mask_B = (ranges_B > 0.5) & (ranges_B < 20)

    # 3. 用遮罩篩選出對應的 X, Y 座標
    A = point_clouds[:, i, :][mask_A]  # 形狀會從 (1081, 2) 變成類似 (850, 2)
    B = point_clouds[:, i+step, :][mask_B]  # 形狀會從 (1081, 2) 變成類似 (862, 2)

    tree = KDTree(A)
    # for i in range(num_pc):
    #     target_pc = to_o3d_pcd(load_pc(obj_name, i))
    #use difference bewteen consecutive encoder counts to estimate initial pose change
    T_init = np.eye(3)
    
    dt = tau * step
    
    # 2. 透過 Encoder 算出機器人「局部」的變化量 (Local Robot Motion)
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
    
    # 3. 將機器人的局部運動，轉換為 LIDAR 的局部運動，作為 ICP 完美的初始猜測
    T_init = np.linalg.inv(T_robot_lidar) @ T_robot_local @ T_robot_lidar
    # print("Initial T:\n", T_init)
    # print("yaw from encoder:", np.degrees(yaw[i] * tau* step), "degrees")
    best_rmse = float("inf")
    rmse = 5
    variance = 0.5
    iteration_size = 50   
    best_T = T_init.copy()          # fallback transform if ICP fails
    best_rmse = float("inf")
    icp_ok = False
    rmse_prev = None
    for it in range(iteration_size):
        T_new, rmse, n_in = icp_one_iteration_scipy(B, A, T_init, max_corr_dist=20,tree=tree)
        # print(f"iter {it:02d}: rmse={rmse:.4f}, inliers={n_in}, variance={variance:.4f}")
        if n_in < 30:
            print("Too few correspondences, stop.")
            break
        if it > 0 and np.linalg.norm(rmse - rmse_prev)/(rmse_prev+1e-10) < 1e-12 and rmse< 0.01:
            print("Converged.")
            print("rmse change:", np.linalg.norm(rmse - rmse_prev))
            T = T_new
            break
        rmse_prev = rmse
        T = T_new
        T_init = T_new
        # recode the minimum rmse and corresponding T
        if rmse < best_rmse:
            best_rmse = rmse
            best_T = T_new
            icp_ok = True
        
    # If ICP never got a good update, we will use best_T = T_init (already set)
    if not icp_ok:
        # You can also choose identity instead:
        best_T = T_init.copy()
        pass
    # T_new = from t to t+1 -> T+1 = T_new @ T
    R_dot = best_T[:2, :2]
    t_dot = best_T[:2, 2]    
    yaw_dot = np.arctan2(R_dot[1, 0], R_dot[0, 0])    
    if yaw_dot > np.radians(20) or yaw_dot < np.radians(-20):
        print("Unrealistic yaw change detected, ignoring ICP result.")
        print("at interation:", it, "with rmse:", rmse)
        print("yaw_dot (degrees):", np.degrees(yaw_dot))
        best_T = T_init.copy()
        yaw_dot = 0
    print("Estimated yaw change from ICP:", np.degrees(yaw_dot), "degrees")
    print("Lidar old to new:", best_T[:2, 2])
    T_robot_oldtonew = T_robot_lidar @ best_T @ np.linalg.inv(T_robot_lidar) #from lidar frame to robot frame from T to T+1
    print("T_robot_oldtonew:\n", T_robot_oldtonew[:2, 2])
    T_robot_new = T_temp @ T_robot_oldtonew #accumulate transformation from start to current scan
    print("T_robot_new:\n", T_robot_new[:2, 2])
    T_temp = T_robot_new
    # T_robot_new = T_temp @ T_new
    # print("T:\n", T_robot_new)
    # visualize_icp_result(target_pc_data, target_pc_data, T)
    R = T_robot_new[:2, :2]
    t = T_robot_new[:2, 2]   
    # print(t)
    global_yaw = np.arctan2(R[1, 0], R[0, 0])
    yaw_estimate[i+step] = yaw_estimate[i] + yaw_dot
    print("yaw_dot_estimated:", np.degrees(yaw_dot), "degrees") # yaw change from scan t to t+1
    print("yaw_dot from scan:", np.degrees(global_yaw), "degrees") # yaw change from scan t to t+1
    print("yaw_dot from encoder:", np.degrees(yaw[i] * tau* step), "degrees") #yaw change from encoder
    print("Estimated yaw:", np.degrees(yaw_estimate[i+step]), "degrees")
    # Update your plotting variable
    x_true[i+step] = x_true[i] + tau * step * v[i] * np.sinc(yaw[i]*tau*step/2) * np.cos(phi_encoder[i]+yaw[i]*tau*step/2)
    y_true[i+step] = y_true[i] + tau * step * v[i] * np.sinc(yaw[i]*tau*step/2) * np.sin(phi_encoder[i]+yaw[i]*tau*step/2)
    phi_encoder[i+step] = phi_encoder[i] + yaw[i] * tau* step
    print("Encoder yaw:", np.degrees(phi_encoder[i+step]), "degrees")
    # 取得 ICP 在局部坐標系下的平移 delta_x, delta_y
    # x[i+step] = x[i] + dx*np.sinc(delta_yaw) * np.cos(phi[i]+delta_yaw)
    # y[i+step] = y[i] + dy*np.sinc(delta_yaw) * np.sin(phi[i]+delta_yaw)
    x[i+step] = T_robot_new[0, 2]
    y[i+step] = T_robot_new[1, 2]
    # T_robot_new = np.eye(3)
    # T_robot_new[:2, 2] = np.array([x_true[i+step], y_true[i+step]])
    # T_robot_new[:2, :2] = np.array([[np.cos(phi_encoder[i+step]), -np.sin(phi_encoder[i+step])], [np.sin(phi_encoder[i+step]), np.cos(phi_encoder[i+step])]])
    T_world_lidar = T_robot_new @ T_robot_lidar
    world_scan = (T_world_lidar[:2, :2] @ point_clouds[:,i+step,:].T).T + T_world_lidar[:2, 2]

    world_scan_x = world_scan[:, 0]
    world_scan_y = world_scan[:, 1]
    car_px_x = T_world_lidar[0, 2] / grid_res + offset
    car_px_y = T_world_lidar[1, 2] / grid_res + offset
    points_px = np.vstack((world_scan_x / grid_res + offset, 
                       world_scan_y / grid_res + offset)).T.astype(np.int32)
    all_free = []  # collect free cells from all rays
    for grid_x, grid_y in points_px.astype(np.int32):
        ray = bresenham2D(car_px_x, car_px_y, int(grid_x), int(grid_y)).T.astype(np.int32)  # (M, 2) = [[x0,y0],...]
        if ray.shape[0] <= 1:
            continue
        all_free.append(ray[:-1])  # exclude the endpoint (occupied cell)
    if len(all_free) > 0:
        free_points_px = np.vstack(all_free)  # (K, 2)
    else:
        free_points_px = np.zeros((0, 2), dtype=np.int32)
    for grid_x, grid_y in free_points_px:
        if 0 <= grid_x < grid_map.shape[0] and 0 <= grid_y < grid_map.shape[1]:
            grid_map[grid_y, grid_x] -= log_odds
    # update log odds for occupied pointss
    for grid_x, grid_y in points_px:
        if 0 <= grid_x < grid_map.shape[0] and 0 <= grid_y < grid_map.shape[1]:
            grid_map[grid_y, grid_x] += log_odds
    grid_map = np.clip(grid_map, -10, 10)
# recover grid map from log odds
grid_map_pmf = 1.0 / (1.0 + np.exp(-grid_map))

# 轉換軌跡坐標 (Meters -> Pixels) 以便疊加在地圖上
x_true_px = x_true / grid_res + offset
y_true_px = y_true / grid_res + offset
x_px = x / grid_res + offset
y_px = y / grid_res + offset
# visualize grid map
plt.figure()
grid_cmap = LinearSegmentedColormap.from_list(
    "occ_grid",
    [(0.0, "white"), (0.5, "lightgray"), (1.0, "black")]
)
plt.imshow(
    grid_map_pmf,
    cmap=grid_cmap,
    vmin=-1,
    vmax=1,  
)
mask = (x_true != 0) & (y_true != 0)
mask_ = (x != 0) & (y != 0)
# plt.plot(x_true_px[mask], y_true_px[mask], label="True Trajectory", color="red",linewidth=1)
plt.plot(x_px[mask_], y_px[mask_], label="Estimated Trajectory", color="blue",linewidth=1)
plt.title("Grid Map")
plt.show(block=True)

mask = (x_true != 0) & (y_true != 0)
mask_x = (x != 0) & (y != 0)
plt.figure(figsize=(10,5))

plt.plot(x[mask_x], y[mask_x], label="Estimated Trajectory", color="blue",linewidth=1)
plt.plot(x_true[mask],y_true[mask], label="True Trajectory", color="red",linewidth=1)
#plot yaw as arrow on trajectory
plt.quiver(x_true[mask], y_true[mask], np.cos(phi_encoder[mask]), np.sin(phi_encoder[mask]), scale=50, width=0.005)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajectory with Yaw Arrows")
plt.grid(True)
plt.show(block=True)



# plt.figure(figsize=(10,5))
# # plt.plot(lidar_ranges[:,0], label="lidar scan")
# plt.plot(point_clouds_x[:,4920],point_clouds_y[:,4920], label="point clouds")
# plt.plot(point_clouds_x[:,4925],point_clouds_y[:,4925], label="point clouds2")
# plt.legend()
# plt.show(block= True)  
# plt.figure(figsize=(10,5))
# plt.plot(A[:,0], A[:,1], label="A (source)")
# plt.plot(B[:,0], B[:,1], label="B (target)")
# plt.plot(final_A[:,0], final_A[:,1], label="Transformed B (final)")
# plt.legend()
# plt.title("ICP Result")
# plt.show()
mask_phi = (phi_encoder != 0)
plt.figure(figsize=(10,5))

plt.plot(phi_encoder[:len(yaw_estimate)][mask_phi], label="encoder yaw")
plt.plot(yaw_estimate[mask_phi], label="estimated yaw")
plt.legend()
plt.title("Estimated Yaw vs Encoder Yaw")
plt.xlabel("Time Step")
plt.ylabel("Yaw (radians)")
plt.grid(True)
plt.show(block=True)
