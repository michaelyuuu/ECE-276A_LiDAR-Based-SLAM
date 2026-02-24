
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
# tau = 1/40
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
idx_lidar = align_closest(lidar_stamps, data["rgb_stamps"])
idx_disp = align_closest(data["disp_stamps"], data["rgb_stamps"])
# shape: (1081, 4962, 2)
# print("point_clouds shape:", point_clouds.shape)
def printimage(i):
    i# 1. 讀取彩色圖片 (絕對不要縮小 img！因為後面的 u_rgb 對應的是原始 640x480 尺寸)
    img = cv2.imread(f"data/dataRGBD/RGB20/rgb20_{i}.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_rgb, W_rgb = img.shape[:2]

    # 2. 讀取深度圖 (保持原始尺寸)
    i_disp = idx_disp[i]
    disparity = cv2.imread(f"data/dataRGBD/Disparity20/disparity20_{i_disp}.png", cv2.IMREAD_UNCHANGED)
    disp = disparity.astype(np.float32)
    H, W = disp.shape

    # 3. 【關鍵降採樣】：只降採樣「座標網格」與「深度值」
    # 加上步長 ds，這樣數值還是 0~640，只是點的數量變少了
    v, u = np.mgrid[0:H:ds, 0:W:ds].astype(np.float32) 
    disp_ds = disp[0:H:ds, 0:W:ds] 
    H_ds, W_ds = u.shape
    # 4. 開始計算深度與座標 (完全使用降採樣後的 disp_ds，速度會變快)
    dd = -0.00304 * disp_ds + 3.31            
    depth = 1.03 / dd
    u_rgb = (526.37 * u + 19276.0 - 7877.07 * dd) / 585.01 #from disparity to rgb pixel coordinates, using the dataset's formula
    v_rgb = (526.37 * v + 16662.0) / 585.01   #from disparity to rgb pixel coordinates, using the dataset's formula
    xc = depth *(u - 242.94) / 585.01 #from disparity to robot frame coordinates, using the dataset's formula
    yc = depth *(v - 315.84) / 585.01 #from disparity to robot frame coordinates, using the dataset's formula
    zc = depth


    # 1. 建立一張與 RGB 影像大小完全一樣的空白深度圖 (初始化為 0)
    depth_aligned_to_rgb = np.zeros((H_rgb, W_rgb), dtype=np.float32)

    # 2. 確保投影出來的 u_rgb, v_rgb 是整數，並且沒有超出 RGB 影像的邊界
    u_rgb_int = np.round(u_rgb).astype(np.int32)
    v_rgb_int = np.round(v_rgb).astype(np.int32)

    valid_mask = (u_rgb_int >= 0) & (u_rgb_int < W_rgb) & \
                (v_rgb_int >= 0) & (v_rgb_int < H_rgb) & \
                (depth > 0)

    depth_aligned_to_rgb[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] = depth[valid_mask]
    # optional: mask invalid
    R_depth_to_robot = np.eye(3)  # identity, since depth is already in robot frame
    roll = 0
    pitch = 0.36
    yaw =0.021
    t = [0.18,0.005,0.36]
    R_rot_robot = Rotation.from_euler(
        'xyz',        # robot-frame axes
        [roll, pitch, yaw],
        degrees=False
    ).as_matrix()
    R_depth_to_robot = R_rot_robot @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    print("R_depth_to_robot:\n", R_depth_to_robot)
    t_depth_to_robot = np.array(t).reshape(3, 1)
    points_depth = np.stack((xc.flatten(), yc.flatten(), zc.flatten()), axis=0)  # shape: (3, N)
    points_robot = R_depth_to_robot @ points_depth + t_depth_to_robot  # shape: (3, N)
    xc = points_robot[0, :].reshape(H_ds, W_ds)
    yc = points_robot[1, :].reshape(H_ds, W_ds)
    zc = points_robot[2, :].reshape(H_ds, W_ds)
    cv = 315.84
    fsv = 585.05
    tolerance = 0.3
    valid_mask = valid_mask & (zc < tolerance) 
    valid_xc = xc[valid_mask]
    valid_yc = yc[valid_mask]
    valid_zc = zc[valid_mask]
    colors = img[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] / 255.0
    return valid_xc, valid_yc, valid_zc, colors

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
current_accumulated_yaw = 0
T_temp = np.eye(3)
T_robot_lidar = np.eye(3)
T_robot_lidar[:2, 2] = np.array([0.13323, 0])
# for i in range(len(v)-1):
yaw_measured = np.zeros(len(lidar_stamps))
x_true = np.zeros(len(lidar_stamps))
y_true = np.zeros(len(lidar_stamps))
yaw_estimate = np.zeros(len(lidar_stamps))
color_map = np.zeros((map_dim, map_dim, 3), dtype=np.float32)
color_count = np.zeros((map_dim, map_dim), dtype=np.int32)
step = 10
# for i in range(0, len(lidar_stamps)-step*2, step):
for i in range(250, 650, step):
    print(f"Processing scan {i} to {i+step}...")
    ranges_A = lidar_ranges[::ds, i]
    ranges_B = lidar_ranges[::ds, i+step]
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
    
    dt = Te[i+step*2] - Te[i+step]
    
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
    best_rmse = float("inf")
    rmse = 5
    variance = 0.5
    iteration_size = 50   
    best_T = T_init.copy()          # fallback transform if ICP fails
    best_rmse = float("inf")
    icp_ok = False
    rmse_prev = None
    for it in range(iteration_size):
        T_new, rmse, n_in = icp_one_iteration_scipy(B, A, T_init, max_corr_dist=0.5,tree=tree)
        # print(f"iter {it:02d}: rmse={rmse:.4f}, inliers={n_in}, variance={variance:.4f}")
        if n_in < 100:
            print("Too few correspondences, stop.")
            break
        if it > 0 and np.linalg.norm(rmse - rmse_prev) < 1e-4 and rmse< 0.1:
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
    print("yaw_dot from encoder:", np.degrees(yaw[i] * dt), "degrees") #yaw change from encoder
    print("Estimated yaw:", np.degrees(yaw_estimate[i+step]), "degrees")
    # Update your plotting variable
    x_true[i+step] = x_true[i] + dt * v[i] * np.sinc(yaw[i]*dt/2) * np.cos(yaw_measured[i]+yaw[i]*dt/2)
    y_true[i+step] = y_true[i] + dt * v[i] * np.sinc(yaw[i]*dt/2) * np.sin(yaw_measured[i]+yaw[i]*dt/2)
    yaw_measured[i+step] = yaw_measured[i] + yaw[i] * dt
    print("Encoder yaw:", np.degrees(yaw_measured[i+step]), "degrees")
    x[i+step] = T_robot_new[0, 2]
    y[i+step] = T_robot_new[1, 2] 
    T_world_lidar = T_robot_new @ T_robot_lidar # transform from lidar frame to world frame
    world_scan = (T_world_lidar[:2, :2] @ point_clouds[:,i+step,:].T).T + T_world_lidar[:2, 2]
    xc, yc, zc, colors = printimage(i+step)
    posc_world = T_robot_new[:2, :2] @ np.stack((xc, yc), axis=0) + T_robot_new[:2, 2].reshape(2, 1)
    
    xc_world = posc_world[0, :] # transform color points to world frame for color map update
    yc_world = posc_world[1, :] # transform color points to world frame for color map update
    world_scan_x = world_scan[:, 0] # for grid map update, still use lidar scan in world frame
    world_scan_y = world_scan[:, 1]
    car_px_x = T_world_lidar[0, 2] / grid_res + offset
    car_px_y = T_world_lidar[1, 2] / grid_res + offset
    #Transform world scan point cloud to pixel coordinates
    points_px = np.vstack((world_scan_x / grid_res + offset, 
                       world_scan_y / grid_res + offset)).T.astype(np.int32)
    color_points_px = np.vstack((xc_world / grid_res + offset,
                                yc_world / grid_res + offset)).T.astype(np.int32)
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
    #update color map
    for (px, py), c in zip(color_points_px, colors):
        if 0 <= px < map_dim and 0 <= py < map_dim:
            color_map[py, px] = c
            color_count[py, px] += 1
#color map as the newest observation at each cell
# recover grid map from log odds
grid_map_pmf = 1.0 / (1.0 + np.exp(-grid_map)) # convert log-odds to probability, from (0,1)

# 轉換軌跡坐標 (Meters -> Pixels) 以便疊加在地圖上
x_true_px = x_true / grid_res + offset
y_true_px = y_true / grid_res + offset
x_px = x / grid_res + offset
y_px = y / grid_res + offset
occupied = grid_map_pmf > 0.7      # 真正牆 / 障礙物
free     = grid_map_pmf < 0.3
unknown  = ~(occupied | free)

# 只保留 occupied 的 RGB，其它清掉
color_map[~free] = 0.0
plt.figure()
plt.imshow(color_map)   # already in [0,1]
plt.title("Color Ground Map")
plt.show()
# visualize grid map
# visualize grid map
plt.figure(figsize=(10, 10))

grid_cmap = LinearSegmentedColormap.from_list(
    "occ_grid",
    [(0.0, "white"), (0.5, "lightgray"), (1.0, "black")]
)

# 1) Base layer: SLAM 網格地圖 (灰階)
plt.imshow(
    grid_map_pmf,
    cmap=grid_cmap,
    vmin=0,
    vmax=1,  
    origin='lower'
)

# 2) Middle layer: 軌跡
mask_x = (x != 0) | (y != 0)
plt.plot(x_px[mask_x], y_px[mask_x], label="Estimated Trajectory", color="blue", linewidth=1.5)
h, w = grid_map_pmf.shape
rgba_map = np.zeros((h, w, 4), dtype=np.float32)
rgba_map[..., :3] = color_map 
valid_color_mask = (color_count > 0) & free
rgba_map[..., 3] = valid_color_mask.astype(np.float32) * 0.9
# 直接顯示這張 RGBA 圖片，記得 origin 也要對齊 'lower'
plt.imshow(rgba_map, origin='lower')

plt.title("SLAM Map + RGB Floor Overlay + Trajectory")
plt.legend()
plt.axis("off")
plt.show(block=True)
#save figure
plt.savefig(f"slam_map_with_rgb_and_trajectory_dataset{dataset}.png", dpi=300, bbox_inches='tight')
# plt.figure()
# grid_cmap = LinearSegmentedColormap.from_list(
#     "occ_grid",
#     [(0.0, "white"), (0.5, "lightgray"), (1.0, "black")]
# )
# plt.imshow(
#     grid_map_pmf,
#     cmap=grid_cmap,
#     vmin=-1,
#     vmax=1,  
# )
# mask = (x_true != 0) & (y_true != 0)
# mask_ = (x != 0) & (y != 0)
# # plt.plot(x_true_px[mask], y_true_px[mask], label="True Trajectory", color="red",linewidth=1)
# plt.plot(x_px[mask_], y_px[mask_], label="Estimated Trajectory", color="blue",linewidth=1)
# plt.title("Grid Map")
# plt.show(block=True)

mask = (x_true != 0) & (y_true != 0)
mask_x = (x != 0) & (y != 0)
plt.figure(figsize=(10,5))

plt.plot(x[mask_x], y[mask_x], label="Estimated Trajectory", color="blue",linewidth=1)
plt.plot(x_true[mask],y_true[mask], label="True Trajectory", color="red",linewidth=1)
#plot yaw as arrow on trajectory
plt.quiver(x_true[mask], y_true[mask], np.cos(yaw_measured[mask]), np.sin(yaw_measured[mask]), scale=50, width=0.005)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.title("Trajectory with Yaw Arrows")
plt.grid(True)
plt.show(block=True)
#save figure
plt.savefig(f"trajectory_with_yaw_arrows_dataset{dataset}.png", dpi=300, bbox_inches='tight')



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
mask_yaw = (yaw_measured != 0)
plt.figure(figsize=(10,5))

plt.plot(yaw_measured[:len(yaw_estimate)][mask_yaw], label="measured yaw")
plt.plot(yaw_estimate[mask_yaw], label="estimated yaw")
plt.legend()
plt.title("Estimated Yaw vs Measured Yaw")
plt.xlabel("Time Step")
plt.ylabel("Yaw (radians)")
plt.grid(True)
plt.show(block=True)
#save figure
plt.savefig(f"estimated_yaw_vs_measured_yaw_dataset{dataset}.png", dpi=300, bbox_inches='tight')

