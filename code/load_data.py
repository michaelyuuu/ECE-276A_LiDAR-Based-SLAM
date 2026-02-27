import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
def align_closest(t_1, t_2):
    idx = np.searchsorted(t_1, t_2)

    idx = np.clip(idx, 1, len(t_1)-1)

    prev = t_1[idx-1]
    next = t_1[idx]

    closer = np.abs(t_2 - prev) <= np.abs(next - t_2)
    idx[closer] -= 1
    return idx
def load_dataset(dataset=21, data_dir="../data"):
  with np.load(f"{data_dir}/Encoders{dataset}.npz") as data:
    encoder_counts = data["counts"]
    encoder_stamps = data["time_stamps"]

  with np.load(f"{data_dir}/Hokuyo{dataset}.npz") as data:
    lidar_angle_min = data["angle_min"]
    lidar_angle_max = data["angle_max"]
    lidar_angle_increment = data["angle_increment"]
    lidar_range_min = data["range_min"]
    lidar_range_max = data["range_max"]
    lidar_ranges = data["ranges"]
    lidar_stamps = data["time_stamps"]

  with np.load(f"{data_dir}/Imu{dataset}.npz") as data:
    imu_angular_velocity = data["angular_velocity"]
    imu_linear_acceleration = data["linear_acceleration"]
    imu_stamps = data["time_stamps"]

    

  with np.load(f"{data_dir}/Kinect{dataset}.npz") as data:
    disp_stamps = data["disparity_time_stamps"]
    rgb_stamps = data["rgb_time_stamps"]
    idx_disp = align_closest(disp_stamps, rgb_stamps)
    idx_lidar = align_closest(lidar_stamps, rgb_stamps)
    lidar_stamps = lidar_stamps[idx_lidar]
    lidar_ranges = lidar_ranges[:, idx_lidar]
    disp_stamps = disp_stamps[idx_disp]
    idx_imu = align_closest(imu_stamps, rgb_stamps)
    imu_stamps = imu_stamps[idx_imu]
    imu_angular_velocity = imu_angular_velocity[:,idx_imu]
    imu_linear_acceleration = imu_linear_acceleration [:,idx_imu]
    idx_encoder = align_closest(encoder_stamps, rgb_stamps)
    encoder_stamps = encoder_stamps[idx_encoder]
    encoder_counts = encoder_counts[:, idx_encoder]
    print(f"Loaded dataset {dataset} with {len(encoder_counts)} encoder counts, {len(lidar_ranges)} lidar scans, {len(imu_angular_velocity)} IMU readings, {len(disp_stamps)} disparity frames, and {len(rgb_stamps)} RGB frames.")
  
  return {
    "encoder_counts": encoder_counts,
    "encoder_stamps": encoder_stamps,
    "lidar_angle_min": lidar_angle_min,
    "lidar_angle_max": lidar_angle_max,
    "lidar_angle_increment": lidar_angle_increment,
    "lidar_range_min": lidar_range_min,
    "lidar_range_max": lidar_range_max,
    "lidar_ranges": lidar_ranges,
    "lidar_stamps": lidar_stamps,
    "imu_angular_velocity": imu_angular_velocity,
    "imu_linear_acceleration": imu_linear_acceleration,
    "imu_stamps": imu_stamps,
    "disp_stamps": disp_stamps,
    "rgb_stamps": rgb_stamps,
  }


if __name__ == '__main__':
  data = load_dataset(dataset=20, data_dir="data")
  idx_lidar = align_closest(data["lidar_stamps"],data["rgb_stamps"])

  idx_disp = align_closest(data["disp_stamps"],data["rgb_stamps"])
  # i as index to read disparity image
  i = 522
  i_disp = idx_disp[i]
  img = cv2.imread(f"data/dataRGBD/RGB20/rgb20_{i}.png")
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  disparity = cv2.imread(f"data/dataRGBD/Disparity20/disparity20_{i_disp}.png", cv2.IMREAD_UNCHANGED)
  # print("Image shape:", img.shape)
  print("Disparity shape:", disparity.shape)
  disp = disparity.astype(np.float32)   # important (uint16 will bite you)
  dd = -0.00304 * disp + 3.31           # "disparity" in whatever units the dataset uses
  depth = 1.03 / dd                     # depth (same units as dataset formula implies)
  H, W = disp.shape
  v, u = np.mgrid[0:H, 0:W].astype(np.float32)  # v=row (y), u=col (x)
  K = np.array([[585.01, 0, 315.84],
                [0, 585.01, 242.94],
                [0, 0, 1]])
  dd = -0.00304 * disp + 3.31
  u_rgb = (526.37 * u + 19276.0 - 7877.07 * dd) / 585.01 #from disparity to rgb pixel coordinates, using the dataset's formula
  v_rgb = (526.37 * v + 16662.0) / 585.01   #from disparity to rgb pixel coordinates, using the dataset's formula
  xc = depth *(u - 315.84) / 585.01 #from disparity to robot frame coordinates, using the dataset's formula
  yc = depth *(v - 242.94) / 585.01 #from disparity to robot frame coordinates, using the dataset's formula
  zc = depth
  H_rgb, W_rgb = img.shape[:2]

  # 1. 建立一張與 RGB 影像大小完全一樣的空白深度圖 (初始化為 0)
  depth_aligned_to_rgb = np.zeros((H_rgb, W_rgb), dtype=np.float32)

  # 2. 確保投影出來的 u_rgb, v_rgb 是整數，並且沒有超出 RGB 影像的邊界
  u_rgb_int = np.round(u_rgb).astype(np.int32)
  v_rgb_int = np.round(v_rgb).astype(np.int32)

  valid_mask = (u_rgb_int >= 0) & (u_rgb_int < W_rgb) & \
               (v_rgb_int >= 0) & (v_rgb_int < H_rgb) & \
               (depth > 0) & (disp > 0)

  depth_aligned_to_rgb[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] = depth[valid_mask]
# optional: mask invalid
  R_depth_to_robot = np.eye(3)  # identity, since depth is already in robot frame
  roll = 0
  pitch = 0.36
  yaw =0.021
  t = [0.18,0.005,0.36]
  R_rot_robot = R.from_euler(
      'xyz',        # robot-frame axes
      [roll, pitch, yaw],
      degrees=False
  ).as_matrix()
  R_depth_to_robot = R_rot_robot @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
  print("R_depth_to_robot:\n", R_depth_to_robot)
  t_depth_to_robot = np.array(t).reshape(3, 1)
  points_depth = np.stack((xc.flatten(), yc.flatten(), zc.flatten()), axis=0)  # shape: (3, N)
  points_robot = R_depth_to_robot @ points_depth + t_depth_to_robot  # shape: (3, N)
  xc = points_robot[0, :].reshape(H, W)
  yc = points_robot[1, :].reshape(H, W)
  zc = points_robot[2, :].reshape(H, W)
  cv = 315.84
  fsv = 585.05
  tolerance = 0.1
  valid_mask = valid_mask & (zc < tolerance) 
  valid_xc = xc[valid_mask]
  valid_yc = yc[valid_mask]
  valid_zc = zc[valid_mask]
  # plt.show()
  # cv2.imshow("img", img)
  # cv2.imshow("disparity", disparity)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  # valid_mask = (u_rgb_int >= 0) & (u_rgb_int < W_rgb) & \
  #            (v_rgb_int >= 0) & (v_rgb_int < H_rgb) & \
  #            (depth > 0)
  # # depth[~valid] = 0
  # plt.figure(figsize=(10,5))

  # # 全部改用 valid_mask 和整數索引
  # plt.scatter(
  #     u_rgb[valid_mask], 
  #     v_rgb[valid_mask], 
  #     c=img[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] / 255.0, 
  #     s=1
  # )
  # plt.grid()
  # plt.gca().invert_yaxis() 
  # plt.title("Depth Map Projected on RGB")
  # plt.show()


  # plot 3d xc, yc, zc
  # colors = img[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] / 255.0 
  # fig = plt.figure(figsize=(10,5))
  # ax = fig.add_subplot(111, projection='3d')
  # #use rgb map for xc yc zc
  # ax.scatter(valid_xc.flatten(), -valid_yc.flatten(), valid_zc.flatten(), c=colors.reshape(-1, 3), s=0.5)
  # ax.set_xlabel('X (m)')
  # ax.set_ylabel('Y (m)')
  # ax.set_zlabel('Z (m)')
  # ax.set_title("3D Point Cloud from Depth Map")

  plt.show()
  plt.figure(figsize=(10,5))
  colors = img[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] / 255.0  
  plt.scatter(-valid_yc.flatten(), valid_xc.flatten(), c=colors.reshape(-1, 3), s=1)
  plt.grid()
  # plt.gca().invert_yaxis()
  plt.title("Point Cloud Projection")
  plt.show()
  
  plt.figure(figsize=(10,5))
  colors = img[v_rgb_int[valid_mask], u_rgb_int[valid_mask]] / 255.0  
  plt.scatter(-valid_yc.flatten(), valid_zc.flatten(), c=colors.reshape(-1, 3), s=1)
  plt.grid()
  # plt.gca().invert_yaxis()
  plt.title("image i" + str(i) + " Point Cloud Projection")
  plt.show()
  
  #plot depth map as image
  # plt.figure(figsize=(10,5))
  # plt.imshow(depth, cmap='plasma')
  # plt.colorbar(label='Depth (m)')
  # plt.title("Depth Map")  
  # plt.show()
  # plt.figure(figsize=(10,5))
  # plt.imshow(depth_aligned_to_rgb, cmap='plasma')
  # plt.colorbar(label='Depth (m)')
  # plt.title("Depth Map Aligned to RGB")  
  # plt.show()
  
  print("data i", i)
  print ("Aligned Disp index to RGB frame:", idx_disp[i])
  print ("Disparity timestamp:", data["disp_stamps"][idx_disp[i]])
  print ("RGB timestamp:", data["rgb_stamps"][i])
  print("Loaded keys:", sorted(data.keys()))
  print("Encoder counts shape:", data["encoder_counts"].shape)
  print("Lidar ranges shape:", data["lidar_ranges"].shape)
  print("Disparity stamps shape:", data["disp_stamps"].shape)
  print("RGB stamps shape:", data["rgb_stamps"].shape)
  print("IMUs shape:", data["imu_angular_velocity"].shape, data["imu_linear_acceleration"].shape)

  print("Encoder timestamps", data["encoder_stamps"][15]-data["imu_stamps"][15])
  print("Lidar timestamps", data["lidar_stamps"][15:20])
  print("IMU timestamps", data["imu_stamps"][15:20])

