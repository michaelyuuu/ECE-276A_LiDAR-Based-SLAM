import numpy as np


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
  print("Loaded keys:", sorted(data.keys()))
  print("Encoder counts shape:", data["encoder_counts"].shape)
  print("Lidar ranges shape:", data["lidar_ranges"].shape)
  print("Disparity stamps shape:", data["disp_stamps"].shape)
  print("RGB stamps shape:", data["rgb_stamps"].shape)

