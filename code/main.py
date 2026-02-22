
import math as m
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
from load_data import *
from pr2_utils import *
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
grid_map = np.zeros((500,500))
map_max = 60 # define map limit in (m)
#mapping from world coordinates to grid coordinates
grid_size = map_max / grid_map.shape[0] # m per grid cell
observation_prior = np.log(80/20) # inverse of observation model
maps_prior = 1 # map prior odd
log_odds = observation_prior*maps_prior
for i in range(len(v)-1):
    x[i+1] = x[i] + tau * v[i] * np.sinc(yaw[i]*tau/2) * np.cos(phi[i]+yaw[i]*tau/2)
    y[i+1] = y[i] + tau * v[i] * np.sinc(yaw[i]*tau/2) * np.sin(phi[i]+yaw[i]*tau/2)
    phi[i+1] = phi[i] + yaw[i] * tau
    #for every car passed grid cell, update log odds
    grid_x = int((x[i+1]) / grid_size) + grid_map.shape[0]//2
    grid_y = -int((y[i+1]) / grid_size) + grid_map.shape[1]//2
    if 0 <= grid_x < grid_map.shape[0] and 0 <= grid_y < grid_map.shape[1]:
        grid_map[grid_y, grid_x] += log_odds
# recover grid map from log odds
for i in range(grid_map.shape[0]):
    for j in range(grid_map.shape[1]):
        map_pmf = m.exp(grid_map[i,j]) / (1 + m.exp(grid_map[i,j]))
        grid_map[i,j] = map_pmf
# visualize grid map
plt.figure()
plt.imshow(grid_map, cmap='gray', vmin=-1, vmax=1)
plt.title("Grid Map")
plt.show(block=True)
# for i in range(len(x)):
# print(type(imud), len(imud) if hasattr(imud, "__len__") else "no len")
# print(type(vicd), len(vicd) if hasattr(vicd, "__len__") else "no len")
# print("imu_angular_velocity keys:", imu_angular_velocity.keys() if hasattr(imu_angular_velocity, "keys") else "no keys")
plt.plot(x,y)
#plot yaw as arrow on trajectory
plt.quiver(x[::10], y[::10], np.cos(phi[::10]), np.sin(phi[::10]), scale=50, width=0.005)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajectory with Yaw Arrows")
plt.grid(True)
plt.show(block=True)