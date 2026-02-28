# ECE 276A LiDAR-Based SLAM

This repository contains the coursework for the LiDAR-based SLAM pipeline developed in ECE 276A. It fuses encoder, IMU, Hokuyo LiDAR, and Kinect RGB-D data to build a pose graph, perform ICP-based scan matching, and paint the map with color information projected from the RGB camera.

## Repository Layout

```
code/               # Algorithms, ICP helpers, and visualizers
data/               # Raw npz sensor logs + camera data
dataset20/21/       # Parsed excerpts from the robot experiments
docs/               # Supporting writeups and analysis
result/             # Generated outputs (maps, logs)
```

### Key Scripts

- `code/main.py` – Entry point for the SLAM system: loads synchronized data, downsamples the LiDAR sweep, estimates odometry from wheel encoders, refines poses with ICP (using scipy + KDTree), adds loop closure factors, solves a gtsam pose graph, and paints an occupancy grid with projected RGB colors.
- `code/load_data.py` – Convenience loader that aligns encoder, IMU, LiDAR, disparity, and RGB timestamps for dataset 20/21 and exposes sensor arrays. Run this script standalone to inspect timestamp alignment and project Kinect disparity into robot-frame points.
- `code/utils.py` – Supporting utilities for visualizing canonical object models and ICP output using Open3D.
- `code/icp_warm_up/` – Helper notebooks and legacy tests for iterating on ICP and lidar preprocessing.
- `code/icp_warm_up/test_icp_ver3.py` – ICP Warm-up Ver3: Advanced point cloud registration testing and validation script. Uses pre-loaded point cloud data for drill and liquid container objects, implements iterative closest point (ICP) alignment, and benchmarks registration accuracy against ground truth model data. Useful for validating ICP convergence, debugging registration failures, and testing different point cloud preprocessing strategies.

## Data Organization

Put the downloaded KITTI-like logs under the `data/` directory (the repo already contains `Encoders*.npz`, `Hokuyo*.npz`, `Imu*.npz`, `Kinect*.npz`, and the associated RGB/Disparity folders). The RGB-D frames live in `data/dataRGBD/RGB20` etc. The loader aligns all modalities to the RGB timestamps so the SLAM pipeline can reference images, depth maps, and LiDAR scans together.

## Dependencies

Install the Python packages before running the SLAM pipeline (recommended inside a virtual environment):

```
pip install numpy scipy matplotlib opencv-python open3d gtsam
```

Additional libraries used in `code/main.py` include `scipy.spatial.transform.Rotation`, `cv2`, and `matplotlib` for visualization.

## Running the Pipeline

1. Activate your Python 3.11+ virtual environment and install the dependencies above.
2. (Optional) Adjust the `dataset` variable at the top of `code/main.py` if you want to switch between dataset 20 and 21.
3. Run the SLAM pipeline:

```
python code/main.py
```

The script prints progress for each scan pair, builds the pose graph with gtsam priors + odometry/loop constraints, executes ICP for scan matching, and maintains an occupancy grid that can be visualized with matplotlib.

## Output

- Pose estimates are stored in the gtsam `Values()` object handled inside `code/main.py` and printed to the console as the graph grows.
- The occupancy grid map (`grid_map`) is built in metric space (5 cm resolution) with color accumulation from the RGB frames.
- Loop closures are triggered either by geometric proximity or a fixed index difference; their success/failure is logged along with ICP RMSE values.

## Tips & Next Steps

1. Use `code/load_data.py` to explore the dataset alignment and validate sensor calibration before running SLAM.
2. Extend `code/main.py` to dump the optimized pose graph or publish the colored map meshes for downstream visualization.
3. Run the ICP warmup scripts under `code/icp_warm_up` to experiment with different preprocessing/ICP variants.