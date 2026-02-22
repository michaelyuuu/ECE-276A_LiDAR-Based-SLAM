import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from scipy.spatial import KDTree
import numpy as np
import open3d as o3d
import copy
def Tz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0 , 0],
        [np.sin(theta),  np.cos(theta), 0 , 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
def to_o3d_pcd(pc):
    # pc can be numpy (N,3) OR already an Open3D PointCloud
    if isinstance(pc, o3d.geometry.PointCloud):
        return pc
    pc = np.asarray(pc, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return pcd
# def visualize_icp_result(source_pc, target_pc, pose):
#     source_pcd = to_o3d_pcd(source_pc)
#     target_pcd = to_o3d_pcd(target_pc)
#     source_pcd.transform(pose)
#     o3d.visualization.draw_geometries([source_pcd, target_pcd])
def draw_registration_result(source, target, transformation):
      source_temp = copy.deepcopy(source)
      target_temp = copy.deepcopy(target)
      source_temp.paint_uniform_color([1, 0.0706, 0])
      target_temp.paint_uniform_color([0, 0.651, 0.929])
      source_temp.transform(transformation)
      o3d.visualization.draw_plotly([source_temp, target_temp])
if __name__ == "__main__":
    obj_name = 'drill' # drill or liq_container
    num_pc = 4 # number of point clouds
    source_pc = to_o3d_pcd(read_canonical_model(obj_name))
    # print("Source Point Cloud:", type(source_pc), "with", np.asarray(source_pc.points).shape[0], "points")  # *** Initial Transformation ***
    trans_init = np.asarray([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0.0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
    threshold = 10
    stepsize = 30
    rmse = np.zeros(num_pc)
    #iterate yaw angle from -pi to pi
    for j in range(stepsize):
        trans_init = trans_init @ Tz(j*np.pi/stepsize-np.pi/2) # rotate around z axis from -pi/2 to pi/2
        for i in range(num_pc):
            target_pc = to_o3d_pcd(load_pc(obj_name, i))
            # print("Initial Alignment")
            evaluation = o3d.pipelines.registration.evaluate_registration(
                source_pc, target_pc, threshold, trans_init
            )
            # print(evaluation)

            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pc, target_pc, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            rmse[i] = reg_p2p.inlier_rmse
            # print(reg_p2p)
            # print("Transformation is:")
            # print(reg_p2p.transformation)
            pose = reg_p2p.transformation
            source_pct = np.asarray(source_pc.points)
            target_pct = np.asarray(target_pc.points)
            if rmse[i] < rmse[i-1]:
                best_pose = pose
                best_target_pc = target_pc
    
    visualize_icp_result(source_pct, best_target_pc, best_pose)
