import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
import open3d as o3d
import copy
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("TkAgg")  # 或 Qt5Agg
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# parameters 
obj_num = 3 # 0,1,2,3

def to_o3d_pcd(pc):
    # pc can be numpy (N,3) OR already an Open3D PointCloud
    if isinstance(pc, o3d.geometry.PointCloud):
        return pc
    pc = np.asarray(pc, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return pcd
def best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve R,t that minimizes || R A + t - B ||^2 for corresponding points.
    A, B: (N,3)
    Returns:
        T: (4,4) homogeneous transform mapping A -> B
    """
    assert A.shape == B.shape and A.shape[1] == 3
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
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def icp_one_iteration_scipy(src_pts, dst_pts, T_init, max_corr_dist, tree):
    # 1. 初始化與座標變換
    if T_init is None: T_init = np.eye(4)
    
    # 將 source 點雲套用目前的變換矩陣 (Ns, 3)
    src_tf = (src_pts @ T_init[:3, :3].T) + T_init[:3, 3]

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
    # R = dT[:3, :3]
    # t = dT[:3, 3]
    # yaw_angle = np.arctan2(R[1, 0], R[0, 0])
    # R_yaw = np.array([
    #     [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
    #     [np.sin(yaw_angle),  np.cos(yaw_angle), 0],
    #     [0,                  0,                 1]
    # ])
    # # 2. 從 R 中提取 Yaw (繞 Z 軸)
    # # 在 3D 旋轉矩陣中，R[1,0] 是 sin(theta), R[0,0] 是 cos(theta)
    
    # # 6. 更新總變換矩陣
    # centroid_A = matched_src.mean(axis=0)
    # centroid_B = matched_dst.mean(axis=0)
    # t_new = centroid_B - R_yaw @ centroid_A
    # dT_new = np.eye(4)
    # dT_new[:3, :3] = R_yaw
    # dT_new[:3, 3] = t_new
    T_new = dT @ T_init
    
    rmse = np.sqrt(np.mean(distances[mask]**2))
    #calculate varianence of inlier distances
    inlier_distances = distances[mask]
    variance = np.var(inlier_distances)
    return T_new, rmse, n_in, variance
if __name__ == "__main__":
    obj_name = 'liq_container' # drill or liq_container
    num_pc = 4 # number of point clouds
    source_pc = to_o3d_pcd(read_canonical_model(obj_name))
    print("Source Point Cloud:", type(source_pc), "with", np.asarray(source_pc.points).shape[0], "points")  # *** Initial Transformation ***
    trans_init = np.asarray([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0.0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
    target_pc = to_o3d_pcd(load_pc(obj_name, obj_num))
    tree = KDTree(np.asarray(target_pc.points))
# for i in range(num_pc):
#     target_pc = to_o3d_pcd(load_pc(obj_name, i))
    T_init = np.eye(4)
    best_rmse = float("inf")
    rmse = 5
    variance = 0.5
    size= 1*2
    iteration_size = 50
    for j in range(size+1):
        T = T_init @ np.asarray([[np.cos(j*2*np.pi/size-np.pi), -np.sin(j*2*np.pi/size-np.pi), 0, 0],
                            [np.sin(j*2*np.pi/size-np.pi), np.cos(j*2*np.pi/size-np.pi), 0, 0],
                            [0,              0,             1, 0],
                            [0,              0,             0, 1]])
        print("Yaw interation:", j, "with initial T:\n", T)
        rmse = 5
        variance = 0.5
        for it in range(iteration_size):
            T_new, rmse, n_in, variance = icp_one_iteration_scipy(np.asarray(source_pc.points), np.asarray(target_pc.points), T, max_corr_dist=max(0.008, 2/(it+1)**1.3),tree=tree)
            print(f"iter {it:02d}: rmse={rmse:.4f}, inliers={n_in}, variance={variance:.4f}")
            if n_in < 30:
                print("Too few correspondences, stop.")
                break
            if it > 0 and np.linalg.norm(rmse - rmse_prev)/rmse_prev < 1e-6:
                print("Converged.")
                T = T_new
                break
            rmse_prev = rmse
            T = T_new
            # recode the minimum rmse and corresponding T
            if rmse < best_rmse:
                best_rmse = rmse
                best_T = T
    print("Best RMSE:", best_rmse)
    print("rmse:", rmse, "inliers:", n_in)
    print("T:\n", T)
    source_pc_data = np.asarray(source_pc.points)
    target_pc_data = np.asarray(target_pc.points)
    final_A = (best_T[:3, :3] @ target_pc_data.T).T + best_T[:3, 3]
    # visualize_icp_result(target_pc_data, target_pc_data, T)
    visualize_icp_result(source_pc_data, target_pc_data, best_T)
    #save figure as jpg
    plt.savefig(f"icp_result_{obj_name}_{obj_num}.jpg", dpi=300, bbox_inches='tight')
    # o3d.io.write_image(f"icp_result_{obj_name}_{obj_num}.jpg", o3d.geometry.Image(np.asarray(visualize_icp_result(source_pc_data, target_pc_data, best_T))))









