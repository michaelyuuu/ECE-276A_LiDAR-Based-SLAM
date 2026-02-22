import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors
def to_o3d_pcd(pc):
    # pc can be numpy (N,3) OR already an Open3D PointCloud
    if isinstance(pc, o3d.geometry.PointCloud):
        return pc
    pc = np.asarray(pc, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return pcd
def best_fit_transform(A, B):
    """
    Calculates the best-fit transform that maps points A onto points B.
    Input:
        A: Nxm numpy array of source points
        B: Nxm numpy array of destination points
    Output:
        T: (m+1)x(m+1) homogeneous transformation matrix
    """
    
    # Check if A and B have same dimensions
    assert A.shape == B.shape
    
    # Get number of dimensions
    m = A.shape[1]
    
    # Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    
    # Rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Translation
    t = centroid_B.reshape(-1,1) - np.dot(R, centroid_A.reshape(-1,1))
    
    # Homogeneous transformation
    T = np.eye(m+1)
    T[:m, :m] = R
    T[:m, -1] = t.ravel()
    
    return T

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def iterative_closest_point(A, B, max_iterations=100, tolerance=0.0002):

    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source points
        B: Nxm numpy array of destination points
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        finalA: Aligned points A; Source points A after getting mapped to destination points B
        final_error: Sum of euclidean distances (errors) of the nearest neighbors
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    print(A.shape, B.shape)
    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error (stop if error is less than specified tolerance)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation, error, and mapped source points
    T = best_fit_transform(A, src[:m,:].T)
    final_error = prev_error
    print("Final RMSE:", final_error)
    # get final A 
    rot = T[0:-1,0:-1]
    t = T[:-1,-1]
    finalA = np.dot(rot, A.T).T + t
    return T, finalA, final_error, i
if __name__ == "__main__":
    obj_name = 'drill' # drill or liq_container
    num_pc = 4 # number of point clouds
    source_pc = to_o3d_pcd(read_canonical_model(obj_name))
    print("Source Point Cloud:", type(source_pc), "with", np.asarray(source_pc.points).shape[0], "points")  # *** Initial Transformation ***
    trans_init = np.asarray([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0.0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
    target_pc = to_o3d_pcd(load_pc(obj_name, 0))
    final_T, final_A, final_error, i = iterative_closest_point(np.asarray(target_pc.points), np.asarray(source_pc.points))
    print("Final Transformation Matrix:")
    print(final_T)
    print("Final Error:", final_error)
    print("Number of iterations:", i)
    target_pc_data = np.asarray(target_pc.points)
    source_pc_data = np.asarray(source_pc.points)
    visualize_icp_result(source_pc_data, final_A, final_T)
# for i in range(num_pc):
#     target_pc = to_o3d_pcd(load_pc(obj_name, i))
    




#     visualize_icp_result(source_pct, target_pct, pose)

