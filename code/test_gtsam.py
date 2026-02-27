#****** Test Script to test GTSAM - python installation ******#
import gtsam
import numpy as np


# class UnaryFactor(gtsam.NoiseModelFactor1):
#   def __init__(self, key, mx, my, model):
#     super().__init__(model, key)
#     self.mx = mx
#     self.my = my

#   def evaluateError(self, pose: gtsam.Pose2, H=None):
#     if H is not None:
#       R = pose.rotation()
#       H[0][:] = [R.c(), -R.s(), 0.0]
#       H[1][:] = [R.s(),  R.c(), 0.0]
#     return np.array([pose.x() - self.mx, pose.y() - self.my])

def test_create_pose2():
  # Create a 2D pose with x, y, and theta (rotation)
  pose = gtsam.Pose2(1.0, 2.0, 0.5)
  print("Pose2 created:", pose)

  return pose

def test_create_prior():
  # Create a prior factor on a Pose2
  prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
  pose_key = gtsam.symbol('x', 1)
  prior_factor = gtsam.PriorFactorPose2(pose_key, gtsam.Pose2(0, 0, 0), prior_noise)
  print("Prior factor created:", prior_factor)
  
  return prior_factor

if __name__ == "__main__":
  # Run basic tests
  pose = test_create_pose2()
  prior = test_create_prior()
  graph = gtsam.NonlinearFactorGraph()

  prior_mean = gtsam.Pose2(0.0, 0.0, 0.0)   
  prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.01])) #
  graph.add(gtsam.PriorFactorPose2(1, prior_mean, prior_noise))

  odometry = gtsam.Pose2(2.0, 0.0, 0.0)
  turn = gtsam.Pose2(2.0, 0.0, np.pi/2)
  odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1])) #noiseModel.Diagonal.Sigmas
  graph.add(gtsam.BetweenFactorPose2(1, 2, odometry, odometry_noise))
  graph.add(gtsam.BetweenFactorPose2(2, 3, turn, odometry_noise))
  graph.add(gtsam.BetweenFactorPose2(3, 4, turn, odometry_noise))
  graph.add(gtsam.BetweenFactorPose2(4, 5, turn, odometry_noise))

  # loop closure from pose 5 back to pose 2
  graph.add(gtsam.BetweenFactorPose2(5, 2, turn, odometry_noise))
  initial = gtsam.Values()
  initial.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
  initial.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
  initial.insert(3, gtsam.Pose2(2.3, 1.9, np.pi/2 - 0.1))
  initial.insert(4, gtsam.Pose2(0.6, 2.0, np.pi - 0.1))
  initial.insert(5, gtsam.Pose2(-1.4, 1.8, -np.pi/2 + 0.1))
  # add landmark/bearing-range measurements
  i1, i2, i3 = 1, 2, 3
  j1 = gtsam.symbol('l', 1)
  j2 = gtsam.symbol('l', 2)
  degrees = np.pi / 180.0
  br_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
  graph.add(gtsam.BearingRangeFactor2D(i1, j1, gtsam.Rot2(45 * degrees), np.sqrt(8), br_noise))
  graph.add(gtsam.BearingRangeFactor2D(i2, j1, gtsam.Rot2(90 * degrees), 2.0, br_noise))
  graph.add(gtsam.BearingRangeFactor2D(i3, j2, gtsam.Rot2(90 * degrees), 2.0, br_noise))
  initial.insert(j1, gtsam.Point2(2.0, 0.0))
  initial.insert(j2, gtsam.Point2(4.0, 0.0))

  result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
  marginals = gtsam.Marginals(graph, result)
  print("x1 covariance:\n", marginals.marginalCovariance(1))
  print("x2 covariance:\n", marginals.marginalCovariance(2))
  print("x3 covariance:\n", marginals.marginalCovariance(3))
  # print("Optimized result:")
  # for i in range(1, 4):
  #   print(f"x{i}:", result.atPose2(i))
  print("result of x5", result.atPose2(5))
  print("GTSAM installation seems to be working!")

