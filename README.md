# Master-thesis-Lidar-localization
Accurate and robust localization is a very important part of the autonomous driving
problem. LiDAR-based positioning is a very important part of the composite positioning
system. This thesis presents two localization systems using different LiDAR features.
Both of these two localization systems can be roughly divided into three modules, namely
the feature extraction module, odometer module and map matching module. In the feature
extraction module, the point cloud features of each frame will be extracted and passed to

subsequent modules. In the odometry module, these extracted features will first be com-
bined with features to calculate the relative transformation relationship of adjacent frames.

These transformation relationships are continued to be passed to the map-matching mod-
ule as initial values. In the map matching module, the features extracted from each frame

will be matched with the built feature map and the optimization problem of the pose
will be established. Finally, by solving this optimization problem, the precise vehicle pose
can be estimated. In addition, in order to achieve map-based localization, after the two
systems have extracted features, these features will first be fused into a global map. These
maps are further processed before being used by the positioning system.
In the first localization approach, in order to improve localization accuracy and stability,
a segmentation-based LiDAR localization system is proposed. Firstly, the four different

features are extracted from the current frame using a series of proposed efficient low-
level semantic segmentation-based multiple types feature extraction algorithms, including

ground, road-curb, edge, and surface. Next, we calculate the transform of the adjacent
frame in the LiDAR odometry module and matched the current frame with the pre-build
feature point cloud map in the LiDAR localization module based on the extracted features
to precisely estimate the 6DoF pose, through the proposed priori information considered

category matching algorithm and multi-group-step L-M (Levenberg-Marquardt) optimiza-
tion algorithm.

In the second localization approach, in order to reduce the map size and improve the feature
association accuracy, a primitives-based localization system is proposed. Firstly, based on
the accuracy of the localization problem, three geometric features that are conducive to
localization are proposed and parameterized as geometric models. These three types of
features are, respectively, facades, poles and ground. For poles and ground, a fast and novel
simultaneous detection and clustering algorithm are proposed. For facades, an efficient
detection algorithm based on LiDAR is proposed. These features are then parameterized,
and the parameterized model is used to build a parametric map to reduce the map size.
Then a novel geometric model-based feature association algorithm is proposed. The final
odometry and map-matching outputs are fed into pose graph optimization to calculate
accurate poses.
Finally, these two localization systems achieve high localization accuracy with a mean
position error of less than 6cm.
<img width="708" alt="image" src="https://user-images.githubusercontent.com/77051392/208889797-e4e14e58-2f53-4e12-8708-d70c39613cf7.png">

