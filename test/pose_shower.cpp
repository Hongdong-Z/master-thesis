#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <boost/thread/thread.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>

#include <calib_storage/calib_storage.h>
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"

#include "common.h"

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


using std::sin;
using std::cos;
using std::atan2;


int main()
{
    std::string poseFileNew = "/home/zhao/zhd_ws/src/localization_lidar/pose_primitives/1_5000/pose_1_5000.txt";
//    std::string poseFile = "/home/zhao/zhd_ws/src/localization_lidar/input/odom_qi.txt";
//    std::string poseFileNew = "/home/zhao/zhd_ws/src/localization_lidar/pose/pose_1_5000.txt";
    std::string poseFile = "/home/zhao/zhd_ws/src/localization_lidar/input/odom_os_opt.txt";
//    std::string poseFile = "/home/zhao/zhd_ws/src/localization_lidar/pose_primitives/odom_os_opt.txt";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar/pose_primitives/1_5000";
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());

    std::ifstream fin(poseFile);
    std::ifstream finNew(poseFileNew);

    int i = 0;
    double distance = 0.;
    while (!fin.eof()) {
        double a, b, c, x, e, f, g, y, p, h, j, z;
        double m, n, o;
        fin >> a >> b >> c >> x >> e >> f >> g >> y >> p >> h >> j >> z;
        PointType thisPoint;
        thisPoint.x = x;
        thisPoint.y = y;
        thisPoint.z = z;
        thisPoint.intensity = 500;
        cloud->push_back(thisPoint);


        finNew >> a >> b >> c >> m >> e >> f >> g >> n >> p >> h >> j >> o;


        thisPoint.x = m;
        thisPoint.y = n;
        thisPoint.z = o;
        thisPoint.intensity = 900;
        cloud->push_back(thisPoint);


        distance += sqrt((x-m) * (x-m) + (y-n) * (y-n) + (z-o) * (z-o));

        if (i == 4930) {
            break;
        }
        i = i + 1;
    }
    std::cout << "mean err(m):\n" << distance / i << std::endl;
    pcl::io::savePCDFileASCII(outPath / "result_1_5000.pcd", *cloud);
//    pcl::PointCloud<PointType>::Ptr totalCloud(new pcl::PointCloud<PointType>());
//    pcl::PointCloud<PointType>::Ptr map(new pcl::PointCloud<PointType>());
//    pcl::io::loadPCDFile("/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/SAVE/WHOLE_MAP_NEW.pcd", *map);
//    *totalCloud += *map;
//    *totalCloud += *cloud;
//    totalCloud->width = 1;
//    totalCloud->height = totalCloud->points.size();
//    pcl::io::savePCDFileASCII(outPath / "result_with_pose.pcd", *totalCloud);
    return 0;
}