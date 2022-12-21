//
// Created by zhao on 15.03.22.
//

#pragma once
#include <pcl/kdtree/kdtree_flann.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>


#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <boost/thread/thread.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <boost/thread/thread.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <folder_based_data_storage/folder_based_data_storage.hpp>



typedef pcl::PointXYZI PointType;

inline double rad2deg(double radians)
{
    return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
    return degrees * M_PI / 180.0;
}


