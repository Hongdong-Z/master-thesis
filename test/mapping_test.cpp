//
// Created by zhao on 09.06.22.
//
#include "map_matching/mapping.h"

#include<pcl/visualization/pcl_visualizer.h>
#include<boost/thread/thread.hpp>

int main() {
//    mapping::Mapping mp(50);
//    mp.run();


    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
//    pcl::io::loadPCDFile("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/FA/10_edge_fa.pcd", *cloud);
    pcl::io::loadPCDFile("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/2edge.pcd", *cloud);
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));


    while (!viewer->wasStopped()) {

        viewer->setBackgroundColor(255, 255, 255);
        viewer->addCoordinateSystem (1.0);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color(cloud, 0, 0, 0);

//        std::vector<int> index;
//        pcl::PointCloud<PointType>::Ptr temCloud(new  pcl::PointCloud<PointType>());
//        for (int i = 0; i < cloud->points.size(); ++i) {
//            for (int j = 0; j < cloud->points.size(); ++j) {
//                if (i != j && cloud->points[i].intensity == cloud->points[j].intensity) {
//                    temCloud->push_back(cloud->points[i]);
//                    temCloud->push_back(cloud->points[j]);
//                }
//            }
//        }
        for (int i = 0; i < cloud->points.size(); ++i) {
            for (int j = 0; j < cloud->points.size(); ++j) {
                if (i != j && cloud->points[i].intensity == cloud->points[j].intensity) {

//                    viewer->addPointCloud<PointType>(temCloud, single_color, "sim_cloud");
                    viewer->addLine(cloud->points[i], cloud->points[j], 255, 0, i*30, "surface" + std::to_string(i));

                }
            }
        }
        viewer->spinOnce(1000);

    }





    std::cout << "******************mapping finish!********************" << std::endl;

    return 0;
}