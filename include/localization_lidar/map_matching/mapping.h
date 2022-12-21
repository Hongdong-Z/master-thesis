//
// Created by zhao on 09.06.22.
//

#ifndef SRC_MAPPING_H
#define SRC_MAPPING_H

#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iostream>
#include <typeinfo>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <seamseg_fbds/reader.h>
#include <calib_storage/calib_storage.h>
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"
#include "localization_lidar/load_datastorage.hpp"
#include "localization_lidar/parameter_server.hpp"

#include "feature_extraction/imageProjection.h"
#include "feature_extraction/load_pose.h"


namespace mapping{
class Mapping {
private:
    pcl::PointCloud<PointType>::Ptr mapGround;
    pcl::PointCloud<PointType>::Ptr mapCurb;
    pcl::PointCloud<PointType>::Ptr mapSurface;
    pcl::PointCloud<PointType>::Ptr mapEdge;

    pcl::PointCloud<PointType>::Ptr mapGroundDS;
    pcl::PointCloud<PointType>::Ptr mapCurbDS;
    pcl::PointCloud<PointType>::Ptr mapSurfaceDS;
    pcl::PointCloud<PointType>::Ptr mapEdgeDS;

    pcl::PointCloud<PointType>::Ptr tempGround;
    pcl::PointCloud<PointType>::Ptr tempCurb;
    pcl::PointCloud<PointType>::Ptr tempSurface;
    pcl::PointCloud<PointType>::Ptr tempEdge;

    pcl::PointCloud<PointType>::Ptr tempGroundTF;
    pcl::PointCloud<PointType>::Ptr tempCurbTF;
    pcl::PointCloud<PointType>::Ptr tempSurfaceTF;
    pcl::PointCloud<PointType>::Ptr tempEdgeTF;

    pcl::PointCloud<PointType>::Ptr currentPointCloud;

    pcl::VoxelGrid<PointType> downSizeFilterGround;
    pcl::VoxelGrid<PointType> downSizeFilterSurface;
    pcl::VoxelGrid<PointType> downSizeFilterCurb;
    pcl::VoxelGrid<PointType> downSizeFilterEdge;

    double distanceCount;

    int lengthMap;


    void initialValue();
    void resetPara();

public:
    Mapping(const int lengthMap_);

    void run();


};














} // end of namespace mapping


#endif // SRC_MAPPING_H
