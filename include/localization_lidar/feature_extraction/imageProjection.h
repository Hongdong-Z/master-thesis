//
// Created by zhao on 02.05.22.
//

#ifndef SRC_IMAGE_PROJECTION_H
#define SRC_IMAGE_PROJECTION_H

#include "common.h"
#include "utility/dbscan.h"
#include "utility/tic_toc.h"

namespace llo{


class ImageProjection{

private:
    // default param for lidar
    const int N_SCAN = 128;
    const int Horizon_SCAN = 1800;
    const int groundScanInd = 65;
    float segmentTheta = 1.0472;
    int segmentValidLineNum = 3;
    int segmentValidPointNum = 5;
    const float segmentAlphaX = 0.2 / 180.0 * M_PI;
    const float segmentAlphaY = 0.11 / 180.0 * M_PI;

    // input and output cloud
    pcl::PointCloud<PointType>::Ptr laserCloudIn;

    pcl::PointCloud<PointType>::Ptr fullCloud;
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;

    std::shared_ptr<pcl::PointCloud<PointType>> segmentedCloud;
    std::shared_ptr<pcl::PointCloud<PointType>> segmentedCloudPure;
    std::shared_ptr<pcl::PointCloud<PointType>> outlierCloud;

    //downSizeFilter
    pcl::VoxelGrid<PointType> downSizeFilter;

    // four features point cloud
    ////Ground param
    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr groundCloudDS;
    pcl::PointCloud<PointType>::Ptr groundFeatureFlat;
    pcl::PointCloud<PointType>::Ptr groundFeatureFlatDS; // NOT USE

    ////Surface param
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloudDS; // voxel filtered surfaceFeatures
    pcl::PointCloud<PointType>::Ptr nonSurfaceCloud; // cloud for edge extraction

    ////Surface segmentation param
    pcl::PointCloud<PointType>::Ptr cloud;
    pcl::PointCloud<PointType>::Ptr cloud_f;
    pcl::PointCloud<PointType>::Ptr cloud_filtered;
    pcl::PointIndices::Ptr inliers;
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::PointCloud<PointType>::Ptr cloud_plane;
    pcl::PointCloud<PointType>::Ptr lagerSurface;

    ////Curb param
    pcl::PointCloud<PointType>::Ptr curbCloud;
    pcl::NormalEstimation<pcl::PointXYZ , pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr pcNormal;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals;
    pcl::PointCloud<pcl::PointXYZ>::Ptr copyCloud;


    //// Edge param
    pcl::PointCloud<PointType>::Ptr edgeCloud;
    int length = 300; //length of whole grid
    int width = 80; // width of whole grid
    float meshSize = 0.15; // each grid size
    int minNumPoint = 40; // min number of point in cell 50

    class mesh { // definition of grid cell
    public:
        mesh() {}
        float high = 0.7;
        int density = 0; // the number of point in this mesh
        int state = 0; // state of this mesh 1: active, 0: inactive
        std::vector<int> pointsIndex;
    };



    PointType nanPoint;

    cv::Mat rangeMat;
    cv::Mat labelMat; // ground: 1 surface: 3 curb: 2 edge: 4
    cv::Mat groundMat;
    cv::Mat surfaceMat;
    cv::Mat curbMat;
    cv::Mat edgeMat;

    int labelCount;

    std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

//    uint16_t *allPushedIndX;
//    uint16_t *allPushedIndY;
//
//    uint16_t *queueIndX;
//    uint16_t *queueIndY;

    void allocateMemory();

    void resetParameters();
    void cloudCopy(const pcl::PointCloud<PointType> &cloud_in);
    void projectPointCloud();

    void groundRemoval();
    void groundFeatureExtraction();


    void curbRemoval();

    void surfaceRemoval();
    void surfaceSegmentation();


    void edgeRemoval();
    void edgeRemovalDBSAC();
//    void cloudSegmentation();
//
//    void labelComponents(int row, int col);



public:
    ImageProjection()
    {
        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }


    ~ImageProjection(){}


    void run(const pcl::PointCloud<PointType> &cloud_in){

        cloudCopy(cloud_in);
        projectPointCloud();
        TicToc featureExTime;
        groundRemoval();
        curbRemoval();
        surfaceRemoval();
        edgeRemoval();
        std::cerr << "Feature Extraction Time: " << featureExTime.toc() << "ms"<< std::endl;

    }

    pcl::PointCloud<PointType>::Ptr getGroundFeature() { // return ground features
//        std::cout << "The size of ground features: " << groundFeatureFlat->points.size() << std::endl;
        std::cout << "The size of ground features: " << groundCloudDS->points.size() << std::endl;
//        return groundFeatureFlat;
        return groundCloudDS;
    }
    pcl::PointCloud<PointType>::Ptr getGroundLessFlatFeature() { // return ground features
        std::cout << "The size of ground features: " << groundFeatureFlat->points.size() << std::endl;
//        std::cout << "The size of ground features: " << groundCloudDS->points.size() << std::endl;
        return groundFeatureFlat;
//        return groundCloudDS;
    }


    pcl::PointCloud<PointType>::Ptr getCurbFeature() { // return curb features
        std::cout << "The size of curb features: " << curbCloud->points.size() << std::endl;
        return curbCloud;
    }

    pcl::PointCloud<PointType>::Ptr getSurfaceFeature() { // return surface features
        std::cout << "The size of surface features: " << surfaceCloudDS->points.size() << std::endl;
        return surfaceCloudDS;
    }
    pcl::PointCloud<PointType>::Ptr getNonSurfaceFeature() { // return surface features
        std::cout << "The size of nonSurface features: " << nonSurfaceCloud->points.size() << std::endl;
        return nonSurfaceCloud;
    }

    pcl::PointCloud<PointType>::Ptr getEdgeFeature() { // return edge features
        std::cout << "The size of edge features: " << edgeCloud->points.size() << std::endl;
        return edgeCloud;
    }



};



}// end of namespace llo

#endif // SRC_SEGMENTATION_H
