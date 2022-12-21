//
// Created by zhao on 08.03.22.
//

#ifndef SRC_FEATURE_EXTRACTION_H
#define SRC_FEATURE_EXTRACTION_H


#include "common.h"
#include "localization_lidar/parameter_server.hpp"

namespace llo{

//! Process point cloud in each frame, return Ground,
//! Curb, Surface, Edge four features point cloud.
class FeaturesExtraction{
public:
    FeaturesExtraction(std::shared_ptr<pcl::PointCloud<PointType>> pointCloudIn, std::string featureName = "allFeatures")
            : pointCloud_(pointCloudIn) {
        std::cout << "----------Start Processing----------" << std::endl;
        initializationValue();

        if (featureName == "Ground"){
            groundExtraction();
        }else if (featureName == "Curb"){
            curbExtraction();
        }else if (featureName == "Surface"){
            surfaceExtraction();
        } else if (featureName == "Edge"){
            edgeExtraction();
        }else{
            edgeExtraction();
        }
    };

    void initializationValue();



    pcl::PointCloud<PointType> getGround() {
//        groundExtraction();
        return *pointCloudGround_;
    }
    pcl::PointCloud<PointType> getCurb() {
//        curbExtraction();
        return *curbCluster_;
    }
    pcl::PointCloud<PointType> getSurface() {
//        surfaceExtraction();
        return *pointCloudSurface_;
    }
    pcl::PointCloud<PointType> getEdge() {
//        edgeExtraction();
        return *pointCloudEdge_;
    }


    pcl::PointCloud<PointType> getNonGround(){
//        groundExtraction();
        return *nonPointCloudGround_;
    }
    pcl::PointCloud<PointType> getNonCurb(){
//        curbExtraction();
        return *nonPointCloudCurb_;
    }
    pcl::PointCloud<PointType> getNonSurface(){
//        surfaceExtraction();
        return *nonPointCloudSurface_;
    }
//    pcl::PointCloud<PointType> getNonEdge(){
//        edgeExtraction();
//        return *nonPointCloudEdge_;
//    }

    void getGroundFeatureSize(){
//        groundExtraction();
        std::cout << " Ground Feature Size: " << pointCloudGround_->points.size() << std::endl;
    }
    void getCurbFeatureSize(){
//        curbExtraction();
        std::cout << " Curb Feature Size:  " << pointCloudCurb_->points.size() << std::endl;
    }
    void getSurfaceFeatureSize(){
//        surfaceExtraction();
        std::cout << " Surface Feature Size:  " << pointCloudSurface_->points.size() << std::endl;
    }
    void getEdgeFeatureSize(){
//        edgeExtraction();
        std::cout << " Edge Feature Size:  " << pointCloudEdge_->points.size() << std::endl;
    }


private:

    void groundExtraction();
    void curbExtraction();
    void surfaceExtraction();
    void edgeExtraction();
    void surfaceClustering();
    bool showCoefficient = false;
    std::shared_ptr<pcl::PointCloud<PointType>> pointCloud_ = std::make_shared<pcl::PointCloud<PointType>>();

    std::shared_ptr<pcl::PointCloud<PointType>> pointCloudGround_;
    std::shared_ptr<pcl::PointCloud<PointType>> nonPointCloudGround_;
    std::shared_ptr<pcl::PointCloud<PointType>> pointCloudCurb_;
    std::shared_ptr<pcl::PointCloud<PointType>> curbCluster_;
    std::shared_ptr<pcl::PointCloud<PointType>> nonPointCloudCurb_;
    std::shared_ptr<pcl::PointCloud<PointType>> pointCloudSurface_;
    std::shared_ptr<pcl::PointCloud<PointType>> nonPointCloudSurface_;
    std::shared_ptr<pcl::PointCloud<PointType>> pointCloudEdge_;
//    std::shared_ptr<pcl::PointCloud<PointType>> nonPointCloudEdge_;

    int N_SCAN_ = 128;
    int Horizon_SCAN_;
    const int groundScanInd = 65;
    float horizontalRevolution = 0.1; // 0.1 deg

    cv::Mat groundMat_;
    cv::Mat curbMat_;
    cv::Mat surfaceMat_;
//    cv::Mat edgeMat_;

    //parameter for ground feature
    size_t lowerInd, upperInd, rightInd;
    float angle, angleThreshold = 0.95;
    float normalThreshold = 0.01;
    Eigen::Vector3f vectorDU_ground = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorLR_ground = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f surfaceNormal_ground = Eigen::Vector3f(0, 0, 0);

    //parameter for curb feature
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr pcNormal;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals;
    pcl::PointCloud<pcl::PointXYZ>::Ptr copyCloud;

    unsigned int minPoints_curb = 3;
    float epsilon_curb = 0.12*0.12;

    float curbThreshold = 0.55;

    //parameter for surface feature
    Eigen::Vector3f vectorLR = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorDU = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorLR2 = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorDU2 = Eigen::Vector3f(0, 0, 0);

    Eigen::Vector3f surfaceNormal = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f surfaceNormal2 = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f normalProduct = Eigen::Vector3f(0, 0, 0);

    float parallelThreshold = 0.01;
    float horizontalThreshold = 0.03;

    //parameter for edge feature
    unsigned int minPoints = 50;
    float epsilon = 0.15*0.15;
};


}// end of namespace llo
#endif // SRC_FEATURE_EXTRACTION_H
