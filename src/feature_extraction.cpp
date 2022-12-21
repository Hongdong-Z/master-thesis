//
// Created by zhao on 08.03.22.
//
#include "feature_extraction/feature_extraction.h"
#include "utility/dbscan.h"

namespace llo{

void FeaturesExtraction::initializationValue() {
    //initialization of PC
    pointCloudGround_ = std::make_shared<pcl::PointCloud<PointType>>();
    pointCloudCurb_ = std::make_shared<pcl::PointCloud<PointType>>();
    curbCluster_ = std::make_shared<pcl::PointCloud<PointType>>();
    pointCloudSurface_ = std::make_shared<pcl::PointCloud<PointType>>();
    pointCloudEdge_ = std::make_shared<pcl::PointCloud<PointType>>();

    nonPointCloudGround_ = std::make_shared<pcl::PointCloud<PointType>>();
    nonPointCloudCurb_ = std::make_shared<pcl::PointCloud<PointType>>();
    nonPointCloudSurface_ = std::make_shared<pcl::PointCloud<PointType>>();
//    nonPointCloudEdge_ = std::make_shared<pcl::PointCloud<PointType>>();

    Horizon_SCAN_ = pointCloud_->points.size() / N_SCAN_;

    groundMat_ = cv::Mat(N_SCAN_, Horizon_SCAN_, CV_8S, cv::Scalar::all(0));
    curbMat_ = cv::Mat(N_SCAN_, Horizon_SCAN_, CV_8S, cv::Scalar::all(0));
    surfaceMat_ = cv::Mat(N_SCAN_, Horizon_SCAN_, CV_8S, cv::Scalar::all(0));


    //Param for Curb Extraction initialization
    pcNormal.reset(new pcl::PointCloud<pcl::Normal>());
    tree.reset(new pcl::search::KdTree<pcl::PointXYZ>());
    cloud_with_normals.reset(new pcl::PointCloud<pcl::PointNormal>());
    copyCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());


}

void FeaturesExtraction::groundExtraction() {
    for (int i = 0; i < Horizon_SCAN_-1; ++i) {
        for (int j = 0; j < groundScanInd; ++j) {
            lowerInd = j + i * N_SCAN_; // row + column
            upperInd = (j + 1) + i * N_SCAN_;
            rightInd = j + (i + 1) * N_SCAN_;

            //param for ground extraction
            vectorDU_ground(0) = pointCloud_->points[upperInd].x - pointCloud_->points[lowerInd].x;
            vectorDU_ground(1) = pointCloud_->points[upperInd].y - pointCloud_->points[lowerInd].y;
            vectorDU_ground(2) = pointCloud_->points[upperInd].z - pointCloud_->points[lowerInd].z;

            vectorLR_ground(0) = pointCloud_->points[rightInd].x - pointCloud_->points[lowerInd].x;
            vectorLR_ground(1) = pointCloud_->points[rightInd].y - pointCloud_->points[lowerInd].y;
            vectorLR_ground(2) = pointCloud_->points[rightInd].z - pointCloud_->points[lowerInd].z;

            surfaceNormal_ground = vectorDU_ground.cross(vectorLR_ground);

//            angle = atan2(surfaceNormal_ground[2], sqrt(surfaceNormal_ground[0] * surfaceNormal_ground[0] + surfaceNormal_ground[1] * surfaceNormal_ground[1]));
            angle = surfaceNormal_ground[2] / surfaceNormal_ground.norm();
            if (abs(angle) > angleThreshold && abs(vectorLR[2]) < normalThreshold) {
                groundMat_.at<int8_t>(j, i) = 1;
                groundMat_.at<int8_t>(j+1, i) = 1;
                groundMat_.at<int8_t>(j, i+1) = 1;
            }

        }
    }

    for (size_t i = 0; i < Horizon_SCAN_; ++i){
        for (size_t j = 0; j < N_SCAN_; ++j){
            if (groundMat_.at<int8_t>(j,i) == 1 ){
                pointCloudGround_->push_back(pointCloud_->points[j + i * N_SCAN_]);
            }else{
                nonPointCloudGround_->push_back(pointCloud_->points[j + i * N_SCAN_]);
            }
        }
    }
    std::cout << " Cloud size of Ground Cloud: " << pointCloudGround_->points.size() << std::endl;
    std::cout << " Cloud size of nonGround Cloud: " << nonPointCloudGround_->points.size() << std::endl;
    std::cout << " Ground Extraction finish! "<< std::endl;
    std::cout << std::endl;

}

void FeaturesExtraction::curbExtraction() {
    groundExtraction();
    for (int i = 0; i < pointCloud_->points.size(); ++i) {
        pcl::PointXYZ tempPoint;
        tempPoint.x = pointCloud_->points[i].x;
        tempPoint.y = pointCloud_->points[i].y;
        tempPoint.z = pointCloud_->points[i].z;
        copyCloud->push_back(tempPoint);
    }

    tree->setInputCloud(copyCloud);
    ne.setInputCloud(copyCloud);
    ne.setSearchMethod(tree);
//    ne.setRadiusSearch (0.03);
    ne.setKSearch(5);
    ne.compute(*pcNormal);
    pcl::concatenateFields(*copyCloud, *pcNormal, *cloud_with_normals);

    for (int i = 0; i < N_SCAN_; ++i) {
        for (int j = 0; j < Horizon_SCAN_; ++j) {
            float yNormal = cloud_with_normals->points[i + j * N_SCAN_].normal_y;
            float zValue = cloud_with_normals->points[i + j * N_SCAN_].z;
            float yValue = cloud_with_normals->points[i + j * N_SCAN_].y;
            float xValue = cloud_with_normals->points[i + j * N_SCAN_].x;
            if (abs(yNormal) > curbThreshold && zValue < -1.5 && yValue < 25 && yValue > -10) {
                curbMat_.at<int8_t>(i, j) = 1;
            }
        }
    }

    for (size_t i = 0; i < Horizon_SCAN_; ++i){
        for (size_t j = 0; j < N_SCAN_; ++j){
            if (curbMat_.at<int8_t>(j,i) == 1 && groundMat_.at<int8_t>(j,i) == 1){
                pointCloudCurb_->push_back(pointCloud_->points[j + i * N_SCAN_]);
            }else{
                nonPointCloudCurb_->push_back(pointCloud_->points[j + i * N_SCAN_]);
            }
        }
    }
    // curb dbscan clustering
    utility::DBSCAN dsCurb(minPoints_curb);
    std::pair<float, pcl::PointCloud<PointType>> curbInputCloud(epsilon_curb, *pointCloudCurb_);
    *curbCluster_ = dsCurb.getClustering_curb(curbInputCloud);

    std::cout << " Cloud size of Curb Cloud after clustering:" << curbCluster_->points.size() << std::endl;
    std::cout << " Cloud size of Curb Cloud before clustering: " << pointCloudCurb_->points.size() << std::endl;
    std::cout << " Cloud size of nonCurb Cloud before clustering: " << nonPointCloudCurb_->points.size() << std::endl;
    std::cout << " Curb Extraction finish! "<< std::endl;
    std::cout << std::endl;
}

void FeaturesExtraction::surfaceExtraction() {
    curbExtraction();
    for (int i = 1; i < N_SCAN_ - 1; ++i) {
        for (int j = 2; j < Horizon_SCAN_ - 2; ++j) {
            PointType topPoint = pointCloud_->points[(i + 1) + j * N_SCAN_];
            PointType downPoint = pointCloud_->points[(i - 1) + j * N_SCAN_];
            PointType leftPoint = pointCloud_->points[i + (j - 1) * N_SCAN_];
            PointType rightPoint = pointCloud_->points[i + (j + 1) * N_SCAN_];

            PointType topPoint2 = pointCloud_->points[(i + 1) + (j + 1) * N_SCAN_];
            PointType downPoint2 = pointCloud_->points[(i - 1) + (j + 1) * N_SCAN_];
            PointType leftPoint2 = pointCloud_->points[i + (j - 1 + 1) * N_SCAN_];
            PointType rightPoint2 = pointCloud_->points[i + (j + 2) * N_SCAN_];


            vectorLR(0) = rightPoint.x - leftPoint.x;
            vectorLR(1) = rightPoint.y - leftPoint.y;
            vectorLR(2) = rightPoint.z - leftPoint.z;

            vectorDU(0) = topPoint.x - downPoint.x;
            vectorDU(1) = topPoint.y - downPoint.y;
            vectorDU(2) = topPoint.z - downPoint.z;
            surfaceNormal = vectorLR.cross(vectorDU);


            vectorLR2(0) = rightPoint2.x - leftPoint2.x;
            vectorLR2(1) = rightPoint2.y - leftPoint2.y;
            vectorLR2(2) = rightPoint2.z - leftPoint2.z;

            vectorDU2(0) = topPoint2.x - downPoint2.x;
            vectorDU2(1) = topPoint2.y - downPoint2.y;
            vectorDU2(2) = topPoint2.z - downPoint2.z;
            surfaceNormal2 = vectorLR2.cross(vectorDU2);

            normalProduct = surfaceNormal.cross(surfaceNormal2);


            if (normalProduct.norm() < parallelThreshold && abs(surfaceNormal[2]) < horizontalThreshold){
                surfaceMat_.at<int8_t>(i, j) = 1;
            }
        }
    }

    for (size_t i = 0; i < Horizon_SCAN_; ++i){
        for (size_t j = 0; j < N_SCAN_; ++j){
            //&& curbMat_.at<int8_t>(j,i) != 1
            if (surfaceMat_.at<int8_t>(j,i) == 1 && groundMat_.at<int8_t>(j,i) != 1 ){
                pointCloudSurface_->push_back(pointCloud_->points[j + i * N_SCAN_]);
            }else if(groundMat_.at<int8_t>(j,i) != 1 && curbMat_.at<int8_t>(j,i) != 1){
                nonPointCloudSurface_->push_back(pointCloud_->points[j + i * N_SCAN_]);
            }
        }
    }
    std::cout << " Cloud size of Original Surface Cloud: " << pointCloudSurface_->points.size() << std::endl;
    surfaceClustering();
    std::cerr << " Cloud size of Original Surface Cloud After Clustering: " << pointCloudSurface_->points.size() << std::endl;
    std::cout << " Cloud size of nonSurface Cloud: " << nonPointCloudSurface_->points.size() << std::endl;
    std::cout << " Surface Extraction finish! "<< std::endl;
    std::cout << std::endl;
}

void FeaturesExtraction::surfaceClustering() {

    std::cerr << " Start Surface Clustering ..... " << std::endl;
    std::cout << std::endl;
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>), cloud_f(new pcl::PointCloud<PointType>);
    *cloud = *pointCloudSurface_;
    std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl;
    pcl::VoxelGrid<PointType> vg;
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.2f, 0.2f, 0.2f);
    vg.filter(*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;

    pcl::SACSegmentation<PointType> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<PointType>::Ptr cloud_plane(new pcl::PointCloud<PointType>());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
//    Eigen::Vector3f zAxis(0,0,1);
//    seg.setAxis(zAxis);
//    seg.setEpsAngle(10);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);

    int i = 0, nr_points = (int)cloud_filtered->points.size();

    pcl::PointCloud<PointType>::Ptr lagerSurface(new pcl::PointCloud<PointType>);

    while (cloud_filtered->points.size() > 50)
    {

        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        pcl::ExtractIndices<PointType> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);
        if (inliers->indices.size() == 0){

            break;
        }

        extract.filter(*cloud_plane);
        if (abs(coefficients->values[2]) < 0.2 ) {
            *lagerSurface += *cloud_plane;
        }
        if (showCoefficient == true){
            std::cout << "PointCloud representing the planar component: " << i << " : " << cloud_plane->points.size() << " data points." << std::endl;
            std::cerr << " Model coefficients: " << i << " : " << coefficients->values[0] << " "
                      << coefficients->values[1] << " "
                      << coefficients->values[2] << " "
                      << coefficients->values[3] << std::endl;
        }

        extract.setNegative(true);
        extract.filter(*cloud_f);
        *cloud_filtered = *cloud_f;
        i++;
    }
    *pointCloudSurface_ = *lagerSurface;
    std::cerr << " End Surface Clustering " << std::endl;
    std::cout << std::endl;
}


void FeaturesExtraction::edgeExtraction(){
    surfaceExtraction();

    utility::DBSCAN ds(minPoints);

    std::pair<float, pcl::PointCloud<PointType>> edgeInputCloud(epsilon, *nonPointCloudSurface_);
    *pointCloudEdge_ = ds.getClustering(edgeInputCloud);
    std::cout << " Cloud size of Edge Cloud: " << pointCloudEdge_->points.size() << std::endl;
    std::cout << " Edge Extraction finish! "<< std::endl;
    std::cout << std::endl;
    std::cout << "-------Finish Processing-------" << std::endl;
    std::cout << std::endl;
}

} // namespace llo