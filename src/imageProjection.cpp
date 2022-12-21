//
// Created by zhao on 02.05.22.
//

#include "feature_extraction/imageProjection.h"


namespace llo{

void ImageProjection::allocateMemory() {


    downSizeFilter.setLeafSize(0.4, 0.4, 0.4);

    laserCloudIn.reset(new pcl::PointCloud<PointType>());

    fullCloud.reset(new pcl::PointCloud<PointType>());
    fullInfoCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);
    fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);


    ////Ground param
    groundCloud.reset(new pcl::PointCloud<PointType>());
    groundCloudDS.reset(new pcl::PointCloud<PointType>());
    groundFeatureFlat.reset(new pcl::PointCloud<PointType>());
    groundFeatureFlatDS.reset(new pcl::PointCloud<PointType>());

    ////Surface param
    surfaceCloud.reset(new pcl::PointCloud<PointType>());
    surfaceCloudDS.reset(new pcl::PointCloud<PointType>());
    nonSurfaceCloud.reset(new pcl::PointCloud<PointType>());

    ////Surface segmentation param
    cloud.reset(new pcl::PointCloud<PointType>());
    cloud_f.reset(new pcl::PointCloud<PointType>());
    cloud_filtered.reset(new pcl::PointCloud<PointType>());
    inliers.reset(new pcl::PointIndices());
    coefficients.reset(new pcl::ModelCoefficients());
    cloud_plane.reset(new pcl::PointCloud<PointType>());
    lagerSurface.reset(new pcl::PointCloud<PointType>());

    ////curb param allocateMemory
    curbCloud.reset(new pcl::PointCloud<PointType>());
    pcNormal.reset(new pcl::PointCloud<pcl::Normal>());
    tree.reset(new pcl::search::KdTree<pcl::PointXYZ>());
    cloud_with_normals.reset(new pcl::PointCloud<pcl::PointNormal>());
    copyCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    ////Edge param
    edgeCloud.reset(new pcl::PointCloud<PointType>());



    segmentedCloud = std::make_shared<pcl::PointCloud<PointType>>();
    segmentedCloudPure = std::make_shared<pcl::PointCloud<PointType>>();
    outlierCloud = std::make_shared<pcl::PointCloud<PointType>>();

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);



}

void ImageProjection::resetParameters() {
    laserCloudIn->clear();

    groundCloud->clear();
    groundCloudDS->clear();
    groundFeatureFlat->clear();
    groundFeatureFlatDS->clear();

    surfaceCloud->clear();
    surfaceCloudDS->clear();
    nonSurfaceCloud->clear();


    curbCloud->clear();
    pcNormal->clear();
    cloud_with_normals->clear();
    copyCloud->clear();


    edgeCloud->clear();


    segmentedCloud->clear();
    segmentedCloudPure->clear();
    outlierCloud->clear();

    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX)); // 32-bit ﬂoating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
    groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0)); //  8-bit signed integers ( -128..127 )
    surfaceMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0)); //  8-bit signed integers ( -128..127 )
    curbMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0)); //  8-bit signed integers ( -128..127 )
    edgeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0)); //  8-bit signed integers ( -128..127 )

    labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0)); // 32-bit signed integers ( -2147483648..2147483647 )
    labelCount = 1;

    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
}

void ImageProjection::cloudCopy(const pcl::PointCloud<PointType> &cloud_in) {
    pcl::copyPointCloud(cloud_in, *laserCloudIn);
}

void ImageProjection::projectPointCloud() { // project into 2d mat and initial fullCloud and fullInfoCloud
                                            // change intensity info for features association
    PointType thisPoint;
    float range;
    size_t index;
    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {

            index = i + j * N_SCAN;
            if (index >= laserCloudIn->points.size()) {
                std::cerr << "Cloud size is less then 128 * 1800 = 230400" << std::endl;
                continue;
            }
            thisPoint.x = laserCloudIn->points[index].x;
            thisPoint.y = laserCloudIn->points[index].y;
            thisPoint.z = laserCloudIn->points[index].z;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            rangeMat.at<float>(i, j) = range;
            thisPoint.intensity = (float)i+ (float)j / 10000.0;

            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range;
        }
    }

}


void ImageProjection::groundRemoval() { // Ground extraction
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    for (size_t j = 0; j < Horizon_SCAN; ++j){
        for (size_t i = 0; i < groundScanInd; ++i){

            lowerInd = i + j * N_SCAN;
            upperInd = (i + 1) + j * N_SCAN;


            if (fullCloud->points[lowerInd].intensity == -1 ||
                fullCloud->points[upperInd].intensity == -1){
                groundMat.at<int8_t>(i,j) = -1;
                continue;
            }


            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

            if (abs(angle) <= 10){
                groundMat.at<int8_t>(i,j) = 1;
                groundMat.at<int8_t>(i+1,j) = 1;
            }
        }
    }


    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                labelMat.at<int>(i,j) = -1;
            }
        }
    }

    for (size_t i = 0; i <= groundScanInd; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1 && rangeMat.at<float>(i,j) > 2.2)
                groundCloud->push_back(fullCloud->points[i + j * N_SCAN]);
        }
    }

    groundCloudDS->clear();
    downSizeFilter.setInputCloud(groundCloud);
    downSizeFilter.filter(*groundCloudDS);

//    groundFeatureExtraction();
//    std::cout << " Ground Extraction finish! "<< std::endl;
//    std::cout << std::endl;
}

void ImageProjection::groundFeatureExtraction() {
    int i1, i2, i3, i4, i5, i6, i7, i8;
    // using leftRight points and upDown points cross product to build surface normal
    Eigen::Vector3f vectorLR = Eigen::Vector3f(0, 0, 0); // left right
    Eigen::Vector3f vectorDU = Eigen::Vector3f(0, 0, 0); // down up

    Eigen::Vector3f surfaceNormal = Eigen::Vector3f(0, 0, 0); //


    for (int i = 1; i < N_SCAN - 1; ++i) {
        for (int j = 2; j < Horizon_SCAN - 2; ++j) {
            i1 = (i + 1) + j * N_SCAN;  i5 = (i + 1) + (j + 1) * N_SCAN; //     -  i1  - i5
            i2 = (i - 1) + j * N_SCAN;  i6 = (i - 1) + (j + 1) * N_SCAN; // i3  -  i7  - i4  - i8
            i3 = i + (j - 1) * N_SCAN;  i7 = i + (j - 1 + 1) * N_SCAN;//        -  i2  - i6
            i4 = i + (j + 1) * N_SCAN;  i8 = i + (j + 2) * N_SCAN;

            PointType topPoint = fullCloud->points[i1];
            PointType downPoint = fullCloud->points[i2];
            PointType leftPoint = fullCloud->points[i3];
            PointType rightPoint = fullCloud->points[i4];

            PointType topPoint2 = fullCloud->points[i5];
            PointType downPoint2 = fullCloud->points[i6];
            PointType leftPoint2 = fullCloud->points[i7];
            PointType rightPoint2 = fullCloud->points[i8];


            vectorLR(0) = rightPoint.x - leftPoint.x;
            vectorLR(1) = rightPoint.y - leftPoint.y;
            vectorLR(2) = rightPoint.z - leftPoint.z;

            vectorDU(0) = topPoint.x - downPoint.x;
            vectorDU(1) = topPoint.y - downPoint.y;
            vectorDU(2) = topPoint.z - downPoint.z;
            surfaceNormal = vectorLR.cross(vectorDU);
            surfaceNormal.normalize();
//            std::cout << "surface Normal" << surfaceNormal << std::endl;

            // remove occluded points


            if (abs(surfaceNormal[2]) < 0.15) {
                labelMat.at<int>(i, j) = 10;
            }
        }
    }
    for (int i = 0; i < N_SCAN; ++i) {
        for (int j = 0; j < Horizon_SCAN; ++j) {
            if (rangeMat.at<float>(i, j ) > 2.2) {
                if (groundMat.at<int8_t>(i, j) == 1 && labelMat.at<int>(i, j) == 10){
                    groundFeatureFlat->push_back(fullCloud->points[i + j * N_SCAN]);
                }
            }
        }
    }

}


void ImageProjection::curbRemoval() {
    pcl::PointXYZ tempPoint;

    for (int i = 0; i < fullCloud->points.size(); ++i) {
        tempPoint.x = fullCloud->points[i].x;
        tempPoint.y = fullCloud->points[i].y;
        tempPoint.z = fullCloud->points[i].z;
        if (fullCloud->points[i].intensity == -1) {
            std::cerr << "automatic jump nan point" << std::endl;
            continue;
        }
        copyCloud->push_back(tempPoint);
    }

    tree->setInputCloud(copyCloud);
    ne.setInputCloud(copyCloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(5);
    ne.compute(*pcNormal);
    pcl::concatenateFields(*copyCloud, *pcNormal, *cloud_with_normals);
//    pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/show/curb_3700.pcd", *cloud_with_normals);

    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            if ((i + j * N_SCAN) > cloud_with_normals->points.size()) {
                std::cerr << "automatic jump cloud normal" << std::endl;
                continue;
            }
            float normalY = cloud_with_normals->points[i + j * N_SCAN].normal_y;
            float zValue = cloud_with_normals->points[i + j * N_SCAN].z;
            float yValue = cloud_with_normals->points[i + j * N_SCAN].y;
            float xValue = cloud_with_normals->points[i + j * N_SCAN].x;


            if (abs(normalY) > 0.55 && yValue < 22 && yValue > -6 && abs(xValue) < 50 && zValue < -1.5) {
                curbMat.at<int8_t>(i, j) = 1;

            }
        }
    }


    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (rangeMat.at<float>(i, j) > 2.2){
                if (curbMat.at<int8_t>(i, j) == 1 && groundMat.at<int8_t>(i, j) == 1){
                    curbCloud->push_back(fullCloud->points[i + j * N_SCAN]);
                }
            }
        }
    }
    // DBSCAN remove outliers
    utility::DBSCAN dsCurb(3);
    std::pair<float, pcl::PointCloud<PointType>> curbInputCloud(0.12*0.12, *curbCloud);
    *curbCloud = dsCurb.getClustering_curb(curbInputCloud);
//    std::cout << " Curb Extraction finish! "<< std::endl;
//    std::cout << std::endl;
}


void ImageProjection::surfaceRemoval() {
    int i1, i2, i3, i4, i5, i6, i7, i8;
    Eigen::Vector3f vectorLR = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorDU = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorLR2 = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorDU2 = Eigen::Vector3f(0, 0, 0);

    Eigen::Vector3f surfaceNormal = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f surfaceNormal2 = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f normalProduct = Eigen::Vector3f(0, 0, 0);

    float parallelThreshold = 0.01;
    float horizontalThreshold = 0.03;

    for (int i = 1; i < N_SCAN - 1; ++i) {
        for (int j = 2; j < Horizon_SCAN - 2; ++j) {
            i1 = (i + 1) + j * N_SCAN;  i5 = (i + 1) + (j + 1) * N_SCAN; //     -  i1  - i5
            i2 = (i - 1) + j * N_SCAN;  i6 = (i - 1) + (j + 1) * N_SCAN; // i3  -  i7  - i4  - i8
            i3 = i + (j - 1) * N_SCAN;  i7 = i + (j - 1 + 1) * N_SCAN;//        -  i2  - i6
            i4 = i + (j + 1) * N_SCAN;  i8 = i + (j + 2) * N_SCAN;

            PointType topPoint = fullCloud->points[i1];
            PointType downPoint = fullCloud->points[i2];
            PointType leftPoint = fullCloud->points[i3];
            PointType rightPoint = fullCloud->points[i4];

            PointType topPoint2 = fullCloud->points[i5];
            PointType downPoint2 = fullCloud->points[i6];
            PointType leftPoint2 = fullCloud->points[i7];
            PointType rightPoint2 = fullCloud->points[i8];


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


            if (normalProduct.norm() < parallelThreshold && abs(surfaceNormal[2]) < horizontalThreshold) {
                surfaceMat.at<int8_t>(i, j) = 1;
            }
        }
    }

    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            //&& curbMat_.at<int8_t>(j,i) != 1
            if (surfaceMat.at<int8_t>(i, j) == 1 && groundMat.at<int8_t>(i, j) != 1) {
                surfaceCloud->push_back(fullCloud->points[i + j * N_SCAN]);
            }
            if (rangeMat.at<float>(i, j) > 2.2) {
                if (surfaceMat.at<int8_t>(i, j) != 1 && groundMat.at<int8_t>(i, j) != 1) {
                    // save nonSurface features for edge extraction
                    nonSurfaceCloud->push_back(fullCloud->points[i + j * N_SCAN]);
                }
            }
        }
    }
    // surface segmentation remove small number of points clustering
//    surfaceSegmentation();
    // voxel filter to reduce size of surface features
    downSizeFilter.setInputCloud(surfaceCloud);
    downSizeFilter.filter(*surfaceCloudDS);
//    std::cout << " Surface Extraction finish! "<< std::endl;
//    std::cout << std::endl;
}

void ImageProjection::surfaceSegmentation() { // surface segmentation remove small number of points clustering

//    std::cout << " Start Surface Clustering ..... " << std::endl;
//    std::cout << std::endl;
//    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>), cloud_f(new pcl::PointCloud<PointType>);
    *cloud = *surfaceCloud;
//    std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl;
    pcl::VoxelGrid<PointType> vg;
//    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.2f, 0.2f, 0.2f);
    vg.filter(*cloud_filtered);
//    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;

    pcl::SACSegmentation<PointType> seg;
//    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

//    pcl::PointCloud<PointType>::Ptr cloud_plane(new pcl::PointCloud<PointType>());
//    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
//    Eigen::Vector3f zAxis(0,0,1);
//    seg.setAxis(zAxis);
//    seg.setEpsAngle(10);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);

    int i = 0, nr_points = (int)cloud_filtered->points.size();

//    pcl::PointCloud<PointType>::Ptr lagerSurface(new pcl::PointCloud<PointType>);

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
        bool showCoefficient = false;
        if (showCoefficient){
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
    *surfaceCloud = *lagerSurface;
//    std::cout << " End Surface Clustering " << std::endl;
//    std::cout << std::endl;

}


void ImageProjection::edgeRemoval() { // edge extraction based on grid and RANSAC methods
    int row = std::floor(width/meshSize);
    int column = std::floor(length/meshSize);

    TicToc t_edge; // count edge extraction consumption
//    cv::Mat edgeMat = cv::Mat(row, column, CV_8S, cv::Scalar::all(0)); //(y, x) mat for show the result
    std::vector<std::vector<mesh>> allMesh(column + 1, std::vector<mesh>(row + 1)); // column for x row for y
//    for (int i = 0; i < column + 1; ++i) {
//        allMesh[i].resize(row + 1);
//    }
//    std::cerr << "row  column: " << allMesh[0].size() << " " << allMesh.size() << std::endl;

    // fill the mesh with points
    for (int i = 0; i < nonSurfaceCloud->points.size(); ++i) {
        float x = nonSurfaceCloud->points[i].x;
        float y = nonSurfaceCloud->points[i].y;
        float z = nonSurfaceCloud->points[i].z;
        if (abs(x) < length/2 && abs(y) < width/2 && z < 0.7) {  // range of selected area
            int c = std::floor((x - (-length / 2)) / meshSize); // x
            int r = std::floor((y - (-width / 2)) / meshSize); // y
            if (c > column || r > row) {
                std::cerr << "c and r: " << c << r << std::endl;
            }
            allMesh[c][r].pointsIndex.push_back(i);
            allMesh[c][r].density++;
        }
    }

    // label valid mesh
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].density > minNumPoint) {
                allMesh[i][j].state = 1;
//                edgeMat.at<int8_t>(j,i) = allMesh[i][j].density;
            }
        }
    }
    // for each candidate push in RANSAC to fit a cylinder
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].state == 1) {

//                pcl::PointCloud<PointType>::Ptr edgeCandidate(new pcl::PointCloud<PointType>);
//                pcl::PointCloud<pcl::PointXYZ>::Ptr candidateXYZ(new pcl::PointCloud<pcl::PointXYZ>);
//                for (std::vector<int>::iterator iter = allMesh[i][j].pointsIndex.begin(); iter != allMesh[i][j].pointsIndex.end(); iter++) {
//                    edgeCandidate->push_back(nonSurfaceCloud->points[*iter]); // push in candidate for cylinder ransac
//                }
                for (int k = 0; k < allMesh[i][j].pointsIndex.size(); k++) {
                    edgeCloud->push_back(nonSurfaceCloud->points[allMesh[i][j].pointsIndex[k]]); // push in candidate for cylinder ransac
                }
            }
        }
    }
//    std::cout << " Edge Extraction time: " << t_edge.toc() << std::endl;
//    std::cout << " Cloud size of Edge Cloud: " << edgeCloud->points.size() << std::endl;
//    std::cout << " Edge Extraction finish! "<< std::endl;
//    std::cout << std::endl;
}


void ImageProjection::edgeRemovalDBSAC() { // edge extraction with DBSACN TWO SLOW NOT USE
    utility::DBSCAN ds(100); // min number of points for DBSCAN
    float epsilon = 0.15 *0.15;
    std::pair<float, pcl::PointCloud<PointType>> edgeInputCloud(epsilon, *nonSurfaceCloud);
    TicToc t_edge;
    *edgeCloud = ds.getClustering(edgeInputCloud);
//    printf("Edge Extraction time %f \n", t_edge.toc());
//    std::cout << " Cloud size of Edge Cloud: " << edgeCloud->points.size() << std::endl;
//    std::cout << " Edge Extraction finish! "<< std::endl;
    std::cout << "******** FeatureExtraction Finish **********" << std::endl;
}

//void ImageProjection::cloudSegmentation() {
//
//    for (size_t i = 0; i < N_SCAN; ++i)
//        for (size_t j = 0; j < Horizon_SCAN; ++j)
//            // 如果labelMat[i][j]=0,表示没有对该点进行过分类
//            // 需要对该点进行聚类
//            if (labelMat.at<int>(i,j) == 0)
//                labelComponents(i, j);
//
//    int sizeOfSegCloud = 0;
//    for (size_t i = 0; i < N_SCAN; ++i) {
//
//        // segMsg.startRingIndex[i]
//        // segMsg.endRingIndex[i]
//        // 表示第i线的点云起始序列和终止序列
//        // 以开始线后的第6线为开始，以结束线前的第6线为结束
////        segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;
//
//        for (size_t j = 0; j < Horizon_SCAN; ++j) {
//            // 找到可用的特征点或者地面点(不选择labelMat[i][j]=0的点)
//            if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
//                // labelMat数值为999999表示这个点是因为聚类数量不够30而被舍弃的点
//                // 需要舍弃的点直接continue跳过本次循环，
//                // 当列数为5的倍数，并且行数较大，可以认为非地面点的，将它保存进异常点云(界外点云)中
//                // 然后再跳过本次循环
//                if (labelMat.at<int>(i,j) == 999999){
//                    if (i > groundScanInd && j % 5 == 0){
//                        outlierCloud->push_back(fullCloud->points[i + j * N_SCAN]);
//                        continue;
//                    }else{
//                        continue;
//                    }
//                }
//
//                // 如果是地面点,对于列数不为5的倍数的，直接跳过不处理
//                if (groundMat.at<int8_t>(i,j) == 1){
//                    if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
//                        continue;
//                }
//                // 上面多个if语句已经去掉了不符合条件的点，这部分直接进行信息的拷贝和保存操作
//                // 保存完毕后sizeOfSegCloud递增
////                segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
////                segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
////                segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
//                segmentedCloud->push_back(fullCloud->points[i + j * N_SCAN]);
//                ++sizeOfSegCloud;
//            }
//        }
//
//        // 以结束线前的第5线为结束
////        segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
//    }
//
//    // 如果有节点订阅SegmentedCloudPure,
//    // 那么把点云数据保存到segmentedCloudPure中去
//    for (size_t i = 0; i < N_SCAN; ++i){
//        for (size_t j = 0; j < Horizon_SCAN; ++j){
//            // 需要选择不是地面点(labelMat[i][j]!=-1)和没被舍弃的点
//            if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
//                segmentedCloudPure->push_back(fullCloud->points[i + j * N_SCAN]);
//                segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
//            }
//        }
//    }
//}
//
//void ImageProjection::labelComponents(int row, int col) {
//    float d1, d2, alpha, angle;
//    int fromIndX, fromIndY, thisIndX, thisIndY;
//    bool lineCountFlag[N_SCAN];
//    lineCountFlag[N_SCAN] = {false};
//
//    queueIndX[0] = row;
//    queueIndY[0] = col;
//    int queueSize = 1;
//    int queueStartInd = 0;
//    int queueEndInd = 1;
//
//    allPushedIndX[0] = row;
//    allPushedIndY[0] = col;
//    int allPushedIndSize = 1;
//
//    // 标准的BFS
//    // BFS的作用是以(row，col)为中心向外面扩散，
//    // 判断(row,col)是否是这个平面中一点
//    while(queueSize > 0){
//        fromIndX = queueIndX[queueStartInd];
//        fromIndY = queueIndY[queueStartInd];
//        --queueSize;
//        ++queueStartInd;
//        // labelCount的初始值为1，后面会递增
//        labelMat.at<int>(fromIndX, fromIndY) = labelCount;
//
//        // neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
//        // 遍历点[fromIndX,fromIndY]边上的四个邻点
//        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
//
//            thisIndX = fromIndX + (*iter).first;
//            thisIndY = fromIndY + (*iter).second;
//
//            if (thisIndX < 0 || thisIndX >= N_SCAN)
//                continue;
//
//            // 是个环状的图片，左右连通
//            if (thisIndY < 0)
//                thisIndY = Horizon_SCAN - 1;
//            if (thisIndY >= Horizon_SCAN)
//                thisIndY = 0;
//
//            // 如果点[thisIndX,thisIndY]已经标记过
//            // labelMat中，-1代表无效点，0代表未进行标记过，其余为其他的标记
//            // 如果当前的邻点已经标记过，则跳过该点。
//            // 如果labelMat已经标记为正整数，则已经聚类完成，不需要再次对该点聚类
//            if (labelMat.at<int>(thisIndX, thisIndY) != 0)
//                continue;
//
//            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
//                          rangeMat.at<float>(thisIndX, thisIndY));
//            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
//                          rangeMat.at<float>(thisIndX, thisIndY));
//
//            // alpha代表角度分辨率，
//            // X方向上角度分辨率是segmentAlphaX(rad)
//            // Y方向上角度分辨率是segmentAlphaY(rad)
//            if ((*iter).first == 0)
//                alpha = segmentAlphaX;
//            else
//                alpha = segmentAlphaY;
//
//            // 通过下面的公式计算这两点之间是否有平面特征
//            // atan2(y,x)的值越大，d1，d2之间的差距越小,越平坦
//            angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));
//
//            if (angle > segmentTheta){
//                // segmentTheta=1.0472<==>60度
//                // 如果算出角度大于60度，则假设这是个平面
//                queueIndX[queueEndInd] = thisIndX;
//                queueIndY[queueEndInd] = thisIndY;
//                ++queueSize;
//                ++queueEndInd;
//
//                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
//                lineCountFlag[thisIndX] = true;
//
//                allPushedIndX[allPushedIndSize] = thisIndX;
//                allPushedIndY[allPushedIndSize] = thisIndY;
//                ++allPushedIndSize;
//            }
//        }
//    }
//
//
//    bool feasibleSegment = false;
//
//    // 如果聚类超过30个点，直接标记为一个可用聚类，labelCount需要递增
//    if (allPushedIndSize >= 30)
//        feasibleSegment = true;
//    else if (allPushedIndSize >= segmentValidPointNum){
//        // 如果聚类点数小于30大于等于5，统计竖直方向上的聚类点数
//        int lineCount = 0;
//        for (size_t i = 0; i < N_SCAN; ++i)
//            if (lineCountFlag[i] == true)
//                ++lineCount;
//
//        // 竖直方向上超过3个也将它标记为有效聚类
//        if (lineCount >= segmentValidLineNum)
//            feasibleSegment = true;
//    }
//
//    if (feasibleSegment == true){
//        ++labelCount;
//    }else{
//        for (size_t i = 0; i < allPushedIndSize; ++i){
//            // 标记为999999的是需要舍弃的聚类的点，因为他们的数量小于30个
//            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
//        }
//    }
//}








}//end of namespace llo