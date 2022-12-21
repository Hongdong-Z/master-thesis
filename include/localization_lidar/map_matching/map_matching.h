//
// Created by zhao on 06.06.22.
//

#ifndef SRC_MAP_MATCHING_H
#define SRC_MAP_MATCHING_H
#include "common.h"

#include "nlohmann/json.hpp"
//#include "localization_lidar/load_datastorage.hpp"
//#include "localization_lidar/parameter_server.hpp"
#include "utility/tic_toc.h"

namespace mm{


class MapMatching{ // MapMatching receive frame features and local corresponding map to perform
                                                // map frame matching in order to fine tune pose


private:

    //input: four features from odo
    pcl::PointCloud<PointType>::Ptr laserCloudGroundLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfaceLast;
    pcl::PointCloud<PointType>::Ptr laserCloudEdgeLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCurbLast;

    //input: local map of four features
    pcl::PointCloud<PointType>::Ptr localMapGround;
    pcl::PointCloud<PointType>::Ptr localMapSurface;
    pcl::PointCloud<PointType>::Ptr localMapEdge;
    pcl::PointCloud<PointType>::Ptr localMapCurb;

    //input: calculated pose from odo
    Eigen::Quaterniond q_wodom_curr = Eigen::Quaterniond (1, 0, 0, 0);
    Eigen::Vector3d t_wodom_curr = Eigen::Vector3d (0, 0, 0);

    //output: fine tuned pose
    double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
    double para_q[4] = {0, 0, 0, 1};
    double para_t[3] = {0, 0, 0};

    // parameters calculated by new method
    double para_q_new[4] = {0, 0, 0, 1};
    double para_t_new[3] = {0, 0, 0};

    Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond> (para_q);
    Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d> (para_t);

    // parameter inside class
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGroundFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeEdgeFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCurbFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfaceFromMap;

    //sub map
    pcl::PointCloud<PointType>::Ptr subMapGround;
    pcl::PointCloud<PointType>::Ptr subMapSurface;
    pcl::PointCloud<PointType>::Ptr subMapEdge;
    pcl::PointCloud<PointType>::Ptr subMapCurb;

    pcl::VoxelGrid<PointType> downSizeFilterGround;
    pcl::VoxelGrid<PointType> downSizeFilterSurface;
    pcl::VoxelGrid<PointType> downSizeFilterCurb;
    pcl::VoxelGrid<PointType> downSizeFilterEdge;

    pcl::PointCloud<PointType>::Ptr laserCloudGroundStack;
    pcl::PointCloud<PointType>::Ptr laserCloudCurbStack;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfaceStack;
    pcl::PointCloud<PointType>::Ptr laserCloudEdgeStack;

    // for debug

    bool saveAssociation = false;

    pcl::PointCloud<PointType>::Ptr testCloudGround;
    pcl::PointCloud<PointType>::Ptr testCloudSurface;
    pcl::PointCloud<PointType>::Ptr testCloudEdge;
    pcl::PointCloud<PointType>::Ptr testCloudCurb;

    PointType pointOri, pointSel;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    float subMapSize = 100.;

    // wmap_T_odom * odom_T_curr = wmap_T_curr;
    // transformation between odom's world and map's world frame
    Eigen::Quaterniond q_wmap_wodom = Eigen::Quaterniond (1, 0, 0, 0);
    Eigen::Vector3d t_wmap_wodom = Eigen::Vector3d (0, 0, 0);


    //set initial guess
    void transformAssociateToMap();

    void  transformUpdate();

    void pointAssociateToMap(PointType const *const pi, PointType *const po);

    void getSubMap();

    void loadMap();

    void initialValue();
    void toEulerAngle(const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw);



public:
    MapMatching(const pcl::PointCloud<PointType>::Ptr laserCloudGroundLast_,
                const pcl::PointCloud<PointType>::Ptr laserCloudCurbLast_,
                const pcl::PointCloud<PointType>::Ptr laserCloudSurfaceLast_,
                const pcl::PointCloud<PointType>::Ptr laserCloudEdgeLast_,
                const pcl::PointCloud<PointType>::Ptr groundMap,
                const pcl::PointCloud<PointType>::Ptr curbMap,
                const pcl::PointCloud<PointType>::Ptr surfaceMap,
                const pcl::PointCloud<PointType>::Ptr edgeMap,
                const Eigen::Quaterniond q_wodom_curr_,
                const Eigen::Vector3d t_wodom_curr_,
                const Eigen::Quaterniond q_wmap_wodom_,
                const Eigen::Vector3d t_wmap_wodom_);

    Eigen::Map<Eigen::Quaterniond> getFineTunedQuaterniond() {
        return q_w_curr;
    }


    Eigen::Map<Eigen::Vector3d> getFineTunedTranslation() {
        return t_w_curr;
    }

    Eigen::Quaterniond getQuaterniondGuess() {
//        std::cout << "q_wmap_wodom:\n" << q_wmap_wodom.matrix() << std::endl;
        return q_wmap_wodom;
    }


    Eigen::Vector3d getTranslationGuess() {
//        std::cout << "t_wmap_wodom:\n" << t_wmap_wodom << std::endl;
        return t_wmap_wodom;
    }
    void run();


    pcl::PointCloud<PointType>::Ptr getSubMapGround() {
        return subMapGround;
    }
    pcl::PointCloud<PointType>::Ptr getSubMapCurb() {
        return subMapCurb;
    }
    pcl::PointCloud<PointType>::Ptr getSubMapSurface() {
        return subMapSurface;
    }
    pcl::PointCloud<PointType>::Ptr getSubMapEdge() {
        return subMapEdge;
    }

    pcl::PointCloud<PointType>::Ptr getAssocGround() {
        return testCloudGround;
    }
    pcl::PointCloud<PointType>::Ptr getAssocCurb() {
        return testCloudCurb;
    }
    pcl::PointCloud<PointType>::Ptr getAssocSurface() {
        return testCloudSurface;
    }
    pcl::PointCloud<PointType>::Ptr getAssocEdge() {
        return testCloudEdge;
    }

    Eigen::Quaterniond publishQuaterniond() {
        return q_w_curr;
    }
    Eigen::Vector3d publishTranslation() {
        return t_w_curr;
    }







};


} // end of namespace mm

















#endif // SRC_MAP_MATCHING_H
