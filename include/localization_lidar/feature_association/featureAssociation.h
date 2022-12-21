//
// Created by zhao on 03.05.22.
//

#ifndef SRC_FEATURE_ASSOCIATION_H
#define SRC_FEATURE_ASSOCIATION_H
#include "common.h"
#include "feature_extraction/imageProjection.h"
#include "nlohmann/json.hpp"
//#include "localization_lidar/load_datastorage.hpp"
//#include "localization_lidar/parameter_server.hpp"
#include "utility/tic_toc.h"

namespace odo {

class FeatureAssociation : llo::ImageProjection { // calculate transform between two frames

private:
    const int N_SCAN = 128;
    const int Horizon_SCAN = 1800;
    const int groundScanInd = 65;

    bool systemInited = false;

    // param for correspondence find
    int ground_correspondence = 0, curb_correspondence = 0,
        surface_correspondence = 0, edge_correspondence = 0;

    // current frame features from imageProjection
    pcl::PointCloud<PointType>::Ptr groundFeature;
    pcl::PointCloud<PointType>::Ptr curbFeature;
    pcl::PointCloud<PointType>::Ptr surfaceFeature;
    pcl::PointCloud<PointType>::Ptr edgeFeature;

    // feature association cloud for debug
    pcl::PointCloud<PointType>::Ptr pcGround;
    pcl::PointCloud<PointType>::Ptr pcCurb;
    pcl::PointCloud<PointType>::Ptr pcSurface;
    pcl::PointCloud<PointType>::Ptr pcEdge;
    pcl::PointCloud<PointType>::Ptr pcTwoFrame;
    pcl::PointCloud<PointType>::Ptr lastFrame;
    pcl::PointCloud<PointType>::Ptr currentFrame;
    // if debug turn to true
    bool saveFeatureAssociation = false;

    // last frame features from imageProjection
    pcl::PointCloud<PointType>::Ptr groundFeatureLast;
    pcl::PointCloud<PointType>::Ptr curbFeatureLast;
    pcl::PointCloud<PointType>::Ptr surfaceFeatureLast;
    pcl::PointCloud<PointType>::Ptr edgeFeatureLast;

    // kdTree to store last frame features
    pcl::KdTreeFLANN<PointType>::Ptr kdTreeGroundLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdTreeCurbLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdTreeSurfaceLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdTreeEdgeLast;

    // param for kdTree
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // Projected point
    PointType point_;

    // nearest feature search distance
    float nearestFeatureSearchSqDist;



    // Transformation from current frame to world frame
    Eigen::Quaterniond q_w_curr = Eigen::Quaterniond (1, 0, 0, 0);
    Eigen::Vector3d t_w_curr = Eigen::Vector3d (0, 0, 0);

    // q_curr_last(x, y, z, w), t_curr_last
    double para_q[4] = {0, 0, 0, 1};
    double para_t[3] = {0, 0, 0};

    // parameters calculated by new method
    double para_q_new[4] = {0, 0, 0, 1};
    double para_t_new[3] = {0, 0, 0};

    Eigen::Map<Eigen::Quaterniond> q_last_curr = Eigen::Map<Eigen::Quaterniond> (para_q);
    Eigen::Map<Eigen::Vector3d> t_last_curr = Eigen::Map<Eigen::Vector3d> (para_t);



    // parameter to calculate
//    double params_[6]; // x y z r p y

    // transform between para_q para_t and params_
    // now calculate params x y z r p y to get pose
    void inverseInnerTransform();


    void initializationValue();

    // undistort lidar point
    void TransformToStart(PointType const *const pi, PointType *const po);
    // transform all lidar points to the start of the next frame



    void updateTransformation();

    void updateFeatures();

    void toEulerAngle(const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw);


public:

    FeatureAssociation(const pcl::PointCloud<PointType>::Ptr oriCloudLast,
                       const pcl::PointCloud<PointType>::Ptr oriCloudCur,
                       Eigen::Quaterniond qCurr = Eigen::Quaterniond (1, 0, 0, 0),
                       Eigen::Vector3d tCurr = Eigen::Vector3d (0, 0, 0),
                       Eigen::Quaterniond q_last_curr_ = Eigen::Quaterniond (1, 0, 0, 0),
                       Eigen::Vector3d t_last_curr_ = Eigen::Vector3d (0, 0, 0));

    void run();



    // Current frame pose
    Eigen::Quaterniond publishQuaterniond();

    Eigen::Vector3d publishTranslation();

    // transform from last frame for better initial guess
    Eigen::Quaterniond quaterniodGuess();

    Eigen::Vector3d translationGuess();




    pcl::PointCloud<PointType>::Ptr getFeatureAssociationGround() { // edge feature association for debug
        return pcGround;
    }
    pcl::PointCloud<PointType>::Ptr getFeatureAssociationCurb() { // edge feature association for debug
        return pcCurb;
    }
    pcl::PointCloud<PointType>::Ptr getFeatureAssociationSurface() { // edge feature association for debug
        return pcSurface;
    }
    pcl::PointCloud<PointType>::Ptr getFeatureAssociationEdge() { // edge feature association for debug
        return pcEdge;
    }
    pcl::PointCloud<PointType>::Ptr getFusionFrame() { // edge feature association for debug
        return pcTwoFrame;
    }

    pcl::PointCloud<PointType>::Ptr getGroundFeatureLast() {
        return groundFeatureLast;
    }
    pcl::PointCloud<PointType>::Ptr getSurfaceFeatureLast() {
        return surfaceFeatureLast;
    }
    pcl::PointCloud<PointType>::Ptr getEdgeFeatureLast() {
        return edgeFeatureLast;
    }
    pcl::PointCloud<PointType>::Ptr getCurbFeatureLast() {
        return curbFeatureLast;
    }

    ////return current feature for map matching
    pcl::PointCloud<PointType>::Ptr getEdgeFeature() {
        return edgeFeature;
    }
    pcl::PointCloud<PointType>::Ptr getGroundFeature() {
        return groundFeature;
    }
    pcl::PointCloud<PointType>::Ptr getSurfaceFeature() {
        return surfaceFeature;
    }
    pcl::PointCloud<PointType>::Ptr getCurbFeature() {
        return curbFeature;
    }


};



} //end of namespace odo
#endif // SRC_FEATURE_ASSOCIATION_H

