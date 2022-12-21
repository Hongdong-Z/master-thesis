//
// Created by zhao on 24.06.22.
//

#ifndef SRC_PRIMITIVESEXTRACTION_H
#define SRC_PRIMITIVESEXTRACTION_H

#include <set>
#include <stdint.h>
#include <vector>

#include <algorithm>
#include <chrono>
#include <limits>
#include <math.h>
#include <random>
#include <thread>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/don.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>


#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <ceres/ceres.h>

#include <nlopt.hpp>

#include <google_mv/levenberg_marquardt.h>

#include <ros/console.h>

#include <rviz_visual_tools/rviz_visual_tools.h>

#include <glog/logging.h>
#include "utility/dbscan.h"



namespace primitivesExtraction {


using Point = pcl::PointXYZI;
using PointXYZ = pcl::PointXYZ;
using PointINormal = pcl::PointXYZINormal;
using Cloud = pcl::PointCloud<Point>;
using CloudXYZ = pcl::PointCloud<PointXYZ>;

using PointColor = pcl::PointXYZRGB;
using CloudColor = pcl::PointCloud<PointColor>;

struct Point_Cluster {
    Point_Cluster() : cloud(new Cloud()), area_rough(0.0f), centroid(Point()) {
    }
    Cloud::Ptr cloud;
    float area_rough; // In meter^2.
    float bb_width;
    float bb_height;
    Point centroid;
};

struct Line_2D {
    Line_2D()
        : direction(Eigen::Vector2f(0.0f, 0.0f)), p(Eigen::Vector2f(0.0f, 0.0f)),
          start_p(Eigen::Vector2f(0.0f, 0.0f)), end_p(Eigen::Vector2f(0.0f, 0.0f)),
          top_match_id(std::vector<int>()), checked(false), support(new Cloud) {
    }
    Eigen::Vector2f direction;
    Eigen::Vector2f p;
    Eigen::Vector2f start_p;
    Eigen::Vector2f end_p;
    std::vector<int> top_match_id;
    std::vector<int> bottom_match_id;
    bool checked;
    Cloud::Ptr support;
};


struct Circle_2D {
    Circle_2D() : center(Eigen::Vector2f(0.0f, 0.0f)), diameter(0.0f), support(new Cloud) {
    }
    Eigen::Vector2f center;
    float diameter;
    Cloud::Ptr support;
};

struct Line_Inf_Param {
    Eigen::Vector3f p;
    Eigen::Vector3f dir;
};

struct Line_Fin_Param {
    Eigen::Vector3f p_bottom;
    Eigen::Vector3f p_top;
};

struct Line_Fin_Param_2D {
    Eigen::Vector2f p_start;
    Eigen::Vector2f p_end;
};

struct Plane_Param { // n_1 +n_2 + n_3 + d = 0
    Eigen::Vector3f n;
    float d;
    std::vector<Eigen::Vector3f> edge_poly; // tl, tr, dr, dl 3, 2, 1, 0
    Line_Fin_Param_2D line_2D;
    float ground_z_start;
    float ground_z_end;
};

struct Cylinder_Fin_Param {
    Line_Fin_Param center_line;
    Eigen::Vector3f line_dir;
    Eigen::Vector3f top_point;
    Eigen::Vector3f down_point;
    Eigen::Vector2f pos_2D;
    float radius;
    float ground_z;
};

// Residual = || (sample-p) x dir ||/||dir|| - r .
struct CylinderResidual {
    CylinderResidual(const Point& sample) : sample_(sample.x, sample.y, sample.z) {
    }

    template <typename T>
    bool operator()(const T* const p_x,
                    const T* const p_y,
                    const T* const dir_x,
                    const T* const dir_y,
                    const T* const r,
                    T* residual) const {

        Eigen::Matrix<T, 3, 1> p(p_x[0], p_y[0], T(0.0));       // p_z = 0 by default.
        Eigen::Matrix<T, 3, 1> dir(dir_x[0], dir_y[0], T(1.0)); // dir_z = 1 by default.

        residual[0] = ((sample_.cast<T>() - p).cross(dir)).norm() / dir.norm() - r[0];

        return true;
    }

private:
    // Observations for a sample.
    const Eigen::Vector3d sample_;
};

inline double cylinder_objective_nlopt(const std::vector<double>& x, std::vector<double>& grad, void* points_in) {
    std::vector<Eigen::Vector3d>* points = reinterpret_cast<std::vector<Eigen::Vector3d>*>(points_in);
    Eigen::Vector3d p(x[0], x[1], 0);
    Eigen::Vector3d dir(x[2], x[3], 1);
    double dir_norm = dir.norm();
    double r = x[4];
    double residual_sum = 0;

    grad.resize(5, 0);

    double res_temp;
    for (int i = 0; i < (int)points->size(); ++i) {


        // Add distance of point to cylinder hull.
        res_temp = (((*points)[i] - p).cross(dir)).norm() / dir_norm - r;
        residual_sum += res_temp * res_temp;
    }

    return residual_sum;
}

class CylinderResidual_google_mv {
public:
    // The residual input type (parameters).
    using XMatrixType = Eigen::Matrix<double, 5, 1>;
    // The residual output type (residuals).
    using FMatrixType = Eigen::VectorXd;

    CylinderResidual_google_mv(const Eigen::Matrix3Xd& points) : points_(points) {
        //      float min_x = points(0,0);
        //      float max_x = points(0,0);
        //      float min_y = points(1,0);
        //      float max_y = points(1,0);

        //      for(int i = 0; i < points.cols(); ++i)
        //      {

        //      }
    }

    Eigen::VectorXd operator()(const Eigen::Matrix<double, 5, 1>& parameters) const {
        // Extract parameters (support point, direction, radius).
        const Eigen::Vector3d supportPoint_1(parameters(0), parameters(1), 0.0);
        const Eigen::Vector3d directionRaw(parameters(2), parameters(3), 1.0);
        const double radius = parameters[4];

        // Evaluate normalized direction.
        const double squaredNorm = directionRaw.squaredNorm();
        const Eigen::Vector3d direction = directionRaw / std::sqrt(squaredNorm);

        const Eigen::Vector3d supportPoint_2 = supportPoint_1 + direction;

        Eigen::VectorXd dists;
        dists.resize(points_.cols());
        for (int i = 0; i < dists.size(); ++i) {
            dists[i] = (points_.col(i) - supportPoint_1).cross(points_.col(i) - supportPoint_2).norm();
        }
        Eigen::VectorXd residuals(points_.cols());


        for (int i = 0; i < residuals.size(); ++i) {
            double diff = std::abs(dists(i) - radius);

            if (diff > 0.15)
                residuals[i] = 0.5;
            else {
                double diff_2 = diff * diff;
                double diff_3 = diff_2 * diff;
                residuals[i] = 11.66667 * diff + 277.7778 * diff_2 - 2222.222 * diff_3;
            }
        }

        return residuals;
    }

private:
    // The measurements.
    const Eigen::Matrix3Xd& points_;
    int max_dist_;
};

class Time {
public:
    Time(const std::string& name = "") : name_(name) {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    ~Time() {
        auto stop_time = std::chrono::high_resolution_clock::now();
        std::cout << "Time " << name_ << ": "
                  << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time_).count() *
                     1e-6
                  << "ms" << std::endl;
    }

private:
    std::string name_;
    decltype(std::chrono::high_resolution_clock::now()) start_time_;
};


class PrimitiveExtractor {

private:
    int N_SCAN = 128; // 128 64
    int Horizon_SCAN = 1800; //vlp 128 1800 kitti1950
    int groundScanInd = 65;

    Cloud::ConstPtr cloudIn;
    Cloud::Ptr fullCloud;
    Point nanPoint;

    Cloud::Ptr groundCloud;
    Cloud::Ptr nonGroundCloud;
    Cloud::Ptr facadeCloudRough;
    Cloud::Ptr nonFacadeCloudRough;
    Cloud::Ptr poleCloudRough;
    Cloud::Ptr poleCloudGrow;

    std::vector<Cloud::Ptr> poleCandidates;
    std::vector<Cloud::Ptr> facadeCandidates;
    std::vector<Cloud::Ptr> groundCandidates;

    // Region growing segmentation for facades
    std::vector<pcl::PointIndices> clustersFacades;

    // Cloud after region growing and PCA filter out non facade feature using for mapping
    Cloud::Ptr facadeCloudCandidates;

    std::vector<float> range;

    std::vector<Cloud::Ptr> zSplits; // store clouds into different splits.
    std::vector<cv::Mat> zSplitsMats; // project clouds into images
    cv::Mat z_profile_ground_;
    Plane_Param ground_plane_;
    float ground_average_z_;

    std::vector<std::vector<Point_Cluster>> pointClusters;        // [layer_id][cluster_id] clusters before filtering.
    std::vector<std::pair<int, int>> clustersIndex; //[3D_cluster_id] pair: first->N_SCAN, second->Horizon_SCAN in this layer.
    std::vector<std::vector<std::pair<int, int>>>
        clusters_3D_; //[3D_cluster_id][2D_cluster index] pair: first->layer, second->2D_cluster_id in this layer
    std::vector<std::vector<Point_Cluster>> pointClustersFiltered; // clusters after filtering.

    // clustering
    pcl::search::KdTree<Point>::Ptr tree;
    pcl::search::KdTree<Point>::Ptr treeEC;
    std::vector<pcl::PointIndices> clusterIndices;

    std::vector<std::vector<uint8_t>> colors;

    cv::Mat labelMat;
    cv::Mat rangeMat;
    cv::Mat groundMat;
    cv::Mat facadeMat;
    cv::Mat poleMat;
    cv::Mat poleCandidatesMat;

    // Save pole params
    std::vector<Cylinder_Fin_Param> polesParam;
    // Facades params
    std::vector<Plane_Param> facadesParam;
    // ground params
    std::vector<Plane_Param> groundsParam;



    int length; //length of whole grid
    int width; // width of whole grid
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


    void allocateMemory();
    void resetParameters();
    void initColors();

    void projectPointCloud();
    void groundSegmentation();
    void ground_segmentation(const std::vector<Cloud::Ptr>& z_splits,
                             std::vector<std::vector<int>>& non_ground_indices);


    void splitCloud();
    void facadeRoughExtraction();
    void facadeExtraction();
    void facadeSegmentation();
    void facadeSegmentationNew();
    void groundSegmentationRG();
    void groundExtractionPCA();
    void polesRoughExtraction();


    void polesExtractionWithFullCloud();
    void polesExtractionWithFullCloudWithPCA();
    void polesClusteringWithFullCloud();


    void estimatePolesModel();
    bool findCircleCeres(Cloud::Ptr& cloud,
                         int numIterations,
                         float maxDiameter,
                         float thresholdDistance,
                         float& radius,
                         Eigen::Vector2f& centerOut);
    void estimateFacadesModel();
    bool findFacadesCeres(Cloud::Ptr& cloud,
                          Eigen::Vector3f& initN,
                          double& d,
                          int& maxIteration,
                           Plane_Param& pParam);
    bool findFacadesRANSAC(Cloud::Ptr& cloud,
                          Eigen::Vector3f& normal,
                           float& d);
    void estimateGroundModel();
    void polesGrowing();
    void cloudClustering();

    static bool customRegionGrowing(const PointINormal& pointA, const PointINormal& pointB, float squaredDistance);
    void facadeSegmentationCEC(); // Conditional euclidean clustering
    bool find_circle_RANSAC(Cloud::Ptr& cloud,
                            const int& num_samples,
                            const float& support_dist,
                            const float& min_circle_support_percentage,
                            const float& neg_support_dist,
                            const float& early_stop_circle_quality,
                            const float& min_circle_quality,
                            const float& small_radius,
                            const float& circle_max_radius,
                            float& radius,
                            Eigen::Vector2f& center_out);

    bool find_circle_RANSAC_fixed_radius(Cloud::Ptr& cloud,
                                         const int& num_samples,
                                         const float& support_dist,
                                         const float& min_circle_support_percentage,
                                         const float& neg_support_dist,
                                         const float& min_circle_quality,
                                         const float& small_radius,
                                         const float& radius,
                                         Eigen::Vector2f& center_out);

    // Circle Residual
    // Residual = ||sample-center|| - r .
    struct CircleResidual {
        CircleResidual(const Point& sample) : sample_(sample.x, sample.y) {
        }

        template <typename T>
        bool operator()(const T* const center_x, const T* const center_y, const T* const r, T* residual) const {
            Eigen::Matrix<T, 2, 1> center(center_x[0], center_y[0]);
            residual[0] = (sample_.cast<T>() - center).norm() - r[0];
            if (residual[0] < (T)0.0) {
                residual[0] *= (T)-1.0;
            }
            return true;
        }

    private:
        // Observations for a sample.
        const Eigen::Vector2d sample_;
    };

    // Facade Residual
    // Residual = (ax1 + by1 + cz1 + d) ^ 2 / (a^2 + b^2 + c^2)
    struct FacadeResidual {
        FacadeResidual(const Point& sample) : sample_(sample.x, sample.y, sample.z) {
        }

        template <typename T>
        bool operator()(const T* const n1,
                        const T* const n2,
                        const T* const n3,
                        const T* const d,
                        T* residual) const {
//            Eigen::Matrix<T, 3, 1> normal(n1[0], n2[0], n3[0]);
            T temp = n1[0] * sample_[0] + n2[0] * sample_[1] + n3[0] * sample_[2] + d[0];
            residual[0] = temp * temp / (n1[0] * n1[0] + n2[0] * n2[0] + n3[0] * n3[0]);
//            residual[0] = temp * temp;
            if (residual[0] < (T)0.0) {
                residual[0] *= (T)-1.0;
            }
            return true;
        }

    private:
        // Observations for a sample.
        const Eigen::Vector3d sample_;
    };




public:
    PrimitiveExtractor() {
        range.resize(6);
        std::fill(range.begin(), range.end(), 0);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        initColors();
        allocateMemory();
        resetParameters();
    }

    void setInputCloud(Cloud::ConstPtr cloud_in);
    void setPara();
    void setRange(const float& x_min,
                   const float& x_max,
                   const float& y_min,
                   const float& y_max,
                   const float& z_min,
                   const float& z_max);

    bool run();

    // get ground cloud
    void getGroundCloud(Cloud::Ptr groundCloud_);

    // get non facade rough cloud
    void getNonFacadeRoughCloud(Cloud::Ptr nonFacadeCloud_);

    // get pole after growing
    void getBiggerPoleCloud(Cloud::Ptr biggerPole);


    // get split clouds and images according to N_SCANs
    void getSplitClouds(std::vector<Cloud::Ptr>& clouds);
    void getSplitCloudsImages(std::vector<cv::Mat>& layerProjections);


    //// get candidate poles after fine pole extraction and clustering and segmentation
    void get3DPoleCandidateCloudsColored(std::vector<CloudColor::Ptr>& pole_candidates);
    void get3DFacadeCandidateCloudsColored(std::vector<CloudColor::Ptr>& facade_candidates);
    void get3DGroundCandidateCloudsColored(std::vector<CloudColor::Ptr>& ground_candidates);
    //// Interface for Mapping get parameter cloud of facades and poles
    void get3DPoleCandidateClouds(Cloud::Ptr& pole_candidates);
    void get3DFacadeCandidateClouds(Cloud::Ptr& facade_candidates);
    //// get poles and poles param after model fitting
    void getPolesFinParam(std::vector<Cylinder_Fin_Param>& poles_param);
    void getFacadesFinParam(std::vector<Plane_Param>& facades_param);
    void getGroundFinParam(std::vector<Plane_Param>& ground_param);


        /// Mapping interface
    void runForMapping();
    void getFusionFacade(Cloud::Ptr& fusionFacadeCloud,std::vector<Plane_Param>& mapFacadeParam);
    void getFusionPole(Cloud::Ptr& fusionPoleCloud,std::vector<Cylinder_Fin_Param>& mapPoleParam);
    void getFacadeRoughCloud(Cloud::Ptr facadeCloud_);

    void getPoleRoughCloud(Cloud::Ptr poleCloud_);

    void get_3D_clusters(std::vector<CloudColor::Ptr>& clouds);
    void testNonGround(std::vector<std::vector<int>>& non_ground_indices);


    void testGetGround();
    void testNewPoleExtractor();
    void createTheoryCircle(Eigen::Vector3f& normal, Eigen::Vector3f& center, float R,
                            Cloud::Ptr& cloud, int color);
    void createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, CloudColor::Ptr& cloud, int length);
    void createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, Cloud::Ptr& cloud, int length, int color);
    void createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, Cloud::Ptr& cloud);
    void createTheoryCylinder(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, const float R, Cloud::Ptr& cloud);
    void createTheoryCylinder(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, const float R, Cloud::Ptr& cloud,
                              int numCyl, int length, int color);


};








} // end of primitivesExtraction


#endif // SRC_PRIMITIVESEXTRACTION_H
