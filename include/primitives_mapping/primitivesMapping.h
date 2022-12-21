//
// Created by zhao on 11.08.22.
//

#ifndef SRC_PRIMITIVESMAPPING_H
#define SRC_PRIMITIVESMAPPING_H

#include <pcl/surface/concave_hull.h>
//#include <rviz_visual_tools/rviz_visual_tools.h>
#include "primitives_extraction/primitivesExtraction.h"
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"
#include "localization_lidar/load_datastorage.hpp"
#include "localization_lidar/parameter_server.hpp"

#include "feature_extraction/load_pose.h"

namespace primitivesMapping {

using Point = pcl::PointXYZI;
using PointXYZ = pcl::PointXYZ;
using PointINormal = pcl::PointXYZINormal;
using Cloud = pcl::PointCloud<Point>;
using CloudXYZ = pcl::PointCloud<PointXYZ>;
using Cylinder_Fin_Param = primitivesExtraction::Cylinder_Fin_Param;
using Plane_Param = primitivesExtraction::Plane_Param;

struct Primitives
{
    std::vector<Plane_Param> planes;
    std::vector<Plane_Param> grounds;
    std::vector<Cylinder_Fin_Param> cylinders;
};



using PointColor = pcl::PointXYZRGB;
using CloudColor = pcl::PointCloud<primitivesExtraction::PointColor>;

class PrimitivesMapping : primitivesExtraction::PrimitiveExtractor{

private:
    Cloud::Ptr fusionFacades; // Primitives facades cloud map
    Cloud::Ptr mapFacadesPara;
    Cloud::Ptr mapPolesPara;
    Cloud::Ptr fusionPoles;// Primitives poles cloud map

    Cloud::Ptr curFacades; // frame facades
    Cloud::Ptr curPoles;   // frame Poles
    Cloud::Ptr curFacadesTF; // frame facades after transformer
    Cloud::Ptr curPolesTF; // frame poles after transformer
    Cloud::Ptr currentCloud; // original frame cloud


    std::vector<Cylinder_Fin_Param> polesParam;
    // Facades params
    std::vector<Plane_Param> facadesParam;
    // Grounds params
    std::vector<Plane_Param> groundsParam;

    int lengthMap;

    /// map manager
    Primitives map_;
    Primitives new_map_;
    std::vector<Cloud::Ptr> fusionFacadesClusters;
    std::vector<Cloud::Ptr> fusionGroundsClusters;
    bool findNewMergedFacadesRANSAC(Cloud::Ptr& cloud, Eigen::Vector3f& normal, float& d);


    int cylinder_min_num_detections_;
    float cylinder_cluster_pos_thresh_;
    float cylinder_max_radius_tolerance_from_median_percentage_;
    float cylinder_max_ground_tolerance_from_median_;
    float cylinder_max_xy_diff_;
    float cylinder_max_angle_diff_from_average_;
    float plane_cluster_pos_thresh_;
    float plane_cluster_angle_thresh_;
    int min_num_planes_in_cluster_;
    float plane_sample_step_size_;
    float new_plane_inlier_max_dist_;





    void initialValue();
    void resetPara();

    void transformFacades(Plane_Param& facade, Eigen::Transform<double, 3, Eigen::Affine>& TF);
    void transformPoles(Cylinder_Fin_Param& pole, Eigen::Transform<double, 3, Eigen::Affine>& TF);

    bool point_in_poly(int nvert, float *vertx, float *verty, float testx, float testy);
    void fit_plane_pcl(const Eigen::Matrix3Xd& points, Eigen::Vector3d& n, double& d);

    float dist_plane_plane(const Eigen::Vector3f& first_fa_dl, const Eigen::Vector3f& first_fa_normal,
                           const Eigen::Vector3f& second_fa_dl, const Eigen::Vector3f& second_fa_tr);

    float dist2D_Segment_to_Segment( const Eigen::Vector2f& seg_1_p0, const Eigen::Vector2f& seg_1_p1,
                                     const Eigen::Vector2f& seg_2_p0, const Eigen::Vector2f& seg_2_p1, bool& edgePointsUsed );


//    rviz_visual_tools::RvizVisualToolsPtr visual_tools_;



public:
    PrimitivesMapping(const int lengthMap_);
    void run();
    void readFacadeMap(std::vector<Plane_Param>& facades_param, std::string& facadesMapPath);
    void readGroundMap(std::vector<Plane_Param>& grounds_param, std::string& groundsMapPath);
    void readPolesMap(std::vector<Cylinder_Fin_Param>& poles_param, std::string& polesMapPath);
    void build_map();
    void save_map(const std::string& save_path_pole, const std::string& save_path_facade, const std::string& save_path_ground);
    void mergeFacade(int iterationNumber, std::vector<Plane_Param>& Facades);
    void getPrimitiveFacadeCloud(Cloud::Ptr& cloud, std::vector<Plane_Param>& Facades, int color);
    void getPrimitiveFacadeCloud(CloudColor::Ptr& cloudColor, std::vector<Plane_Param>& Facades);
    void getPrimitivePoleCloud(Cloud::Ptr& cloud, std::vector<Cylinder_Fin_Param>& Poles);






};





} // end of namespace
#endif // SRC_PRIMITIVESMAPPING_H
