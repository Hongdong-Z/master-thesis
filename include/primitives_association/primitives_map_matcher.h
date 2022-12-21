//
// Created by zhao on 14.08.22.
//

#ifndef SRC_PRIMITIVES_MAP_MATCHER_H
#define SRC_PRIMITIVES_MAP_MATCHER_H
#include "primitives_extraction/primitivesExtraction.h"
#include "primitives_mapping/primitivesMapping.h"

namespace primitivesMapMatcher {
using Plane = primitivesExtraction::Plane_Param;
using Cylinder = primitivesExtraction::Cylinder_Fin_Param;
using Point = primitivesExtraction::Point;
using Cloud = primitivesExtraction::Cloud;

struct Primitives
{
    std::vector<Plane> planes;
    std::vector<Plane> grounds;
    std::vector<Cylinder> cylinders;
};

struct PrimitivesIndices
{
    std::vector<int> plane_ids;
    std::vector<int> cylinder_ids;
};

struct Pose3D
{
    Eigen::Vector3f pos;
    Eigen::Quaternion<float> q;
};


struct PrimitiveAssociation
{
    std::vector<std::pair<int,int>> plane_asso;  // first: detection id ; second: matching id of map primitive
    std::vector<std::pair<int,int>> cylinder_asso; // first: detection id ; second: matching id of map primitive
    std::vector<std::pair<int,int>> ground_asso;  // first: detection id ; second: matching id of map primitive
};

struct MatchingCosts
{
    std::vector<float> plane_costs;
    std::vector<float> cylinder_costs;
    std::vector<float> ground_costs;
};


struct PoleFactor
{
    PoleFactor(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_)
        : curr_point(curr_point_), last_point(last_point_) {}
//    , cur_dir(cur_dir_), last_dir(last_dir_)
    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
//        Eigen::Matrix<T, 3, 1> cDir{T(cur_dir.x()), T(cur_dir.y()), T(cur_dir.z())};

        Eigen::Matrix<T, 3, 1> lastP{T(last_point.x()), T(last_point.y()), T(last_point.z())}; // bottom point of pole in last frame

        Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

        Eigen::Matrix<T, 3, 1> lp;

        lp = q_last_curr * cp + t_last_curr; // bottom point of current pole transformed to last frame

        residual[0] = T(1.0) * ( lp.x() - T(lastP.x()) ); // bottom point distance
        residual[1] = T(1.0) * ( lp.y() - T(lastP.y()) );
        residual[2] = T(1.0) * ( lp.z() - T(lastP.z()) );

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_)
    {
        return (new ceres::AutoDiffCostFunction<
            PoleFactor, 3, 4, 3>(
            new PoleFactor(curr_point_, last_point_)));
    }

    Eigen::Vector3d curr_point, last_point;
};
struct PlaneResidual { // in LiDAR odo line is only be used to parallel
    PlaneResidual(const Eigen::Vector3d& last_fa_dl, // plane in last frame dl dr tr points
                  const Eigen::Vector3d& last_fa_dr,
                  const Eigen::Vector3d& last_fa_normal,
                  const Eigen::Vector3d& cur_fa_dl, // plane in current frame dl dr tr points
                  const Eigen::Vector3d& cur_fa_dr,
                  const Eigen::Vector3d& cur_fa_tr)
        : last_fa_dl_(last_fa_dl), last_fa_dr_(last_fa_dr), last_fa_normal_(last_fa_normal),
          cur_fa_dl_(cur_fa_dl), cur_fa_dr_(cur_fa_dr), cur_fa_tr_(cur_fa_tr) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        // transformed to last frame
        Eigen::Matrix<T, 3, 1> cur_fa_dl_tran = q_w_curr * cur_fa_dl_.cast<T>() + t_w_curr;
        Eigen::Matrix<T, 3, 1> cur_fa_dr_tran = q_w_curr * cur_fa_dr_.cast<T>() + t_w_curr;
        Eigen::Matrix<T, 3, 1> cur_fa_tr_tran = q_w_curr * cur_fa_tr_.cast<T>() + t_w_curr;
        // The center of current facade transformed
        Eigen::Matrix<T, 3, 1> cur_center_tran = cur_fa_dl_tran + T(0.5) * (cur_fa_tr_tran - cur_fa_dl_tran);
        Eigen::Matrix<T, 3, 1> last_fa_dl_cur_center_tran = cur_center_tran - last_fa_dl_.cast<T>();

        // The center to last facade should be minimize.
        residual[0] = last_fa_normal_.cast<T>().dot(last_fa_dl_cur_center_tran);

        // The current facade normal transformed.
        Eigen::Matrix<T, 3, 1> dl_dr = cur_fa_dr_tran - cur_fa_dl_tran;
        Eigen::Matrix<T, 3, 1> dl_tr = cur_fa_tr_tran - cur_fa_dl_tran;
        Eigen::Matrix<T, 3, 1> cur_fa_normal_tran = dl_dr.cross(dl_tr);

        // Normalize normals
        Eigen::Matrix<T, 3, 1> cur_fa_normal_tran_norm = cur_fa_normal_tran.normalized();
        Eigen::Matrix<T, 3, 1> last_fa_normal_norm = last_fa_normal_.cast<T>().normalized();
        T direction_diff = (last_fa_normal_norm.cross(cur_fa_normal_tran_norm)).norm();

        // The direction residual is less then 1 so should be weighted.
        residual[1] = T(2.0) * direction_diff;

        // constrain of plane bottom line
        Eigen::Matrix<T, 3, 1> cur_dl_dr_norm = dl_dr.normalized();
        Eigen::Matrix<T, 3, 1> last_dl_dr_norm = (last_fa_dr_.cast<T>() - last_fa_dl_.cast<T>()).normalized();

//        residual[2] = T(2.0) * ( cur_dl_dr_norm.cross(last_dl_dr_norm) ).norm();


        return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector3d& last_fa_dl, // plane in last frame dl dr tr points
                                       const Eigen::Vector3d& last_fa_dr,
                                       const Eigen::Vector3d& last_fa_normal,
                                       const Eigen::Vector3d& cur_fa_dl, // plane in current frame dl dr tr points
                                       const Eigen::Vector3d& cur_fa_dr,
                                       const Eigen::Vector3d& cur_fa_tr)
    {
        return ( new ceres::AutoDiffCostFunction<
            PlaneResidual, 2, 4, 3>(new PlaneResidual(last_fa_dl, last_fa_dr, last_fa_normal,
                                                      cur_fa_dl, cur_fa_dr, cur_fa_tr) ) );
    }
    Eigen::Vector3d last_fa_dl_; // plane in last frame dl dr tr points
    Eigen::Vector3d last_fa_dr_;
    Eigen::Vector3d last_fa_normal_;
    Eigen::Vector3d cur_fa_dl_; // plane in current frame dl dr tr points
    Eigen::Vector3d cur_fa_dr_;
    Eigen::Vector3d cur_fa_tr_;
};
struct GroundResidual { // in LiDAR odo line is only be used to parallel
    GroundResidual(const Eigen::Vector3d& last_fa_dl, // plane in last frame dl dr tr points
                   const Eigen::Vector3d& last_fa_dr,
                   const Eigen::Vector3d& last_fa_normal,
                   const Eigen::Vector3d& cur_fa_dl, // plane in current frame dl dr tr points
                   const Eigen::Vector3d& cur_fa_dr,
                   const Eigen::Vector3d& cur_fa_tr)
        : last_fa_dl_(last_fa_dl), last_fa_dr_(last_fa_dr), last_fa_normal_(last_fa_normal),
          cur_fa_dl_(cur_fa_dl), cur_fa_dr_(cur_fa_dr), cur_fa_tr_(cur_fa_tr) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        // transformed to last frame
        Eigen::Matrix<T, 3, 1> cur_fa_dl_tran = q_w_curr * cur_fa_dl_.cast<T>() + t_w_curr;
        Eigen::Matrix<T, 3, 1> cur_fa_dr_tran = q_w_curr * cur_fa_dr_.cast<T>() + t_w_curr;
        Eigen::Matrix<T, 3, 1> cur_fa_tr_tran = q_w_curr * cur_fa_tr_.cast<T>() + t_w_curr;
        // The center of current facade transformed
        Eigen::Matrix<T, 3, 1> cur_center_tran = cur_fa_dl_tran + T(0.5) * (cur_fa_tr_tran - cur_fa_dl_tran);
        Eigen::Matrix<T, 3, 1> last_fa_dl_cur_center_tran = cur_center_tran - last_fa_dl_.cast<T>();

        // The center to last facade should be minimize.
        residual[0] = last_fa_normal_.cast<T>().dot(last_fa_dl_cur_center_tran);

        // The current facade normal transformed.
        Eigen::Matrix<T, 3, 1> dl_dr = cur_fa_dr_tran - cur_fa_dl_tran;
        Eigen::Matrix<T, 3, 1> dl_tr = cur_fa_tr_tran - cur_fa_dl_tran;
        Eigen::Matrix<T, 3, 1> cur_fa_normal_tran = dl_dr.cross(dl_tr);

        // Normalize normals
        Eigen::Matrix<T, 3, 1> cur_fa_normal_tran_norm = cur_fa_normal_tran.normalized();
        Eigen::Matrix<T, 3, 1> last_fa_normal_norm = last_fa_normal_.cast<T>().normalized();
        T direction_diff = (last_fa_normal_norm.cross(cur_fa_normal_tran_norm)).norm();

        // The direction residual is less then 1 so should be weighted.
//        residual[1] = T(2.0) * direction_diff;

        // constrain of plane bottom line
        Eigen::Matrix<T, 3, 1> cur_dl_dr_norm = dl_dr.normalized();
        Eigen::Matrix<T, 3, 1> last_dl_dr_norm = (last_fa_dr_.cast<T>() - last_fa_dl_.cast<T>()).normalized();


        return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector3d& last_fa_dl, // plane in last frame dl dr tr points
                                       const Eigen::Vector3d& last_fa_dr,
                                       const Eigen::Vector3d& last_fa_normal,
                                       const Eigen::Vector3d& cur_fa_dl, // plane in current frame dl dr tr points
                                       const Eigen::Vector3d& cur_fa_dr,
                                       const Eigen::Vector3d& cur_fa_tr)
    {
        return ( new ceres::AutoDiffCostFunction<
            GroundResidual, 1, 4, 3>(new GroundResidual(last_fa_dl, last_fa_dr, last_fa_normal,
                                                        cur_fa_dl, cur_fa_dr, cur_fa_tr) ) );
    }
    Eigen::Vector3d last_fa_dl_; // plane in last frame dl dr tr points
    Eigen::Vector3d last_fa_dr_;
    Eigen::Vector3d last_fa_normal_;
    Eigen::Vector3d cur_fa_dl_; // plane in current frame dl dr tr points
    Eigen::Vector3d cur_fa_dr_;
    Eigen::Vector3d cur_fa_tr_;
};

//struct PriorPoseResidual {
//    PriorPoseResidual(const Eigen::Vector3d& prior_pos,     // Position of cylinder in map in the map frame.
//                      const Eigen::Quaternion<double>& prior_q,
//                      const double& weighting_factor_pos, const double& weighting_factor_q )
//        : prior_pos_(prior_pos),prior_q_(prior_q),
//          weighting_factor_pos_(weighting_factor_pos),
//          weighting_factor_q_(weighting_factor_q) {}
//
//    template <typename T>
//    bool operator()(const T *q, const T *t, T *residual) const {
//
//        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
//        Eigen::Matrix<T,3,1> t_w_cur{t[0], t[1], t[2]};
//        Eigen::Matrix<T,3,1> prior_pos(prior_pos_.cast<T>());
//
//        T pos_diff = (prior_pos - t_w_cur).norm();
//
//        T q_diff = q_w_curr * prior_q_.inverse();
//
//        // weighing_factor = cost_weighting_cylinder_/max_cylinder_association_pos_diff_
//        residual[0] = T(weighting_factor_pos_)*pos_diff + T(weighting_factor_q_)*q_diff;
//
//        return true;
//    }
//    static ceres::CostFunction *Create(const Eigen::Vector3d& prior_pos,     // Position of cylinder in map in the map frame.
//                                       const Eigen::Quaternion<double>& prior_q,
//                                       const double& weighting_factor_pos, const double& weighting_factor_q )
//    {
//        return (new ceres::AutoDiffCostFunction<
//            PriorPoseResidual, 3, 4, 3>(new PriorPoseResidual(prior_pos, prior_q, weighting_factor_pos, weighting_factor_q)));
//    }
//    const Eigen::Vector3d prior_pos_;
//    const Eigen::Quaternion<double> prior_q_;
//    const double weighting_factor_pos_;
//    const double weighting_factor_q_;
//
//};

class PrimitiveMatcher {
private:

    std::string mapPolesPath;
    std::string mapFacadesPath;
    std::string mapGroundPath;
    float search_radius_;
    float search_radius_squared_;
    double cylinder_weighting_factor_;
    double plane_distance_factor_;
    double plane_direction_factor_;
    double plane_overlap_factor_;
    float max_cylinder_association_pos_diff_;
    float max_cylinder_association_radius_diff_;
    float max_cylinder_association_angle_diff_;
    float max_plane_association_pos_diff_;
    float max_plane_association_angle_diff_;
    int ceres_solver_max_num_iterations_;

    Primitives map_; // whole map of primitives
    Primitives detections_; // detection each frame
    Primitives subMap_;
    PrimitivesIndices map_primitives_in_range_; // whole map index in given range
    PrimitiveAssociation association_;


    bool find_primitives_in_range();

    void getSubMap();
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeFacadeFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreePoleFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGroundFromMap;
    //sub map
    pcl::PointCloud<PointType>::Ptr subMapFacade;
    pcl::PointCloud<PointType>::Ptr subMapPole;
    pcl::PointCloud<PointType>::Ptr subMapGround;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    bool check_distance_plane_2_pos_3D( const int map_plane_id, const Eigen::Vector3f& pos );
    bool check_distance_cylinder_2_pos_3D( const int map_cylinder_id, const Eigen::Vector3f& pos );

    void find_associations();
    bool check_distance_planes( const int map_plane_id, const int detection_plane_id );

    float SqDistancePtSegment( Eigen::Vector3f a, Eigen::Vector3f b, Eigen::Vector3f p );
    float dist3D_Segment_to_Segment( Eigen::Vector3f seg_1_p0, Eigen::Vector3f seg_1_p1,
                                     Eigen::Vector3f seg_2_p0, Eigen::Vector3f seg_2_p1 );
    void calc_costs_associations();

    float calPolesMatchingCost(const Cylinder mapPole, const Cylinder curPole);
    float calFacadesMatchingCost(const Plane mapFacade, const Plane curFacade);
    float planeToPlaneDistanceCost(const Plane& plane1, const Plane& plane2);
    float planeToPlaneMatchingCost(const Plane& plane1, const Plane& plane2);

    void transformPlaneToMap(Plane& detected_plane, Plane& mapped_plane);

    void read_map_file( const std::string& pole_map_path, const std::string& facade_map_path, const std::string& ground_map_path);




    //input: calculated pose from odo
    Eigen::Quaterniond q_wodom_curr = Eigen::Quaterniond (1, 0, 0, 0);
    Eigen::Vector3d t_wodom_curr = Eigen::Vector3d (0, 0, 0);
    double para_q[4] = {0, 0, 0, 1};
    double para_t[3] = {0, 0, 0};
    Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond> (para_q);
    Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d> (para_t);
    // transformation between odom's world and map's world frame
    Eigen::Quaterniond q_wmap_wodom = Eigen::Quaterniond (1, 0, 0, 0);
    Eigen::Vector3d t_wmap_wodom = Eigen::Vector3d (0, 0, 0);
    void transformAssociateToMap();
    void  transformUpdate();
    void detect_primitives_to_map(const Eigen::Vector3f * const pi, Eigen::Vector3f* const po);
    // kdtree to save pole map in range
    void put_pole_in_tree();
    pcl::KdTreeFLANN<Point>::Ptr kdtree_pole_map_in_range;


    std::vector<int> pole_search_index;
    std::vector<float> pole_search_distance;

    // association para
    Eigen::Vector3f pole_center;
    Eigen::Vector3f pole_center_transformed;

    Eigen::Vector3f facade_center;
    Eigen::Vector3f facade_center_transformed;

    Eigen::Vector3f ground_center;
    Eigen::Vector3f ground_center_transformed;

    Eigen::Vector3f line_start;
    Eigen::Vector3f line_end;
    Eigen::Vector3f line_start_transformed;
    Eigen::Vector3f line_end_transformed;

    void getPrimitiveFacadeCloud(Cloud::Ptr& cloud, std::vector<Plane>& Facades, int color);
    void getPrimitivePoleCloud(Cloud::Ptr& cloud, std::vector<Cylinder>& Poles, int color);

    void createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, Cloud::Ptr& cloud, int length, int color);
    void createTheoryCylinder(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, const float R, Cloud::Ptr& cloud,
                              int numCyl, int length, int color);
    void createTheoryCircle(Eigen::Vector3f& normal, Eigen::Vector3f& center, float R,
                            Cloud::Ptr& cloud, int color);

public:
    PrimitiveMatcher();
    void match();


    void set_pose_prior(const Eigen::Vector3d& t, const Eigen::Quaterniond& q,
                        const Eigen::Vector3d& t_guess, const Eigen::Quaterniond& q_guess);
    void set_detections( const std::vector<Plane>& planes,
                         const std::vector<Cylinder>& cylinders,
                         const std::vector<Plane>& grounds);

    /// Interface to get initial guess
    Eigen::Quaterniond getQuaterniondGuess() {
        return q_wmap_wodom;
    }
    Eigen::Vector3d getTranslationGuess() {
        return t_wmap_wodom;
    }

    Eigen::Quaterniond getOptimizedQuaterniond() {
        return q_w_curr;
    }
    Eigen::Vector3d getOptimizedTranslation() {
        return t_w_curr;
    }

    void testAssociation();


    void testCeres(Cloud::Ptr cloud);
    void getAssociationMM(Cloud::Ptr cloud);
    void getMapPrimitives(Primitives& primitivesInMap);
    void getMapPrimitives(Cloud::Ptr& primitivesInMap, int color);







}; // end of class PrimitiveMatcher




























} // end of namespace primitivesMapMatcher
#endif // SRC_PRIMITIVES_MAP_MATCHER_H
