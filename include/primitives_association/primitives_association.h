//
// Created by zhao on 04.08.22.
//

#ifndef SRC_PRIMITIVES_ASSOCIATION_H
#define SRC_PRIMITIVES_ASSOCIATION_H
#include "primitives_extraction/primitivesExtraction.h"
#include "feature_association/lidarFactor.h"
// Input two features poles and facades after extraction
// Output the transformation those two frames

namespace primitivesFa{

using Cylinder = primitivesExtraction::Cylinder_Fin_Param;
using Plane = primitivesExtraction::Plane_Param;
using Point = primitivesExtraction::Point;
using Time = primitivesExtraction::Time;
using Cloud = primitivesExtraction::Cloud;
using CloudColor = primitivesExtraction::CloudColor;
struct PrimitiveAssociation
{
    std::vector<std::pair<int,int>> plane_asso;  // first: detection id ; second: matching id of map primitive
    std::vector<std::pair<int,int>> cylinder_asso; // first: detection id ; second: matching id of map primitive
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
//        residual[2] = T(1.0) * ( lp.z() - T(lastP.z()) );

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_)
    {
        return (new ceres::AutoDiffCostFunction<
            PoleFactor, 2, 4, 3>(
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
        residual[1] = T(2.0) * direction_diff;

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
            GroundResidual, 2, 4, 3>(new GroundResidual(last_fa_dl, last_fa_dr, last_fa_normal,
                                                      cur_fa_dl, cur_fa_dr, cur_fa_tr) ) );
    }
    Eigen::Vector3d last_fa_dl_; // plane in last frame dl dr tr points
    Eigen::Vector3d last_fa_dr_;
    Eigen::Vector3d last_fa_normal_;
    Eigen::Vector3d cur_fa_dl_; // plane in current frame dl dr tr points
    Eigen::Vector3d cur_fa_dr_;
    Eigen::Vector3d cur_fa_tr_;
};

class PrimitivesAssociation : primitivesExtraction::PrimitiveExtractor {

private:
    // input pole param and facade param from last frame
    vector<Cylinder> prePoles;
    vector<Plane> preFacades;
    vector<Plane> preGrounds;
    // input pole param and facade param from current frame
    vector<Cylinder> curPoles;
    vector<Plane> curFacades;
    vector<Plane> curGrounds;

    // cloud for storing the points of poles and facades
    pcl::PointCloud<Point>::Ptr polesGroundPointsCloud;
    pcl::PointCloud<Point>::Ptr facadesCenterPointsCloud;
    pcl::PointCloud<Point>::Ptr groundsCenterPointsCloud;

    // kdTree features
    pcl::KdTreeFLANN<Point>::Ptr kdTreePolesLast;
    pcl::KdTreeFLANN<Point>::Ptr kdTreeFacadesLast;
    pcl::KdTreeFLANN<Point>::Ptr kdTreeGroundsLast;
    std::vector<int> pointSearchInd_pole;
    std::vector<int> pointSearchInd_facade;
    std::vector<int> pointSearchInd_ground;
    std::vector<float> pointSearchSqDis_pole;
    std::vector<float> pointSearchSqDis_facade;
    std::vector<float> pointSearchSqDis_ground;

    Point point_pole;
    Eigen::Vector3f point_facade;
    Eigen::Vector3f point_ground;
    float pole_position_threshold = 3; // search in 10m range
    float facade_position_threshold = 10;
    float ground_position_threshold = 5;

    // nearest feature search distance
//    float nearestFeatureSearchSqDist;

    // Transformation from current frame to world frame
    Eigen::Quaterniond q_w_curr = Eigen::Quaterniond (1, 0, 0, 0);
    Eigen::Vector3d t_w_curr = Eigen::Vector3d (0, 0, 0);

    // q_curr_last(x, y, z, w), t_curr_last
    double para_q[4] = {0, 0, 0, 1};
    double para_t[3] = {0, 0, 0};

    // 2D localization
    // (x, y, 0)
    double paraT[2] = {0,0};
    // (0, 0, z, w)
    double paraQ[2] = {0, 1};
//    Eigen::Map<Eigen::Quaterniond> qLastCurr = Eigen::Map<Eigen::Quaterniond> (0, 0, paraQ[0], paraQ[1]);
//    Eigen::Map<Eigen::Vector3d> tLastCurr = Eigen::Map<Eigen::Vector3d> (paraT[0], paraT[1], 0);


    Eigen::Map<Eigen::Quaterniond> q_last_curr = Eigen::Map<Eigen::Quaterniond> (para_q);
    Eigen::Map<Eigen::Vector3d> t_last_curr = Eigen::Map<Eigen::Vector3d> (para_t);


    void initializationValue();
    // undistort lidar point
    void TransformToStart(Point const *const pi, Point *const po);
    void TransformToStart(const Eigen::Vector3f * const pi, Eigen::Vector3f* const po);
    void transformPlaneToLast(Plane& detected_plane, Plane& mapped_plane);
    // transform all lidar points to the start of the next frame


    void addTree();

    void associationCalculation();

    float calPolesMatchingCost(const Cylinder lastPole, const Cylinder curPole);

    float calFacadesMatchingCost(const Plane lastFacade, const Plane curFacade);
    float calGroundsMatchingCost(const Plane lastGround, const Plane curGround);
    float dist3D_Segment_to_Segment( Eigen::Vector3f seg_1_p0, Eigen::Vector3f seg_1_p1,
                                     Eigen::Vector3f seg_2_p0, Eigen::Vector3f seg_2_p1 );

    std::vector<std::pair<int, int>> facades_association; // first: current facade id, second: last facade id
    std::vector<std::pair<int, int>> poles_association; // first: current pole id, second: last pole id
    std::vector<std::pair<int, int>> grounds_association; // first: current ground id, second: last ground id



public:
    PrimitivesAssociation(const vector<Cylinder>& lastPoles, const vector<Plane>& lastFacades, const vector<Plane>& lastGrounds,
                          const vector<Cylinder>& currentPoles, const vector<Plane>& currentFacades, const vector<Plane>& currentGrounds,
                          Eigen::Quaterniond qCurr = Eigen::Quaterniond (1, 0, 0, 0),
                          Eigen::Vector3d tCurr = Eigen::Vector3d (0, 0, 0),
                          Eigen::Quaterniond q_last_curr_ = Eigen::Quaterniond (1, 0, 0, 0),
                          Eigen::Vector3d t_last_curr_ = Eigen::Vector3d (0, 0, 0));

    PrimitivesAssociation(const pcl::PointCloud<Point>::Ptr oriCloudLast,
                          const pcl::PointCloud<Point>::Ptr oriCloudCur,
                          Eigen::Quaterniond qCurr = Eigen::Quaterniond (1, 0, 0, 0),
                          Eigen::Vector3d tCurr = Eigen::Vector3d (0, 0, 0),
                          Eigen::Quaterniond q_last_curr_ = Eigen::Quaterniond (1, 0, 0, 0),
                          Eigen::Vector3d t_last_curr_ = Eigen::Vector3d (0, 0, 0));

    void runAS();
    // Current frame pose

    void getCurrentDetection(std::vector<Cylinder>& detectedPoles, std::vector<Plane>& detectedPlanes, std::vector<Plane>& detectedGround);
    Eigen::Quaterniond publishQuaterniond();

    Eigen::Vector3d publishTranslation();


    Eigen::Quaterniond quaterniodGuess();
    Eigen::Vector3d translationGuess();

    // 2D Localization
    Eigen::Vector2d publishTranslation2D();
    float publishYaw();
    void getAssociation(Cloud::Ptr& cloud);


};


} // end of primitivesFa



#endif // SRC_PRIMITIVES_ASSOCIATION_H
