//
// Created by zhao on 12.05.22.
//

#ifndef SRC_LIDARFACTOR_H
#define SRC_LIDARFACTOR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

struct LidarGroundFactor
{
    LidarGroundFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
                     Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
        : curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
          last_point_m(last_point_m_), s(s_)
    {
        ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
        ljm_norm.normalize();
    }

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
        //Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
        //Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
        Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

        //Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
        Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
        q_last_curr = q_identity.slerp(T(s), q_last_curr);
        Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

        Eigen::Matrix<T, 3, 1> lp;
        lp = q_last_curr * cp + t_last_curr;

        residual[0] = (lp - lpj).dot(ljm);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
                                       const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
                                       const double s_)
    {
        return (new ceres::AutoDiffCostFunction<
            LidarGroundFactor, 1, 4, 3>(
            new LidarGroundFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
    }

    Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
    Eigen::Vector3d ljm_norm;
    double s;
};
struct LidarCurbFactor
{
    LidarCurbFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
                    Eigen::Vector3d last_point_b_, double s_)
        : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
        Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        //Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
        Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
        q_last_curr = q_identity.slerp(T(s), q_last_curr);
        Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

        Eigen::Matrix<T, 3, 1> lp;
        lp = q_last_curr * cp + t_last_curr;

        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        residual[0] = nu.x() / de.norm();
        residual[1] = nu.y() / de.norm();
        residual[2] = nu.z() / de.norm();

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
                                       const Eigen::Vector3d last_point_b_, const double s_)
    {
        return (new ceres::AutoDiffCostFunction<
            LidarCurbFactor, 3, 4, 3>(
            new LidarCurbFactor(curr_point_, last_point_a_, last_point_b_, s_)));
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
};



struct LidarEdgeFactor
{
    LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
                    Eigen::Vector3d last_point_b_, double s_)
        : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
        Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        //Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
        Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
        q_last_curr = q_identity.slerp(T(s), q_last_curr);
        Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

        Eigen::Matrix<T, 3, 1> lp;
        lp = q_last_curr * cp + t_last_curr;

        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        residual[0] = nu.x() / de.norm();
        residual[1] = nu.y() / de.norm();
        residual[2] = nu.z() / de.norm();

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
                                       const Eigen::Vector3d last_point_b_, const double s_)
    {
        return (new ceres::AutoDiffCostFunction<
            LidarEdgeFactor, 3, 4, 3>(
            new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
};

struct LidarPlaneFactor
{
    LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
                     Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
        : curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
          last_point_m(last_point_m_), s(s_)
    {
        ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
        ljm_norm.normalize();
    }

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
        //Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
        //Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
        Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

        //Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
        Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
        q_last_curr = q_identity.slerp(T(s), q_last_curr);
        Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

        Eigen::Matrix<T, 3, 1> lp;
        lp = q_last_curr * cp + t_last_curr;

        residual[0] = (lp - lpj).dot(ljm);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
                                       const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
                                       const double s_)
    {
        return (new ceres::AutoDiffCostFunction<
            LidarPlaneFactor, 1, 4, 3>(
            new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
    }

    Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
    Eigen::Vector3d ljm_norm;
    double s;
};

struct LidarPlaneNormFactor
{

    LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
                         double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                         negative_OA_dot_norm(negative_OA_dot_norm_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> point_w;
        point_w = q_w_curr * cp + t_w_curr;

        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
                                       const double negative_OA_dot_norm_)
    {
        return (new ceres::AutoDiffCostFunction<
            LidarPlaneNormFactor, 1, 4, 3>(
            new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
};

struct LidarSurfaceNormFactor
{

    LidarSurfaceNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
                         double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                         negative_OA_dot_norm(negative_OA_dot_norm_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> point_w;
        point_w = q_w_curr * cp + t_w_curr;

        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
                                       const double negative_OA_dot_norm_)
    {
        return (new ceres::AutoDiffCostFunction<
            LidarPlaneNormFactor, 1, 4, 3>(
            new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
};



struct LidarDistanceFactor
{

    LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_)
        : curr_point(curr_point_), closed_point(closed_point_){}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> point_w;
        point_w = q_w_curr * cp + t_w_curr;


        residual[0] = point_w.x() - T(closed_point.x());
        residual[1] = point_w.y() - T(closed_point.y());
        residual[2] = point_w.z() - T(closed_point.z());
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
    {
        return (new ceres::AutoDiffCostFunction<
            LidarDistanceFactor, 3, 4, 3>(
            new LidarDistanceFactor(curr_point_, closed_point_)));
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d closed_point;
};


class EdgeCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
    EdgeCostFunction(Eigen::Vector3d cp, Eigen::Vector3d lpj, Eigen::Vector3d lpl) : cp_(cp), lpj_(lpj), lpl_(lpl) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Vector3d lp = (Eigen::AngleAxisd(parameters[0][5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(parameters[0][4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(parameters[0][3], Eigen::Vector3d::UnitX())) * cp_ + Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
        double k = std::sqrt(std::pow(lpj_.x() - lpl_.x(), 2) + std::pow(lpj_.y() - lpl_.y(), 2) + std::pow(lpj_.z() - lpl_.z(), 2));
        double a = (lp.y() - lpj_.y()) * (lp.z() - lpl_.z()) - (lp.z() - lpj_.z()) * (lp.y() - lpl_.y());
        double b = (lp.z() - lpj_.z()) * (lp.x() - lpl_.x()) - (lp.x() - lpj_.x()) * (lp.z() - lpl_.z());
        double c = (lp.x() - lpj_.x()) * (lp.y() - lpl_.y()) - (lp.y() - lpj_.y()) * (lp.x() - lpl_.x());
        double m = std::sqrt(a * a + b * b + c * c);

        residuals[0] = m / k;

        double dm_dx = (b * (lpl_.z() - lpj_.z()) + c * (lpj_.y() - lpl_.y())) / m;
        double dm_dy = (a * (lpj_.z() - lpl_.z()) - c * (lpj_.x() - lpl_.x())) / m;
        double dm_dz = (-a * (lpj_.y() - lpl_.y()) + b * (lpj_.x() - lpl_.x())) / m;

        double sr = std::sin(parameters[0][3]);
        double cr = std::cos(parameters[0][3]);
        double sp = std::sin(parameters[0][4]);
        double cp = std::cos(parameters[0][4]);
        double sy = std::sin(parameters[0][5]);
        double cy = std::cos(parameters[0][5]);

        double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
        double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
        double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

        double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
        double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
        double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

        double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
        double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
        double dz_dy = 0.;

        if (jacobians && jacobians[0])
        {
            jacobians[0][0] = dm_dx / k;
            jacobians[0][1] = dm_dy / k;
            jacobians[0][2] = 0.;
            jacobians[0][3] = 0.;
            jacobians[0][4] = 0.;
            jacobians[0][5] = 0.;
            // printf("lp: %.3f, %.3f, %.3f; lpj: %.3f, %.3f, %.3f; lpl: %.3f, %.3f, %.3f", lp.x(), lp.y(), lp.z(), lpj_.x(), lpj_.y(), lpj_.z(), lpl_.x(), lpl_.y(), lpl_.z());
            // printf("residual: %.3f\n", residuals[0]);
            // printf("J: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", jacobians[0][0], jacobians[0][1], jacobians[0][2], jacobians[0][3], jacobians[0][4], jacobians[0][5]);
        }

        return true;
    }

private:
    Eigen::Vector3d cp_;        // under t frame
    Eigen::Vector3d lpj_, lpl_; // under t-1 frame
};

class CurbCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
    CurbCostFunction(Eigen::Vector3d cp, Eigen::Vector3d lpj, Eigen::Vector3d lpl) : cp_(cp), lpj_(lpj), lpl_(lpl) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Vector3d lp = (Eigen::AngleAxisd(parameters[0][5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(parameters[0][4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(parameters[0][3], Eigen::Vector3d::UnitX())) * cp_ + Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
        double k = std::sqrt(std::pow(lpj_.x() - lpl_.x(), 2) + std::pow(lpj_.y() - lpl_.y(), 2) + std::pow(lpj_.z() - lpl_.z(), 2));
        double a = (lp.y() - lpj_.y()) * (lp.z() - lpl_.z()) - (lp.z() - lpj_.z()) * (lp.y() - lpl_.y());
        double b = (lp.z() - lpj_.z()) * (lp.x() - lpl_.x()) - (lp.x() - lpj_.x()) * (lp.z() - lpl_.z());
        double c = (lp.x() - lpj_.x()) * (lp.y() - lpl_.y()) - (lp.y() - lpj_.y()) * (lp.x() - lpl_.x());
        double m = std::sqrt(a * a + b * b + c * c);

        residuals[0] = m / k;

        double dm_dx = (b * (lpl_.z() - lpj_.z()) + c * (lpj_.y() - lpl_.y())) / m;
        double dm_dy = (a * (lpj_.z() - lpl_.z()) - c * (lpj_.x() - lpl_.x())) / m;
        double dm_dz = (-a * (lpj_.y() - lpl_.y()) + b * (lpj_.x() - lpl_.x())) / m;

        double sr = std::sin(parameters[0][3]);
        double cr = std::cos(parameters[0][3]);
        double sp = std::sin(parameters[0][4]);
        double cp = std::cos(parameters[0][4]);
        double sy = std::sin(parameters[0][5]);
        double cy = std::cos(parameters[0][5]);

        double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
        double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
        double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

        double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
        double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
        double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

        double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
        double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
        double dz_dy = 0.;

        if (jacobians && jacobians[0])
        {
            jacobians[0][0] = dm_dx / k;
            jacobians[0][1] = dm_dy / k;
            jacobians[0][2] = dm_dz / k;
            jacobians[0][3] = 0.;
            jacobians[0][4] = 0.;
            jacobians[0][5] = (dm_dx * dx_dy + dm_dy * dy_dy + dm_dz * dz_dy) / k;
            // printf("lp: %.3f, %.3f, %.3f; lpj: %.3f, %.3f, %.3f; lpl: %.3f, %.3f, %.3f", lp.x(), lp.y(), lp.z(), lpj_.x(), lpj_.y(), lpj_.z(), lpl_.x(), lpl_.y(), lpl_.z());
            // printf("residual: %.3f\n", residuals[0]);
            // printf("J: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", jacobians[0][0], jacobians[0][1], jacobians[0][2], jacobians[0][3], jacobians[0][4], jacobians[0][5]);
        }

        return true;
    }

private:
    Eigen::Vector3d cp_;        // under t frame
    Eigen::Vector3d lpj_, lpl_; // under t-1 frame
};

class GroundCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
    GroundCostFunction(Eigen::Vector3d cp, Eigen::Vector3d lpj, Eigen::Vector3d lpl, Eigen::Vector3d lpm) : cp_(cp), lpj_(lpj), lpl_(lpl), lpm_(lpm) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Vector3d lp = (Eigen::AngleAxisd(parameters[0][5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(parameters[0][4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(parameters[0][3], Eigen::Vector3d::UnitX())) * cp_ + Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
        double a = (lpj_.y() - lpl_.y()) * (lpj_.z() - lpm_.z()) - (lpj_.z() - lpl_.z()) * (lpj_.y() - lpm_.y());
        double b = (lpj_.z() - lpl_.z()) * (lpj_.x() - lpm_.x()) - (lpj_.x() - lpl_.x()) * (lpj_.z() - lpm_.z());
        double c = (lpj_.x() - lpl_.x()) * (lpj_.y() - lpm_.y()) - (lpj_.y() - lpl_.y()) * (lpj_.x() - lpm_.x());
        a *= a;
        b *= b;
        c *= c;
        double m = std::sqrt(std::pow((lp.x() - lpj_.x()), 2) * a + std::pow((lp.y() - lpj_.y()), 2) * b + std::pow((lp.z() - lpj_.z()), 2) * c);
        double k = std::sqrt(a + b + c);

        residuals[0] = m / k;

        double tmp = m * k;

        double dm_dx = ((lp.x() - lpj_.x()) * a) / tmp;
        double dm_dy = ((lp.y() - lpj_.y()) * b) / tmp;
        double dm_dz = ((lp.z() - lpj_.z()) * c) / tmp;

        double sr = std::sin(parameters[0][3]);
        double cr = std::cos(parameters[0][3]);
        double sp = std::sin(parameters[0][4]);
        double cp = std::cos(parameters[0][4]);
        double sy = std::sin(parameters[0][5]);
        double cy = std::cos(parameters[0][5]);

        double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
        double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
        double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

        double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
        double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
        double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

        double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
        double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
        double dz_dy = 0.;


        if (jacobians && jacobians[0])
        {
            jacobians[0][0] = 0.;
            jacobians[0][1] = 0.;
            jacobians[0][2] = dm_dz / k;
            jacobians[0][3] = (dm_dx * dx_dr + dm_dy * dy_dr + dm_dz * dz_dr) / k;
            jacobians[0][4] = (dm_dx * dx_dp + dm_dy * dy_dp + dm_dz * dz_dp) / k;
            jacobians[0][5] = 0.;
        }

        return true;
    }

private:
    Eigen::Vector3d cp_;
    Eigen::Vector3d lpj_, lpl_, lpm_;
};

class SurfaceCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
    SurfaceCostFunction(Eigen::Vector3d cp, Eigen::Vector3d lpj, Eigen::Vector3d lpl, Eigen::Vector3d lpm) : cp_(cp), lpj_(lpj), lpl_(lpl), lpm_(lpm) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Vector3d lp = (Eigen::AngleAxisd(parameters[0][5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(parameters[0][4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(parameters[0][3], Eigen::Vector3d::UnitX())) * cp_ + Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
        double a = (lpj_.y() - lpl_.y()) * (lpj_.z() - lpm_.z()) - (lpj_.z() - lpl_.z()) * (lpj_.y() - lpm_.y());
        double b = (lpj_.z() - lpl_.z()) * (lpj_.x() - lpm_.x()) - (lpj_.x() - lpl_.x()) * (lpj_.z() - lpm_.z());
        double c = (lpj_.x() - lpl_.x()) * (lpj_.y() - lpm_.y()) - (lpj_.y() - lpl_.y()) * (lpj_.x() - lpm_.x());
        a *= a;
        b *= b;
        c *= c;
        double m = std::sqrt(std::pow((lp.x() - lpj_.x()), 2) * a + std::pow((lp.y() - lpj_.y()), 2) * b + std::pow((lp.z() - lpj_.z()), 2) * c);
        double k = std::sqrt(a + b + c);

        residuals[0] = m / k;

        double tmp = m * k;

        double dm_dx = ((lp.x() - lpj_.x()) * a) / tmp;
        double dm_dy = ((lp.y() - lpj_.y()) * b) / tmp;
        double dm_dz = ((lp.z() - lpj_.z()) * c) / tmp;

        double sr = std::sin(parameters[0][3]);
        double cr = std::cos(parameters[0][3]);
        double sp = std::sin(parameters[0][4]);
        double cp = std::cos(parameters[0][4]);
        double sy = std::sin(parameters[0][5]);
        double cy = std::cos(parameters[0][5]);

        double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
        double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
        double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

        double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
        double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
        double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

        double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
        double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
        double dz_dy = 0.;


        if (jacobians && jacobians[0])
        {
            jacobians[0][0] = dm_dx / k;
            jacobians[0][1] = dm_dy / k;
            jacobians[0][2] = 0.;
            jacobians[0][3] = (dm_dx * dx_dr + dm_dy * dy_dr + dm_dz * dz_dr) / k;
            jacobians[0][4] = (dm_dx * dx_dp + dm_dy * dy_dp + dm_dz * dz_dp) / k;
            jacobians[0][5] = (dm_dx * dx_dy + dm_dy * dy_dy + dm_dz * dz_dy) / k;
        }

        return true;
    }

private:
    Eigen::Vector3d cp_;
    Eigen::Vector3d lpj_, lpl_, lpm_;
};

//Eigen::Matrix<double,3,3> skew(Eigen::Matrix<double,3,1>& mat_in){
//    Eigen::Matrix<double,3,3> skew_mat;
//    skew_mat.setZero();
//    skew_mat(0,1) = -mat_in(2);
//    skew_mat(0,2) =  mat_in(1);
//    skew_mat(1,2) = -mat_in(0);
//    skew_mat(1,0) =  mat_in(2);
//    skew_mat(2,0) = -mat_in(1);
//    skew_mat(2,1) =  mat_in(0);
//    return skew_mat;
//}



#endif // SRC_LIDARFACTOR_H
