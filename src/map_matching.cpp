//
// Created by zhao on 06.06.22.
//
#include "map_matching/map_matching.h"
#include <feature_association/lidarFactor.h>

namespace mm {

MapMatching::MapMatching(const pcl::PointCloud<PointType>::Ptr laserCloudGroundLast_,
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
                         const Eigen::Vector3d t_wmap_wodom_)
{

    q_wodom_curr = q_wodom_curr_;
    t_wodom_curr = t_wodom_curr_;

    // initial guess from last frame
    q_wmap_wodom = q_wmap_wodom_;
    t_wmap_wodom = t_wmap_wodom_;


    initialValue();

    *laserCloudGroundLast = *laserCloudGroundLast_;
    *laserCloudCurbLast = *laserCloudCurbLast_;

    *laserCloudSurfaceLast = *laserCloudSurfaceLast_;
    *laserCloudEdgeLast = *laserCloudEdgeLast_;

    *localMapGround = *groundMap;
    *localMapCurb = *curbMap;
    *localMapSurface = *surfaceMap;
    *localMapEdge = *edgeMap;

    getSubMap();
}
void MapMatching::initialValue() {

    laserCloudGroundLast.reset(new pcl::PointCloud<PointType>());
    laserCloudCurbLast.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfaceLast.reset(new pcl::PointCloud<PointType>());
    laserCloudEdgeLast.reset(new pcl::PointCloud<PointType>());

    localMapGround.reset(new pcl::PointCloud<PointType>());
    localMapSurface.reset(new pcl::PointCloud<PointType>());
    localMapEdge.reset(new pcl::PointCloud<PointType>());
    localMapCurb.reset(new pcl::PointCloud<PointType>());

    subMapGround.reset(new pcl::PointCloud<PointType>());
    subMapSurface.reset(new pcl::PointCloud<PointType>());
    subMapEdge.reset(new pcl::PointCloud<PointType>());
    subMapCurb.reset(new pcl::PointCloud<PointType>());

    kdtreeGroundFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeEdgeFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeCurbFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfaceFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudGroundStack.reset(new pcl::PointCloud<PointType>());
    laserCloudCurbStack.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfaceStack.reset(new pcl::PointCloud<PointType>());
    laserCloudEdgeStack.reset(new pcl::PointCloud<PointType>());

    downSizeFilterCurb.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilterSurface.setLeafSize(0.8, 0.8, 0.8);
    downSizeFilterGround.setLeafSize(0.8, 0.8, 0.8);
    downSizeFilterEdge.setLeafSize(0.4, 0.4, 0.4);

    testCloudGround.reset(new pcl::PointCloud<PointType>());
    testCloudSurface.reset(new pcl::PointCloud<PointType>());
    testCloudEdge.reset(new pcl::PointCloud<PointType>());
    testCloudCurb.reset(new pcl::PointCloud<PointType>());
}


void MapMatching::getSubMap() { // process local map and extract part which surrounding odo pose
    PointType temPoint;
    //// *************************ground map part**************************
    for (int i = 0; i < localMapGround->points.size(); ++i) {
        temPoint.x = localMapGround->points[i].x;
        temPoint.y = localMapGround->points[i].y;
        temPoint.z = localMapGround->points[i].z;
        temPoint.intensity = localMapGround->points[i].intensity;
        float distance = std::sqrt(std::pow((temPoint.x - t_wodom_curr[0]), 2)
                                   + std::pow((temPoint.y - t_wodom_curr[1]), 2)
                                   + std::pow((temPoint.z - t_wodom_curr[2]), 2));
        if (distance < subMapSize) {
            subMapGround->push_back(temPoint);
        }
    }
    //// *************************curb map part**************************
    for (int i = 0; i < localMapCurb->points.size(); ++i) {
        temPoint.x = localMapCurb->points[i].x;
        temPoint.y = localMapCurb->points[i].y;
        temPoint.z = localMapCurb->points[i].z;
        temPoint.intensity = localMapCurb->points[i].intensity;
        float distance = std::sqrt(std::pow((temPoint.x - t_wodom_curr[0]), 2)
                                   + std::pow((temPoint.y - t_wodom_curr[1]), 2)
                                   + std::pow((temPoint.z - t_wodom_curr[2]), 2));
        if (distance < subMapSize) {
            subMapCurb->push_back(temPoint);
        }
    }
    //// *************************surface map part**************************
    for (int i = 0; i < localMapSurface->points.size(); ++i) {
        temPoint.x = localMapSurface->points[i].x;
        temPoint.y = localMapSurface->points[i].y;
        temPoint.z = localMapSurface->points[i].z;
        temPoint.intensity = localMapSurface->points[i].intensity;
        float distance = std::sqrt(std::pow((temPoint.x - t_wodom_curr[0]), 2)
                                   + std::pow((temPoint.y - t_wodom_curr[1]), 2)
                                   + std::pow((temPoint.z - t_wodom_curr[2]), 2));
        if (distance < subMapSize) {
            subMapSurface->push_back(temPoint);
        }
    }
    //// *************************edge map part**************************
    for (int i = 0; i < localMapEdge->points.size(); ++i) {
        temPoint.x = localMapEdge->points[i].x;
        temPoint.y = localMapEdge->points[i].y;
        temPoint.z = localMapEdge->points[i].z;
        temPoint.intensity = localMapEdge->points[i].intensity;
        float distance = std::sqrt(std::pow((temPoint.x - t_wodom_curr[0]), 2)
                                   + std::pow((temPoint.y - t_wodom_curr[1]), 2)
                                   + std::pow((temPoint.z - t_wodom_curr[2]), 2));
        if (distance < subMapSize) {
            subMapEdge->push_back(temPoint);
        }
    }
    kdtreeGroundFromMap->setInputCloud(subMapGround);
    kdtreeCurbFromMap->setInputCloud(subMapCurb);
    kdtreeSurfaceFromMap->setInputCloud(subMapSurface);
    kdtreeEdgeFromMap->setInputCloud(subMapEdge);
}

void MapMatching::transformAssociateToMap() {
    q_w_curr = q_wmap_wodom * q_wodom_curr;
    t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void MapMatching::transformUpdate() {
    q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
    t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void MapMatching::pointAssociateToMap(const PointType* const pi, PointType* const po) {
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
}

void MapMatching::toEulerAngle(const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw) {
    // roll (x-axis rotation)
    double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
    double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
    roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
    if (fabs(sinp) >= 1)
        pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        pitch = asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    yaw = atan2(siny_cosp, cosy_cosp);
}

//void MapMatching::test() {
//    getSubMap();
//}


void MapMatching::run() {
    transformAssociateToMap();
    int laserCloudGroundFromMapNum = subMapGround->points.size();
    int laserCloudCurbFromMapNum = subMapCurb->points.size();
    int laserCloudSurfaceFromMapNum = subMapSurface->points.size();
    int laserCloudEdgeFromMapNum = subMapEdge->points.size();

    if (laserCloudGroundFromMapNum < 10 || laserCloudCurbFromMapNum < 10
        || laserCloudSurfaceFromMapNum < 10 || laserCloudEdgeFromMapNum < 10) {
        std::cerr << " little Features from map" << std::endl;
        std::cout << "Feature from Map g, c, s, e: " << laserCloudGroundFromMapNum << " , "
        << laserCloudCurbFromMapNum << " , " << laserCloudSurfaceFromMapNum << " , " << laserCloudEdgeFromMapNum << std::endl;
    }

    downSizeFilterGround.setInputCloud(laserCloudGroundLast);
    downSizeFilterGround.filter(*laserCloudGroundStack);
    int laserCloudGroundStackNum = laserCloudGroundStack->points.size();

    downSizeFilterCurb.setInputCloud(laserCloudCurbLast);
    downSizeFilterCurb.filter(*laserCloudCurbStack);
    int laserCloudCurbStackNum = laserCloudCurbStack->points.size();

    downSizeFilterSurface.setInputCloud(laserCloudSurfaceLast);
    downSizeFilterSurface.filter(*laserCloudSurfaceStack);
    int laserCloudSurfaceStackNum = laserCloudSurfaceStack->points.size();

    downSizeFilterEdge.setInputCloud(laserCloudEdgeLast);
    downSizeFilterEdge.filter(*laserCloudEdgeStack);
    int laserCloudEdgeStackNum = laserCloudEdgeStack->points.size();


    TicToc t_opt;

//    for (int iterCount = 0; iterCount < 4; ++iterCount) {

    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *q_parameterization =
        new ceres::EigenQuaternionParameterization();
    ceres::Problem::Options problem_options;

    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_q, 4, q_parameterization);
    problem.AddParameterBlock(para_t, 3);


    ////************************surface association in sub map***************************************
    int surfaceNum = 0;
    for (int i = 0; i < laserCloudSurfaceStackNum; i++)
    {
        if (laserCloudSurfaceFromMapNum < 10) {
            break;
        }
        pointOri = laserCloudSurfaceStack->points[i];
        //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
        pointAssociateToMap(&pointOri, &pointSel);
        kdtreeSurfaceFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        // Surface equation Ax + By + Cz + 1 = 0 <--> matA0 * norm（A, B, C） = matB0
        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0)
        {

            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = subMapSurface->points[pointSearchInd[j]].x;
                matA0(j, 1) = subMapSurface->points[pointSearchInd[j]].y;
                matA0(j, 2) = subMapSurface->points[pointSearchInd[j]].z;
                //printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm();
            norm.normalize();

            // Here n(pa, pb, pc) is unit norm of plane
            bool surfaceValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well distance equation fabs(Ax0 + By0 + Cz0 + D) / sqrt(A^2 + B^2 + C^2)
                if (fabs(norm(0) * subMapSurface->points[pointSearchInd[j]].x +
                         norm(1) * subMapSurface->points[pointSearchInd[j]].y +
                         norm(2) * subMapSurface->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    surfaceValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
            if (surfaceValid)
            {
                if (saveAssociation) {
                    PointType temPoint;
                    temPoint = subMapSurface->points[pointSearchInd[0]];
                    temPoint.intensity = 10*surfaceNum;
                    testCloudSurface->push_back(temPoint);
                    temPoint = subMapSurface->points[pointSearchInd[1]];
                    temPoint.intensity = 10*surfaceNum;
                    testCloudSurface->push_back(temPoint);
                    temPoint = subMapSurface->points[pointSearchInd[2]];
                    temPoint.intensity = 10*surfaceNum;
                    testCloudSurface->push_back(temPoint);
                    temPoint = subMapSurface->points[pointSearchInd[3]];
                    temPoint.intensity = 10*surfaceNum;
                    testCloudSurface->push_back(temPoint);
                    temPoint = subMapSurface->points[pointSearchInd[4]];
                    temPoint.intensity = 10*surfaceNum;
                    testCloudSurface->push_back(temPoint);

                    temPoint = pointSel;
                    temPoint.intensity = 10*surfaceNum;
                    testCloudSurface->push_back(temPoint);
                }
                ceres::CostFunction *cost_function = LidarSurfaceNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                surfaceNum++;
            }
        }
    }
    std::cout << "Surface correspondences in sub map: " << surfaceNum << std::endl;
    if (surfaceNum >= 10) {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 5;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        //// update corresponding para for surface: x,y,r,p,y
        para_q_new[0] = para_q[0];
        para_q_new[1] = para_q[1];
        para_q_new[2] = para_q[2];
        para_q_new[3] = para_q[3];

        para_t_new[0] = para_t[0];
        para_t_new[1] = para_t[1];

        std::cout << summary.BriefReport() << std::endl;
    }
    else {
        std::cerr << "few surface correspondence" << std::endl;
    }



    ////************************edge association in sub map***************************************
    int edgeNum = 0;

    for (int i = 0; i < laserCloudEdgeStackNum; ++i) {
        if (laserCloudEdgeFromMapNum < 10) {
            break;
        }

        pointOri = laserCloudEdgeStack->points[i];

        pointAssociateToMap(&pointOri, &pointSel);
        kdtreeEdgeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < 2.0) {
            Eigen::Vector3d tmp(subMapEdge->points[pointSearchInd[0]].x, subMapEdge->points[pointSearchInd[0]].y, subMapEdge->points[pointSearchInd[0]].z);
            Eigen::Vector3d unit_direction(0, 0, 1);
            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
            Eigen::Vector3d point_a, point_b;
            point_a = 0.1 * unit_direction + tmp;
            point_b = -0.1 * unit_direction + tmp;

            if (saveAssociation) {
                PointType temPoint;
                temPoint = subMapEdge->points[pointSearchInd[0]];
                temPoint.intensity = 10*edgeNum;
                testCloudEdge->push_back(temPoint);
                temPoint = pointSel;
                temPoint.intensity = 10*edgeNum;
                testCloudEdge->push_back(temPoint);
            }
            ceres::CostFunction * cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
            edgeNum++;
        }

    }
    std::cout << "Edge correspondences in sub map: " << edgeNum << std::endl;
    if (edgeNum >= 10) {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 5;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        //// update corresponding para for edge: x,y
        para_t_new[0] = para_t[0];
        para_t_new[1] = para_t[1];



        std::cout << summary.BriefReport() << std::endl;
    }
    else {
        std::cerr << "few edge correspondence" << std::endl;
    }

    ////************************curb association in sub map***************************************
//        TicToc t_data;
    int curbNum = 0;
    for (int i = 0; i < laserCloudCurbStackNum; ++i) {

        if (laserCloudCurbFromMapNum < 10) {
            break;
        }
        pointOri = laserCloudCurbStack->points[i];
        pointAssociateToMap(&pointOri, &pointSel);
        kdtreeCurbFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        if (pointSearchSqDis[4] < 2.0) {
            std::vector<Eigen::Vector3d> nearCurbs;
            Eigen::Vector3d center(0, 0, 0);
            for (int j = 0; j < 5; j++)
            {
                Eigen::Vector3d tmp(subMapCurb->points[pointSearchInd[j]].x,
                                    subMapCurb->points[pointSearchInd[j]].y,
                                    subMapCurb->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCurbs.push_back(tmp);
            }
            center = center / 5.0;

            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int j = 0; j < 5; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCurbs[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            // if is indeed line feature
            // note Eigen library sort eigenvalues in increasing order
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;

                if (saveAssociation) {
                    PointType temPoint;
                    temPoint.x = point_a[0];
                    temPoint.y = point_a[1];
                    temPoint.z = point_a[2];
                    temPoint.intensity = 10*curbNum;
                    testCloudCurb->push_back(temPoint);
                    temPoint.x = point_b[0];
                    temPoint.y = point_b[1];
                    temPoint.z = point_b[2];
                    temPoint.intensity = 10*curbNum;
                    testCloudCurb->push_back(temPoint);
                    temPoint = pointSel;
                    temPoint.intensity = 10*curbNum;
                    testCloudCurb->push_back(temPoint);
                }

                ceres::CostFunction *cost_function = LidarCurbFactor::Create(curr_point, point_a, point_b, 1.0);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                curbNum++;
            }

        }
    }
    std::cout << "Curb correspondences in sub map: " << curbNum << std::endl;
    if (curbNum >= 10) {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 5;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        //// update corresponding para for curb: x,y,z,yaw
        // transform para_q_new to eulerAngle then only change yaw with para_q
        Eigen::Quaterniond q_new(para_q_new[3],para_q_new[0],para_q_new[1],para_q_new[2]);
        Eigen::Quaterniond q_opti(para_q[3],para_q[0],para_q[1],para_q[2]);
        double r,p,y,r_o,p_o,y_o;
        toEulerAngle(q_new,r,p,y);
        toEulerAngle(q_opti,r_o,p_o,y_o);

        // change yaw
        // transform back to quaternion
        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(r,Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(p,Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(y_o,Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond quaternion;
        quaternion=yawAngle*pitchAngle*rollAngle;
        // put in para_q_new

        para_q_new[3] = quaternion.w();
        para_q_new[0] = quaternion.x();
        para_q_new[1] = quaternion.y();
        para_q_new[2] = quaternion.z();

        para_t_new[0] = para_t[0];
        para_t_new[1] = para_t[1];
        para_t_new[2] = para_t[2];



        std::cout << summary.BriefReport() << std::endl;
    }
    else {
        std::cerr << "few curb correspondence" << std::endl;
    }



    ////************************ground association in sub map***************************************
    int groundNum = 0;
    for (int i = 0; i < laserCloudGroundStackNum; ++i) {

        if (laserCloudGroundFromMapNum < 10) {
            break;
        }
        pointOri = laserCloudGroundStack->points[i];

        pointAssociateToMap(&pointOri, &pointSel);
        kdtreeGroundFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

//        if (pointSearchSqDis[0] < 1) {
//            Eigen::Vector3d tmp(subMapGround->points[pointSearchInd[0]].x,
//                                subMapGround->points[pointSearchInd[0]].y,
//                                subMapGround->points[pointSearchInd[0]].z);
//            Eigen::Vector3d norm(0, 0, 1);
//            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
//            double negative_OA_dot_norm = 1 / norm.norm();
//
//            if (saveAssociation) {
//                PointType temPoint;
//                temPoint = subMapGround->points[pointSearchInd[0]];
//                temPoint.intensity = 10*groundNum;
//                testCloudGround->push_back(temPoint);
//                temPoint = pointSel;
//                temPoint.intensity = 10*groundNum;
//                testCloudGround->push_back(temPoint);
//            }
//            ceres::CostFunction * cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
//            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
//            groundNum++;
//        }
        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0)
        {

            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = subMapGround->points[pointSearchInd[j]].x;
                matA0(j, 1) = subMapGround->points[pointSearchInd[j]].y;
                matA0(j, 2) = subMapGround->points[pointSearchInd[j]].z;
                //printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm();
            norm.normalize();

            // Here n(pa, pb, pc) is unit norm of plane
            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * subMapGround->points[pointSearchInd[j]].x +
                         norm(1) * subMapGround->points[pointSearchInd[j]].y +
                         norm(2) * subMapGround->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
            if (planeValid)
            {
                if (saveAssociation && groundNum < 10000) {
                    PointType temPoint;
                    temPoint = subMapGround->points[pointSearchInd[0]];
                    temPoint.intensity = 10*groundNum;
                    testCloudGround->push_back(temPoint);
                    temPoint = subMapGround->points[pointSearchInd[1]];
                    temPoint.intensity = 10*groundNum;
                    testCloudGround->push_back(temPoint);
                    temPoint = subMapGround->points[pointSearchInd[2]];
                    temPoint.intensity = 10*groundNum;
                    testCloudGround->push_back(temPoint);
                    temPoint = subMapGround->points[pointSearchInd[3]];
                    temPoint.intensity = 10*groundNum;
                    testCloudGround->push_back(temPoint);
                    temPoint = subMapGround->points[pointSearchInd[4]];
                    temPoint.intensity = 10*groundNum;
                    testCloudGround->push_back(temPoint);

                    temPoint = pointSel;
                    temPoint.intensity = 10*groundNum;
                    testCloudGround->push_back(temPoint);

                }
//                std::cerr << norm << "\n" << std::endl;
                ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                groundNum++;
            }
        }
    }
    std::cout << "Ground correspondences in sub map: " << groundNum << std::endl;
    if (groundNum >= 10) {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 5;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        //// update corresponding para for ground: z,roll,pitch
        // transform para_q_new to eulerAngle then only change roll,pitch with para_q
        Eigen::Quaterniond q_new(para_q_new[3],para_q_new[0],para_q_new[1],para_q_new[2]);
        Eigen::Quaterniond q_opti(para_q[3],para_q[0],para_q[1],para_q[2]);
        double r,p,y,r_o,p_o,y_o;
        toEulerAngle(q_new,r,p,y);
        toEulerAngle(q_opti,r_o,p_o,y_o);
        // change roll and pitch
        // transform back to quaternion
        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(r_o,Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(p_o,Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(y,Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond quaternion;
        quaternion=yawAngle*pitchAngle*rollAngle;
        // put in para_q_new

        para_q_new[3] = quaternion.w();
        para_q_new[0] = quaternion.x();
        para_q_new[1] = quaternion.y();
        para_q_new[2] = quaternion.z();

        para_t_new[2] = para_t[2]; //z


        std::cout << summary.BriefReport() << std::endl;
    }
    else {
        std::cerr << "few ground correspondence" << std::endl;
    }



    ///test new opti
    q_w_curr = Eigen::Map<Eigen::Quaterniond> (para_q_new);
    t_w_curr = Eigen::Map<Eigen::Vector3d> (para_t_new);


//    TicToc t_solver;
//    ceres::Solver::Options options;
//    options.linear_solver_type = ceres::DENSE_QR;
//    options.max_num_iterations = 4;
//    options.minimizer_progress_to_stdout = false;
//    options.check_gradients = false;
//    options.gradient_check_relative_precision = 1e-4;
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.BriefReport() << std::endl;
//    std::cout << " map matching solver time s: \n " << t_solver.toc() << std::endl;
//    }
    std::cerr << " map matching optimization time s: " << t_opt.toc() << "ms"<< std::endl;
    transformUpdate();

}









}//end of mm