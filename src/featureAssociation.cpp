//
// Created by zhao on 03.05.22.
//

#include <feature_association/featureAssociation.h>
#include <feature_association/lidarFactor.h>


namespace odo {

FeatureAssociation::FeatureAssociation(const pcl::PointCloud<PointType>::Ptr oriCloudLast,
                                       const pcl::PointCloud<PointType>::Ptr oriCloudCur,
                                       Eigen::Quaterniond qCurr,
                                       Eigen::Vector3d tCurr,
                                       Eigen::Quaterniond q_last_curr_,
                                       Eigen::Vector3d t_last_curr_) {


    q_w_curr = qCurr;
    t_w_curr = tCurr;
    // initial guess from last transform
    q_last_curr = q_last_curr_;
    t_last_curr = t_last_curr_;

    initializationValue();
    *lastFrame = *oriCloudLast;
    *currentFrame = *oriCloudCur;
    // get features from imageProjection from last frame and current frame
    // Initialize imageProjection
    llo::ImageProjection IpLast, IpCur;

    IpLast.run(*oriCloudLast);

    IpCur.run(*oriCloudCur);

    // get features
    groundFeatureLast = IpLast.getGroundFeature();
//    groundFeatureLast = IpLast.getGroundLessFlatFeature();
    curbFeatureLast = IpLast.getCurbFeature();
    surfaceFeatureLast = IpLast.getSurfaceFeature();
    edgeFeatureLast = IpLast.getEdgeFeature();

    groundFeature = IpCur.getGroundFeature();
//    groundFeature = IpCur.getGroundLessFlatFeature();
    curbFeature = IpCur.getCurbFeature();
    surfaceFeature = IpCur.getSurfaceFeature();
    edgeFeature = IpCur.getEdgeFeature();
    // set input into kdTree
    kdTreeGroundLast->setInputCloud(groundFeatureLast);
    kdTreeCurbLast->setInputCloud(curbFeatureLast);
    kdTreeSurfaceLast->setInputCloud(surfaceFeatureLast);
    kdTreeEdgeLast->setInputCloud(edgeFeatureLast);


}

void FeatureAssociation::initializationValue() {

    groundFeature.reset(new pcl::PointCloud<PointType>());
    curbFeature.reset(new pcl::PointCloud<PointType>());
    surfaceFeature.reset(new pcl::PointCloud<PointType>());
    edgeFeature.reset(new pcl::PointCloud<PointType>());

    pcGround.reset(new pcl::PointCloud<PointType>());
    pcCurb.reset(new pcl::PointCloud<PointType>());
    pcSurface.reset(new pcl::PointCloud<PointType>());
    pcEdge.reset(new pcl::PointCloud<PointType>());
    pcTwoFrame.reset(new pcl::PointCloud<PointType>());
    lastFrame.reset(new pcl::PointCloud<PointType>());
    currentFrame.reset(new pcl::PointCloud<PointType>());


    groundFeatureLast.reset(new pcl::PointCloud<PointType>());
    curbFeatureLast.reset(new pcl::PointCloud<PointType>());
    surfaceFeatureLast.reset(new pcl::PointCloud<PointType>());
    edgeFeatureLast.reset(new pcl::PointCloud<PointType>());

    kdTreeGroundLast.reset(new pcl::KdTreeFLANN<PointType>());
    kdTreeCurbLast.reset(new pcl::KdTreeFLANN<PointType>());
    kdTreeSurfaceLast.reset(new pcl::KdTreeFLANN<PointType>());
    kdTreeEdgeLast.reset(new pcl::KdTreeFLANN<PointType>());


    nearestFeatureSearchSqDist = 40;

}

void FeatureAssociation::TransformToStart(PointType const *const pi, PointType *const po) {
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(1, q_last_curr);
    Eigen::Vector3d t_point_last = t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void FeatureAssociation::toEulerAngle(const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw) {
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


void FeatureAssociation::run() {

    TicToc t_whole; // count whole time cost
    // initializing
    if (systemInited)
    {
        systemInited = true;
        std::cout << "Initialization finished \n";
    }
    else
    {


//        // ceres opti
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *q_parameterization =
            new ceres::EigenQuaternionParameterization();
        ceres::Problem::Options problem_options;

        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(para_q, 4, q_parameterization);
        problem.AddParameterBlock(para_t, 3);




        // find correspondence features
        TicToc t_opt;  // time consumption for optimization
        TicToc t_data; // time consumption for correspondence find


//        t_data.tic();
        int groundPointNum = groundFeature->points.size();
        int surfacePointNum = surfaceFeature->points.size();
        int edgePointNum = edgeFeature->points.size();
        int curbPointNum = curbFeature->points.size();




        //// find correspondence features for surface features
        for (int i = 0; i < surfacePointNum; ++i) {
            if (surfaceFeature->empty() == true || surfaceFeatureLast->empty() == true) {
                std::cerr << "no edge feature using other features" << std::endl;
                break;
            }
            // transform to last frame
            TransformToStart(&surfaceFeature->points[i], &point_);

            kdTreeSurfaceLast->nearestKSearch(point_, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

            if (pointSearchSqDis[4] < 1.0)
            {

                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = surfaceFeatureLast->points[pointSearchInd[j]].x;
                    matA0(j, 1) = surfaceFeatureLast->points[pointSearchInd[j]].y;
                    matA0(j, 2) = surfaceFeatureLast->points[pointSearchInd[j]].z;
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
                    if (fabs(norm(0) * surfaceFeatureLast->points[pointSearchInd[j]].x +
                             norm(1) * surfaceFeatureLast->points[pointSearchInd[j]].y +
                             norm(2) * surfaceFeatureLast->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                    {
                        surfaceValid = false;
                        break;
                    }
                }

                Eigen::Vector3d curr_point(surfaceFeature->points[i].x, surfaceFeature->points[i].y, surfaceFeature->points[i].z);
                if (surfaceValid)
                {
                    if (saveFeatureAssociation) {
                        PointType temPoint;
                        temPoint.x = curr_point[0];
                        temPoint.y = curr_point[1];
                        temPoint.z = curr_point[2];
                        temPoint.intensity = 30*surface_correspondence;
                        pcSurface->push_back(temPoint);

                        temPoint.x = surfaceFeatureLast->points[pointSearchInd[0]].x;
                        temPoint.y = surfaceFeatureLast->points[pointSearchInd[0]].y;
                        temPoint.z = surfaceFeatureLast->points[pointSearchInd[0]].z;
                        temPoint.intensity = 30*surface_correspondence;
                        pcSurface->push_back(temPoint);
                        temPoint.x = surfaceFeatureLast->points[pointSearchInd[1]].x;
                        temPoint.y = surfaceFeatureLast->points[pointSearchInd[1]].y;
                        temPoint.z = surfaceFeatureLast->points[pointSearchInd[1]].z;
                        temPoint.intensity = 30*surface_correspondence;
                        pcSurface->push_back(temPoint);
                        temPoint.x = surfaceFeatureLast->points[pointSearchInd[2]].x;
                        temPoint.y = surfaceFeatureLast->points[pointSearchInd[2]].y;
                        temPoint.z = surfaceFeatureLast->points[pointSearchInd[2]].z;
                        temPoint.intensity = 30*surface_correspondence;
                        pcSurface->push_back(temPoint);
                        temPoint.x = surfaceFeatureLast->points[pointSearchInd[3]].x;
                        temPoint.y = surfaceFeatureLast->points[pointSearchInd[3]].y;
                        temPoint.z = surfaceFeatureLast->points[pointSearchInd[3]].z;
                        temPoint.intensity = 30*surface_correspondence;
                        pcSurface->push_back(temPoint);
                        temPoint.x = surfaceFeatureLast->points[pointSearchInd[4]].x;
                        temPoint.y = surfaceFeatureLast->points[pointSearchInd[4]].y;
                        temPoint.z = surfaceFeatureLast->points[pointSearchInd[4]].z;
                        temPoint.intensity = 30*surface_correspondence;
                        pcSurface->push_back(temPoint);
                    }
                    ceres::CostFunction *cost_function = LidarSurfaceNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    surface_correspondence++;
                }
            }

        }
        std::cout << "surface correspondence: " << surface_correspondence << std::endl;
//        std::cout << "surface association time: " << t_data.toc() << " s " << std::endl;
        if (surface_correspondence >= 10) {
            TicToc t_solver;
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


            printf("surface association solver time %f s \n", t_solver.toc());
            std::cout << summary.BriefReport() << std::endl;
        }
        else {
            std::cerr << "few surface correspondence" << std::endl;
        }

        //// find correspondence for edge feature

        //to show correspondence

        for (int i = 0; i < edgePointNum; ++i) {
            // transform to last frame
            if (edgeFeature->empty() == true || edgeFeatureLast->empty() == true) {
                std::cerr << "no edge feature using other features" << std::endl;
                break;
            }
            TransformToStart(&edgeFeature->points[i], &point_);

            kdTreeEdgeLast->nearestKSearch(point_, 1, pointSearchInd, pointSearchSqDis);
            int closestPointInd = -1, minPointInd2 = -1;

            if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {
                closestPointInd = pointSearchInd[0];
                // to get scan id
                int closestPointScan = int(edgeFeatureLast->points[closestPointInd].intensity);

                // save last scan to minPointSqDis2, after scan to minPointSqDis3
                float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist;
                // search in last frame of edge features
                for (int j = closestPointInd + 1; j < (int)edgeFeatureLast->points.size(); ++j) {
                    if (int(edgeFeatureLast->points[j].intensity) <= closestPointScan) {
                        continue;
                    }
                    if (int(edgeFeatureLast->points[j].intensity) > closestPointScan + 2.5) {
                        break;
                    }
                    pointSqDis =
                        (edgeFeatureLast->points[j].x - point_.x) * (edgeFeatureLast->points[j].x - point_.x) +
                        (edgeFeatureLast->points[j].y - point_.y) * (edgeFeatureLast->points[j].y - point_.y) +
                        (edgeFeatureLast->points[j].z - point_.z) * (edgeFeatureLast->points[j].z - point_.z);

                    // if the second point valid
                    if (pointSqDis < minPointSqDis2) {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                }

                // search for another valid point int the direction of decreasing scan line
                for (int j = closestPointInd - 1; j >= 0; j--) {
                    if (int(edgeFeatureLast->points[j].intensity) >= closestPointScan) {
                        continue;
                    }
                    if (int(edgeFeatureLast->points[j].intensity) < (closestPointScan - 2.5)) {
                        break;
                    }

                    pointSqDis =
                        (edgeFeatureLast->points[j].x - point_.x) * (edgeFeatureLast->points[j].x - point_.x) +
                        (edgeFeatureLast->points[j].y - point_.y) * (edgeFeatureLast->points[j].y - point_.y) +
                        (edgeFeatureLast->points[j].z - point_.z) * (edgeFeatureLast->points[j].z - point_.z);

                    if (pointSqDis < minPointSqDis2) {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                }
            }
            if (minPointInd2 >= 0)
            {
//                std::cout << "i, closestPointInd, minPointInd2: " << i << "  " << "  " << closestPointInd << "  " << minPointInd2 << std::endl;
                Eigen::Vector3d curr_point_edge = Eigen::Vector3d (edgeFeature->points[i].x,
                                                                   edgeFeature->points[i].y,
                                                                   edgeFeature->points[i].z);
                Eigen::Vector3d last_point_edge_a = Eigen::Vector3d (edgeFeatureLast->points[closestPointInd].x,
                                                                     edgeFeatureLast->points[closestPointInd].y,
                                                                     edgeFeatureLast->points[closestPointInd].z);
                Eigen::Vector3d last_point_edge_b = Eigen::Vector3d (edgeFeatureLast->points[minPointInd2].x,
                                                                     edgeFeatureLast->points[minPointInd2].y,
                                                                     edgeFeatureLast->points[minPointInd2].z);
//                Eigen::Vector3d last_point_edge_test = Eigen::Vector3d (edgeFeatureLast->points[pointSearchInd[0]].x,
//                                                                     edgeFeatureLast->points[pointSearchInd[0]].y,
//                                                                     edgeFeatureLast->points[pointSearchInd[0]].z);
                // refine association
                Eigen::Vector3d ab = last_point_edge_a - curr_point_edge;
                Eigen::Vector3d ac = last_point_edge_b - curr_point_edge;
                Eigen::Vector3d bc = last_point_edge_a - last_point_edge_b;

                double distance = (ab.cross(ac)).norm() / bc.norm();

                if (distance < 1.5) {
                    if (saveFeatureAssociation) {
                        PointType temPoint;
                        temPoint.x = curr_point_edge[0];
                        temPoint.y = curr_point_edge[1];
                        temPoint.z = curr_point_edge[2];
                        temPoint.intensity = edge_correspondence * 10;
                        pcEdge->push_back(temPoint);
                        temPoint.x = last_point_edge_a[0];
                        temPoint.y = last_point_edge_a[1];
                        temPoint.z = last_point_edge_a[2];
                        temPoint.intensity = edge_correspondence * 10;
                        pcEdge->push_back(temPoint);
                        temPoint.x = last_point_edge_b[0];
                        temPoint.y = last_point_edge_b[1];
                        temPoint.z = last_point_edge_b[2];
                        temPoint.intensity = edge_correspondence * 10;
                        pcEdge->push_back(temPoint);
//                    temPoint.x = last_point_edge_test[0];
//                    temPoint.y = last_point_edge_test[1];
//                    temPoint.z = last_point_edge_test[2];
//                    temPoint.intensity = 50;
//                    pcEdge->push_back(temPoint);
                    }

                    ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point_edge, last_point_edge_a, last_point_edge_b, 1.0);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
//                problem.AddResidualBlock(new EdgeCostFunction(curr_point_edge, last_point_edge_a, last_point_edge_b), loss_function, params_);
                    edge_correspondence++;
                }
            }
        }
        std::cout << "edge correspondence: " << edge_correspondence << std::endl;

        if (edge_correspondence >= 10) {
            TicToc t_solver;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 5;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            //// update corresponding para for edge: x,y
            para_t_new[0] = para_t[0];
            para_t_new[1] = para_t[1];



            printf("edge association solver time %f s \n", t_solver.toc());
            std::cout << summary.BriefReport() << std::endl;
        }
        else {
            std::cerr << "few edge correspondence" << std::endl;
        }

//        t_data.tic();

        //// find correspondence curb features

        for (int i = 0; i < curbPointNum; ++i) {
            // transform to last frame

            if (curbFeature->empty() == true || curbFeatureLast->empty() == true) {
                std::cerr << "no curb feature using other features" << std::endl;
                break;
            }
            TransformToStart(&curbFeature->points[i], &point_);

            kdTreeCurbLast->nearestKSearch(point_, 5, pointSearchInd, pointSearchSqDis);


            if (pointSearchSqDis[4] < 1.0) {
                std::vector<Eigen::Vector3d> nearCurbs;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d tmp(curbFeatureLast->points[pointSearchInd[j]].x,
                                        curbFeatureLast->points[pointSearchInd[j]].y,
                                        curbFeatureLast->points[pointSearchInd[j]].z);
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
                Eigen::Vector3d curr_point(curbFeature->points[i].x, curbFeature->points[i].y, curbFeature->points[i].z);
                if (saes.eigenvalues()[2] > 4 * saes.eigenvalues()[1])
                {
                    Eigen::Vector3d point_on_line = center;
                    Eigen::Vector3d point_a, point_b;
                    point_a = 0.1 * unit_direction + point_on_line;
                    point_b = -0.1 * unit_direction + point_on_line;

                    if (saveFeatureAssociation) {
                        PointType temPoint;
                        temPoint.x = point_a[0];
                        temPoint.y = point_a[1];
                        temPoint.z = point_a[2];
                        temPoint.intensity = 10*curb_correspondence;
                        pcCurb->push_back(temPoint);
                        temPoint.x = point_b[0];
                        temPoint.y = point_b[1];
                        temPoint.z = point_b[2];
                        temPoint.intensity = 10*curb_correspondence;
                        pcCurb->push_back(temPoint);
                        temPoint = curbFeature->points[i];
                        temPoint.intensity = 10*curb_correspondence;
                        pcCurb->push_back(temPoint);
                    }

                    ceres::CostFunction *cost_function = LidarCurbFactor::Create(curr_point, point_a, point_b, 1.0);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    curb_correspondence++;
                }

            }
        }
        std::cout << "Curb correspondences in sub map: " << curb_correspondence << std::endl;
        if (curb_correspondence >= 10) {
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


        //// find correspondence features for ground features

        for (int i = 0; i < groundPointNum; ++i) {
            // transform to last frame
            TransformToStart(&groundFeature->points[i], &point_);

            kdTreeGroundLast->nearestKSearch(point_, 5, pointSearchInd, pointSearchSqDis);
            // Surface equation Ax + By + Cz + 1 = 0 <--> matA0 * norm（A, B, C） = matB0
            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

            if (pointSearchSqDis[4] < 1.0)
            {

                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = groundFeatureLast->points[pointSearchInd[j]].x;
                    matA0(j, 1) = groundFeatureLast->points[pointSearchInd[j]].y;
                    matA0(j, 2) = groundFeatureLast->points[pointSearchInd[j]].z;
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
                    // if OX * n > 0.2, then plane is not fit well distance equation fabs(Ax0 + By0 + Cz0 + D) / sqrt(A^2 + B^2 + C^2)
                    if (fabs(norm(0) * groundFeatureLast->points[pointSearchInd[j]].x +
                             norm(1) * groundFeatureLast->points[pointSearchInd[j]].y +
                             norm(2) * groundFeatureLast->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }
                Eigen::Vector3d curr_point(groundFeature->points[i].x, groundFeature->points[i].y, groundFeature->points[i].z);
                if (planeValid)
                {
                    if (saveFeatureAssociation) {
                        PointType temPoint;
                        temPoint = groundFeatureLast->points[pointSearchInd[0]];
                        temPoint.intensity = 10*ground_correspondence;
                        pcGround->push_back(temPoint);
                        temPoint = groundFeatureLast->points[pointSearchInd[1]];
                        temPoint.intensity = 10*ground_correspondence;
                        pcGround->push_back(temPoint);
                        temPoint = groundFeatureLast->points[pointSearchInd[2]];
                        temPoint.intensity = 10*ground_correspondence;
                        pcGround->push_back(temPoint);
                        temPoint = groundFeatureLast->points[pointSearchInd[3]];
                        temPoint.intensity = 10*ground_correspondence;
                        pcGround->push_back(temPoint);
                        temPoint = groundFeatureLast->points[pointSearchInd[4]];
                        temPoint.intensity = 10*ground_correspondence;
                        pcGround->push_back(temPoint);

                        temPoint = groundFeature->points[i];
                        temPoint.intensity = 10*ground_correspondence;
                        pcGround->push_back(temPoint);
                    }
                    ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    ground_correspondence++;
                }
            }


        }

        std::cout << "ground correspondence: " << ground_correspondence << std::endl;
        if (ground_correspondence >= 10) {
            TicToc t_solver;
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


            printf("ground association solver time %f s \n", t_solver.toc());
            std::cout << summary.BriefReport() << std::endl;
        }
        else {
            std::cerr << "few ground correspondence" << std::endl;
        }




        // using params to update t_last_curr and q_last_curr
//        inverseInnerTransform();
        ///test new opti
        q_last_curr = Eigen::Map<Eigen::Quaterniond> (para_q_new);
        t_last_curr = Eigen::Map<Eigen::Vector3d> (para_t_new);

        t_w_curr = t_w_curr + q_w_curr * t_last_curr;
        q_w_curr = q_w_curr * q_last_curr;
        std::cout << "total opt time: " << t_opt.toc() << " s " << std::endl;
    }
    std::cout << "Transform between last frame and current frame: " << std::endl;
    std::cout << " rotation matrix \n " << q_last_curr.toRotationMatrix() << std::endl;
    std::cout << " t_last_cur \n" << t_last_curr << "\n" << std::endl;


    Eigen::Isometry3d T = Eigen::Isometry3d ::Identity();
    T.rotate(q_w_curr.matrix());
    T.pretranslate(t_w_curr);
    std::cout << "current frame pose in world frame: " << std::endl;
    std::cout << "  "<< T.matrix() << std::endl;

    std::cerr << "feature association time: " << t_whole.toc() << "ms" << std::endl;


    if (saveFeatureAssociation) {
        PointType temP;
//        *pcTwoFrame = *lastFrame;
        for (int i = 0; i < currentFrame->points.size(); ++i) {
            TransformToStart(&currentFrame->points[i], &temP);
//            temP = currentFrame->points[i];
            temP.intensity = 1000;
            pcTwoFrame->push_back(temP);
        }
        *pcTwoFrame += *lastFrame;
    }

}

Eigen::Quaterniond FeatureAssociation::publishQuaterniond() {
    return q_w_curr;
}
Eigen::Vector3d FeatureAssociation::publishTranslation() {
    return t_w_curr;
}

Eigen::Quaterniond FeatureAssociation::quaterniodGuess() {
    std::cout << "q_last_curr:\n" << q_last_curr.matrix() << std::endl;
    return q_last_curr;
}
Eigen::Vector3d FeatureAssociation::translationGuess() {
    std::cout << "t_last_curr:\n" << t_last_curr << std::endl;
    return t_last_curr;
}



} // end of namespace odo