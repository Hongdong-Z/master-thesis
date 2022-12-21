//
// Created by zhao on 14.08.22.
//
#include "primitives_association/primitives_map_matcher.h"
#include "utility/tic_toc.h"

namespace primitivesMapMatcher {

PrimitiveMatcher::PrimitiveMatcher() {
    search_radius_ = 150.0f;
    search_radius_squared_ = search_radius_ * search_radius_;

    max_cylinder_association_pos_diff_ = 2;
    max_cylinder_association_radius_diff_ = 0.1;
    max_cylinder_association_angle_diff_ = 0.24;

    max_plane_association_pos_diff_ = 3;
    max_plane_association_angle_diff_ = 15/180.0*M_PI;

    mapPolesPath = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/poles_new_map.txt";
    mapFacadesPath = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/facades_new_map.txt";
    mapGroundPath = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/grounds_new_map.txt";

    ceres_solver_max_num_iterations_ = 150;

    pole_search_index.clear();
    pole_search_distance.clear();
    kdtree_pole_map_in_range.reset(new pcl::KdTreeFLANN<Point>());

    kdtreeFacadeFromMap.reset(new pcl::KdTreeFLANN<Point>());
    kdtreePoleFromMap.reset(new pcl::KdTreeFLANN<Point>());
    kdtreeGroundFromMap.reset(new pcl::KdTreeFLANN<Point>());
    //sub map
    subMapFacade.reset(new Cloud());
    subMapPole.reset(new Cloud());
    subMapGround.reset(new Cloud());
}
/// Read map txt
void PrimitiveMatcher::read_map_file(const std::string& pole_map_path, const std::string& facade_map_path, const std::string& ground_map_path) {

    map_.planes.clear();
    map_.cylinders.clear();
    map_.grounds.clear();

    std::ifstream file_reader_pole(pole_map_path);
    if (file_reader_pole.is_open()) {
        std::string line;
        while (std::getline(file_reader_pole, line)) {
            std::stringstream stream_cylinder(line);
            Cylinder c;
            stream_cylinder >> c.center_line.p_bottom(0) >> c.center_line.p_bottom(1) >> c.center_line.p_bottom(2)
                            >> c.center_line.p_top(0)  >> c.center_line.p_top(1) >> c.center_line.p_top(2)
                            >> c.radius;

            map_.cylinders.push_back(c);
        }
    }

    std::ifstream file_reader_facade(facade_map_path);
    if (file_reader_facade.is_open()) {
        std::string line;
        while (std::getline(file_reader_facade, line)) {
            std::stringstream plane_stream(line);
            Plane p;
            p.edge_poly.resize(4);
            plane_stream    >> p.edge_poly[0][0] >> p.edge_poly[0][1] >> p.edge_poly[0][2]
                            >> p.edge_poly[1][0] >> p.edge_poly[1][1] >> p.edge_poly[1][2]
                            >> p.edge_poly[2][0] >> p.edge_poly[2][1] >> p.edge_poly[2][2]
                            >> p.edge_poly[3][0] >> p.edge_poly[3][1] >> p.edge_poly[3][2]
                            >> p.n[0] >> p.n[1] >> p.n[2] >> p.d;

            map_.planes.push_back(p);
        }
    }
    std::ifstream file_reader_ground(ground_map_path);
    if (file_reader_ground.is_open()) {
        std::string line;
        while (std::getline(file_reader_ground, line)) {
            std::stringstream ground_stream(line);
            Plane p;
            p.edge_poly.resize(4);
            ground_stream    >> p.edge_poly[0][0] >> p.edge_poly[0][1] >> p.edge_poly[0][2]
                            >> p.edge_poly[1][0] >> p.edge_poly[1][1] >> p.edge_poly[1][2]
                            >> p.edge_poly[2][0] >> p.edge_poly[2][1] >> p.edge_poly[2][2]
                            >> p.edge_poly[3][0] >> p.edge_poly[3][1] >> p.edge_poly[3][2]
                            >> p.n[0] >> p.n[1] >> p.n[2] >> p.d;

            map_.grounds.push_back(p);
        }
    }
//    std::cout << "Whole map poles number: " << map_.cylinders.size() << std::endl;
//    std::cout << "Whole map facades number: " << map_.planes.size() << std::endl;
//    std::cout << "Whole map grounds number: " << map_.grounds.size() << std::endl;
}
void PrimitiveMatcher::getSubMap() {
    Point tmpPoint;
    /// *******************get poles sub map*************************
    for (int i = 0; i < map_.cylinders.size(); ++i) {
        tmpPoint.x = map_.cylinders[i].center_line.p_bottom[0];
        tmpPoint.y = map_.cylinders[i].center_line.p_bottom[1];
        tmpPoint.z = map_.cylinders[i].center_line.p_bottom[2];
        tmpPoint.intensity = 1000;
        float distance = std::sqrt(std::pow((tmpPoint.x - t_wodom_curr[0]), 2)
                                   + std::pow((tmpPoint.y - t_wodom_curr[1]), 2)
                                   + std::pow((tmpPoint.z - t_wodom_curr[2]), 2));
        if (distance < search_radius_) {
            subMapPole->push_back(tmpPoint);
            subMap_.cylinders.push_back(map_.cylinders[i]);
        }
    }
    /// *******************get facades sub map*************************
    // save the center points as a distance mark
    for (int i = 0; i < map_.planes.size(); ++i) {
        Eigen::Vector3f dl = map_.planes[i].edge_poly[0];
        Eigen::Vector3f dr = map_.planes[i].edge_poly[1];
        Eigen::Vector3f tr = map_.planes[i].edge_poly[2];
        Eigen::Vector3f centerPoint = dl + 0.5 * (tr - dl);

        tmpPoint.x = centerPoint[0];
        tmpPoint.y = centerPoint[1];
        tmpPoint.z = centerPoint[2];
        tmpPoint.intensity = 500;
        float distance = std::sqrt(std::pow((tmpPoint.x - t_wodom_curr[0]), 2)
                                   + std::pow((tmpPoint.y - t_wodom_curr[1]), 2)
                                   + std::pow((tmpPoint.z - t_wodom_curr[2]), 2));
        if (distance < search_radius_) {
            subMapFacade->push_back(tmpPoint);
            subMap_.planes.push_back(map_.planes[i]);
        }
    }
    /// *******************get grounds sub map*************************
    // save the center points as a distance mark
    for (int i = 0; i < map_.grounds.size(); ++i) {
        Eigen::Vector3f dl = map_.grounds[i].edge_poly[0];
        Eigen::Vector3f dr = map_.grounds[i].edge_poly[1];
        Eigen::Vector3f tr = map_.grounds[i].edge_poly[2];
        Eigen::Vector3f centerPoint = dl + 0.5 * (tr - dl);

        tmpPoint.x = centerPoint[0];
        tmpPoint.y = centerPoint[1];
        tmpPoint.z = centerPoint[2];
        tmpPoint.intensity = 200;
        float distance = std::sqrt(std::pow((tmpPoint.x - t_wodom_curr[0]), 2)
                                   + std::pow((tmpPoint.y - t_wodom_curr[1]), 2)
                                   + std::pow((tmpPoint.z - t_wodom_curr[2]), 2));
        if (distance < search_radius_) {
            subMapGround->push_back(tmpPoint);
            subMap_.grounds.push_back(map_.grounds[i]);
        }
    }
    // put in the Kd tree
    if (!subMapPole->points.empty()) {
        kdtreePoleFromMap->setInputCloud(subMapPole);
    } else {
        std::cerr << "No sub poles map found!" << std::endl;
    }
    if (!subMapFacade->points.empty()) {
        kdtreeFacadeFromMap->setInputCloud(subMapFacade);
    } else {
        std::cerr << "No sub facades map found!" << std::endl;
    }
    if (!subMapGround->points.empty()) {
        kdtreeGroundFromMap->setInputCloud(subMapGround);
    } else {
        std::cerr << "No sub grounds map found!" << std::endl;
    }
}

/// Set pose from scan2scan. Interface
void PrimitiveMatcher::set_pose_prior(const Eigen::Vector3d& t, const Eigen::Quaterniond& q,
                                      const Eigen::Vector3d& t_guess, const Eigen::Quaterniond& q_guess) {
    // Output from scan to scan as initial pose
    // using map matcher adjust difference.
    q_wodom_curr = q;
    t_wodom_curr = t;
    // initial guess between adjustment of odo and world frame
    q_wmap_wodom = q_guess;
    t_wmap_wodom = t_guess;
}


/// Get detected primitives.
void PrimitiveMatcher::set_detections(const std::vector<Plane>& planes, const std::vector<Cylinder>& cylinders, const std::vector<Plane>& grounds) {
    detections_.cylinders = cylinders;
    detections_.planes = planes;
    detections_.grounds = grounds;
    std::cout << "Current detected poles number: " << detections_.cylinders.size() << std::endl;
    std::cout << "Current detected facades number: " << detections_.planes.size() << std::endl;
    std::cout << "Current detected grounds number: " << detections_.grounds.size() << std::endl;
}
// Find primitives in range and push into ranged map
bool PrimitiveMatcher::find_primitives_in_range() {
    map_primitives_in_range_.plane_ids.clear();
    map_primitives_in_range_.cylinder_ids.clear();

    // Go through all planes in the map.
    for( int i = 0; i < (int)map_.planes.size(); ++i )
    {
        bool plane_in_range = check_distance_plane_2_pos_3D( i, t_wodom_curr.cast<float>());
        if( plane_in_range )
        {
            map_primitives_in_range_.plane_ids.push_back(i);
        }
    }
    for( int i = 0; i < (int)map_.cylinders.size(); ++i )
    {
        bool cylinder_in_range = check_distance_cylinder_2_pos_3D( i, t_wodom_curr.cast<float>());
        if( cylinder_in_range )
        {
            map_primitives_in_range_.cylinder_ids.push_back(i);
        }
    }

    return true;
}

// Put pole bottom point in kd-tree
void PrimitiveMatcher::put_pole_in_tree() {
    Cloud::Ptr tmpCloud(new Cloud());
    for (int i = 0; i < map_primitives_in_range_.cylinder_ids.size(); ++i) {
        Point p;
        p.x = map_.cylinders[i].center_line.p_bottom[0];
        p.y = map_.cylinders[i].center_line.p_bottom[1];
        p.z = map_.cylinders[i].center_line.p_bottom[2];
        p.intensity = 5000;
        tmpCloud->points.push_back(p);
    }
    kdtree_pole_map_in_range->setInputCloud(tmpCloud);
}

bool PrimitiveMatcher::check_distance_plane_2_pos_3D(const int map_plane_id, const Eigen::Vector3f& pos) {
    Eigen::Vector3f start = map_.planes[map_plane_id].edge_poly[3]; // down left
    Eigen::Vector3f end = map_.planes[map_plane_id].edge_poly[2]; // down right
    if( SqDistancePtSegment( start, end, pos ) < search_radius_squared_ )
    {
        return true;
    }
    else
    {
        return false;
    }
}
float PrimitiveMatcher::SqDistancePtSegment(Eigen::Vector3f a, Eigen::Vector3f b, Eigen::Vector3f p) {
    Eigen::Vector3f n = b - a;
    Eigen::Vector3f pa = a - p;

    float c = n.dot( pa );

    // Closest point is a
    if ( c > 0.0f )
        return pa.dot( pa );

    Eigen::Vector3f bp = p - b;

    // Closest point is b
    if ( n.dot( bp ) > 0.0f )
        return bp.dot( bp );

    // Closest point is between a and b
    Eigen::Vector3f e = pa - n * (c / n.dot( n ));

    return e.dot( e );

}
bool PrimitiveMatcher::check_distance_cylinder_2_pos_3D(const int map_cylinder_id, const Eigen::Vector3f& pos) {
    // Use 3D position and intersection point of axis to find distance.
    if( (pos - map_.cylinders[map_cylinder_id].center_line.p_bottom).squaredNorm() < search_radius_squared_ )
    {
        return true;
    }
    else
    {
        return false;
    }

}

void PrimitiveMatcher::transformUpdate() {
    // calculated transform will be initial guess for next process
    q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
    t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void PrimitiveMatcher::transformAssociateToMap() {
    // Optima para is transformer between map and odom
    q_w_curr = q_wmap_wodom * q_wodom_curr;
    t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}
void PrimitiveMatcher::detect_primitives_to_map(const Eigen::Vector3f * const pi, Eigen::Vector3f* const po) {
    Eigen::Vector3d point_curr = pi->cast<double>();
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x() = point_w.x();
    po->y() = point_w.y();
    po->z() = point_w.z();
}
void PrimitiveMatcher::transformPlaneToMap(Plane& detected_plane, Plane& mapped_plane) {
    Plane plane_curr = detected_plane;

    mapped_plane.edge_poly.resize(4);
    mapped_plane.edge_poly[0] = (q_w_curr * plane_curr.edge_poly[0].cast<double>() + t_w_curr).cast<float>();
    mapped_plane.edge_poly[1] = (q_w_curr * plane_curr.edge_poly[1].cast<double>() + t_w_curr).cast<float>();
    mapped_plane.edge_poly[2] = (q_w_curr * plane_curr.edge_poly[2].cast<double>() + t_w_curr).cast<float>();
    mapped_plane.edge_poly[3] = (q_w_curr * plane_curr.edge_poly[3].cast<double>() + t_w_curr).cast<float>();

}

void PrimitiveMatcher::match() {

    // whole process:
    //      1. read map txt
    //      2. set transform from odo
    //      3. set current detection
    //          a. put poles bottom in tree
    //          b. find local range map
    //          c. find association of two primitive features
    //          d. association put in optimizer to get fine pose

    // First load map

    read_map_file(mapPolesPath, mapFacadesPath, mapGroundPath);

    // Second get sub map and put in the kd tree
    TicToc t_map_matching;
    getSubMap();
    /// test submap
//
//    Cloud::Ptr testCloud1(new Cloud());
//    Cloud::Ptr testCloud2(new Cloud());
//
//
//
//    getPrimitiveFacadeCloud(testCloud1, map_.planes, 100);
//    getPrimitivePoleCloud(testCloud1, map_.cylinders, 100);
//
//    for (int i = 0; i < testCloud1->points.size(); ++i) {
//        Point p1 = testCloud1->points[i];
//        for (int j = 0; j < testCloud2->points.size() ; ++j) {
//            Point p2 = testCloud2->points[j];
//            if (p1.x == p2.x) {
//                testCloud1->points[i].intensity = 300;
//            }
//        }
//    }


    // Then find association for current frame
    find_associations();

//    Cloud::Ptr testCloud(new Cloud());
//    std::vector<Plane> detectedPlane;
//    std::vector<Plane> mapPlane;
//    for (int i = 0; i < association_.plane_asso.size(); ++i) {
//        int detectedIndex = association_.plane_asso[i].first;
//        int mapIndex = association_.plane_asso[i].second;
//
//        detectedPlane.push_back(detections_.planes[detectedIndex]);
//        mapPlane.push_back(subMap_.planes[mapIndex]);
//    }
//    std::vector<Cylinder> detectedPoles;
//    std::vector<Cylinder> map_Poles;
//    for (int i = 0; i < association_.cylinder_asso.size(); ++i) {
//        int dInd = association_.cylinder_asso[i].first;
//        int mInd = association_.cylinder_asso[i].second;
//        detectedPoles.push_back(detections_.cylinders[dInd]);
//        map_Poles.push_back(subMap_.cylinders[mInd]);
//        createTheoryLine(detections_.cylinders[dInd].center_line.p_bottom,
//                         subMap_.cylinders[mInd].center_line.p_bottom, testCloud, 20, 4000);
//    }

//    getPrimitivePoleCloud(testCloud, detectedPoles, 2000);
//    getPrimitivePoleCloud(testCloud, map_Poles, 1000);
//
//    pcl::io::savePCDFile("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/localizationResult/test_association_poles.pcd",
//                         *testCloud);

    // Finally optimize transform between odo and map
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *q_parameterization =
        new ceres::EigenQuaternionParameterization();
    ceres::Problem::Options problem_options;

    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_q, 4, q_parameterization);
    problem.AddParameterBlock(para_t, 3);

    std::cerr << "Pole Association Number in map match: " << association_.cylinder_asso.size() << std::endl;
    std::cerr << "Facade Association Number in map match: " << association_.plane_asso.size() << std::endl;
    std::cerr << "Ground Association Number in map match: " << association_.ground_asso.size() << std::endl;

    // Costs for associated cylinders.
    double weighting_factor_cylinder = cylinder_weighting_factor_;
    for (int i = 0; i < association_.cylinder_asso.size(); ++i) {
        if (association_.cylinder_asso.empty()) {
            break;
        }
        int detect_pole_id = association_.cylinder_asso[i].first;
        int map_pole_id = association_.cylinder_asso[i].second;

        Eigen::Vector3d detectBottomPoint = detections_.cylinders[detect_pole_id].center_line.p_bottom.cast<double>();

        Eigen::Vector3d mapBottomPoint = subMap_.cylinders[map_pole_id].center_line.p_bottom.cast<double>();


        ceres::CostFunction *cost_function = PoleFactor::Create(detectBottomPoint, mapBottomPoint);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
    }

    // Cost for associated planes.
    double weighting_factor_plane_distance = 2 * plane_distance_factor_; //0.53
    double weighting_factor_plane_direction = 2 * plane_direction_factor_; // 1.54
    double weighting_factor_plane_overlap = 2 * plane_overlap_factor_; // 0.6
    for (int i = 0; i < (int)association_.plane_asso.size(); ++i) {
        if (association_.plane_asso.empty()) {
            break;
        }
        int detect_plane_id = association_.plane_asso[i].first;
        int map_plane_id = association_.plane_asso[i].second;

        Eigen::Vector3d detectedLineDl = detections_.planes[detect_plane_id].edge_poly[0].cast<double>();  // input in residual
        Eigen::Vector3d detectedLineDr = detections_.planes[detect_plane_id].edge_poly[1].cast<double>(); // input in residual
        Eigen::Vector3d detectedLineTr = detections_.planes[detect_plane_id].edge_poly[2].cast<double>(); // input in residual



        Eigen::Vector3d mapLineDl = subMap_.planes[map_plane_id].edge_poly[0].cast<double>(); // input in residual
        Eigen::Vector3d mapLineDr = subMap_.planes[map_plane_id].edge_poly[1].cast<double>(); // input in residual
        Eigen::Vector3d mapLineTr = subMap_.planes[map_plane_id].edge_poly[2].cast<double>();

        Eigen::Vector3d mapFacadeNormal = (mapLineDr - mapLineDl).cross(mapLineTr - mapLineDl);
        Eigen::Vector3d mapFacadeNormalNormalized = mapFacadeNormal.normalized(); // input in residual


        ceres::CostFunction *cost_function = PlaneResidual::Create(mapLineDl, mapLineDr, mapFacadeNormalNormalized,
                                                                   detectedLineDl, detectedLineDr, detectedLineTr);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
    }
    for (int i = 0; i < (int)association_.ground_asso.size(); ++i) {
        if (association_.ground_asso.empty()) {
            break;
        }
        int detect_ground_id = association_.ground_asso[i].first;
        int map_ground_id = association_.ground_asso[i].second;

        Eigen::Vector3d detectedLineDl = detections_.grounds[detect_ground_id].edge_poly[0].cast<double>();  // input in residual
        Eigen::Vector3d detectedLineDr = detections_.grounds[detect_ground_id].edge_poly[1].cast<double>(); // input in residual
        Eigen::Vector3d detectedLineTr = detections_.grounds[detect_ground_id].edge_poly[2].cast<double>(); // input in residual



        Eigen::Vector3d mapLineDl = subMap_.grounds[map_ground_id].edge_poly[0].cast<double>(); // input in residual
        Eigen::Vector3d mapLineDr = subMap_.grounds[map_ground_id].edge_poly[1].cast<double>(); // input in residual
        Eigen::Vector3d mapLineTr = subMap_.grounds[map_ground_id].edge_poly[2].cast<double>();

        Eigen::Vector3d mapGroundNormal = (mapLineDr - mapLineDl).cross(mapLineTr - mapLineDl);
        Eigen::Vector3d mapGroundNormalNormalized = mapGroundNormal.normalized(); // input in residual


        ceres::CostFunction *cost_function = GroundResidual::Create(mapLineDl, mapLineDr, mapGroundNormalNormalized,
                                                                   detectedLineDl, detectedLineDr, detectedLineTr);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
    }

    // Solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = ceres_solver_max_num_iterations_;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.BriefReport() << std::endl;

    transformUpdate();
    std::cerr << "map matching time: " << t_map_matching.toc() << "ms" << std::endl;


}
float PrimitiveMatcher::calPolesMatchingCost(const Cylinder mapPole, const Cylinder curPole) {
    // transformer to map frame
    Eigen::Vector3f cur_center = curPole.center_line.p_bottom;
    Eigen::Vector3f cur_top = curPole.center_line.p_top;

    detect_primitives_to_map(&cur_center, &cur_center);
    detect_primitives_to_map(&cur_top, &cur_top);
    Eigen::Vector3f cur_axis_dir = cur_center - cur_top;

    // cal the difference of detected pole and pole in map
    float difRadius = std::abs(mapPole.radius - curPole.radius);
    if (difRadius > max_cylinder_association_radius_diff_) {
        difRadius = 999;
    }

    Eigen::Vector3f map_pole_line_dir_normalized = (mapPole.center_line.p_bottom - mapPole.center_line.p_top).normalized();
    Eigen::Vector3f cur_pole_line_dir_normalized = cur_axis_dir.normalized();

    float difAngel = 1 - std::abs(map_pole_line_dir_normalized.dot(cur_pole_line_dir_normalized));

    if (difAngel > max_cylinder_association_angle_diff_) {
        difAngel = 999;
    }
    float difDistance = (mapPole.center_line.p_bottom - cur_center).norm();
    if (difDistance > max_cylinder_association_pos_diff_) {
        difDistance = 999;
    }
    return 5.0 * difAngel + difDistance + difRadius;
}
float PrimitiveMatcher::planeToPlaneDistanceCost(const Plane& plane1, const Plane& plane2) {
    // plane1 parameter
    Eigen::Vector3f p1_dl = plane1.edge_poly[0];
    Eigen::Vector3f p1_dr = plane1.edge_poly[1];
    Eigen::Vector3f p1_tr = plane1.edge_poly[2];
    Eigen::Vector3f p1_center = p1_dl + 0.5 * (p1_tr - p1_dl);
    Eigen::Vector3f p1_normal_normalized = ( (p1_dr - p1_dl).cross(p1_tr - p1_dl) ).normalized();

    // plane2 parameter
    Eigen::Vector3f p2_dl = plane2.edge_poly[0];
    Eigen::Vector3f p2_dr = plane2.edge_poly[1];
    Eigen::Vector3f p2_tr = plane2.edge_poly[2];
    Eigen::Vector3f p2_center = p2_dl + 0.5 * (p2_tr - p2_dl);
    Eigen::Vector3f p2_normal_normalized = ( (p2_dr - p2_dl).cross(p2_tr - p2_dl) ).normalized();

    Eigen::Vector3f p2_center_p1_dl = p1_dl - p2_center;

    float dist_diff = std::abs(p1_normal_normalized.dot(p2_center_p1_dl));

    return dist_diff;
}
float PrimitiveMatcher::calFacadesMatchingCost(const Plane mapFacade, const Plane curFacade) {
    // transform to map frame
    Eigen::Vector3f cur_fa_dl = curFacade.edge_poly[0];
    Eigen::Vector3f cur_fa_dr = curFacade.edge_poly[1];
    Eigen::Vector3f cur_fa_tr = curFacade.edge_poly[2];

    Eigen::Vector3f cur_fa_dl_tr = cur_fa_tr - cur_fa_dl;
    Eigen::Vector3f cur_fa_dl_dr = cur_fa_dr - cur_fa_dl;

    // cur center and normal
    Eigen::Vector3f cur_fa_center = cur_fa_dl + 0.5 * cur_fa_dl_tr;
    Eigen::Vector3f cur_fa_normal = (cur_fa_dl_dr.cross(cur_fa_dl_tr)).normalized();
    // map line in base frame
    Eigen::Vector3f base_fa_dl = mapFacade.edge_poly[0]; // dl
    Eigen::Vector3f base_fa_dr = mapFacade.edge_poly[1]; // dr
    Eigen::Vector3f base_fa_tr = mapFacade.edge_poly[2]; // tr

    Eigen::Vector3f base_dl_tr = base_fa_tr - base_fa_dl;
    Eigen::Vector3f base_dl_dr = base_fa_dr - base_fa_dl;
    Eigen::Vector3f base_fa_center = base_fa_dl + 0.5 * (base_fa_tr - base_fa_dl);
    // base normal
    Eigen::Vector3f base_fa_normal = (base_dl_dr.cross(base_dl_tr)).normalized();

    Eigen::Vector3f base_fa_dl_cur_center = cur_fa_center - base_fa_dl;

    // Calculate transformed center to the plane in base

    float distance_dif = std::abs(base_fa_normal.dot(base_fa_dl_cur_center));
    // Calculate direction difference
    float direction_dif = (cur_fa_normal.cross(base_fa_normal)).norm();

//    std::cout << "distance: " << distance_dif << "direction: " << direction_dif << std::endl;
    float cost = distance_dif + 10 * direction_dif;
    if (direction_dif > 0.17) {
        cost += 9999;
    }
//    if (distance_dif > 3.0) {
//        cost += 999;
//    }
    return cost;

}

float PrimitiveMatcher::planeToPlaneMatchingCost(const Plane& plane1, const Plane& plane2) { // plane1 : plane in map

                                                                             // plane2 : plane in detection and transformed
    // plane1 parameter
    Eigen::Vector3f cur_fa_dl = plane2.edge_poly[0];
    Eigen::Vector3f cur_fa_dr = plane2.edge_poly[1];
    Eigen::Vector3f cur_fa_tr = plane2.edge_poly[2];

    Eigen::Vector3f cur_fa_dl_tr = cur_fa_tr - cur_fa_dl;
    Eigen::Vector3f cur_fa_dl_dr = cur_fa_dr - cur_fa_dl;

    // cur center and normal
    Eigen::Vector3f cur_fa_center = cur_fa_dl + 0.5 * cur_fa_dl_tr;
    Eigen::Vector3f cur_fa_normal = (cur_fa_dl_dr.cross(cur_fa_dl_tr)).normalized();
    // map line in base frame
    Eigen::Vector3f base_fa_dl = plane1.edge_poly[0]; // dl
    Eigen::Vector3f base_fa_dr = plane1.edge_poly[1]; // dr
    Eigen::Vector3f base_fa_tr = plane1.edge_poly[2]; // tr

    Eigen::Vector3f base_dl_tr = base_fa_tr - base_fa_dl;
    Eigen::Vector3f base_dl_dr = base_fa_dr - base_fa_dl;
    // base normal
    Eigen::Vector3f base_fa_normal = (base_dl_dr.cross(base_dl_tr)).normalized();

    Eigen::Vector3f base_fa_dl_cur_center = cur_fa_center - base_fa_dl;

    // Calculate transformed center to the plane in base

    float distance_dif = std::abs(base_fa_normal.dot(base_fa_dl_cur_center));
    // Calculate direction difference
    float direction_dif = (cur_fa_normal.cross(base_fa_normal)).norm();

//    std::cout << "distance: " << distance_dif << "direction: " << direction_dif << std::endl;
    float cost = distance_dif + 10 * direction_dif;
    if (direction_dif > 0.13) {
        cost += 999;
    }
//    if (distance_dif > 0.2) {
//        cost += 999;
//    }
//    if (std::abs(cur_fa_dl[2] - base_fa_dl[2]) > 0.2) {
//        cost += 999;
//    }
    return cost;
}

// Find the best association for facades and pole in current pose from odo and primitives in ranged map
void PrimitiveMatcher::find_associations() {

    transformAssociateToMap();

    // Find association for each detected pole a the similar pole in map.

    for (int i = 0; i < (int)detections_.cylinders.size(); ++i) {
        if (detections_.cylinders.empty()) {
            std::cerr << "No poles detected skip poles association" << std::endl;
            break;
        }
        pole_center = detections_.cylinders[i].center_line.p_bottom;
        // transform detect to map frame
        detect_primitives_to_map(&pole_center, &pole_center_transformed);
        Point tmpP;
        tmpP.x = pole_center_transformed[0];
        tmpP.y = pole_center_transformed[1];
        tmpP.z = pole_center_transformed[2];
        tmpP.intensity = 2000; // associated point has intensity 2000
        kdtreePoleFromMap->radiusSearch(tmpP, 2., pointSearchInd, pointSearchSqDis);



        if (pointSearchInd.size() == 0) {
            continue;
        }
        float machCost = 9999;
        float costThreshold = 999;
        int closeIndex; // Store the best candidate index of pole in map.
        bool valid_pole = false;

        // select the pole with the best match cost
        for (auto& poleIndexInSubMap : pointSearchInd) {
            // select the best pole around

            float cost = calPolesMatchingCost(subMap_.cylinders[poleIndexInSubMap], detections_.cylinders[i]);
//            std::cerr << "cost: " << cost << std::endl;
//            if (cost > costThreshold) {
//                continue;
//            }
            if (cost < machCost) {
                machCost = cost;
                closeIndex = poleIndexInSubMap;
                valid_pole = true;
            }
        }
        if (valid_pole && machCost < costThreshold) {
            association_.cylinder_asso.push_back(std::make_pair(i, closeIndex)); // first detect index. second index in map
        }
    }
//    std::cerr << "Pole correspondences in map before filter: " << association_.cylinder_asso.size() << std::endl;

    // Filter out outliers
    // get average dis and filter out location outliers
    if (!association_.cylinder_asso.empty()) {
        double middleDistance = 0;
        int costCount = 0;
        std::vector<double> distanceDiff;
        std::vector<double> zValueDiff;
        distanceDiff.resize(association_.cylinder_asso.size());
        zValueDiff.resize(association_.cylinder_asso.size());
        double middleZ = 0;
        for (int i = 0; i < association_.cylinder_asso.size(); ++i) {
            Eigen::Vector3d curPoint = detections_.cylinders[association_.cylinder_asso[i].first].center_line.p_bottom.cast<double>();
            Eigen::Vector3d prePoint = subMap_.cylinders[association_.cylinder_asso[i].second].center_line.p_bottom.cast<double>();
            double distance = (curPoint - prePoint).norm();
            double difZ = std::abs(curPoint[2] - prePoint[2]);
            distanceDiff[i] = distance;
            zValueDiff[i] = difZ;
            costCount++;
        }
        std::partial_sort(distanceDiff.begin(), distanceDiff.begin() + distanceDiff.size() / 2 + 1, distanceDiff.end());
        std::partial_sort(zValueDiff.begin(), zValueDiff.begin() + zValueDiff.size() / 2 + 1, zValueDiff.end());
        int mid = association_.cylinder_asso.size() / 2;
//        std::cout << mid << std::endl;
        middleDistance = distanceDiff[mid];
        middleZ = zValueDiff[mid];
//    std::cerr << "middleDistance : " << middleDistance << std::endl;
//    std::cerr << "middleZ : " << middleZ << std::endl;

        std::vector<std::pair<int, int>> tmpCylinderAs;
        for (int i = 0; i < association_.cylinder_asso.size(); ++i) {
            // The associated bottom point and axis direction
            int lastPoleIndex = association_.cylinder_asso[i].second;
            int currentPoleIndex = association_.cylinder_asso[i].first;

            Eigen::Vector3d prePoint = subMap_.cylinders[lastPoleIndex].center_line.p_bottom.cast<double>();
            Eigen::Vector3d preDir = subMap_.cylinders[lastPoleIndex].line_dir.cast<double>();
            Eigen::Vector3d preTopPoint = subMap_.cylinders[lastPoleIndex].center_line.p_top.cast<double>();

            Eigen::Vector3d curPoint = detections_.cylinders[currentPoleIndex].center_line.p_bottom.cast<double>();
            Eigen::Vector3d curDir = detections_.cylinders[currentPoleIndex].line_dir.cast<double>();
            Eigen::Vector3d curTopPoint = detections_.cylinders[currentPoleIndex].center_line.p_top.cast<double>();

            double distance = (curPoint - prePoint).norm();
            double verticalDiff = std::abs(curPoint[2] - prePoint[2]);
//        std::cerr << verticalDiff << std::endl;
            if (distance < 1.2 * middleDistance && verticalDiff < 1.2 * middleZ) { // filter out location outliers
                //association_.cylinder_asso.erase(association_.cylinder_asso.begin() + i);
                //continue;
                tmpCylinderAs.push_back(std::make_pair(currentPoleIndex, lastPoleIndex));
            }
        }
        association_.cylinder_asso = tmpCylinderAs;
    }
//    std::cerr << "Pole correspondences in map: " << association_.cylinder_asso.size() << std::endl;

    // Find association for each detected facade a the similar facade in map.
    for (int i = 0; i < (int)detections_.planes.size(); ++i) {
        if (detections_.planes.empty()) {
            std::cerr << "No facades detected skip facades association" << std::endl;
            break;
        }
        // current frame detected facade center
//        facade_center = detections_.planes[i].edge_poly[0]
//                                        + 0.5 * (detections_.planes[i].edge_poly[2] - detections_.planes[i].edge_poly[0]);
//        detect_primitives_to_map(&facade_center, &facade_center_transformed);
//        Point tmpP;
//        tmpP.x = facade_center_transformed[0];
//        tmpP.y = facade_center_transformed[1];
//        tmpP.z = facade_center_transformed[2];
//        tmpP.intensity = 2000; // associated point has intensity 2000
//        kdtreeFacadeFromMap->radiusSearch(tmpP, 10.0, pointSearchInd, pointSearchSqDis);
//
//        if (pointSearchInd.size() == 0) {
//            continue;
//        }
//        float machCost = 9999;
//        int index_facade;
//        float costThreshold = 2.0;
//        bool valid_facade = false;
//        // Search the facade with the smallest match cost
//        for (auto& closeFacadeIndex : pointSearchInd) {
//            // detected plane line
//            Plane detected_plane = detections_.planes[i];
//            // transformed line
//            Plane detected_plane_in_map;
//            transformPlaneToMap(detected_plane, detected_plane_in_map);
//            // map line
//            Plane map_plane = subMap_.planes[closeFacadeIndex];
//
//            float cost = planeToPlaneMatchingCost(map_plane, detected_plane_in_map);
//
////            if (cost > costThreshold) {
////                continue;
////            }
//            if (cost < machCost) {
//                machCost = cost;
//                index_facade = closeFacadeIndex;
//                valid_facade = true;
//            }
//
//
//        }
//        if (valid_facade) {
//            association_.plane_asso.push_back(std::make_pair(i, index_facade)); // first detect facade index.
//                                                                                // second map facade index.
//        }
        float cost_facade = 9999;
        int index_facade;
        float facade_threshold = 999;
        bool valid_facade = false;
        for (int indexLastFacade = 0; indexLastFacade < subMap_.planes.size(); ++indexLastFacade) {
            // last frame facades
            Eigen::Vector3f plane_pos_start_last(subMap_.planes[indexLastFacade].edge_poly[0]); // dl
            Eigen::Vector3f plane_pos_end_last(subMap_.planes[indexLastFacade].edge_poly[1]); // dr
            // current frame facades
            Eigen::Vector3f plane_pos_start_cur(detections_.planes[i].edge_poly[0]);
            Eigen::Vector3f plane_pos_end_cur(detections_.planes[i].edge_poly[1]);
            // Transform to last frame
            detect_primitives_to_map(&plane_pos_start_cur, &plane_pos_start_cur);
            detect_primitives_to_map(&plane_pos_end_cur, &plane_pos_end_cur);


            // calculate distance between line segmentation
            float dist = dist3D_Segment_to_Segment(plane_pos_start_last.cast<float>(), plane_pos_end_last.cast<float>(),
                                                   plane_pos_start_cur.cast<float>(), plane_pos_end_cur.cast<float>());
//            std::cout << "dist: " << dist << std::endl;
            if (dist < 2.0) { // Segment distance less than 2 m
                Plane cur_plane_map;
                transformPlaneToMap(detections_.planes[i], cur_plane_map);
                float cost = calFacadesMatchingCost(subMap_.planes[indexLastFacade], cur_plane_map);
//                std::cerr << "cost facade: " << cost << std::endl;
                if (cost < cost_facade) {
                    cost_facade = cost;
                    index_facade = indexLastFacade;
                    valid_facade = true;
                }
            }
        }
//        std::cerr << "cost_facade in map: " << cost_facade << std::endl;
        if (valid_facade && cost_facade < facade_threshold) {

            association_.plane_asso.push_back(std::make_pair(i, index_facade));
        }
    }
//    std::cerr << "Facade correspondences in map: " << association_.plane_asso.size() << std::endl;

    // Find association for each detected ground a the similar ground in map.
    for (int i = 0; i < (int)detections_.grounds.size(); ++i) {
        if (detections_.grounds.empty()) {
            std::cerr << "No grounds detected skip grounds association" << std::endl;
            break;
        }
        // current frame detected grounds center
        ground_center = detections_.grounds[i].edge_poly[0]
                        + 0.5 * (detections_.grounds[i].edge_poly[2] - detections_.grounds[i].edge_poly[0]);
        detect_primitives_to_map(&ground_center, &ground_center_transformed);
        Point tmpP;
        tmpP.x = ground_center_transformed[0];
        tmpP.y = ground_center_transformed[1];
        tmpP.z = ground_center_transformed[2];
        tmpP.intensity = 2000; // associated point has intensity 2000
        kdtreeGroundFromMap->nearestKSearch(tmpP, 5, pointSearchInd, pointSearchSqDis);

        if (pointSearchInd.size() == 0) {
            continue;
        }
        float machCost = 9999;
        int index_ground;
        float costThreshold = 999;
        bool valid_ground = false;
        // Search the ground with the smallest match cost
        for (auto& closeGroundIndex : pointSearchInd) {
            // detected plane line
            Plane detected_ground = detections_.grounds[i];
            // transformed line
            Plane detected_ground_in_map;
            transformPlaneToMap(detected_ground, detected_ground_in_map);
            // map line
            Plane map_ground = subMap_.grounds[closeGroundIndex];

            float cost = planeToPlaneMatchingCost(map_ground, detected_ground_in_map);

//            if (cost > costThreshold) {
//                continue;
//            }
            if (cost < machCost) {
                machCost = cost;
                index_ground = closeGroundIndex;
                valid_ground = true;
            }

        }
        if (valid_ground && machCost < costThreshold) {
            association_.ground_asso.push_back(std::make_pair(i, index_ground)); // first detect facade index.
            // second map facade index.
        }
    }
//    std::cerr << "Ground correspondences in map: " << association_.ground_asso.size() << std::endl;
}

float PrimitiveMatcher::dist3D_Segment_to_Segment(Eigen::Vector3f seg_1_p0, Eigen::Vector3f seg_1_p1,
                                                  Eigen::Vector3f seg_2_p0, Eigen::Vector3f seg_2_p1) {
    Eigen::Vector2f u = seg_1_p1.block<2,1>(0,0) - seg_1_p0.block<2,1>(0,0);
    Eigen::Vector2f v = seg_2_p1.block<2,1>(0,0) - seg_2_p0.block<2,1>(0,0);
    Eigen::Vector2f w = seg_1_p0.block<2,1>(0,0) - seg_2_p0.block<2,1>(0,0);
    float SMALL_NUM = 0.0001;
    float    a = u.dot(u);         // always >= 0
    float    b = u.dot(v);
    float    c = v.dot(v);         // always >= 0
    float    d = u.dot(w);
    float    e = v.dot(w);
    float    D = a*c - b*b;        // always >= 0
    float    sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
    float    tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

    // compute the line parameters of the two closest points
    if (D < SMALL_NUM) { // the lines are almost parallel
        sN = 0.0;         // force using point P0 on segment S1
        sD = 1.0;         // to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    }
    else {                 // get the closest points on the infinite lines
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
        tN = 0.0;
        // recompute sc for this edge
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    }
    else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
        tN = tD;
        // recompute sc for this edge
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d +  b);
            sD = a;
        }
    }
    // finally do the division to get sc and tc
    sc = (std::abs(sN) < SMALL_NUM ? 0.0 : sN / sD);
    tc = (std::abs(tN) < SMALL_NUM ? 0.0 : tN / tD);

    // get the difference of the two closest points
    Eigen::Vector2f dP = w + (sc * u) - (tc * v);  // =  S1(sc) - S2(tc)

    return dP.norm();   // return the closest distance
}

void PrimitiveMatcher::createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point,
                                          Cloud::Ptr& cloud, int length, int color) {
    Eigen::Vector3f lineDir = end_point - start_point;
    int index = 0;
    std::vector<Eigen::Vector3f> pointVector;
    Eigen::Vector3f tmpPoint;
    int stepLength = length;
    float step = lineDir.norm() / stepLength;
    while (index < stepLength) {
        tmpPoint = start_point + step * index * lineDir / lineDir.norm();
        pointVector.push_back(tmpPoint);
        index++;
    }
    for (int i = 0; i < pointVector.size(); ++i) {
        Point p;
        p.x = pointVector[i][0];
        p.y = pointVector[i][1];
        p.z = pointVector[i][2];
        p.intensity = color;
        cloud->points.push_back(p);
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();
}
void PrimitiveMatcher::createTheoryCylinder(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, const float R,
                                              Cloud::Ptr& cloud, int numCyl, int length, int color) {
    Eigen::Vector3f lineDir = end_point - start_point;
    int index = 0;
    Eigen::Vector3f axisPoint;
    float step = lineDir.norm() / numCyl;
    // Axis
    Point center;
    while (index < length) {
        axisPoint = start_point + step * index * lineDir / lineDir.norm();
        center.x = axisPoint[0];
        center.y = axisPoint[1];
        center.z = axisPoint[2];
        center.intensity = color;
        createTheoryCircle(lineDir, axisPoint, R, cloud, color);
        cloud->points.push_back(center);
        index++;
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();

}
void PrimitiveMatcher::createTheoryCircle(Eigen::Vector3f& normal, Eigen::Vector3f& center, float R, Cloud::Ptr& cloud, int color) {

    double nx = normal[0], ny = normal[1], nz = normal[2];
    double cx = center[0], cy = center[1], cz = center[2];
    double r = R;

    double ux = ny, uy = -nx, uz = 0;
    double vx = nx*nz,
        vy = ny*nz,
        vz = -nx*nx - ny*ny;

    double sqrtU = sqrt(ux*ux + uy*uy + uz*uz);
    double sqrtV = sqrt(vx*vx + vy*vy + vz*vz);

    double ux_ = (1 / sqrtU)*ux;
    double uy_ = (1 / sqrtU)*uy;
    double uz_ = (1 / sqrtU)*uz;

    double vx_ = (1 / sqrtV)*vx;
    double vy_ = (1 / sqrtV)*vy;
    double vz_ = (1 / sqrtV)*vz;

    double xi, yi, zi;
    double t = 0;
    double angle = (t / 180.0)*M_PI;
    vector<double> x, y, z;

    while (t < 360.0)
    {
        xi = cx + r*(ux_*cos(angle) + vx_*sin(angle));
        yi = cy + r*(uy_*cos(angle) + vy_*sin(angle));
        zi = cz + r*(uz_*cos(angle) + vz_*sin(angle));
        x.push_back(xi);
        y.push_back(yi);
        z.push_back(zi);

        t = t + 20;
        angle = (t / 180.0)*M_PI;
    }

    for (int i = 0; i < x.size(); i++){
        Point p;
        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        p.intensity = color;
        cloud->points.push_back(p);
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();
}
void PrimitiveMatcher::getPrimitivePoleCloud(Cloud::Ptr& cloud, std::vector<Cylinder>& Poles, int color) {
    for (auto& pole : Poles) {
        createTheoryCylinder(pole.center_line.p_bottom, pole.center_line.p_top, pole.radius, cloud, 20, 20, color);
    }
}
void PrimitiveMatcher::getPrimitiveFacadeCloud(Cloud::Ptr& cloud, std::vector<Plane>& Facades, int color) {
    for (auto& facade : Facades) {
        createTheoryLine(facade.edge_poly[0], facade.edge_poly[1], cloud, 50, color); // dl - > dr
        createTheoryLine(facade.edge_poly[1], facade.edge_poly[2], cloud, 50, color); // dr - > tr
        createTheoryLine(facade.edge_poly[2], facade.edge_poly[3], cloud, 50, color); // tr - > tl
        createTheoryLine(facade.edge_poly[3], facade.edge_poly[0], cloud, 50, color); // tl - > dl
        createTheoryLine(facade.edge_poly[2], facade.edge_poly[0], cloud, 50, color); // tr - > dl
        createTheoryLine(facade.edge_poly[1], facade.edge_poly[3], cloud, 50, color); // dr - > tl

    }
}
void PrimitiveMatcher::getAssociationMM(Cloud::Ptr cloud) {
    for (int i = 0; i < association_.cylinder_asso.size(); ++i) {
        int indexLastPole = association_.cylinder_asso[i].second;
        int indexCurPole = association_.cylinder_asso[i].first;
        Eigen::Vector3f last_p_bottom = subMap_.cylinders[indexLastPole].center_line.p_bottom;
        Eigen::Vector3f last_p_top = subMap_.cylinders[indexLastPole].center_line.p_top;
        float last_r = subMap_.cylinders[indexLastPole].radius;
        Eigen::Vector3f cur_p_bottom = detections_.cylinders[indexCurPole].center_line.p_bottom;
        Eigen::Vector3f cur_p_top = detections_.cylinders[indexCurPole].center_line.p_top;
        float cur_r = detections_.cylinders[indexCurPole].radius;
        detect_primitives_to_map(&cur_p_top, &cur_p_top);
        detect_primitives_to_map(&cur_p_bottom, &cur_p_bottom);
        // last pole
        createTheoryCylinder(last_p_bottom, last_p_top, last_r, cloud, 10, 10, 5000);
        // current pole
        createTheoryCylinder(cur_p_bottom, cur_p_top, cur_r, cloud, 10, 10, 2000);
        // association line
        createTheoryLine(last_p_bottom, cur_p_bottom, cloud, 1000, 10000);
    }
    for (int i = 0; i < association_.plane_asso.size(); ++i) {
        int indexLastFacade = association_.plane_asso[i].second;
        int indexCurrentFacade = association_.plane_asso[i].first;
        Eigen::Vector3f last_dl = subMap_.planes[indexLastFacade].edge_poly[0];
        Eigen::Vector3f last_dr = subMap_.planes[indexLastFacade].edge_poly[1];
        Eigen::Vector3f last_tl = subMap_.planes[indexLastFacade].edge_poly[3];
        Eigen::Vector3f last_tr = subMap_.planes[indexLastFacade].edge_poly[2];

        Eigen::Vector3f last_center = 0.5 * (last_tr - last_dl);
        // last facade
        createTheoryLine(last_dl, last_dr, cloud,100, 5000);
        createTheoryLine(last_dr, last_tr, cloud,100, 5000);
        createTheoryLine(last_tr, last_tl, cloud,100, 5000);
        createTheoryLine(last_tl, last_dl, cloud,100, 5000);
        createTheoryLine(last_tl, last_dr, cloud,100, 5000);
        createTheoryLine(last_tr, last_dl, cloud,100, 5000);

        Eigen::Vector3f cur_dl = detections_.planes[indexCurrentFacade].edge_poly[0];
        Eigen::Vector3f cur_dr = detections_.planes[indexCurrentFacade].edge_poly[1];
        Eigen::Vector3f cur_tl = detections_.planes[indexCurrentFacade].edge_poly[3];
        Eigen::Vector3f cur_tr = detections_.planes[indexCurrentFacade].edge_poly[2];

        detect_primitives_to_map(&cur_dl, &cur_dl);
        detect_primitives_to_map(&cur_dr, &cur_dr);
        detect_primitives_to_map(&cur_tl, &cur_tl);
        detect_primitives_to_map(&cur_tr, &cur_tr);

        Eigen::Vector3f cur_center = 0.5 * (cur_tr - cur_dl) + cur_dl;
        // current facade
        createTheoryLine(cur_dl, cur_dr, cloud,100, 1000);
        createTheoryLine(cur_dr, cur_tr, cloud,100, 1000);
        createTheoryLine(cur_tr, cur_tl, cloud,100, 1000);
        createTheoryLine(cur_tl, cur_dl, cloud,100, 1000);
        createTheoryLine(cur_tl, cur_dr, cloud,100, 1000);
        createTheoryLine(cur_tr, cur_dl, cloud,100, 1000);
        // association line
        createTheoryLine(cur_center, last_dl, cloud, 200, 20000);
        createTheoryLine(cur_center, last_tr, cloud, 200, 30000);
        createTheoryLine(cur_center, last_dr, cloud, 200, 40000);

    }
    for (int i = 0; i < association_.ground_asso.size(); ++i) {
        int indexLastFacade = association_.ground_asso[i].second;
        int indexCurrentFacade = association_.ground_asso[i].first;
        Eigen::Vector3f last_dl = subMap_.grounds[indexLastFacade].edge_poly[0];
        Eigen::Vector3f last_dr = subMap_.grounds[indexLastFacade].edge_poly[1];
        Eigen::Vector3f last_tl = subMap_.grounds[indexLastFacade].edge_poly[3];
        Eigen::Vector3f last_tr = subMap_.grounds[indexLastFacade].edge_poly[2];

        Eigen::Vector3f last_center = 0.5 * (last_tr - last_dl);
        // last facade
        createTheoryLine(last_dl, last_dr, cloud,100, 5000);
        createTheoryLine(last_dr, last_tr, cloud,100, 5000);
        createTheoryLine(last_tr, last_tl, cloud,100, 5000);
        createTheoryLine(last_tl, last_dl, cloud,100, 5000);
        createTheoryLine(last_tl, last_dr, cloud,100, 5000);
        createTheoryLine(last_tr, last_dl, cloud,100, 5000);

        Eigen::Vector3f cur_dl = detections_.grounds[indexCurrentFacade].edge_poly[0];
        Eigen::Vector3f cur_dr = detections_.grounds[indexCurrentFacade].edge_poly[1];
        Eigen::Vector3f cur_tl = detections_.grounds[indexCurrentFacade].edge_poly[3];
        Eigen::Vector3f cur_tr = detections_.grounds[indexCurrentFacade].edge_poly[2];

        detect_primitives_to_map(&cur_dl, &cur_dl);
        detect_primitives_to_map(&cur_dr, &cur_dr);
        detect_primitives_to_map(&cur_tl, &cur_tl);
        detect_primitives_to_map(&cur_tr, &cur_tr);

        Eigen::Vector3f cur_center = 0.5 * (cur_tr - cur_dl) + cur_dl;
        // current facade
        createTheoryLine(cur_dl, cur_dr, cloud,100, 2500);
        createTheoryLine(cur_dr, cur_tr, cloud,100, 2500);
        createTheoryLine(cur_tr, cur_tl, cloud,100, 2500);
        createTheoryLine(cur_tl, cur_dl, cloud,100, 2500);
        createTheoryLine(cur_tl, cur_dr, cloud,100, 2500);
        createTheoryLine(cur_tr, cur_dl, cloud,100, 2500);
        // association line
        createTheoryLine(cur_center, last_dl, cloud, 200, 20000);
        createTheoryLine(cur_center, last_tr, cloud, 200, 30000);
        createTheoryLine(cur_center, last_dr, cloud, 200, 40000);

    }
    // add all primitive
//    for (int i = 0; i < subMap_.cylinders.size(); ++i) {
//        Eigen::Vector3f p_b = subMap_.cylinders[i].center_line.p_bottom;
//        Eigen::Vector3f p_t = subMap_.cylinders[i].center_line.p_top;
//        float r = subMap_.cylinders[i].radius;
//        createTheoryCylinder(p_b, p_t, r, cloud, 10, 10, 100);
//
//    }
//    for (int i = 0; i < detections_.cylinders.size(); ++i) {
//        Eigen::Vector3f p_b = detections_.cylinders[i].center_line.p_bottom;
//        Eigen::Vector3f p_t = detections_.cylinders[i].center_line.p_top;
//        float r = detections_.cylinders[i].radius;
//        createTheoryCylinder(p_b, p_t, r, cloud, 10, 10, 200);
//
//    }
//    for (int i = 0; i < subMap_.planes.size(); ++i) {
//        Eigen::Vector3f last_dl = subMap_.planes[i].edge_poly[0];
//        Eigen::Vector3f last_dr = subMap_.planes[i].edge_poly[1];
//        Eigen::Vector3f last_tl = subMap_.planes[i].edge_poly[3];
//        Eigen::Vector3f last_tr = subMap_.planes[i].edge_poly[2];
//        // last facade
//        createTheoryLine(last_dl, last_dr, cloud,10, 500);
//        createTheoryLine(last_dr, last_tr, cloud,10, 500);
//        createTheoryLine(last_tr, last_tl, cloud,10, 500);
//        createTheoryLine(last_tl, last_dl, cloud,10, 500);
//    }
//    for (int i = 0; i < detections_.planes.size(); ++i) {
//        Eigen::Vector3f cur_dl = detections_.planes[i].edge_poly[0];
//        Eigen::Vector3f cur_dr = detections_.planes[i].edge_poly[1];
//        Eigen::Vector3f cur_tl = detections_.planes[i].edge_poly[3];
//        Eigen::Vector3f cur_tr = detections_.planes[i].edge_poly[2];
//        // last facade
//        createTheoryLine(cur_dl, cur_dr, cloud,10, 1000);
//        createTheoryLine(cur_dr, cur_tr, cloud,10, 1000);
//        createTheoryLine(cur_tr, cur_tl, cloud,10, 1000);
//        createTheoryLine(cur_tl, cur_dl, cloud,10, 1000);
//
//    }
//    for (int i = 0; i < subMap_.grounds.size(); ++i) {
//        Eigen::Vector3f last_dl = subMap_.grounds[i].edge_poly[0];
//        Eigen::Vector3f last_dr = subMap_.grounds[i].edge_poly[1];
//        Eigen::Vector3f last_tl = subMap_.grounds[i].edge_poly[3];
//        Eigen::Vector3f last_tr = subMap_.grounds[i].edge_poly[2];
//        // last facade
//        createTheoryLine(last_dl, last_dr, cloud,10, 2000);
//        createTheoryLine(last_dr, last_tr, cloud,10, 2000);
//        createTheoryLine(last_tr, last_tl, cloud,10, 2000);
//        createTheoryLine(last_tl, last_dl, cloud,10, 2000);
//    }
//    for (int i = 0; i < detections_.grounds.size(); ++i) {
//        Eigen::Vector3f cur_dl = detections_.grounds[i].edge_poly[0];
//        Eigen::Vector3f cur_dr = detections_.grounds[i].edge_poly[1];
//        Eigen::Vector3f cur_tl = detections_.grounds[i].edge_poly[3];
//        Eigen::Vector3f cur_tr = detections_.grounds[i].edge_poly[2];
//        // last facade
//        createTheoryLine(cur_dl, cur_dr, cloud,10, 2500);
//        createTheoryLine(cur_dr, cur_tr, cloud,10, 2500);
//        createTheoryLine(cur_tr, cur_tl, cloud,10, 2500);
//        createTheoryLine(cur_tl, cur_dl, cloud,10, 2500);
//    }
}
void PrimitiveMatcher::getMapPrimitives(Primitives& primitivesInMap) {
    for (int i = 0; i < association_.cylinder_asso.size(); i++) {
        int mapCylinderIndex = association_.cylinder_asso[i].second;

        primitivesInMap.cylinders.push_back(subMap_.cylinders[mapCylinderIndex]);
    }
    for (int i = 0; i < association_.plane_asso.size(); i++) {
        int mapPlaneIndex = association_.plane_asso[i].second;

        primitivesInMap.planes.push_back(subMap_.planes[mapPlaneIndex]);
    }
    for (int i = 0; i < association_.ground_asso.size(); i++) {
        int mapGroundIndex = association_.ground_asso[i].second;

        primitivesInMap.grounds.push_back(subMap_.grounds[mapGroundIndex]);
    }
}
void PrimitiveMatcher::getMapPrimitives(Cloud::Ptr& primitivesInMap, int color) {
    std::vector<Plane> planes, grounds;
    std::vector<Cylinder> poles;
    for (int i = 0; i < association_.cylinder_asso.size(); i++) {
        int mapCylinderIndex = association_.cylinder_asso[i].second;
        poles.push_back(subMap_.cylinders[mapCylinderIndex]);
    }
    getPrimitivePoleCloud(primitivesInMap, poles, color * 4);
    for (int i = 0; i < association_.plane_asso.size(); i++) {
        int mapPlaneIndex = association_.plane_asso[i].second;
        planes.push_back(subMap_.planes[mapPlaneIndex]);
    }
    getPrimitiveFacadeCloud(primitivesInMap, planes, color * 2);
    for (int i = 0; i < association_.ground_asso.size(); i++) {
        int mapGroundIndex = association_.ground_asso[i].second;
        grounds.push_back(subMap_.grounds[mapGroundIndex]);
    }
    getPrimitiveFacadeCloud(primitivesInMap, grounds, color * 1);
    primitivesInMap->width = 1;
    primitivesInMap->height = primitivesInMap->points.size();
}






} // End of namespace primitivesMapMatcher
