//
// Created by zhao on 04.08.22.
//

#include "primitives_association/primitives_association.h"
#include "utility/tic_toc.h"

namespace primitivesFa {
PrimitivesAssociation::PrimitivesAssociation(const vector<Cylinder>& lastPoles, const vector<Plane>& lastFacades, const vector<Plane>& lastGrounds,
                                             const vector<Cylinder>& currentPoles, const vector<Plane>& currentFacades, const vector<Plane>& currentGrounds,
                                             Eigen::Quaterniond qCurr, Eigen::Vector3d tCurr,
                                             Eigen::Quaterniond q_last_curr_, Eigen::Vector3d t_last_curr_) {
    q_w_curr = qCurr;
    t_w_curr = tCurr;
    // initial guess from last transform
    q_last_curr = q_last_curr_;
    t_last_curr = t_last_curr_;

    initializationValue();
    prePoles = lastPoles;
    preFacades = lastFacades;
    preGrounds = lastGrounds;
    curPoles = currentPoles;
    curFacades = currentFacades;
    curGrounds = currentGrounds;

}
PrimitivesAssociation::PrimitivesAssociation(const pcl::PointCloud<Point>::Ptr oriCloudLast, // Cloud last frame
                                             const pcl::PointCloud<Point>::Ptr oriCloudCur, // Cloud current frame
                                             Eigen::Quaterniond qCurr, Eigen::Vector3d tCurr,
                                             Eigen::Quaterniond q_last_curr_, Eigen::Vector3d t_last_curr_) {

    q_w_curr = qCurr;
    t_w_curr = tCurr;
    // initial guess from last transform
    q_last_curr = q_last_curr_;
    t_last_curr = t_last_curr_;

    initializationValue();

    // get the para of pole and facades from extractor
    primitivesExtraction::PrimitiveExtractor PriExtractorLast, PriExtractorCurrent;
    PriExtractorLast.setInputCloud(oriCloudLast);
    PriExtractorLast.setRange(-200, 200,
                              -200, 200,
                              -2.5, 10);
    TicToc t_ex;
    PriExtractorLast.run();
    std::cerr << "feature extraction time: " << t_ex.toc() << "ms" << std::endl;
    PriExtractorLast.getPolesFinParam(prePoles);
    PriExtractorLast.getFacadesFinParam(preFacades);
    PriExtractorLast.getGroundFinParam(preGrounds);

    PriExtractorCurrent.setInputCloud(oriCloudCur);
    PriExtractorCurrent.setRange(-200, 200,
                              -200, 200,
                              -2.5, 10);
    PriExtractorCurrent.run();
    PriExtractorCurrent.getPolesFinParam(curPoles);
    PriExtractorCurrent.getFacadesFinParam(curFacades);
    PriExtractorCurrent.getGroundFinParam(curGrounds);


}

void PrimitivesAssociation::initializationValue() {
    prePoles.clear();
    preFacades.clear();
    preGrounds.clear();
    curPoles.clear();
    curFacades.clear();
    curGrounds.clear();

    kdTreePolesLast.reset(new pcl::KdTreeFLANN<Point>());
    kdTreeFacadesLast.reset(new pcl::KdTreeFLANN<Point>());
    kdTreeGroundsLast.reset(new pcl::KdTreeFLANN<Point>());

    polesGroundPointsCloud.reset(new pcl::PointCloud<Point>());
    facadesCenterPointsCloud.reset(new pcl::PointCloud<Point>());
    groundsCenterPointsCloud.reset(new pcl::PointCloud<Point>());

    point_pole.intensity = 500;



}

void PrimitivesAssociation::TransformToStart(const Point* const pi, Point* const po) {

    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(1, q_last_curr);
    Eigen::Vector3d t_point_last = t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}


void PrimitivesAssociation::TransformToStart(const Eigen::Vector3f * const pi, Eigen::Vector3f* const po) {

    Eigen::Vector3d point_curr = pi->cast<double>();
    Eigen::Vector3d point_w = q_last_curr * point_curr + t_last_curr;
    po->x() = point_w.x();
    po->y() = point_w.y();
    po->z() = point_w.z();
}
void PrimitivesAssociation::transformPlaneToLast(Plane& detected_plane, Plane& mapped_plane) {
    Plane plane_curr = detected_plane;

    mapped_plane.edge_poly.resize(4);
    mapped_plane.edge_poly[0] = (q_last_curr * plane_curr.edge_poly[0].cast<double>() + t_last_curr).cast<float>();
    mapped_plane.edge_poly[1] = (q_last_curr * plane_curr.edge_poly[1].cast<double>() + t_last_curr).cast<float>();
    mapped_plane.edge_poly[2] = (q_last_curr * plane_curr.edge_poly[2].cast<double>() + t_last_curr).cast<float>();
    mapped_plane.edge_poly[3] = (q_last_curr * plane_curr.edge_poly[3].cast<double>() + t_last_curr).cast<float>();

}

void PrimitivesAssociation::addTree() {
    // in the poles tree will only store the intersection poles and ground
    // in the facades tree will store the center points of facade line and z is 0

    // poles
    std::cout << "Poles and Facades and Ground blocks Number in last frame: " << prePoles.size() << "  "
              << preFacades.size() << " " << preGrounds.size() << std::endl;

    for (int i = 0; i < prePoles.size(); ++i) {
//        if (prePoles.empty()) {
//            break;
//        }
        Point p;
        p.x = prePoles[i].center_line.p_bottom[0];  // store the lowest point of pole
        p.y = prePoles[i].center_line.p_bottom[1];
        p.z = prePoles[i].center_line.p_bottom[2];
        p.intensity = 500;
        polesGroundPointsCloud->points.push_back(p);
    }
    if (!polesGroundPointsCloud->points.empty()) {
        kdTreePolesLast->setInputCloud(polesGroundPointsCloud);
    } else {
        std::cerr << "No pole in last frame" << std::endl;
    }

    Point tmpPoint;
    for (int i = 0; i < preGrounds.size(); ++i) {
        Eigen::Vector3f dl = preGrounds[i].edge_poly[0];
        Eigen::Vector3f dr = preGrounds[i].edge_poly[1];
        Eigen::Vector3f tr = preGrounds[i].edge_poly[2];
        Eigen::Vector3f centerPoint = dl + 0.5 * (tr - dl);

        tmpPoint.x = centerPoint[0];
        tmpPoint.y = centerPoint[1];
        tmpPoint.z = centerPoint[2];
        tmpPoint.intensity = 500;
        groundsCenterPointsCloud->points.push_back(tmpPoint);
    }
    if (!groundsCenterPointsCloud->points.empty()) {
        kdTreeGroundsLast->setInputCloud(groundsCenterPointsCloud);
    } else {
        std::cerr << "No ground blocks in last frame" << std::endl;
    }

    for (int i = 0; i < preFacades.size(); ++i) {
        Eigen::Vector3f dl = preFacades[i].edge_poly[0];
        Eigen::Vector3f dr = preFacades[i].edge_poly[1];
        Eigen::Vector3f tr = preFacades[i].edge_poly[2];
        Eigen::Vector3f centerPoint = dl + 0.5 * (tr - dl);

        tmpPoint.x = centerPoint[0];
        tmpPoint.y = centerPoint[1];
        tmpPoint.z = centerPoint[2];
        tmpPoint.intensity = 500;
        facadesCenterPointsCloud->points.push_back(tmpPoint);
    }
    if (!facadesCenterPointsCloud->points.empty()) {
        kdTreeFacadesLast->setInputCloud(facadesCenterPointsCloud);
    } else {
        std::cerr << "No facade blocks in last frame" << std::endl;
    }


}

float PrimitivesAssociation::calPolesMatchingCost(const Cylinder lastPole, const Cylinder curPole) {


    // transformer to map frame
    Eigen::Vector3f cur_center = curPole.center_line.p_bottom;
    Eigen::Vector3f cur_top = curPole.center_line.p_top;

    TransformToStart(&cur_center, &cur_center);
    TransformToStart(&cur_top, &cur_top);
    Eigen::Vector3f cur_axis_dir = cur_center - cur_top;

    // cal the difference of detected pole and pole in map
    float difRadius = std::abs(lastPole.radius - curPole.radius);
//    std::cerr << "DiffRadius: " << difRadius << std::endl;
    if (difRadius > 0.1) {
        difRadius = 999;
    }
    Eigen::Vector3f map_pole_line_dir_normalized = lastPole.line_dir.normalized();
    Eigen::Vector3f cur_pole_line_dir_normalized = cur_axis_dir.normalized();
    float difAngel = 1 - std::abs(map_pole_line_dir_normalized.dot(cur_pole_line_dir_normalized));
    float max_cylinder_association_angle_diff_ = 40/180.0*M_PI; // which is < 0.24
    if (difAngel > 0.13) {
        difAngel = 999;
    }
    float difDistance = (lastPole.center_line.p_bottom - cur_center).norm();
    if (difDistance > 5) {
        difDistance = 999;
    }
    return difAngel + difDistance + difRadius;
}

float PrimitivesAssociation::calFacadesMatchingCost(const Plane lastFacade, const Plane curFacade) {
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
    Eigen::Vector3f base_fa_dl = lastFacade.edge_poly[0]; // dl
    Eigen::Vector3f base_fa_dr = lastFacade.edge_poly[1]; // dr
    Eigen::Vector3f base_fa_tr = lastFacade.edge_poly[2]; // tr

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

//    std::cout << "distance: " << distance_dif << " direction: " << direction_dif << std::endl;
    float cost = distance_dif + 10 * direction_dif;
    if (direction_dif > 0.17) { // sin(10deg)
        cost += 9999;
    }
//    if (distance_dif > 3.0) {
//        cost += 999;
//    }
    return cost;
}

float PrimitivesAssociation::calGroundsMatchingCost(const Plane lastGround, const Plane curGround) {
    // transform to map frame
    Eigen::Vector3f cur_fa_dl = curGround.edge_poly[0];
    Eigen::Vector3f cur_fa_dr = curGround.edge_poly[1];
    Eigen::Vector3f cur_fa_tr = curGround.edge_poly[2];

    Eigen::Vector3f cur_fa_dl_tr = cur_fa_tr - cur_fa_dl;
    Eigen::Vector3f cur_fa_dl_dr = cur_fa_dr - cur_fa_dl;

    // cur center and normal
    Eigen::Vector3f cur_fa_center = cur_fa_dl + 0.5 * cur_fa_dl_tr;
    Eigen::Vector3f cur_fa_normal = (cur_fa_dl_dr.cross(cur_fa_dl_tr)).normalized();
    // map line in base frame
    Eigen::Vector3f base_fa_dl = lastGround.edge_poly[0]; // dl
    Eigen::Vector3f base_fa_dr = lastGround.edge_poly[1]; // dr
    Eigen::Vector3f base_fa_tr = lastGround.edge_poly[2]; // tr

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

float PrimitivesAssociation::dist3D_Segment_to_Segment(Eigen::Vector3f seg_1_p0, Eigen::Vector3f seg_1_p1,
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
void PrimitivesAssociation::associationCalculation() {
//    Time t1("Association time");

    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *q_parameterization =
        new ceres::EigenQuaternionParameterization();
    ceres::Problem::Options problem_options;

    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_q, 4, q_parameterization);
    problem.AddParameterBlock(para_t, 3);



    // Find facades correspondences
    int curFacadesNum = curFacades.size();
    // search for each current facade a correspondence in last facade
    for (int indexCurrentFacade = 0; indexCurrentFacade < curFacadesNum; ++indexCurrentFacade) {
        // No Facade break
        if (curFacadesNum == 0) break;

//        point_facade = curFacades[indexCurrentFacade].edge_poly[0]
//                       + 0.5 * (curFacades[indexCurrentFacade].edge_poly[2] - curFacades[indexCurrentFacade].edge_poly[0]);
//        TransformToStart(&point_facade, &point_facade);
//        Point tmpP;
//        tmpP.x = point_facade[0];
//        tmpP.y = point_facade[1];
//        tmpP.z = point_facade[2];
//        tmpP.intensity = 1000;
//        kdTreeFacadesLast->nearestKSearch(tmpP, 5, pointSearchInd_facade, pointSearchSqDis_facade);
//        if (pointSearchInd_facade.size() == 0) {
////            std::cerr << "DEBUG: " << std::endl;
//            continue;
//        }
//
//        float cost_facade = 9999;
//        int index_facade;
//        float facade_threshold = 999;
//        bool valid_facade = false;
//        for (auto& closeFacadeIndex : pointSearchInd_facade) {
//            Plane current_facade = curFacades[indexCurrentFacade];
//            Plane current_facade_last;
//            transformPlaneToMap(current_facade, current_facade_last);
//            Plane last_facade = preFacades[closeFacadeIndex];
//
//            float cost = calFacadesMatchingCost(last_facade, current_facade_last);
//
//            if (cost < cost_facade) {
//                cost_facade = cost;
//                index_facade = closeFacadeIndex;
//                valid_facade = true;
//            }
//        }
//        if (valid_facade && cost_facade < facade_threshold) {
//            facades_association.push_back(std::make_pair(indexCurrentFacade, index_facade));
//        }

        float cost_facade = 9999;
        int index_facade;
        float facade_threshold = 999;
        bool valid_facade = false;
        for (int indexLastFacade = 0; indexLastFacade < preFacades.size(); ++indexLastFacade) {
            // last frame facades
            Eigen::Vector3f plane_pos_start_last(preFacades[indexLastFacade].edge_poly[0]); // dl
            Eigen::Vector3f plane_pos_end_last(preFacades[indexLastFacade].edge_poly[1]); // dr
            // current frame facades
            Eigen::Vector3f plane_pos_start_cur(curFacades[indexCurrentFacade].edge_poly[0]);
            Eigen::Vector3f plane_pos_end_cur(curFacades[indexCurrentFacade].edge_poly[1]);
            // Transform to last frame
            TransformToStart(&plane_pos_start_cur, &plane_pos_start_cur);
            TransformToStart(&plane_pos_end_cur, &plane_pos_end_cur);


            // calculate distance between line segmentation
            float dist = dist3D_Segment_to_Segment(plane_pos_start_last.cast<float>(), plane_pos_end_last.cast<float>(),
                                                   plane_pos_start_cur.cast<float>(), plane_pos_end_cur.cast<float>());
//            std::cout << "dist: " << dist << std::endl;
            if (dist < 2.0) { // Segment distance less than 2 m
                Plane cur_plane_tran;
                transformPlaneToLast(curFacades[indexCurrentFacade], cur_plane_tran);
                float cost = calFacadesMatchingCost(preFacades[indexLastFacade], cur_plane_tran);
//                std::cerr << "less then 2.0: " << cost << std::endl;
                if (cost < cost_facade) {
                    cost_facade = cost;
                    index_facade = indexLastFacade;
                    valid_facade = true;
                }
            }
        }
//        std::cerr << "cost_facade in odo: " << cost_facade << std::endl;
        if (valid_facade && cost_facade < facade_threshold) {

            facades_association.push_back(std::make_pair(indexCurrentFacade, index_facade));
        }
    }
    std::cout << "Facades Correspondences in LiDAR odo: " << facades_association.size() << std::endl;
    if (facades_association.size() > 0) {
        for (int i = 0; i < facades_association.size(); ++i) {
            int indexCurrentFacade = facades_association[i].first;
            int indexLastFacade = facades_association[i].second;
            Eigen::Vector3d lastLineDl = preFacades[indexLastFacade].edge_poly[0].cast<double>();  // input in residual
            Eigen::Vector3d lastLineDr = preFacades[indexLastFacade].edge_poly[1].cast<double>(); // input in residual
            Eigen::Vector3d lastLineTr = preFacades[indexLastFacade].edge_poly[2].cast<double>();

            Eigen::Vector3d lastFacadeNormal = (lastLineDr - lastLineDl).cross(lastLineTr - lastLineDl);
            Eigen::Vector3d lastFacadeNormalNormalized = lastFacadeNormal.normalized(); // input in residual

            Eigen::Vector3d currLineDl = curFacades[indexCurrentFacade].edge_poly[0].cast<double>(); // input in residual
            Eigen::Vector3d currLineDr = curFacades[indexCurrentFacade].edge_poly[1].cast<double>(); // input in residual
            Eigen::Vector3d currLineTr = curFacades[indexCurrentFacade].edge_poly[2].cast<double>(); // input in residual

            ceres::CostFunction *cost_function = PlaneResidual::Create(lastLineDl, lastLineDr, lastFacadeNormalNormalized,
                                                                       currLineDl, currLineDr, currLineTr);
            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
        }
    } else {
        std::cout << "No Facades Association Found!" << std::endl;
    }


    int polesNum = curPoles.size();
    for (int i = 0; i < polesNum; ++i) {
        if (prePoles.empty()) {
            break; // no poles detected
        }
        Point bottomPoint;
        bottomPoint.x = curPoles[i].center_line.p_bottom[0];
        bottomPoint.y = curPoles[i].center_line.p_bottom[1];
        bottomPoint.z = curPoles[i].center_line.p_bottom[2];
        bottomPoint.intensity = 500;
        TransformToStart(&bottomPoint, &point_pole); // transformer current pole to last frame
        // search in last frame the most similar pole
        kdTreePolesLast->radiusSearch(point_pole, pole_position_threshold, pointSearchInd_pole, pointSearchSqDis_pole);
        // Association in search range
        float machCost = 9999; // count the cost
        float pole_threshold = 999;
        int closeIndex;  // store the best candidate
        if (pointSearchInd_pole.size() == 0) {
//            std::cerr << "no Pole fund!" << std::endl;
            continue;
        }
        bool validPole = false;
        float costThreshold = 2.0;
        for (auto& closeElem : pointSearchInd_pole) {  // Already chose the range in 3 m pole candidates next compare
                                                        // radius and axis direction
            float cost = calPolesMatchingCost(prePoles[closeElem], curPoles[i]);
//            std::cout << "cost pole: " << cost << std::endl;

            if (cost < machCost) {
                machCost = cost;
                closeIndex = closeElem;
                validPole = true;
            }

                // chose the pole with tne smallest cost

//            std::cout << "cost: " << machCost << std::endl;
        }
        if (validPole && machCost < pole_threshold) {
            poles_association.push_back(std::make_pair(i, closeIndex)); // first: current pole id. second: last pole id
        }
    }
    std::cout << "Pole correspondences before filtering: " << poles_association.size() << std::endl;

    // Filter out outliers
    // get average dis and filter out location outliers
    float averageDis = 0;
    int costCount = 0;
    for (int i = 0; i < poles_association.size(); ++i) {
        Eigen::Vector3d curPoint = curPoles[poles_association[i].first].center_line.p_bottom.cast<double>();
        Eigen::Vector3d prePoint = prePoles[poles_association[i].second].center_line.p_bottom.cast<double>();
        double distance = (curPoint - prePoint).norm();
        averageDis += distance;
        costCount++;
    }
    averageDis = averageDis / costCount;
//    std::cerr << averageDis << std::endl;
    for (int i = 0; i < poles_association.size(); ++i) {
        // The associated bottom point and axis direction
        int lastPoleIndex = poles_association[i].second;
        int currentPoleIndex = poles_association[i].first;

        Eigen::Vector3d prePoint = prePoles[lastPoleIndex].center_line.p_bottom.cast<double>();
        Eigen::Vector3d preDir = prePoles[lastPoleIndex].line_dir.cast<double>();
        Eigen::Vector3d preTopPoint = prePoles[lastPoleIndex].center_line.p_top.cast<double>();

        Eigen::Vector3d curPoint = curPoles[currentPoleIndex].center_line.p_bottom.cast<double>();
        Eigen::Vector3d curDir = curPoles[currentPoleIndex].line_dir.cast<double>();
        Eigen::Vector3d curTopPoint = curPoles[currentPoleIndex].center_line.p_top.cast<double>();

        double distance = (curPoint - prePoint).norm();
        if (distance > 1.5 * averageDis) { // filter out location outliers
            poles_association.erase(poles_association.begin() + i);
            continue;
        }

        ceres::CostFunction *cost_function = PoleFactor::Create(curPoint, prePoint);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
    }


    std::cout << "Pole correspondences in LiDAR odo: " << poles_association.size() << std::endl;


    // ground correspondence find
    for (int i = 0; i < (int)curGrounds.size(); ++i) {
        if (curGrounds.empty()) {
            std::cerr << "NO GROUND IN CURRENT SKIP" << std::endl;
            break;
        }
        point_ground = curGrounds[i].edge_poly[0]
                       + 0.5 * (curGrounds[i].edge_poly[2] - curGrounds[i].edge_poly[0]);
        TransformToStart(&point_ground, &point_ground);
        Point tmpP;
        tmpP.x = point_ground[0];
        tmpP.y = point_ground[1];
        tmpP.z = point_ground[2];
        tmpP.intensity = 1000;
        kdTreeGroundsLast->nearestKSearch(tmpP, 5, pointSearchInd_ground, pointSearchSqDis_ground);
        if (pointSearchInd_ground.size() == 0) {
//            std::cerr << "DEBUG: " << std::endl;
            continue;
        }
        float machCost = 9999;
        int index_ground;
        float costThreshold = 999;
        bool valid_ground = false;
        for (auto& closeGroundIndex : pointSearchInd_ground) {
            Plane current_ground = curGrounds[i];
            Plane current_ground_last;
            transformPlaneToLast(current_ground, current_ground_last);
            Plane last_ground = preGrounds[closeGroundIndex];

            float cost = calGroundsMatchingCost(last_ground, current_ground_last);

            if (cost < machCost) {
                machCost = cost;
                index_ground = closeGroundIndex;
                valid_ground = true;
            }
        }
        if (valid_ground && machCost < costThreshold) {
//            std::cerr << "machCost: " << machCost << std::endl;
            grounds_association.push_back(std::make_pair(i, index_ground));
        }
    }
    std::cout << "Ground Correspondences in LiDAR odo: " << grounds_association.size() << std::endl;
    if (grounds_association.size() > 0) {
        for (int i = 0; i < grounds_association.size(); ++i) {
            int indexCurrentGround = grounds_association[i].first;
            int indexLastGround = grounds_association[i].second;
            Eigen::Vector3d lastLineDl = preGrounds[indexLastGround].edge_poly[0].cast<double>();  // input in residual
            Eigen::Vector3d lastLineDr = preGrounds[indexLastGround].edge_poly[1].cast<double>(); // input in residual
            Eigen::Vector3d lastLineTr = preGrounds[indexLastGround].edge_poly[2].cast<double>();

            Eigen::Vector3d lastGroundNormal = (lastLineDr - lastLineDl).cross(lastLineTr - lastLineDl);
            Eigen::Vector3d lastGroundNormalNormalized = lastGroundNormal.normalized(); // input in residual

            Eigen::Vector3d currLineDl = curGrounds[indexCurrentGround].edge_poly[0].cast<double>(); // input in residual
            Eigen::Vector3d currLineDr = curGrounds[indexCurrentGround].edge_poly[1].cast<double>(); // input in residual
            Eigen::Vector3d currLineTr = curGrounds[indexCurrentGround].edge_poly[2].cast<double>(); // input in residual

            ceres::CostFunction *cost_function = GroundResidual::Create(lastLineDl, lastLineDr, lastGroundNormalNormalized,
                                                                       currLineDl, currLineDr, currLineTr);
            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
        }
    } else {
        std::cerr << "No Grounds Association Found!" << std::endl;
    }



    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

//        std::cout << summary.BriefReport() << std::endl;

    // update pose
//    std::cout << q_last_curr.toRotationMatrix() << std::endl;

    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
    q_w_curr = q_w_curr * q_last_curr;

//    std::cout << "Transform between last frame and current frame:\n " << std::endl;
//    std::cout << " rotation matrix \n " << q_last_curr.toRotationMatrix() << std::endl;
//    std::cout << " t_last_cur \n" << t_last_curr << "\n" << std::endl;

}


void PrimitivesAssociation::runAS() {
    Time t("Odometry Time");
    // put last frame feature in the corresponding kdTree
    addTree();
    associationCalculation();

}
Eigen::Quaterniond PrimitivesAssociation::publishQuaterniond() {
    return q_w_curr;
}
Eigen::Vector3d PrimitivesAssociation::publishTranslation() {
    return t_w_curr;
}

Eigen::Quaterniond PrimitivesAssociation::quaterniodGuess() {
//    std::cout << "q_last_curr:\n" << q_last_curr.matrix() << std::endl;
    return q_last_curr;
}
Eigen::Vector3d PrimitivesAssociation::translationGuess() {
//    std::cout << "t_last_curr:\n" << t_last_curr << std::endl;
    return t_last_curr;
}
void PrimitivesAssociation::getCurrentDetection(std::vector<Cylinder>& detectedPoles,
                                                std::vector<Plane>& detectedPlanes,
                                                std::vector<Plane>& detectedGround) {
    for (auto& pole : curPoles) {
        detectedPoles.push_back(pole);
    }
    for (auto& plane : curFacades) {
        detectedPlanes.push_back(plane);
    }
    for (auto& plane : curGrounds) {
        detectedGround.push_back(plane);
    }
}
void PrimitivesAssociation::getAssociation(Cloud::Ptr& cloud) {

    for (int i = 0; i < poles_association.size(); ++i) {
        int indexLastPole = poles_association[i].second;
        int indexCurPole = poles_association[i].first;
        Eigen::Vector3f last_p_bottom = prePoles[indexLastPole].center_line.p_bottom;
        Eigen::Vector3f last_p_top = prePoles[indexLastPole].center_line.p_top;
        float last_r = prePoles[indexLastPole].radius;
        Eigen::Vector3f cur_p_bottom = curPoles[indexCurPole].center_line.p_bottom;
        Eigen::Vector3f cur_p_top = curPoles[indexCurPole].center_line.p_top;
        float cur_r = curPoles[indexCurPole].radius;
        // last pole
        createTheoryCylinder(last_p_bottom, last_p_top, last_r, cloud, 10, 10, 2500);
        // current pole
        createTheoryCylinder(cur_p_bottom, cur_p_top, cur_r, cloud, 10, 10, 2000);
        // association line
        createTheoryLine(last_p_bottom, cur_p_bottom, cloud, 1000, 10000);
    }
    for (int i = 0; i < facades_association.size(); ++i) {
        int indexLastFacade = facades_association[i].second;
        int indexCurrentFacade = facades_association[i].first;
        Eigen::Vector3f last_dl = preFacades[indexLastFacade].edge_poly[0];
        Eigen::Vector3f last_dr = preFacades[indexLastFacade].edge_poly[1];
        Eigen::Vector3f last_tl = preFacades[indexLastFacade].edge_poly[3];
        Eigen::Vector3f last_tr = preFacades[indexLastFacade].edge_poly[2];

        Eigen::Vector3f last_center = 0.5 * (last_tr - last_dl);
        // last facade
        createTheoryLine(last_dl, last_dr, cloud,100, 1500);
        createTheoryLine(last_dr, last_tr, cloud,100, 1500);
        createTheoryLine(last_tr, last_tl, cloud,100, 1500);
        createTheoryLine(last_tl, last_dl, cloud,100, 1500);
        createTheoryLine(last_tl, last_dr, cloud,100, 1500);
        createTheoryLine(last_tr, last_dl, cloud,100, 1500);

        Eigen::Vector3f cur_dl = curFacades[indexCurrentFacade].edge_poly[0];
        Eigen::Vector3f cur_dr = curFacades[indexCurrentFacade].edge_poly[1];
        Eigen::Vector3f cur_tl = curFacades[indexCurrentFacade].edge_poly[3];
        Eigen::Vector3f cur_tr = curFacades[indexCurrentFacade].edge_poly[2];

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
    for (int i = 0; i < grounds_association.size(); ++i) {
        int indexLastFacade = grounds_association[i].second;
        int indexCurrentFacade = grounds_association[i].first;
        Eigen::Vector3f last_dl = preGrounds[indexLastFacade].edge_poly[0];
        Eigen::Vector3f last_dr = preGrounds[indexLastFacade].edge_poly[1];
        Eigen::Vector3f last_tl = preGrounds[indexLastFacade].edge_poly[3];
        Eigen::Vector3f last_tr = preGrounds[indexLastFacade].edge_poly[2];

        Eigen::Vector3f last_center = 0.5 * (last_tr - last_dl);
        // last facade
        createTheoryLine(last_dl, last_dr, cloud,100, 1500);
        createTheoryLine(last_dr, last_tr, cloud,100, 1500);
        createTheoryLine(last_tr, last_tl, cloud,100, 1500);
        createTheoryLine(last_tl, last_dl, cloud,100, 1500);
        createTheoryLine(last_tl, last_dr, cloud,100, 1500);
        createTheoryLine(last_tr, last_dl, cloud,100, 1500);

        Eigen::Vector3f cur_dl = curGrounds[indexCurrentFacade].edge_poly[0];
        Eigen::Vector3f cur_dr = curGrounds[indexCurrentFacade].edge_poly[1];
        Eigen::Vector3f cur_tl = curGrounds[indexCurrentFacade].edge_poly[3];
        Eigen::Vector3f cur_tr = curGrounds[indexCurrentFacade].edge_poly[2];

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
    // add all primitive
    for (int i = 0; i < prePoles.size(); ++i) {
        Eigen::Vector3f p_b = prePoles[i].center_line.p_bottom;
        Eigen::Vector3f p_t = prePoles[i].center_line.p_top;
        float r = prePoles[i].radius;
        createTheoryCylinder(p_b, p_t, r, cloud, 10, 10, 5);

    }
    for (int i = 0; i < curPoles.size(); ++i) {
        Eigen::Vector3f p_b = curPoles[i].center_line.p_bottom;
        Eigen::Vector3f p_t = curPoles[i].center_line.p_top;
        float r = curPoles[i].radius;
        createTheoryCylinder(p_b, p_t, r, cloud, 10, 10, 5);

    }
    for (int i = 0; i < preFacades.size(); ++i) {
        Eigen::Vector3f last_dl = preFacades[i].edge_poly[0];
        Eigen::Vector3f last_dr = preFacades[i].edge_poly[1];
        Eigen::Vector3f last_tl = preFacades[i].edge_poly[3];
        Eigen::Vector3f last_tr = preFacades[i].edge_poly[2];
        // last facade
        createTheoryLine(last_dl, last_dr, cloud,10, 5000);
        createTheoryLine(last_dr, last_tr, cloud,10, 5000);
        createTheoryLine(last_tr, last_tl, cloud,10, 5000);
        createTheoryLine(last_tl, last_dl, cloud,10, 5000);
    }
    for (int i = 0; i < curFacades.size(); ++i) {
        Eigen::Vector3f cur_dl = curFacades[i].edge_poly[0];
        Eigen::Vector3f cur_dr = curFacades[i].edge_poly[1];
        Eigen::Vector3f cur_tl = curFacades[i].edge_poly[3];
        Eigen::Vector3f cur_tr = curFacades[i].edge_poly[2];
        // last facade
        createTheoryLine(cur_dl, cur_dr, cloud,10, 1000);
        createTheoryLine(cur_dr, cur_tr, cloud,10, 1000);
        createTheoryLine(cur_tr, cur_tl, cloud,10, 1000);
        createTheoryLine(cur_tl, cur_dl, cloud,10, 1000);

    }
    for (int i = 0; i < preGrounds.size(); ++i) {
        Eigen::Vector3f last_dl = preGrounds[i].edge_poly[0];
        Eigen::Vector3f last_dr = preGrounds[i].edge_poly[1];
        Eigen::Vector3f last_tl = preGrounds[i].edge_poly[3];
        Eigen::Vector3f last_tr = preGrounds[i].edge_poly[2];
        // last facade
        createTheoryLine(last_dl, last_dr, cloud,10, 5000);
        createTheoryLine(last_dr, last_tr, cloud,10, 5000);
        createTheoryLine(last_tr, last_tl, cloud,10, 5000);
        createTheoryLine(last_tl, last_dl, cloud,10, 5000);
    }
    for (int i = 0; i < curGrounds.size(); ++i) {
        Eigen::Vector3f cur_dl = curGrounds[i].edge_poly[0];
        Eigen::Vector3f cur_dr = curGrounds[i].edge_poly[1];
        Eigen::Vector3f cur_tl = curGrounds[i].edge_poly[3];
        Eigen::Vector3f cur_tr = curGrounds[i].edge_poly[2];
        // last facade
        createTheoryLine(cur_dl, cur_dr, cloud,10, 1000);
        createTheoryLine(cur_dr, cur_tr, cloud,10, 1000);
        createTheoryLine(cur_tr, cur_tl, cloud,10, 1000);
        createTheoryLine(cur_tl, cur_dl, cloud,10, 1000);
    }

}




} // end of primitivesFa