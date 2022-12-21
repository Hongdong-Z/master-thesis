//
// Created by zhao on 11.08.22.
//

#include "primitives_mapping/primitivesMapping.h"
namespace primitivesMapping {

PrimitivesMapping::PrimitivesMapping(const int lengthMap_) {
    lengthMap = lengthMap_;
    cylinder_min_num_detections_ = 3;
    cylinder_cluster_pos_thresh_ = 0.5;
    cylinder_max_radius_tolerance_from_median_percentage_ = 0.1;
    cylinder_max_ground_tolerance_from_median_ = 0.1;
    cylinder_max_xy_diff_ = 0.10;
    cylinder_max_angle_diff_from_average_ = 1 - cos(10.0/180.0*M_PI);

    plane_cluster_pos_thresh_ = 0.1;
    plane_cluster_angle_thresh_ = cos(5.0 / 180 * M_PI);
    min_num_planes_in_cluster_ = 5;
    plane_sample_step_size_ = 0.5;

    new_plane_inlier_max_dist_ = 0.15;
//    visual_tools_.reset(new rviz_visual_tools::RvizVisualTools(std::to_string(lengthMap_),"/debug_visu_map_manager"));

    initialValue();
}

void PrimitivesMapping::initialValue() {
    fusionFacades.reset(new Cloud());
    fusionPoles.reset(new Cloud());

    curFacades.reset(new Cloud());
    curPoles.reset(new Cloud());

    curFacadesTF.reset(new Cloud());
    curPolesTF.reset(new Cloud());

    mapFacadesPara.reset(new Cloud());
    mapPolesPara.reset(new Cloud());

    currentCloud.reset(new Cloud());

    facadesParam.clear();
    polesParam.clear();
    groundsParam.clear();
}
void PrimitivesMapping::resetPara() {

    curFacades->clear();
    curPoles->clear();
    curFacadesTF->clear();
    curPolesTF->clear();
    currentCloud->clear();
}
void PrimitivesMapping::transformFacades(Plane_Param& facade, Eigen::Transform<double, 3, Eigen::Affine>& TF) {
    Eigen::Vector4f ptl(facade.edge_poly[0][0], facade.edge_poly[0][1], facade.edge_poly[0][2], 1.0f);
    Eigen::Vector4f ptr(facade.edge_poly[1][0], facade.edge_poly[1][1], facade.edge_poly[1][2], 1.0f);
    Eigen::Vector4f pdr(facade.edge_poly[2][0], facade.edge_poly[2][1], facade.edge_poly[2][2], 1.0f);
    Eigen::Vector4f pdl(facade.edge_poly[3][0], facade.edge_poly[3][1], facade.edge_poly[3][2], 1.0f);
    Eigen::Vector4f n(facade.n[0], facade.n[1], facade.n[2], 1.0f);
    // transform
    facade.edge_poly[0] = (TF.matrix()*ptl.cast<double>()).block<3,1>(0,0).cast<float>();
    facade.edge_poly[1] = (TF.matrix()*ptr.cast<double>()).block<3,1>(0,0).cast<float>();
    facade.edge_poly[2] = (TF.matrix()*pdr.cast<double>()).block<3,1>(0,0).cast<float>();
    facade.edge_poly[3] = (TF.matrix()*pdl.cast<double>()).block<3,1>(0,0).cast<float>();
    facade.n = (TF.matrix()*n.cast<double>()).block<3,1>(0, 0).cast<float>();
}
void PrimitivesMapping::transformPoles(Cylinder_Fin_Param& pole, Eigen::Transform<double, 3, Eigen::Affine>& TF) {
    Eigen::Vector4f bottom_p(pole.center_line.p_bottom[0], pole.center_line.p_bottom[1], pole.center_line.p_bottom[2], 1.0f);
    Eigen::Vector4f top_p(pole.center_line.p_top[0], pole.center_line.p_top[1], pole.center_line.p_top[2], 1.0f);
    //transform
    pole.center_line.p_bottom = (TF.matrix()*bottom_p.cast<double>()).block<3,1>(0, 0).cast<float>();
    pole.center_line.p_top = (TF.matrix()*top_p.cast<double>()).block<3,1>(0, 0).cast<float>();
}

void PrimitivesMapping::run() {

    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    localization_lidar::ParameterServer::initialize(configPath);

    std::filesystem::path dataRootPath(localization_lidar::ParameterServer::get()["main"]["dataRoot"]);
    std::filesystem::path outputMapPath(localization_lidar::ParameterServer::get()["main"]["outputFolderPrimitivesMap"]);
//    std::filesystem::path facadesMapPath = outputMapPath / "facadesMap";
    std::filesystem::path facadesParaMapPath = outputMapPath / "facadesParaMap";
    std::filesystem::path polesParaMapPath = outputMapPath / "polesParaMap";
    std::filesystem::path groundsParaMapPath = outputMapPath / "groundsParaMap";
    std::filesystem::create_directory(outputMapPath);
    std::filesystem::create_directory(groundsParaMapPath);
//    std::filesystem::create_directory(facadesParaMapPath);
//    std::filesystem::create_directory(polesParaMapPath);


    auto mainDataStorage1 = loadDataStorage(dataRootPath);


    uint64_t dataRange[2] = {localization_lidar::ParameterServer::get()["main"]["startDataRange"], localization_lidar::ParameterServer::get()["main"]["stopDataRange"]};


    std::vector<size_t> validDataIndices; // Only frames with both point cloud is used

    // load new pose file
    std::string poseConfig = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::shared_ptr<localization_lidar::PoseLoader> poseLoader = std::make_shared<localization_lidar::PoseLoader>(poseConfig);
    poseLoader->loadNewPoseFile();

    std::ofstream facadesMap( "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/facadesParaMap/facadesMap_ori.txt");
//    std::ofstream polesMap( "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/polesParaMap/polesMap_ori.txt");
//    std::ofstream groundsMap( "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/groundsParaMap/groundsMap_ori.txt");


    size_t tempI = 0;
    double distanceCountName = 0;
    double distanceCountNameStart = 0;
    std::vector<std::pair<int,int>> fewPoleFrames;
    for (size_t i = dataRange[0]; i <= dataRange[1]; ++i) {

        // start with second frame
        std::optional<fbds::FrameData> frameData = mainDataStorage1[i];
        if(!frameData) { std::cerr << "\nFrameData or lastFrameData " << i << " not found " << std::endl; continue; }

        std::cout << "\nProcessing Frame " << i << std::endl;

        if(!frameData->pose()) { std::cerr << "Did not find pose. ~> Skipping frame. " << std::endl; continue; }

        validDataIndices.emplace_back(i);
        // set start frame
        auto pose_w_first = poseLoader->getPose(tempI);
        // true first frame
        auto pose_w_first_fix = poseLoader->getPose(0);

        auto pose_w_current = poseLoader->getPose(i);

        Eigen::Vector3d firstPosition = pose_w_first.matrix().block<3, 1>(0, 3);
        Eigen::Vector3d trueFirstPosition = pose_w_first_fix.matrix().block<3, 1>(0, 3);
        Eigen::Vector3d currentPosition = pose_w_current.matrix().block<3, 1>(0, 3);


        auto velodyneFLDatum = (*frameData)[fbds::FrameSource::VelodyneFL];

        *currentCloud = velodyneFLDatum->asPointcloud<PointType>();


        // process single frame and sum together for mapping
        primitivesExtraction::PrimitiveExtractor PriExtractor;
        PriExtractor.setInputCloud(currentCloud);
        PriExtractor.setRange(-200, 200,
                              -200, 200,
                              -2.5, 10);
        /// ********** Get cloud primitives candidates **********
        PriExtractor.runForMapping();

//        PriExtractor.getFacadeRoughCloud(curFacades);
//        PriExtractor.getPoleRoughCloud(curPoles);

        // Transform between first frame and current frame
        Eigen::Transform<double, 3, Eigen::Affine> transformFrame = pose_w_current;
//        pcl::transformPointCloud(*curFacades, *curFacadesTF, transformFrame);
//        pcl::transformPointCloud(*curPoles, *curPolesTF, transformFrame);

        /// ************** Fusion the primitives clouds **********

//        *fusionFacades += *curFacadesTF;
//        *fusionPoles += *curPolesTF;

     // Get Facades and Poles param
//        std::vector<Cylinder_Fin_Param> tempPolesParam;
        std::vector<Plane_Param> tempFacadesParam;
//        std::vector<Plane_Param> tempGroundsParam;
//        PriExtractor.getPolesFinParam(tempPolesParam);
//        PriExtractor.getGroundFinParam(tempGroundsParam);
        PriExtractor.getFacadesFinParam(tempFacadesParam);
//        if (tempPolesParam.size() < 5) {
//            fewPoleFrames.push_back(std::make_pair(i, tempPolesParam.size()));
//        }
//        PriExtractor.getFacadesFinParam(tempFacadesParam);
//
//        for (auto& pole : tempPolesParam) {
//            transformPoles(pole, transformFrame);
//            polesParam.push_back(pole);
//        }
        for (auto& facade : tempFacadesParam) {
            transformFacades(facade, transformFrame);
            facadesParam.push_back(facade);
        }
//        for (auto& ground : tempGroundsParam) {
//            transformFacades(ground, transformFrame);
//            groundsParam.push_back(ground);
//        }




        if (i % lengthMap == 0) {

//            std::cout << "Number of fusion poles: " << fusionPoles->points.size() << std::endl;
//            std::cout << "Number of fusion facades: " << fusionFacades->points.size() << std::endl;
//            pcl::io::savePCDFileASCII(polesParaMapPath / ("SAVE/fusionPolesRough_" + std::to_string(i) + ".pcd"),
//                                      *fusionPoles);
//            pcl::io::savePCDFileASCII(facadesParaMapPath / ("SAVE/fusionFacadesRough_" + std::to_string(i) + ".pcd"),
//                                      *fusionFacades);
//            PriExtractor.getFusionFacade(fusionFacades, facadesParam);

//            std::vector<CloudColor::Ptr> facade_candidates;
//            PriExtractor.get3DFacadeCandidateCloudsColored(facade_candidates);
//            CloudColor::Ptr clusteredCloud(new CloudColor());
//            for (auto& cloud : facade_candidates) {
//                *clusteredCloud += *cloud;
//            }

//
//            PriExtractor.getFusionPole(fusionPoles, polesParam);
//
//            getPrimitivePoleCloud(fusionPoles, polesParam);
//            getPrimitiveFacadeCloud(fusionFacades, facadesParam, 50);
//            // colored
//            getPrimitiveFacadeCloud(clusteredCloud, facadesParam);
//            // pure para
//            Cloud::Ptr paraCloud(new Cloud());
//            getPrimitiveFacadeCloud(paraCloud, facadesParam, 50);
//
//            pcl::io::savePCDFileASCII(facadesParaMapPath / ("SAVE/fusionFacades_" + std::to_string(i) + ".pcd"),
//                                      *fusionFacades);
//            // colored
//            pcl::io::savePCDFileASCII(facadesParaMapPath / ("SAVE/clusteredFacade.pcd"),
//                                      *clusteredCloud);
//            pcl::io::savePCDFileASCII(facadesParaMapPath / ("SAVE/paraFacade.pcd"),
//                                      *paraCloud);

//            pcl::io::savePCDFileASCII(polesParaMapPath / ("SAVE/fusionPoles_" + std::to_string(i) + ".pcd"),
//                                      *fusionPoles);


//            for (auto& pole : polesParam) {
//                // bottom point << top point << radius
//                polesMap << pole.center_line.p_bottom[0] << " " << pole.center_line.p_bottom[1] << " " << pole.center_line.p_bottom[2] <<" "
//                         << pole.center_line.p_top[0] << " " << pole.center_line.p_top[1] << " " << pole.center_line.p_top[2] << " "
//                         << pole.radius << std::endl;
//            }
            for (auto& facade : facadesParam) {
                // down left << down right << top right << top left << n << d
                facadesMap << facade.edge_poly[0][0] << " " << facade.edge_poly[0][1] << " " << facade.edge_poly[0][2] << " "
                           << facade.edge_poly[1][0] << " " << facade.edge_poly[1][1] << " " << facade.edge_poly[1][2] << " "
                           << facade.edge_poly[2][0] << " " << facade.edge_poly[2][1] << " " << facade.edge_poly[2][2] << " "
                           << facade.edge_poly[3][0] << " " << facade.edge_poly[3][1] << " " << facade.edge_poly[3][2] << " "
                           << facade.n[0] << " " << facade.n[1] << " " << facade.n[2] << " "
                           << facade.d << std::endl;
            }
//            for (auto& ground : groundsParam) {
//                // down left << down right << top right << top left << n << d
//                groundsMap << ground.edge_poly[0][0] << " " << ground.edge_poly[0][1] << " " << ground.edge_poly[0][2] << " "
//                           << ground.edge_poly[1][0] << " " << ground.edge_poly[1][1] << " " << ground.edge_poly[1][2] << " "
//                           << ground.edge_poly[2][0] << " " << ground.edge_poly[2][1] << " " << ground.edge_poly[2][2] << " "
//                           << ground.edge_poly[3][0] << " " << ground.edge_poly[3][1] << " " << ground.edge_poly[3][2] << " "
//                           << ground.n[0] << " " << ground.n[1] << " " << ground.n[2] << " "
//                           << ground.d << std::endl;
//            }

            initialValue();

            std::cerr << "save sub map between frames: " << tempI << " and " << i << std::endl;
            tempI = i;
            distanceCountNameStart = distanceCountName;
            continue;
        }
        resetPara();

    }
//    for (int i = 0; i < fewPoleFrames.size(); ++i) {
//        std::cerr << "frame: " << fewPoleFrames[i].first << " has: " << fewPoleFrames[i].second << " poles " << std::endl;
//    }
}
void PrimitivesMapping::save_map(const std::string& save_path_pole,
                                 const std::string& save_path_facade,
                                 const std::string& save_path_ground) {
    std::ofstream file_pole_writer(save_path_pole);
    if (file_pole_writer.is_open())
    {

        for( auto& c : new_map_.cylinders )
        {
            file_pole_writer <<c.center_line.p_bottom(0) << " "
                        <<c.center_line.p_bottom(1) << " "
                        <<c.center_line.p_bottom(2) << " "
                        <<c.center_line.p_top(0) << " "
                        <<c.center_line.p_top(1) << " "
                        <<c.center_line.p_top(2) << " "
                             << c.radius << std::endl;
        }
        file_pole_writer.close();
    }
    else
    {
        std::cout << "File "<< save_path_pole <<" could not be opened." <<std::endl;
    }

    std::ofstream file_facade_writer(save_path_facade);
    if (file_facade_writer.is_open()) {
        for( Plane_Param& p : new_map_.planes )
        {
             // DL DR TR TL
            file_facade_writer << p.edge_poly[0][0] << " " << p.edge_poly[0][1] << " " << p.edge_poly[0][2] << " "
                                << p.edge_poly[1][0] << " " << p.edge_poly[1][1] << " " << p.edge_poly[1][2] << " "
                                << p.edge_poly[2][0] << " " << p.edge_poly[2][1] << " " << p.edge_poly[2][2] << " "
                                << p.edge_poly[3][0] << " " << p.edge_poly[3][1] << " " << p.edge_poly[3][2] << " "
                                << p.n[0] << " " << p.n[1] << " " << p.n[2] << " "
                                << p.d << std::endl;
        }
        file_facade_writer.close();
    } else {
        std::cout << "File "<< save_path_facade <<" could not be opened." <<std::endl;
    }

    std::ofstream file_ground_writer(save_path_ground);
    if (file_ground_writer.is_open()) {
        for( Plane_Param& p : new_map_.grounds )
        {
            // DL DR TR TL
            file_ground_writer << p.edge_poly[0][0] << " " << p.edge_poly[0][1] << " " << p.edge_poly[0][2] << " "
                               << p.edge_poly[1][0] << " " << p.edge_poly[1][1] << " " << p.edge_poly[1][2] << " "
                               << p.edge_poly[2][0] << " " << p.edge_poly[2][1] << " " << p.edge_poly[2][2] << " "
                               << p.edge_poly[3][0] << " " << p.edge_poly[3][1] << " " << p.edge_poly[3][2] << " "
                               << p.n[0] << " " << p.n[1] << " " << p.n[2] << " "
                               << p.d << std::endl;
        }
        file_ground_writer.close();
    } else {
        std::cout << "File "<< save_path_ground <<" could not be opened." <<std::endl;
    }

}
void PrimitivesMapping::readPolesMap(std::vector<Cylinder_Fin_Param>& poles_param, std::string& polesMapPath) {
    std::cout << "start poles loading" << std::endl;
    std::ifstream file(polesMapPath);
    map_.cylinders.clear();
    while (!file.eof()){
        float bp0, bp1, bp2, tp0, tp1, tp2, r;
        file >> bp0 >> bp1 >> bp2 >> tp0 >> tp1 >> tp2 >> r;
        Cylinder_Fin_Param pole;
        pole.radius = r;
        pole.center_line.p_top << tp0, tp1, tp2;
        pole.center_line.p_bottom << bp0, bp1, bp2;
        poles_param.push_back(pole);
        // push in map
        map_.cylinders.push_back(pole);
    }
    std::cout << "finish poles loading" << std::endl;
}

void PrimitivesMapping::readGroundMap(std::vector<Plane_Param>& grounds_param, std::string& groundsMapPath) {
    std::cout << "start grounds loading" << std::endl;
    std::ifstream file(groundsMapPath);
    map_.grounds.clear();
    while (!file.eof()) {
        float ptl0, ptl1, ptl2, ptr0, ptr1, ptr2, pdr0, pdr1, pdr2, pdl0, pdl1, pdl2, n0, n1, n2, d;
        // dl >> dr >> tr >> tl >> n >> d
        file >> pdl0 >> pdl1 >> pdl2 >> pdr0 >> pdr1 >> pdr2 >> ptr0 >> ptr1 >> ptr2 >> ptl0 >> ptl1 >> ptl2 >> n0 >> n1 >> n2 >> d;
        Plane_Param ground;
        ground.n << n0, n1, n2;
        ground.d = d;
        ground.edge_poly.resize(4);
        ground.edge_poly[3] << ptl0, ptl1, ptl2;
        ground.edge_poly[2] << ptr0, ptr1, ptr2;
        ground.edge_poly[1] << pdr0, pdr1, pdr2;
        ground.edge_poly[0] << pdl0, pdl1, pdl2;
        grounds_param.push_back(ground);
        // push in map
        map_.grounds.push_back(ground);
//        std::cout << map_.planes.size() << std::endl;
    }

    std::cout << "finish grounds loading" << std::endl;
}
void PrimitivesMapping::readFacadeMap(std::vector<Plane_Param>& facades_param, std::string& facadesMapPath) {
    std::cout << "start facades loading" << std::endl;
    std::ifstream file(facadesMapPath);
    map_.planes.clear();
    while (!file.eof()) {
        float ptl0, ptl1, ptl2, ptr0, ptr1, ptr2, pdr0, pdr1, pdr2, pdl0, pdl1, pdl2, n0, n1, n2, d;
        // dl >> dr >> tr >> tl >> n >> d
        file >> pdl0 >> pdl1 >> pdl2 >> pdr0 >> pdr1 >> pdr2 >> ptr0 >> ptr1 >> ptr2 >> ptl0 >> ptl1 >> ptl2 >> n0 >> n1 >> n2 >> d;
        Plane_Param facade;
        facade.n << n0, n1, n2;
        facade.d = d;
        facade.edge_poly.resize(4);
        facade.edge_poly[3] << ptl0, ptl1, ptl2;
        facade.edge_poly[2] << ptr0, ptr1, ptr2;
        facade.edge_poly[1] << pdr0, pdr1, pdr2;
        facade.edge_poly[0] << pdl0, pdl1, pdl2;
        facades_param.push_back(facade);
        // push in map
        map_.planes.push_back(facade);
//        std::cout << map_.planes.size() << std::endl;
    }

//    if (file.is_open()) {
//        std::string line;
//        std::getline(file, line);
//        std::cout << line << std::endl;
//        std::stringstream stream(line);
//        int num_facades;
//        stream >> num_facades;
//        std::cout << "Num Facades: " << num_facades <<std::endl;
//        for( int i = 0; i < num_facades; ++i )
//        {
//            std::getline(file,line);
//            std::stringstream facade_stream(line);
//
//            Plane_Param plane;
//            plane.edge_poly.resize(4);
//            facade_stream >> plane.edge_poly[0][0] >> plane.edge_poly[0][1] >> plane.edge_poly[0][2]
//                            >> plane.edge_poly[1][0] >> plane.edge_poly[1][1] >> plane.edge_poly[1][2]
//                            >> plane.edge_poly[2][0] >> plane.edge_poly[2][1] >> plane.edge_poly[2][2]
//                            >> plane.edge_poly[3][0] >> plane.edge_poly[3][1] >> plane.edge_poly[3][2]
//                            >> plane.n[0] >> plane.n[1] >> plane.n[2]
//                            >> plane.d;
//
//            map_.planes.push_back(plane);
//        }
//
//
//    }
    std::cout << "finish facades loading" << std::endl;
}

float PrimitivesMapping::dist2D_Segment_to_Segment(const Eigen::Vector2f& seg_1_p0, const Eigen::Vector2f& seg_1_p1,
                                                  const Eigen::Vector2f& seg_2_p0, const Eigen::Vector2f& seg_2_p1,
                                                  bool& edgePointsUsed) {
    float SMALL_NUM = 0.0001;

    Eigen::Vector2f u = seg_1_p1 - seg_1_p0;
    Eigen::Vector2f v = seg_2_p1 - seg_2_p0;
    Eigen::Vector2f w = seg_1_p0 - seg_2_p0;
    float    a = u.dot(u);         // always >= 0
    float    b = u.dot(v);
    float    c = v.dot(v);         // always >= 0
    float    d = u.dot(w);
    float    e = v.dot(w);
    float    D = a*c - b*b;        // always >= 0
    float    sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
    float    tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0
    bool useEdgePointS = false;
    bool useEdgePointT = false;

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
            useEdgePointS = true;
        }
        else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
            useEdgePointS = true;
        }
    }

    if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
        useEdgePointT = true;
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
        useEdgePointT = true;
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

void PrimitivesMapping::build_map() {
    new_map_.cylinders.clear();
    new_map_.planes.clear();
    new_map_.grounds.clear();


    std::cout << " pole number in map before process: " << map_.cylinders.size() << std::endl;

    std::cout <<"Build map with following parameter: "<<std::endl
              <<"cylinder_min_num_detections: "<<cylinder_min_num_detections_<<std::endl
              <<"cylinder_cluster_pos_thresh: "<<cylinder_cluster_pos_thresh_<<std::endl
              <<"cylinder_max_radius_tolerance_from_median_percentage: "<<cylinder_max_radius_tolerance_from_median_percentage_<<std::endl
              <<"cylinder_max_ground_tolerance_from_median: "<<cylinder_max_ground_tolerance_from_median_<<std::endl
              <<"cylinder_max_xy_diff: "<<cylinder_max_xy_diff_<<std::endl
              <<"cylinder_max_angle_diff_from_average: "<<cylinder_max_angle_diff_from_average_<<std::endl
              <<"plane_cluster_pos_thresh: "<<plane_cluster_pos_thresh_<<std::endl
              <<"plane_cluster_angle_thresh: "<<plane_cluster_angle_thresh_<<std::endl
              <<"min_num_planes_in_cluster: "<<min_num_planes_in_cluster_<<std::endl
              <<"plane_sample_step_size: "<<plane_sample_step_size_<<std::endl
              <<"new_plane_inlier_max_dist: "<<new_plane_inlier_max_dist_<<std::endl;


    ///  Process Cylinder
    // First cluster cylinder.
    if (map_.cylinders.size() > cylinder_min_num_detections_) {

        // Check the rotation and the ratio of the cylinders. If too much tilted then remove it.
        std::vector<Cylinder_Fin_Param> temp;
        int bad_count = 0;
        Eigen::Vector3f z_dir(0.f, 0.f, 1.f);
        for (Cylinder_Fin_Param& c : map_.cylinders) {
            Eigen::Vector3f axis = c.center_line.p_top - c.center_line.p_bottom;
            double height = axis.norm();
            Eigen::Vector3f dir = axis.normalized();
            double angle = acos(dir.dot(z_dir));
            double ratio = height / c.radius;
            if (angle > 40 / 180.0 * M_PI || ratio < 1.5) {
                ++bad_count;
            } else {
                temp.push_back(c);
            }
        }
        std::cout << "#BAD CYLINDERS: " << bad_count << std::endl;

        map_.cylinders = temp;

        {
            std::cout << "Form cylinder clusters..." << std::endl;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (size_t i = 0; i < map_.cylinders.size(); ++i) {
                pcl::PointXYZ p;
                p.x = map_.cylinders[i].center_line.p_bottom[0];
                p.y = map_.cylinders[i].center_line.p_bottom[1];
                p.z = map_.cylinders[i].center_line.p_bottom[2];
                cloud->points.push_back(p);
            }
            kdtree.setInputCloud(cloud);


            // Vector which saves the unprocessed ids.
            std::vector<bool> processed(map_.cylinders.size(), false);

            // Vector which holds the clusters.
            struct Cluster {
                std::vector<int> ids;
            };
            std::vector<Cluster> cylinder_clusters;


            // AS long as there are unprocessed cylinders, create clusters.
            std::vector<bool>::iterator it;
            while ((it = std::find(processed.begin(), processed.end(), false)) != processed.end()) {

                std::cout << "Processed cylinders: " << std::count(processed.begin(), processed.end(), true) << "("
                          << processed.size() << ")" << std::endl;

                Cluster c;
                std::queue<size_t> expand_ids; // stores all ids on which to expand on --> radius search.
                expand_ids.push(std::distance(processed.begin(), it));
                c.ids.push_back(expand_ids.back());
                processed[expand_ids.back()] = true;
                while (!expand_ids.empty()) // Breadth first search.
                {
                    size_t id = expand_ids.back();
                    expand_ids.pop();

                    // Find close neighbors.
                    std::vector<int> pointIdxRadiusSearch;
                    std::vector<float> pointRadiusSquaredDistance;
                    pcl::PointXYZ root_point;
                    root_point.x = map_.cylinders[id].center_line.p_bottom[0];
                    root_point.y = map_.cylinders[id].center_line.p_bottom[1];
                    root_point.z = map_.cylinders[id].center_line.p_bottom[2];
                    kdtree.radiusSearch(
                        root_point, cylinder_cluster_pos_thresh_, pointIdxRadiusSearch, pointRadiusSquaredDistance);

                    // All close neighbors which are were not processed yet are added to the queue.
                    for (auto& close_elem : pointIdxRadiusSearch) {
                        const double& r1 = map_.cylinders[id].radius;
                        const double& r2 = map_.cylinders[close_elem].radius;
                        double rad_diff = std::abs(r1 - r2);
                        double max_rad_diff = std::max(0.05, std::max(r1, r2) * 0.25);
                        if (processed[close_elem] == false /*&& rad_diff < max_rad_diff*/) {
                            expand_ids.push(close_elem);
                            c.ids.push_back(close_elem);
                            processed[close_elem] = true;
                        }
                    }
                }

                cylinder_clusters.push_back(c);
            }
//            std::cout << "Found Cylinder Clusters: " << cylinder_clusters.size() << std::endl;



            // Merge the cylinder clusters.
            // Pick a cluster.
            std::cout << "Merge cylinder clusters..." << std::endl;



            for (auto& cluster : cylinder_clusters) {
                //    std::cout <<"---Cluster start size: " <<cluster.ids.size() <<std::endl;

                // Find median of cylinder radii in cluster.
                std::vector<float> radii;
                for (auto id : cluster.ids) {
                    radii.push_back(map_.cylinders[id].radius);
                }
                std::nth_element(radii.begin(), radii.begin() + radii.size() / 2, radii.end());
                float rad_median = radii[radii.size() / 2];


                // Filter out cylinders which have very different ground_z estimation.
                // Find median of cylinder ground_z in cluster.
                std::vector<float> ground_zs;
                for(auto id:cluster.ids)
                {
                    ground_zs.push_back(map_.cylinders[id].center_line.p_bottom[2]);
                }
                std::nth_element(ground_zs.begin(), ground_zs.begin() + ground_zs.size()/2, ground_zs.end());
                float ground_z_median = ground_zs[ground_zs.size()/2];
                {
                    std::vector<int> valid_ids;
                    for(auto id:cluster.ids)
                    {
                        if( std::abs(map_.cylinders[id].center_line.p_bottom[2] - ground_z_median) < cylinder_max_ground_tolerance_from_median_)
                        {
                            valid_ids.push_back(id);
                        }
                    }
                    cluster.ids = valid_ids;
                }


                // Filter out cylinders which have very different radius.
                {
                    float cylinder_max_radius_tolerance_from_median =
                        std::max(cylinder_max_radius_tolerance_from_median_percentage_ * rad_median, 0.05f);
                    std::vector<int> valid_ids;
                    for (auto id : cluster.ids) {
                        if (std::abs(map_.cylinders[id].radius - rad_median) <
                            cylinder_max_radius_tolerance_from_median) {
                            valid_ids.push_back(id);
                        }
                    }
                    cluster.ids = valid_ids;
                }

                //    std::cout <<"---Cluster radius size: " <<cluster.ids.size() <<std::endl;

                // Calculate the median of top and bottom points
                // Find xy-average of middle points.
                Eigen::Vector2f average_middle(0.0f, 0.0f);
                for (auto id : cluster.ids) {
                    average_middle +=
                        ((map_.cylinders[id].center_line.p_top + map_.cylinders[id].center_line.p_bottom) / 2.0)
                            .topRows(2);
                }
                average_middle /= (float)cluster.ids.size();

                // Filter out cylinders which are too far away.
                {
                    std::vector<int> valid_ids;
                    for (auto id : cluster.ids) {
                        Eigen::Vector2f middle =
                            ((map_.cylinders[id].center_line.p_top + map_.cylinders[id].center_line.p_bottom) / 2.0)
                                .topRows(2);
                        if ((middle - average_middle).norm() < cylinder_max_xy_diff_) {
                            valid_ids.push_back(id);
                        }
                    }
                    cluster.ids = valid_ids;
                }


                //    std::cout <<"---Cluster distance size: " <<cluster.ids.size() <<std::endl;

                // Find average direction vector.
                Eigen::Vector3f average_dir(0.0f, 0.0f, 0.0f);
                for (auto id : cluster.ids) {
                    average_dir +=
                        (map_.cylinders[id].center_line.p_top - map_.cylinders[id].center_line.p_bottom).normalized();
                    //      std::cout <<(map_.cylinders[id].center_line.p_top - map_.cylinders[id].center_line.p_bottom).normalized()<<std::endl;
                }
                average_dir = average_dir.normalized();
                //    std::cout <<average_dir<<std::endl;

                // Filter out cylinders which have too large angle to average dir vector.
                {
                    std::vector<int> valid_ids;
                    for (auto id : cluster.ids) {
                        Eigen::Vector3f dir_vec =
                            (map_.cylinders[id].center_line.p_top - map_.cylinders[id].center_line.p_bottom)
                                .normalized();
                        float angle_diff = 1 - std::abs(dir_vec.dot(average_dir));
                        //        std::cout <<"Angle diff " <<angle_diff <<std::endl;
                        if (angle_diff < cylinder_max_angle_diff_from_average_) {
                            valid_ids.push_back(id);
                        }
                    }
                    cluster.ids = valid_ids;
                }
                //    std::cout <<"---Cluster angle size: " <<cluster.ids.size() <<std::endl;

                // If there are still enough supporting cylinders then merge.
                if ((int)cluster.ids.size() >= cylinder_min_num_detections_) {

                    Cylinder_Fin_Param c;

                    // Find average radius.
                    float average_radius = 0;
                    for (auto id : cluster.ids) {
                        average_radius += map_.cylinders[id].radius;
                    }
                    average_radius /= (float)cluster.ids.size();


                    // Find direction.
                    average_dir = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
                    for (auto id : cluster.ids) {
                        average_dir += (map_.cylinders[id].center_line.p_top - map_.cylinders[id].center_line.p_bottom)
                                           .normalized();
                    }
                    average_dir.normalize();


                    // Find top.
                    float average_z_top = 0;
                    for (auto id : cluster.ids) {
                        average_z_top += map_.cylinders[id].center_line.p_top(2);
                    }
                    average_z_top /= (float)cluster.ids.size();

                    // Find bottom.
                    float average_z_bottom;
                    for (auto id : cluster.ids) {
                        average_z_bottom += map_.cylinders[id].center_line.p_bottom(2);
                    }
                    average_z_bottom /= (float)cluster.ids.size();

                    // Average ground height.
                    float average_ground_z = average_z_bottom;

                    average_middle = Eigen::Vector2f(0.0f,0.0f);
                    for(auto id:cluster.ids)
                    {
                        average_middle += ((map_.cylinders[id].center_line.p_top + map_.cylinders[id].center_line.p_bottom)/2.0).topRows(2);
                    }
                    average_middle /= (float)cluster.ids.size();

                    // Find 3D center.
                    Eigen::Vector3f center_3D = Eigen::Vector3f(average_middle(0), average_middle(1),
                                                                (average_z_top-average_z_bottom)/2.0 + average_z_bottom);

                    // Calc p_top.
                    c.center_line.p_top = center_3D + average_dir*((average_z_top-average_z_bottom)/2.0)/average_dir(2);
                    // Calc p_bottom.
                    c.center_line.p_bottom = center_3D + average_dir*((average_z_bottom-average_z_top)/2.0)/average_dir(2);
                    // Set rest of parameters.
                    c.radius = average_radius;

//                    int color = rand()%20000;
//                    createTheoryCylinder(c.center_line.p_bottom, c.center_line.p_top, c.radius, tmp_cloud, 4, 5, color);

                    new_map_.cylinders.push_back(c);
                    //        std::cout <<"---MERGED CYLINDER---" <<std::endl;
                }
            }
        }
    }


    /// Process planes.
    // First cluster planes.
    {
        std::cout <<"Form plane clusters..."<<std::endl;

        // 1. For each facade find a neighbor which fulfill those three conditions:
        //      a. Distance of selected facade center point to candidates facades less than threshold
        //      b. Two facades normal should be parallel
        //      c. Two facades should be overlapped


        std::vector<Plane_Param> showFacades;
        std::vector<bool> processed (map_.planes.size(),false); // iteration end sign
        std::vector<bool>::iterator it;


        std::vector<std::vector<int>> clusteredFacades; // First: index of cluster
                                                        // Second: index for each cluster number
        while((it = std::find( processed.begin(),processed.end(),false)) != processed.end()) {
            for (int i = 0; i < map_.planes.size(); ++i) { // The selected facade

                if (processed[i] == true) continue;


                std::vector<int> clusterIndex;
                clusterIndex.push_back(i);

                // vector used to store suitable facade index
                for (int j = 0; j < map_.planes.size(); ++j) { // To find facade which fulfill the three conditions
                    // turn the selected facade to true
                    if (processed[j] == true) continue;

                    // the selected facade value
                    Eigen::Vector3f first_fa_dl = map_.planes[i].edge_poly[0];
                    Eigen::Vector3f first_fa_dr = map_.planes[i].edge_poly[1];
                    Eigen::Vector3f first_fa_tr = map_.planes[i].edge_poly[2];

                    Eigen::Vector3f first_fa_normal = (first_fa_dr - first_fa_dl).cross(first_fa_tr - first_fa_dl);
                    Eigen::Vector3f first_fa_normal_norma = first_fa_normal.normalized();
                    Eigen::Vector3f first_center = first_fa_dl + 0.5 * (first_fa_tr - first_fa_dl);

                    // the facade to find
                    Eigen::Vector3f second_fa_dl = map_.planes[j].edge_poly[0];
                    Eigen::Vector3f second_fa_dr = map_.planes[j].edge_poly[1];
                    Eigen::Vector3f second_fa_tr = map_.planes[j].edge_poly[2];

                    Eigen::Vector3f second_fa_normal_norma = ((second_fa_dr - second_fa_dl).cross(second_fa_tr - second_fa_dl)).normalized();
                    Eigen::Vector3f second_center = second_fa_dl + 0.5 * (second_fa_tr - second_fa_dl);

                    // flag to verify conditions
                    double angle_diff = ( first_fa_normal_norma.cross(second_fa_normal_norma) ).norm();
                    float dis_diff = dist_plane_plane(first_fa_dl, first_fa_normal_norma, second_fa_dl, second_fa_tr);

                    float diagonal_len = (first_center - second_center).norm();
                    float non_overlap_dia_len = 0.5 * ((first_fa_tr - first_fa_dl).norm() + (second_fa_tr - second_fa_dl).norm());

                    bool dis_flag = dis_diff < plane_cluster_pos_thresh_ ? true : false;
                    bool dir_flag = angle_diff > 0.13 ? true : false;
                    bool overlap_flag = diagonal_len < 2.0 * non_overlap_dia_len ? true : false;


                    if (dis_flag && dir_flag && overlap_flag) {
                        processed[j] = true;
                        clusterIndex.push_back(j);
                    }
                }
                clusteredFacades.push_back(clusterIndex);
                clusterIndex.clear();
                processed[i] = true;
                // if suitable facades is less than 2 then do not merge.
//                if (clusteredFacades[i].empty()) {
//                    std::
//                }
            }
        }
        std::cerr << "Found: " << clusteredFacades.size() << " clusters " << std::endl;


        // Put clustered facades in vector for next step merging
        fusionFacadesClusters.resize(clusteredFacades.size());
        int clusterIndex = 0;
        for (auto& cluster : clusteredFacades) {
            int intensity = rand()%10000;
            std::vector<Plane_Param> planes;
            for (int i = 0; i < cluster.size(); ++i) {
                Plane_Param p = map_.planes[cluster[i]];
                planes.push_back(p);
            }
            Cloud::Ptr tmpCluster(new Cloud);
            getPrimitiveFacadeCloud(tmpCluster, planes, intensity);
            fusionFacadesClusters[clusterIndex] = tmpCluster;
            clusterIndex++;
        }

        // Merge clustered facades and find new plane equation and edge poly
        for (int facadeCloudId = 0; facadeCloudId < fusionFacadesClusters.size(); facadeCloudId++) {
            //        std::cout << "size " << facadeCandidates.size() << std::endl;
            Cloud::Ptr facadeCloud(fusionFacadesClusters[facadeCloudId]);
            // Check if the cloud has reasonable large xy dimension.
            float x_min = 9999;
            float x_max = -9999;
            float y_min = 9999;
            float y_max = -9999;
            for (auto& p : facadeCloud->points) {
                if (p.x < x_min)
                    x_min = p.x;
                else if (p.x > x_max)
                    x_max = p.x;
                if (p.y < y_min)
                    y_min = p.y;
                else if (p.y > y_max)
                    y_max = p.y;
            }
            if ((x_max - x_min < 1.0) && (y_max - y_min < 1.0)) {
                continue;
            }
            // sample five points and PCA filter out horizontal plane
            // PCA
            Eigen::Vector4f pcaCentroid;
            Eigen::Matrix3f covariance;
            Cloud::Ptr pcaCloud(new Cloud());
            // sample five points randomly
            int numSample = 5;
            while (numSample) {
                pcaCloud->points.push_back(facadeCloud->points[std::rand() % facadeCloud->points.size()]);
                numSample--;
            }
            pcl::compute3DCentroid(*pcaCloud, pcaCentroid);
            pcl::computeCovarianceMatrixNormalized(*pcaCloud, pcaCentroid, covariance);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance, Eigen::ComputeEigenvectors);
            Eigen::Matrix3f eigenVectorsPCA = eigenSolver.eigenvectors();
            Eigen::Vector3f eigenValuesPCA = eigenSolver.eigenvalues();
            // If the smallest eigen value is much smaller than the others -> facade.
            // And the smallest eigen vector should be vertical to z axis.
            Eigen::Vector3f smallEigenVector = eigenVectorsPCA.block<3, 1>(0, 0);
            // cos between z axis should be less 0.03
            float cosAngle = abs(smallEigenVector[2]);
            float angleThreshold = (float)85 / 180 * M_PI; // angle dif threshold 5 deg.
            if (cosAngle > cos(angleThreshold)) {
                // this cluster is not facade
                continue;
            }


            // Estimate facades model for each cluster
            Plane_Param facadeParam;
            // Accept initial norm from PCA and d for ceres
            Eigen::Vector3f normal;
            float d;
            bool validFacadeFlag = findNewMergedFacadesRANSAC(facadeCloud, normal, d);
            if (validFacadeFlag == true) {
                facadeParam.n = normal;
                facadeParam.d = d;
            }
            // Find projected position
            Eigen::Vector3f width_dir = (facadeParam.n.cross(Eigen::Vector3f(0.0f, 0.0f, 1.0f))).normalized();
            Eigen::Vector3f height_dir = (width_dir.cross(facadeParam.n)).normalized();
            Eigen::Vector3f p1(
                facadeCloud->points[8].x, facadeCloud->points[8].y, facadeCloud->points[8].z); // sample a point in facade
            Eigen::Vector3f projection;
            Eigen::Vector3f diff_to_p1;
            float pixel_size_m = 0.2f;
            float pixel_size_factor = 1.0f / pixel_size_m; // pixel size 0.2
            float dist;
            int proj_height_pixel, proj_width_pixel;
            std::vector<int> supporters;
            std::vector<float> distances;
            std::vector<std::pair<int, int>> proj_pos_pixel; // Relative to support point. First->height, Second->width.
            int id = 0;
            for (auto& p : facadeCloud->points) {
                diff_to_p1 = Eigen::Vector3f(p.x, p.y, p.z) - p1;
                dist = facadeParam.n.dot(diff_to_p1);
                distances.push_back(dist);

                // Find a 2D coordinate for projection to a 2D plane image.
                projection = diff_to_p1 - facadeParam.n * dist;
                proj_height_pixel = (int)std::round(projection.dot(height_dir) * pixel_size_factor);
                proj_width_pixel = (int)std::round(projection.dot(width_dir) * pixel_size_factor);
                proj_pos_pixel.push_back(std::make_pair(proj_height_pixel, proj_width_pixel));

                // If the point is a plane supporter.
                if (std::abs(dist) < 0.1) {
                    supporters.push_back(id);
                }
                ++id;
            }

            // Put projected points on image.
            // First find image boundaries.
            int max_height = proj_pos_pixel[supporters[0]].first;
            int min_height = max_height;
            int max_width = proj_pos_pixel[supporters[0]].second;
            int min_width = max_width;

            for (auto& s : supporters) {
                if (proj_pos_pixel[s].first > max_height)
                    max_height = proj_pos_pixel[s].first;
                else if (proj_pos_pixel[s].first < min_height)
                    min_height = proj_pos_pixel[s].first;
                if (proj_pos_pixel[s].second > max_width)
                    max_width = proj_pos_pixel[s].second;
                else if (proj_pos_pixel[s].second < min_width)
                    min_width = proj_pos_pixel[s].second;
            }
            int img_height = max_height - min_height + 1;
            int img_width = max_width - min_width + 1;


            // Check if plane large enough.
            float plane_height = img_height * pixel_size_m;
            float plane_width = img_width * pixel_size_m;
            float plane_min_height = 1.5f;
            float plane_long_width = 1;
            float plane_min_width = 1.0f;
            if ((plane_height < plane_min_height && plane_width < plane_long_width) ||
                (plane_height < plane_min_height / 3.0 && plane_width >= plane_long_width) ||
                plane_width < plane_min_width) {
                continue;
            }

            // Now work on the image.

            cv::Mat projection_image = cv::Mat::zeros(img_height, img_width, CV_8U);

            // Fill the image.
            int pos_x, pos_y;
            for (auto& s : supporters) {
                pos_y = proj_pos_pixel[s].first - min_height;
                pos_x = proj_pos_pixel[s].second - min_width;
                if (projection_image.ptr<uchar>(pos_y)[pos_x] < 255) {
                    (projection_image.ptr<uchar>(pos_y)[pos_x])++;
                }
            }
            // Get binary occupancy image.
            cv::Mat occupancy_img = cv::Mat::zeros(img_height, img_width, CV_8U);
            uchar* img_ptr;
            for (int row_id = 0; row_id < img_height; ++row_id) {
                img_ptr = projection_image.ptr<uchar>(row_id);
                for (int col_id = 0; col_id < img_width; ++col_id) {
                    if (img_ptr[col_id] != 0) {
                        occupancy_img.ptr<uchar>(row_id)[col_id] = 1;
                    }
                }
            }
            cv::Mat integral_img(occupancy_img.rows + 1, occupancy_img.cols + 1, CV_32S);
            cv::integral(occupancy_img, integral_img, CV_32S);


            // Check if there are big vertical gaps in the plane image.
            float max_gap_meter = 2.0;
            int max_gap_pixel = std::max(1, (int)std::round(max_gap_meter * pixel_size_factor));
            int gap_counter = max_gap_pixel;
            float min_column_occupany_meter = 0.3;
            int min_column_occupany_pixel = (int)std::round(min_column_occupany_meter * pixel_size_factor);
            std::vector<int> start_col;
            std::vector<int> end_col;
            int* integral_img_ptr = integral_img.ptr<int>(img_height);
            bool end_set = true;
            // Go through all columns.
            for (int col_id = 0; col_id < img_width; ++col_id) // Column id of real image not integral img!
            {
                // If there are too few points in the current column.
                if (integral_img_ptr[col_id + 1] - integral_img_ptr[col_id] <= min_column_occupany_pixel) {
                    // Increment gap counter.
                    ++gap_counter;
                    // If the gap is large and the end of the last split is not stored yet.
                    if (gap_counter >= max_gap_pixel && !end_set) {
                        // Store the end of the split.
                        end_col.push_back(col_id - max_gap_pixel);
                        end_set = true;
                    }
                }
                    // If there is at least one point in the column.
                else {
                    // If the gap counter is still high (the last column was part of a gap).
                    if (gap_counter >= max_gap_pixel) {
                        // Define this column as the start of a new split.
                        start_col.push_back(col_id);
                        // Store that a end was not set yet.
                        end_set = false;
                    }
                    // Reset the gap counter.
                    gap_counter = 0;
                }
            }
            if (end_set == false) {
                end_col.push_back(img_width - 1 - gap_counter);
            }


            // Create new images for splits.
            std::vector<cv::Mat> splits;
            int min_split_width_pixel = (int)std::round(plane_min_width * pixel_size_factor);
            std::vector<int> valid_width_split_ids;
            for (int s = 0; s < (int)start_col.size(); ++s) {
                // Check if split is wide enough.

                int split_width = end_col[s] - start_col[s] + 1;
                if (split_width < min_split_width_pixel) {
                    //        std::cout <<"SPLIT TOO NARROW" <<start_col[s] <<"  " <<end_col[s] <<std::endl;
                    continue;
                }

                // Store split.
                splits.push_back(occupancy_img(cv::Rect(start_col[s], 0, split_width, img_height)));
                valid_width_split_ids.push_back(s);
            }



            // Improve the vertical dimension of the splits.
            std::vector<int> top_row(splits.size());
            std::vector<int> bottom_row(splits.size());
            for (int s = 0; s < (int)splits.size(); ++s) {
                // Start at the top and look whether the rows are empty or not.
                bool finished_top = false;
                for (int row_id = 0; row_id < splits[s].rows; ++row_id) {
                    top_row[s] = row_id;
                    uchar* split_ptr = splits[s].ptr<uchar>(row_id);
                    int counter = 0;
                    for (int col_id = 0; col_id < splits[s].cols; ++col_id) {
                        if (split_ptr[col_id] != 0) {
                            counter++;
                        }
                        if (counter >= max_gap_pixel) {
                            finished_top = true;
                            break;
                        }
                    }
                    if (finished_top) {
                        break;
                    }
                }

                // Start at the bottom and look whether the rows are empty or not.
                bool finished_bottom = false;
                for (int row_id = splits[s].rows - 1; row_id >= 0; --row_id) {
                    bottom_row[s] = row_id;
                    uchar* split_ptr = splits[s].ptr<uchar>(row_id);
                    int counter = 0;
                    for (int col_id = 0; col_id < splits[s].cols; ++col_id) {
                        if (split_ptr[col_id] != 0) {
                            counter++;
                        }
                        if (counter >= max_gap_pixel) {
                            finished_bottom = true;
                            break;
                        }
                    }
                    if (finished_bottom) {
                        break;
                    }
                }
            }

            // Filter out splits which are not high enough or the occupancy is too low.
            float large_height_thresh_m = 5.0f;
            float min_plane_2D_occupancy_percantage = 0.1;
            int plane_min_height_pixel = (int)std::round(plane_min_height * pixel_size_factor);
            int plane_min_width_pixel = (int)std::round(plane_min_width * pixel_size_factor);
            int plane_long_width_pixel = (int)std::round(plane_long_width * pixel_size_factor);
            int large_height_thresh_pixel = (int)std::round(large_height_thresh_m * pixel_size_factor);

            std::vector<std::pair<int, int>>
                col_row_ids_for_valid_splits; // Stores indices of start/end_col (first) and top/bottom_row (second) which
            // correspond to valid split.
            std::vector<cv::Mat> valid_dimensions_splits; // Stores splits which are valid in width and height.
            for (int s = 0; s < (int)splits.size(); ++s) {
                int plane_height_pixel = bottom_row[s] - top_row[s];
                int plane_width_pixel = splits[s].cols;

                // Calc occupancy for split.
                int occupancy = integral_img.ptr<int>(bottom_row[s] + 1)[end_col[valid_width_split_ids[s]] + 1] -
                                integral_img.ptr<int>(top_row[s])[end_col[valid_width_split_ids[s]] + 1] -
                                integral_img.ptr<int>(bottom_row[s] + 1)[start_col[valid_width_split_ids[s]]] +
                                integral_img.ptr<int>(top_row[s])[start_col[valid_width_split_ids[s]]];

                // If occupancy is high enough.
                int large_height_reduction_factor = 1; // Compensates the lower density of the lidar in large heights.
                if (bottom_row[s] - top_row[s] + 1 > large_height_thresh_pixel)
                    large_height_reduction_factor = 3;
                if (large_height_reduction_factor * occupancy >
                    std::round(min_plane_2D_occupancy_percantage * (bottom_row[s] - top_row[s] + 1) * splits[s].cols)) {
                    if ((plane_height_pixel >= plane_min_height_pixel && plane_width_pixel > plane_min_width_pixel) ||
                        (plane_height_pixel >= std::round(plane_min_height_pixel / 3.0) &&
                         plane_width_pixel >= plane_long_width_pixel)) {
                        valid_dimensions_splits.push_back(
                            splits[s](cv::Rect(0, top_row[s], splits[s].cols, bottom_row[s] - top_row[s] + 1)));
                        col_row_ids_for_valid_splits.push_back(std::make_pair(valid_width_split_ids[s], s));
                    }
                }
            }
            // Store the planes.
            for (auto& id : col_row_ids_for_valid_splits) {
                Plane_Param plane_param;
                plane_param.n = facadeParam.n;
                plane_param.d = facadeParam.d;

                Eigen::Vector3f top_left, top_right, bottom_left, bottom_right;
                top_left = p1 + (min_height + top_row[id.second]) * pixel_size_m * height_dir +
                           (min_width + start_col[id.first]) * pixel_size_m * width_dir;
                top_right = p1 + (min_height + top_row[id.second]) * pixel_size_m * height_dir +
                            (min_width + end_col[id.first]) * pixel_size_m * width_dir;
                bottom_left = p1 + (min_height + bottom_row[id.second]) * pixel_size_m * height_dir +
                              (min_width + start_col[id.first]) * pixel_size_m * width_dir;
                bottom_right = p1 + (min_height + bottom_row[id.second]) * pixel_size_m * height_dir +
                               (min_width + end_col[id.first]) * pixel_size_m * width_dir;

                // The bottom right and left are used for localization
                plane_param.edge_poly.push_back(top_left);
                plane_param.edge_poly.push_back(top_right);
                plane_param.edge_poly.push_back(bottom_right);
                plane_param.edge_poly.push_back(bottom_left);


                float ground_z_start = bottom_left[2];
                float ground_z_end = bottom_right[2];

                plane_param.ground_z_start = ground_z_start;
                plane_param.ground_z_end = ground_z_end;
                // The 2D line left -> right
                plane_param.line_2D.p_start = Eigen::Vector2f(bottom_left(0), bottom_left(1));
                plane_param.line_2D.p_end = Eigen::Vector2f(bottom_right(0), bottom_right(1));
                new_map_.planes.push_back(plane_param);
            }
        }

    }





    /// Process grounds.
    // First cluster grounds.
    {
        std::cout <<"Form ground clusters..."<<std::endl;


        std::vector<bool> processed (map_.grounds.size(),false); // iteration end sign
        std::vector<bool>::iterator it;


        std::vector<std::vector<int>> clusteredGround; // First: index of cluster
        // Second: index for each cluster number
        while((it = std::find( processed.begin(),processed.end(),false)) != processed.end()) {

            for (int i = 0; i < map_.grounds.size(); ++i) { // The selected facade

                if (processed[i] == true) continue;


                std::vector<int> clusterIndex;
                clusterIndex.push_back(i);

                // vector used to store suitable facade index
                for (int j = 0; j < map_.grounds.size(); ++j) { // To find facade which fulfill the three conditions
                    // turn the selected facade to true
                    if (processed[j] == true) continue;

                    // the selected facade value
                    Eigen::Vector3f first_fa_dl = map_.grounds[i].edge_poly[0];
                    Eigen::Vector3f first_fa_dr = map_.grounds[i].edge_poly[1];
                    Eigen::Vector3f first_fa_tr = map_.grounds[i].edge_poly[2];

                    Eigen::Vector3f first_fa_normal = (first_fa_dr - first_fa_dl).cross(first_fa_tr - first_fa_dl);
                    Eigen::Vector3f first_fa_normal_norma = first_fa_normal.normalized();
                    Eigen::Vector3f first_center = first_fa_dl + 0.5 * (first_fa_tr - first_fa_dl);

                    // the facade to find
                    Eigen::Vector3f second_fa_dl = map_.grounds[j].edge_poly[0];
                    Eigen::Vector3f second_fa_dr = map_.grounds[j].edge_poly[1];
                    Eigen::Vector3f second_fa_tr = map_.grounds[j].edge_poly[2];

                    Eigen::Vector3f second_fa_normal_norma = ((second_fa_dr - second_fa_dl).cross(second_fa_tr - second_fa_dl)).normalized();
                    Eigen::Vector3f second_center = second_fa_dl + 0.5 * (second_fa_tr - second_fa_dl);

                    // flag to verify conditions
                    double angle_diff = ( first_fa_normal_norma.cross(second_fa_normal_norma) ).norm();
                    float dis_diff = dist_plane_plane(first_fa_dl, first_fa_normal_norma, second_fa_dl, second_fa_tr);

                    float diagonal_len = (first_center - second_center).norm();
                    float non_overlap_dia_len = 0.5 * ((first_fa_tr - first_fa_dl).norm() + (second_fa_tr - second_fa_dl).norm());

                    bool dis_flag = dis_diff < 0.05 ? true : false;
                    bool dir_flag = angle_diff > 0.09 ? true : false;
                    bool overlap_flag = diagonal_len < 4.0 * non_overlap_dia_len ? true : false;


                    if (dis_flag && dir_flag && overlap_flag) {
                        processed[j] = true;
                        clusterIndex.push_back(j);
                    }
                }
                clusteredGround.push_back(clusterIndex);
                clusterIndex.clear();
                processed[i] = true;
            }
        }
        std::cerr << "Found: " << clusteredGround.size() << " clusters " << std::endl;


        fusionGroundsClusters.resize(clusteredGround.size());
        int clusterIndex = 0;
        for (auto& cluster : clusteredGround) {
            int intensity = rand()%10000;
            std::vector<Plane_Param> planes;
            for (int i = 0; i < cluster.size(); ++i) {
                Plane_Param p = map_.grounds[cluster[i]];
                planes.push_back(p);
            }
            Cloud::Ptr tmpCluster(new Cloud);
            getPrimitiveFacadeCloud(tmpCluster, planes, intensity);
            fusionGroundsClusters[clusterIndex] = tmpCluster;
            clusterIndex++;
        }

        // Merge clustered facades and find new plane equation and edge poly
        for (int facadeCloudId = 0; facadeCloudId < fusionGroundsClusters.size(); facadeCloudId++) {
            //        std::cout << "size " << facadeCandidates.size() << std::endl;
            Cloud::Ptr facadeCloud(fusionGroundsClusters[facadeCloudId]);


            // Estimate facades model for each cluster
            Plane_Param facadeParam;
            // Accept initial norm from PCA and d for ceres
            Eigen::Vector3f normal;
            float d;
            bool validFacadeFlag = findNewMergedFacadesRANSAC(facadeCloud, normal, d);
            if (validFacadeFlag == true) {
                facadeParam.n = normal;
                facadeParam.d = d;
            }
            // Find projected position
            Eigen::Vector3f width_dir = (facadeParam.n.cross(Eigen::Vector3f(1.0f, 0.0f, 0.0f))).normalized();
            Eigen::Vector3f height_dir = (width_dir.cross(facadeParam.n)).normalized();
            Eigen::Vector3f p1(
                facadeCloud->points[8].x, facadeCloud->points[8].y, facadeCloud->points[8].z); // sample a point in facade
            Eigen::Vector3f projection;
            Eigen::Vector3f diff_to_p1;
            float pixel_size_m = 0.2f;
            float pixel_size_factor = 1.0f / pixel_size_m; // pixel size 0.2
            float dist;
            int proj_height_pixel, proj_width_pixel;
            std::vector<int> supporters;
            std::vector<float> distances;
            std::vector<std::pair<int, int>> proj_pos_pixel; // Relative to support point. First->height, Second->width.
            int id = 0;
            for (auto& p : facadeCloud->points) {
                diff_to_p1 = Eigen::Vector3f(p.x, p.y, p.z) - p1;
                dist = facadeParam.n.dot(diff_to_p1);
                distances.push_back(dist);

                // Find a 2D coordinate for projection to a 2D plane image.
                projection = diff_to_p1 - facadeParam.n * dist;
                proj_height_pixel = (int)std::round(projection.dot(height_dir) * pixel_size_factor);
                proj_width_pixel = (int)std::round(projection.dot(width_dir) * pixel_size_factor);
                proj_pos_pixel.push_back(std::make_pair(proj_height_pixel, proj_width_pixel));

                // If the point is a plane supporter.
                if (std::abs(dist) < 0.1) {
                    supporters.push_back(id);
                }
                ++id;
            }

            // Put projected points on image.
            // First find image boundaries.
            int max_height = proj_pos_pixel[supporters[0]].first;
            int min_height = max_height;
            int max_width = proj_pos_pixel[supporters[0]].second;
            int min_width = max_width;

            for (auto& s : supporters) {
                if (proj_pos_pixel[s].first > max_height)
                    max_height = proj_pos_pixel[s].first;
                else if (proj_pos_pixel[s].first < min_height)
                    min_height = proj_pos_pixel[s].first;
                if (proj_pos_pixel[s].second > max_width)
                    max_width = proj_pos_pixel[s].second;
                else if (proj_pos_pixel[s].second < min_width)
                    min_width = proj_pos_pixel[s].second;
            }
            int img_height = max_height - min_height + 1;
            int img_width = max_width - min_width + 1;


            // Check if plane large enough.
            float plane_height = img_height * pixel_size_m;
            float plane_width = img_width * pixel_size_m;
            float plane_min_height = 1.0f;
            float plane_long_width = 1;
            float plane_min_width = 1.0f;
            if ((plane_height < plane_min_height && plane_width < plane_long_width) ||
                (plane_height < plane_min_height / 3.0 && plane_width >= plane_long_width) ||
                plane_width < plane_min_width) {
                continue;
            }

            // Now work on the image.

            cv::Mat projection_image = cv::Mat::zeros(img_height, img_width, CV_8U);

            // Fill the image.
            int pos_x, pos_y;
            for (auto& s : supporters) {
                pos_y = proj_pos_pixel[s].first - min_height;
                pos_x = proj_pos_pixel[s].second - min_width;
                if (projection_image.ptr<uchar>(pos_y)[pos_x] < 255) {
                    (projection_image.ptr<uchar>(pos_y)[pos_x])++;
                }
            }

//                    cv::namedWindow("plane_projection",cv::WINDOW_NORMAL);
//                    cv::imshow("plane_projection",projection_image*100);
//                    cv::waitKey();

            // Get binary occupancy image.
            cv::Mat occupancy_img = cv::Mat::zeros(img_height, img_width, CV_8U);
            uchar* img_ptr;
            for (int row_id = 0; row_id < img_height; ++row_id) {
                img_ptr = projection_image.ptr<uchar>(row_id);
                for (int col_id = 0; col_id < img_width; ++col_id) {
                    if (img_ptr[col_id] != 0) {
                        occupancy_img.ptr<uchar>(row_id)[col_id] = 1;
                    }
                }
            }
            cv::Mat integral_img(occupancy_img.rows + 1, occupancy_img.cols + 1, CV_32S);
            cv::integral(occupancy_img, integral_img, CV_32S);

            // ---------------- Visualization ---------------------
            //        {
            //          cv::namedWindow("occupancy img",cv::WINDOW_NORMAL);
            //          cv::Mat visu;
            //          cv::flip(occupancy_img*255,visu,0);
            //          cv::imshow("occupancy img",visu);
            //          cv::waitKey();
            //        }
            //        cv::namedWindow("integral img",cv::WINDOW_NORMAL);
            //        cv::imshow("integral img",integral_img*10);
            //        cv::waitKey();
            // ---------------- Visualization ---------------------


            // Check if there are big vertical gaps in the plane image.
            float max_gap_meter = 0.8;
            int max_gap_pixel = std::max(1, (int)std::round(max_gap_meter * pixel_size_factor));
            int gap_counter = max_gap_pixel;
            float min_column_occupany_meter = 0.3;
            int min_column_occupany_pixel = (int)std::round(min_column_occupany_meter * pixel_size_factor);
            std::vector<int> start_col;
            std::vector<int> end_col;
            int* integral_img_ptr = integral_img.ptr<int>(img_height);
            bool end_set = true;
            // Go through all columns.
            for (int col_id = 0; col_id < img_width; ++col_id) // Column id of real image not integral img!
            {
                // If there are too few points in the current column.
                if (integral_img_ptr[col_id + 1] - integral_img_ptr[col_id] <= min_column_occupany_pixel) {
                    // Increment gap counter.
                    ++gap_counter;
                    // If the gap is large and the end of the last split is not stored yet.
                    if (gap_counter >= max_gap_pixel && !end_set) {
                        // Store the end of the split.
                        end_col.push_back(col_id - max_gap_pixel);
                        end_set = true;
                    }
                }
                    // If there is at least one point in the column.
                else {
                    // If the gap counter is still high (the last column was part of a gap).
                    if (gap_counter >= max_gap_pixel) {
                        // Define this column as the start of a new split.
                        start_col.push_back(col_id);
                        // Store that a end was not set yet.
                        end_set = false;
                    }
                    // Reset the gap counter.
                    gap_counter = 0;
                }
            }
            if (end_set == false) {
                end_col.push_back(img_width - 1 - gap_counter);
            }


            // Create new images for splits.
            std::vector<cv::Mat> splits;
            int min_split_width_pixel = (int)std::round(plane_min_width * pixel_size_factor);
            std::vector<int> valid_width_split_ids;
            for (int s = 0; s < (int)start_col.size(); ++s) {
                // Check if split is wide enough.

                int split_width = end_col[s] - start_col[s] + 1;
                if (split_width < min_split_width_pixel) {
                    //        std::cout <<"SPLIT TOO NARROW" <<start_col[s] <<"  " <<end_col[s] <<std::endl;
                    continue;
                }

                // Store split.
                splits.push_back(occupancy_img(cv::Rect(start_col[s], 0, split_width, img_height)));
                valid_width_split_ids.push_back(s);
            }


            //---------------- Visualization -------------------
            //            for( int i = 0; i < (int)splits.size(); ++i )
            //            {
            //              std::string win_name = "split " + i;
            //              cv::namedWindow(win_name,cv::WINDOW_NORMAL);
            //              cv::imshow(win_name,splits[i]);
            //              cv::waitKey();
            //            }
            //---------------- Visualization -------------------


            // Improve the vertical dimension of the splits.
            std::vector<int> top_row(splits.size());
            std::vector<int> bottom_row(splits.size());
            for (int s = 0; s < (int)splits.size(); ++s) {
                // Start at the top and look whether the rows are empty or not.
                bool finished_top = false;
                for (int row_id = 0; row_id < splits[s].rows; ++row_id) {
                    top_row[s] = row_id;
                    uchar* split_ptr = splits[s].ptr<uchar>(row_id);
                    int counter = 0;
                    for (int col_id = 0; col_id < splits[s].cols; ++col_id) {
                        if (split_ptr[col_id] != 0) {
                            counter++;
                        }
                        if (counter >= max_gap_pixel) {
                            finished_top = true;
                            break;
                        }
                    }
                    if (finished_top) {
                        break;
                    }
                }

                // Start at the bottom and look whether the rows are empty or not.
                bool finished_bottom = false;
                for (int row_id = splits[s].rows - 1; row_id >= 0; --row_id) {
                    bottom_row[s] = row_id;
                    uchar* split_ptr = splits[s].ptr<uchar>(row_id);
                    int counter = 0;
                    for (int col_id = 0; col_id < splits[s].cols; ++col_id) {
                        if (split_ptr[col_id] != 0) {
                            counter++;
                        }
                        if (counter >= max_gap_pixel) {
                            finished_bottom = true;
                            break;
                        }
                    }
                    if (finished_bottom) {
                        break;
                    }
                }
            }

            // Filter out splits which are not high enough or the occupancy is too low.
            float large_height_thresh_m = 5.0f;
            float min_plane_2D_occupancy_percantage = 0.1;
            int plane_min_height_pixel = (int)std::round(plane_min_height * pixel_size_factor);
            int plane_min_width_pixel = (int)std::round(plane_min_width * pixel_size_factor);
            int plane_long_width_pixel = (int)std::round(plane_long_width * pixel_size_factor);
            int large_height_thresh_pixel = (int)std::round(large_height_thresh_m * pixel_size_factor);

            std::vector<std::pair<int, int>>
                col_row_ids_for_valid_splits; // Stores indices of start/end_col (first) and top/bottom_row (second) which
            // correspond to valid split.
            std::vector<cv::Mat> valid_dimensions_splits; // Stores splits which are valid in width and height.
            for (int s = 0; s < (int)splits.size(); ++s) {
                int plane_height_pixel = bottom_row[s] - top_row[s];
                int plane_width_pixel = splits[s].cols;

                // Calc occupancy for split.
                int occupancy = integral_img.ptr<int>(bottom_row[s] + 1)[end_col[valid_width_split_ids[s]] + 1] -
                                integral_img.ptr<int>(top_row[s])[end_col[valid_width_split_ids[s]] + 1] -
                                integral_img.ptr<int>(bottom_row[s] + 1)[start_col[valid_width_split_ids[s]]] +
                                integral_img.ptr<int>(top_row[s])[start_col[valid_width_split_ids[s]]];

                // If occupancy is high enough.
                int large_height_reduction_factor = 1; // Compensates the lower density of the lidar in large heights.
                if (bottom_row[s] - top_row[s] + 1 > large_height_thresh_pixel)
                    large_height_reduction_factor = 3;
                if (large_height_reduction_factor * occupancy >
                    std::round(min_plane_2D_occupancy_percantage * (bottom_row[s] - top_row[s] + 1) * splits[s].cols)) {
                    if ((plane_height_pixel >= plane_min_height_pixel && plane_width_pixel > plane_min_width_pixel) ||
                        (plane_height_pixel >= std::round(plane_min_height_pixel / 3.0) &&
                         plane_width_pixel >= plane_long_width_pixel)) {
                        valid_dimensions_splits.push_back(
                            splits[s](cv::Rect(0, top_row[s], splits[s].cols, bottom_row[s] - top_row[s] + 1)));
                        col_row_ids_for_valid_splits.push_back(std::make_pair(valid_width_split_ids[s], s));
                    }
                }
            }
            // Store the planes.
            for (auto& id : col_row_ids_for_valid_splits) {
                Plane_Param plane_param;
                plane_param.n = facadeParam.n;
                plane_param.d = facadeParam.d;

                Eigen::Vector3f top_left, top_right, bottom_left, bottom_right;
                top_left = p1 + (min_height + top_row[id.second]) * pixel_size_m * height_dir +
                           (min_width + start_col[id.first]) * pixel_size_m * width_dir;
                top_right = p1 + (min_height + top_row[id.second]) * pixel_size_m * height_dir +
                            (min_width + end_col[id.first]) * pixel_size_m * width_dir;
                bottom_left = p1 + (min_height + bottom_row[id.second]) * pixel_size_m * height_dir +
                              (min_width + start_col[id.first]) * pixel_size_m * width_dir;
                bottom_right = p1 + (min_height + bottom_row[id.second]) * pixel_size_m * height_dir +
                               (min_width + end_col[id.first]) * pixel_size_m * width_dir;

                // The bottom right and left are used for localization
                plane_param.edge_poly.push_back(top_left);
                plane_param.edge_poly.push_back(top_right);
                plane_param.edge_poly.push_back(bottom_right);
                plane_param.edge_poly.push_back(bottom_left);


                float ground_z_start = bottom_left[2];
                float ground_z_end = bottom_right[2];

                plane_param.ground_z_start = ground_z_start;
                plane_param.ground_z_end = ground_z_end;
                // The 2D line left -> right
                plane_param.line_2D.p_start = Eigen::Vector2f(bottom_left(0), bottom_left(1));
                plane_param.line_2D.p_end = Eigen::Vector2f(bottom_right(0), bottom_right(1));
                new_map_.grounds.push_back(plane_param);
            }
        }

    }




    std::cout << "MAP CONSISTS OF " << new_map_.cylinders.size() << " CYLINDERS AND " << new_map_.planes.size()
              << " PLANES " << new_map_.grounds.size() << " GROUND BLOCKS " << std::endl;


}

bool PrimitivesMapping::findNewMergedFacadesRANSAC(Cloud::Ptr& cloud, Eigen::Vector3f& normal, float& d) {
    // PCL RANSAC facade estimation
    pcl::SampleConsensusModelPlane<Point>::Ptr model_plane(new pcl::SampleConsensusModelPlane<Point>(cloud));
    pcl::RandomSampleConsensus<Point> ransac(model_plane);
    ransac.setDistanceThreshold(0.05);
    ransac.computeModel();
    // get param Ax + By + Cz + D = 0
    Eigen::VectorXf coefficient;
    ransac.getModelCoefficients(coefficient);
    // Pass to param vector
    normal[0] = coefficient[0];
    normal[1] = coefficient[1];
    normal[2] = coefficient[2];
    d = coefficient[3];
    return true;
}


bool PrimitivesMapping::point_in_poly(int nvert, float *vertx, float *verty, float testx, float testy)
{
    int i, j, c = 0;
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((verty[i]>testy) != (verty[j]>testy)) &&
             (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
            c = !c;
    }

    if(c%2 != 0) return true;
    else return false;
}

void PrimitivesMapping::fit_plane_pcl(const Eigen::Matrix3Xd& points, Eigen::Vector3d& n, double& d){

    Eigen::Matrix3Xd coord = points;

    // calculate centroid
    Eigen::Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

    // subtract centroid
    coord.row(0).array() -= centroid(0);
    coord.row(1).array() -= centroid(1);
    coord.row(2).array() -= centroid(2);

    // we only need the left-singular matrix here
    //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points

    auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    n = svd.matrixU().rightCols<1>();
    n.normalize();
    d = -n.dot(centroid);
}


float PrimitivesMapping::dist_plane_plane(const Eigen::Vector3f& first_fa_dl,
                                          const Eigen::Vector3f& first_fa_normal,
                                          const Eigen::Vector3f& second_fa_dl,
                                          const Eigen::Vector3f& second_fa_tr) {

    Eigen::Vector3f first_fa_normal_norma = first_fa_normal.normalized();

    Eigen::Vector3f second_center = second_fa_dl + (second_fa_tr - second_fa_dl) * 0.5;
    Eigen::Vector3f dl_center = second_center - first_fa_dl;
    // calculate center of a plane to another plane
    float dist = first_fa_normal_norma.dot(dl_center);

    return std::abs(dist);
}







void PrimitivesMapping::mergeFacade(int iterationNumber, std::vector<Plane_Param>& Facades) {
    int i = 0;

    while (i < iterationNumber) {
        for (int j = 0; j < Facades.size(); ++j) {
            for (int k = 0; k < Facades.size(); ++k) {
                if (j == k) continue;
                // First rectangle
                Eigen::Vector3f R_TL = Facades[j].edge_poly[3];
                Eigen::Vector3f R_TR = Facades[j].edge_poly[2];
                Eigen::Vector3f R_DR = Facades[j].edge_poly[1];
                Eigen::Vector3f R_DL = Facades[j].edge_poly[0];
                Eigen::Vector3f R_N = Facades[j].n;
                // Second rectangle
                Eigen::Vector3f r_TL = Facades[k].edge_poly[3];
                Eigen::Vector3f r_TR = Facades[k].edge_poly[2];
                Eigen::Vector3f r_DR = Facades[k].edge_poly[1];
                Eigen::Vector3f r_DL = Facades[k].edge_poly[0];
                Eigen::Vector3f r_N = Facades[k].n;
                float D = Facades[j].d;

                std::vector<float> xList;
                std::vector<float> yList;
                std::vector<float> zList;
                // save x y z to chose the side value.
                for (int l = 0; l < 4; ++l) {
                    xList.push_back(Facades[j].edge_poly[i][0]);
                    yList.push_back(Facades[j].edge_poly[i][1]);
                    zList.push_back(Facades[j].edge_poly[i][2]);

                    xList.push_back(Facades[k].edge_poly[i][0]);
                    yList.push_back(Facades[k].edge_poly[i][1]);
                    zList.push_back(Facades[k].edge_poly[i][2]);
                }
                // Chose rectangle parallel to each other
                float nThreshold = R_N.cross(r_N).norm();
                float dThreshold = abs(Facades[j].d - Facades[k].d);
                if (nThreshold < 0.01 && dThreshold < 0.05) {
                    float maxX = *(std::max_element(xList.begin(), xList.end()));
                    float minX = *(std::min_element(xList.begin(), xList.end()));

                    float maxY = *(std::max_element(yList.begin(), yList.end()));
                    float minY = *(std::min_element(yList.begin(), yList.end()));

                    float maxZ = *(std::max_element(zList.begin(), zList.end()));
                    float minZ = *(std::min_element(zList.begin(), zList.end()));
                    // erase the two rectangle
                    Facades.erase(Facades.begin() + j);
                    Facades.erase(Facades.begin() + k);

                    // Build new rectangle with new coordinate and bush in Facades
                    Plane_Param newFacade;
                    newFacade.edge_poly.resize(4);
                    // tl
                    newFacade.edge_poly[0][0] = minX;
                    newFacade.edge_poly[0][1] = minY;
                    newFacade.edge_poly[0][2] = maxZ;
                    // tr
                    newFacade.edge_poly[1][0] = maxX;
                    newFacade.edge_poly[1][1] = minY;
                    newFacade.edge_poly[1][2] = maxZ;
                    // dr
                    newFacade.edge_poly[2][0] = maxX;
                    newFacade.edge_poly[2][1] = minY;
                    newFacade.edge_poly[2][2] = minZ;
                    // dl
                    newFacade.edge_poly[3][0] = minX;
                    newFacade.edge_poly[3][1] = minY;
                    newFacade.edge_poly[3][2] = minZ;
                    newFacade.n = R_N;
                    newFacade.d = D;
                    Facades.push_back(newFacade);
                }

            }
        }
        i++;
    }
}

void PrimitivesMapping::getPrimitiveFacadeCloud(Cloud::Ptr& cloud, std::vector<Plane_Param>& Facades, int color) {
    for (auto& facade : Facades) {
        createTheoryLine(facade.edge_poly[0], facade.edge_poly[1], cloud, 5, color); // dl - > dr
        createTheoryLine(facade.edge_poly[1], facade.edge_poly[2], cloud, 5, color); // dr - > tr
        createTheoryLine(facade.edge_poly[2], facade.edge_poly[3], cloud, 5, color); // tr - > tl
        createTheoryLine(facade.edge_poly[3], facade.edge_poly[0], cloud, 5, color); // tl - > dl
        createTheoryLine(facade.edge_poly[2], facade.edge_poly[0], cloud, 5, color); // tr - > dl
        createTheoryLine(facade.edge_poly[1], facade.edge_poly[3], cloud, 5, color); // dr - > tl

    }
}
void PrimitivesMapping::getPrimitiveFacadeCloud(CloudColor::Ptr& cloudColor, std::vector<Plane_Param>& Facades) {
    for (auto& facade : Facades) {
        createTheoryLine(facade.edge_poly[0], facade.edge_poly[1], cloudColor, 100); // dl - > dr
        createTheoryLine(facade.edge_poly[1], facade.edge_poly[2], cloudColor, 100); // dr - > tr
        createTheoryLine(facade.edge_poly[2], facade.edge_poly[3], cloudColor, 100); // tr - > tl
        createTheoryLine(facade.edge_poly[3], facade.edge_poly[0], cloudColor, 100); // tl - > dl
        createTheoryLine(facade.edge_poly[2], facade.edge_poly[0], cloudColor, 100); // tr - > dl
        createTheoryLine(facade.edge_poly[1], facade.edge_poly[3], cloudColor, 100); // dr - > tl

    }
}
void PrimitivesMapping::getPrimitivePoleCloud(Cloud::Ptr& cloud, std::vector<Cylinder_Fin_Param>& Poles) {
    for (auto& pole : Poles) {
        createTheoryCylinder(pole.center_line.p_bottom, pole.center_line.p_top, pole.radius, cloud, 10, 10, 1000);
    }
}


} // end of namespace primitivesMapping
