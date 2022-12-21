//
// Created by zhao on 11.08.22.
//

#include "primitives_mapping/primitivesMapping.h"
int main() {
//    int argc = 0;
//    ros::init(argc, NULL, "Prim");
//    ros::NodeHandle nodeHandle;
//    ros::Publisher publisher;
//    publisher = nodeHandle.advertise<geometry_msgs::Point>("debug", 1, true);
    std::cout << "******************mapping start!********************" << std::endl;

    primitivesMapping::PrimitivesMapping PM(50);

//    PM.run();

    ///******************** read txt map to cloud*********************
//    std::vector<primitivesMapping::Plane_Param> facadesParam;
//    std::vector<primitivesMapping::Plane_Param> groundsParam;
//    primitivesMapping::Cloud::Ptr cloud(new primitivesMapping::Cloud());
//    std::vector<primitivesMapping::Cylinder_Fin_Param> polesParam;

//    std::string filePathF = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/facadesParaMap/facadesMap_ori.txt";
    std::string filePathP = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/polesParaMap/polesMap_ori.txt";
    std::string filePathG = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/groundsParaMap/groundsMap_ori.txt";
//
//
//    PM.readFacadeMap(facadesParam, filePathF);
//    PM.readPolesMap(polesParam,filePathP);
//    PM.readGroundMap(groundsParam, filePathG);

//    PM.getPrimitiveFacadeCloud(cloud, facadesParam, 500);
//    PM.getPrimitivePoleCloud(cloud, polesParam);
//
//    pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/whole_ground_map_ori.pcd",
//                              *cloud);

//    PM.build_map();
    std::string filePathFacade = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/facades_new_map.txt";
    std::string filePathPole = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/poles_new_map.txt";
    std::string filePathGrounds = "/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/grounds_new_map.txt";

//    PM.save_map(filePathPole, filePathFacade, filePathGrounds);


    if (1) {
        std::vector<primitivesMapping::Plane_Param> facadesParamNew;
        std::vector<primitivesMapping::Plane_Param> groundsParamNew;
        std::vector<primitivesMapping::Cylinder_Fin_Param> polesParamNew;

        PM.readFacadeMap(facadesParamNew, filePathFacade);
//        PM.readPolesMap(polesParamNew, filePathPole);
//        PM.readGroundMap(groundsParamNew, filePathGrounds);

        primitivesMapping::Cloud::Ptr cloudNew(new primitivesMapping::Cloud());
        PM.getPrimitiveFacadeCloud(cloudNew, facadesParamNew, 400);
        PM.getPrimitiveFacadeCloud(cloudNew, groundsParamNew, 250);
        PM.getPrimitivePoleCloud(cloudNew, polesParamNew);

        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/whole_merged_map.pcd",
                                  *cloudNew);
    }

    std::cout << "******************mapping finish!********************" << std::endl;

    return 0;
}