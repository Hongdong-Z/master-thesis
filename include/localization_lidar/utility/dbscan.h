//
// Created by rwang on 15.02.22.
// copy from https://github.com/james-yoo/DBSCAN by james/yoo
//

#ifndef SRC_DBSCAN_H
#define SRC_DBSCAN_H

#include <cmath>
#include <filesystem>
#include <omp.h>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "common.h"
#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE_ -3

using namespace std;

namespace utility{

typedef struct Point_
{
    float x, y, z;  // X, Y, Z position
    float time;
    float intensity;
    int clusterID;  //label
}Point;

class DBSCAN {
public:
    DBSCAN(unsigned int minPts) : m_minPoints(minPts), fusionFinal_(new pcl::PointCloud<PointType>) {}
    ~DBSCAN(){}

    int run();
    vector<int> calculateCluster(Point point);
    int expandCluster(Point point, int clusterID);
    inline double calculateDistance(const Point& pointCore, const Point& pointTarget);

    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
    void printResults(vector<Point>& points, int num_points);
//    vector<PointType> getClustering(const std::filesystem::path&, std::pair<float, pcl::PointCloud<PointType>> input);
    pcl::PointCloud<PointType> getClustering(std::pair<float, pcl::PointCloud<PointType>> input);
    pcl::PointCloud<PointType> getClustering_curb(std::pair<float, pcl::PointCloud<PointType>> input);

public:
    vector<Point> m_points;
    int cluster_size;

private:
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
    vector<PointType> centroid_;
    vector<PointType> centroidCopy_;
    pcl::PointCloud<PointType>::Ptr fusionFinal_;
    vector<pcl::PointCloud<PointType>> clustersCloud_;

};

inline void readBenchmarkData(vector<Point>& points, pcl::PointCloud<PointType> inputCloud)
{
    // load point cloud
    unsigned int minpts, num_points, cluster, i = 0;
    double epsilon;
    num_points = inputCloud.size();
    Point *p = (Point *)calloc(num_points, sizeof(Point));

    while (i < num_points) {
        p[i].x = inputCloud.points[i].x;
        p[i].y = inputCloud.points[i].y;
        p[i].z = inputCloud.points[i].z;
        p[i].clusterID = UNCLASSIFIED;
        p[i].intensity = inputCloud.points[i].intensity;
//        p[i].time=inputCloud.points[i].time;
        points.push_back(p[i]);
        ++i;
    }

}

} // namespace utility

#endif // SRC_DBSCAN_H
