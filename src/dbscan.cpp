//
// Created by zhao on 06.04.22.
//
#include "utility/dbscan.h"
#include <stdio.h>

namespace utility {

int DBSCAN::run()
{
    int clusterID = 1;
    vector<Point>::iterator iter;
    for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( iter->clusterID == UNCLASSIFIED )
        {
            if ( expandCluster(*iter, clusterID) != FAILURE_ )
            {
                clusterID += 1;
            }
        }
    }
    return cluster_size = clusterID - 1;
//    return cluster_size = clusterID;

}

int DBSCAN::expandCluster(Point point, int clusterID)
{
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE_;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y)
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

vector<int> DBSCAN::calculateCluster(Point point)
{
    int index = 0;
    vector<Point>::iterator iter;
    vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance(const Point& pointCore, const Point& pointTarget )
{
    return pow(pointCore.x - pointTarget.x,2)+pow(pointCore.y - pointTarget.y,2);
}

void DBSCAN::printResults(vector<Point>& points, int num_points)
{
    int i = 0;
    printf("Number of points: %u\n"
           " x     y     z     cluster_id\n"
           "-----------------------------\n"
        , num_points);
    while (i < num_points)
    {
        printf("%5.2lf %5.2lf %5.2lf: %d\n",
               points[i].x,
               points[i].y, points[i].z,
               points[i].clusterID);
        ++i;
    }
}

pcl::PointCloud<PointType> DBSCAN::getClustering(std::pair<float, pcl::PointCloud<PointType>> input) {
    vector<Point> points;
    m_points = points;
    m_pointSize = points.size();
    m_epsilon = input.first;
    readBenchmarkData(m_points, input.second);
    int numOfCluster = run();
//    std::cout << "cluster size is: " << cluster_size << std::endl;
    centroid_.resize(numOfCluster);

//#pragma omp parallel for num_threads(4)
    for (int j = 0; j < numOfCluster; j++) {
        pcl::PointCloud<PointType>::Ptr final(new pcl::PointCloud<PointType>);
        for (auto& p : m_points) {
            if (p.clusterID == j + 1) {
                PointType tmp;
                tmp.x = p.x;
                tmp.y = p.y;
                tmp.z = p.z;
                tmp.intensity = p.intensity;
//                tmp.time=p.time;
//                tmp.label=p.clusterID;
//                tmp.instance=j;
                if (tmp.z < 0.7){  // distance to ground only take points that under 3 meters
                    final->push_back(tmp);
                }
//                clustersCloud_.push_back(*final);
            }
        }


        float x = 0;
        float y = 0;
        float z = 0;
        if (final->size() > 0){
            clustersCloud_.push_back(*final);
            for (auto& p : *final) {
                x += p.x;
                y += p.y;
                z += p.z;
            }
            float num = final->size();
            centroid_[j].x = x / num;
            centroid_[j].y = y / num;
            centroid_[j].z = z / num;
            centroid_[j].intensity = 1;
//            *fusionFinal_ += *final;
            centroidCopy_.push_back(centroid_[j]);
//            std::cout << " Size of each clustering after cutting: " << final->size() << std::endl;
        }
//        *fusionFinal_ += *final;
    }
//#pragma omp parallel for num_threads(4)
    for (int i = 0; i < centroidCopy_.size() - 1; ++i) {
        float dis = sqrt(pow((centroidCopy_[i].x - centroidCopy_[i+1].x), 2) + pow((centroidCopy_[i].y - centroidCopy_[i+1].y), 2));
        if (dis > 5){
            *fusionFinal_ += clustersCloud_[i];
        }
    }
//    pcl::io::savePCDFileASCII(savePath / "edge_3600.pcd", *fusionFinal_);
    return *fusionFinal_;
}

pcl::PointCloud<PointType> DBSCAN::getClustering_curb(std::pair<float, pcl::PointCloud<PointType>> input) {
    vector<Point> points;
    m_points = points;
    m_pointSize = points.size();
    m_epsilon = input.first;
    readBenchmarkData(m_points, input.second);
    int numOfCluster = run();
//    std::cout << "cluster size is: " << cluster_size << std::endl;
    centroid_.resize(numOfCluster);

//#pragma omp parallel for num_threads(4)
    for (int j = 0; j < numOfCluster; j++) {
        pcl::PointCloud<PointType>::Ptr final(new pcl::PointCloud<PointType>);
        for (auto& p : m_points) {
            if (p.clusterID == j + 1) {
                PointType tmp;
                tmp.x = p.x;
                tmp.y = p.y;
                tmp.z = p.z;
                tmp.intensity = p.intensity;
//                tmp.time=p.time;
//                tmp.label=p.clusterID;
//                tmp.instance=j;
                if (tmp.z < 0.7){  // distance to ground only take points that under 3 meters
                    final->push_back(tmp);
                }
//                clustersCloud_.push_back(*final);
            }
        }


        float x = 0;
        float y = 0;
        float z = 0;
        if (final->size() > 0){
            clustersCloud_.push_back(*final);
            for (auto& p : *final) {
                x += p.x;
                y += p.y;
                z += p.z;
            }
            float num = final->size();
            centroid_[j].x = x / num;
            centroid_[j].y = y / num;
            centroid_[j].z = z / num;
            centroid_[j].intensity = 1;
//            *fusionFinal_ += *final;
            centroidCopy_.push_back(centroid_[j]);
//            std::cout << " Size of each clustering after cutting: " << final->size() << std::endl;
        }
        *fusionFinal_ += *final;
    }
//#pragma omp parallel for num_threads(4)
//    for (int i = 0; i < centroidCopy_.size() - 1; ++i) {
//        float dis = sqrt(pow((centroidCopy_[i].x - centroidCopy_[i+1].x), 2)
//                         + pow((centroidCopy_[i].y - centroidCopy_[i+1].y), 2)
//                         + pow((centroidCopy_[i].z - centroidCopy_[i+1].z), 2));
//        if (dis > 5){
//            *fusionFinal_ += clustersCloud_[i];
//        }
//    }
//    pcl::io::savePCDFileASCII(savePath / "edge_3600.pcd", *fusionFinal_);
    return *fusionFinal_;
}

} // namespace utility