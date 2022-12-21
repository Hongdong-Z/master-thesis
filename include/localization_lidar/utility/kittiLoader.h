//
// Created by zhao on 25.08.22.
//

#ifndef SRC_KITTILOADER_H
#define SRC_KITTILOADER_H

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class KittiLoader
{
public:
    KittiLoader(const std::string& dataset_path);
    ~KittiLoader() {}

    size_t size() const { return num_frames_; }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(size_t i) const;

private:
    int num_frames_;
    std::string dataset_path_;

};


#endif // SRC_KITTILOADER_H
