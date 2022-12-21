//
// Created by zhao on 25.08.22.
//

#include "utility/kittiLoader.h"
KittiLoader::KittiLoader(const std::string& dataset_path)
{
    dataset_path_ = dataset_path;
    for (num_frames_ = 0; ; ++num_frames_)
    {
        std::string filename = (boost::format("%s/%06d.bin") % dataset_path_ % num_frames_).str();
        if (!boost::filesystem::exists(filename))
        {
            break;
        }
    }

    if (num_frames_== 0)
    {
        std::cerr << "error: no files in " << dataset_path_ << std::endl;
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr KittiLoader::cloud(size_t i) const
{
    std::string filename = (boost::format("%s/%06d.bin") % dataset_path_ % i).str();
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file)
    {
        std::cerr << "error: failed to load " << filename << std::endl;
        return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t num_points = fread(reinterpret_cast<char*>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
    fclose(file);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->resize(num_points);

    for (int i = 0; i < num_points; ++i)
    {
        auto& pt = cloud->at(i);
        pt.x = buffer[i * 4];
        pt.y = buffer[i * 4 + 1];
        pt.z = buffer[i * 4 + 2];
        pt.intensity = buffer[i * 4 + 3];
    }

    return cloud;
}

