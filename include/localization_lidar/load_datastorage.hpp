#pragma once
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include <seamseg_fbds/reader.h>

// Loads main data storage
inline fbds::DataStorage loadDataStorage(std::filesystem::path root) {
    // Main data "frame" sources (one frame corresponds to one synchronized capture trigger)
    std::vector<fbds::FrameSourceInfo> mainFrameDataSources = {
        fbds::FSI{
            fbds::FrameSource::Other,
            "frontColorSphere",
            fbds::DataCategory::Image,
            "color_front_sphere",
            "timestamp.front_color_sphere.txt",
            "Undistorted images with sphere model"
        },
        fbds::FSI{
            fbds::FrameSource::SeamsegInstances,
            "frontColorSphereInstances",
            fbds::DataCategory::Image,
            "postproc/seamseg_instances/",
            "timestamp.color_front_sphere.txt",
            "Instance images from front_color_sphere"
        },
        fbds::FSI{
            fbds::FrameSource::SeamsegBoxes,
            "frontColorSphereBoxes",
            fbds::DataCategory::Other,
            "postproc/seamseg_boxes",
            "timestamp.color_front_sphere.txt",
            "YAML bounding box data files for images from front_color_sphere"
        },
        fbds::FSI{
            fbds::FrameSource::VelodyneFL,
            "velodyne",
            fbds::DataCategory::Pointcloud,
            "velodyne_c",
            "timestamp.velodyne_c.txt",
            "path to velodyne data (unskewed)"
        }
    };

    // Static sources (one file)
    std::vector<fbds::StaticSourceInfo> mainStaticDataSources = {
        fbds::SSI{
            fbds::StaticSource::Calibration,
            "calibration",
            fbds::DataCategory::Other,
            "calibration.bin",
            "Calibration file"
        }
    };

    // Configuration object, which is needed for constructing a fbds::DataStorage
    fbds::StorageConfiguration mainStorageConfig;
    mainStorageConfig.frameDataSources = mainFrameDataSources;
    mainStorageConfig.staticDataSources = mainStaticDataSources;
    mainStorageConfig.posesFile = "poses.dump";

    return fbds::DataStorage(
        root,
        mainStorageConfig
    );
}