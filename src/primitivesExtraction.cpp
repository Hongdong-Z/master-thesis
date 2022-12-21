//
// Created by zhao on 24.06.22.
//
#include <primitives_extraction/primitivesExtraction.h>

namespace primitivesExtraction {

void PrimitiveExtractor::allocateMemory() {
    cloudIn.reset(new Cloud);
    fullCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    fullCloud->points.resize(N_SCAN * Horizon_SCAN);

    groundCloud.reset(new Cloud);
    nonGroundCloud.reset(new Cloud);
    facadeCloudRough.reset(new Cloud);
    poleCloudRough.reset(new Cloud);
    poleCloudGrow.reset(new Cloud);
    nonFacadeCloudRough.reset(new Cloud);
    facadeCloudCandidates.reset(new Cloud);


    tree.reset(new pcl::search::KdTree<Point>());
    treeEC.reset(new pcl::search::KdTree<Point>());
}

void PrimitiveExtractor::resetParameters() {


    pointClusters.clear();
    zSplits.clear();
    zSplitsMats.clear();
    clustersIndex.clear();
    polesParam.clear();
    facadesParam.clear();

    zSplits.resize(N_SCAN);

    labelMat =
        cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0)); // 32-bit signed integers ( -2147483648..2147483647 )
    rangeMat = cv::Mat(N_SCAN,
                       Horizon_SCAN,
                       CV_32F,
                       cv::Scalar::all(FLT_MAX)); // 32-bit ﬂoating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
    groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0)); //  8-bit signed integers ( -128..127 )
    facadeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0)); //  8-bit signed integers ( -128..127 )
    poleMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));   //  8-bit signed integers ( -128..127 )
    poleCandidatesMat =
        cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0)); //  8-bit signed integers ( -128..127 )


    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
}
void PrimitiveExtractor::initColors() {
    colors.resize(100000);
    for (int i = 0; i < 100000; ++i) {
        uint8_t r = rand() % 256;
        uint8_t g = rand() % 256;
        uint8_t b = (2 * 255 - r - g) % 256; // For not too bright and not too dark color!
        std::vector<uint8_t> color;
        color.push_back(r);
        color.push_back(g);
        color.push_back(b);
        std::random_shuffle(color.begin(), color.end());
        colors[i] = color;
    }
}
void PrimitiveExtractor::setInputCloud(Cloud::ConstPtr cloud_in) {
    cloudIn = cloud_in;
}

void PrimitiveExtractor::setRange(const float& x_min,
                                  const float& x_max,
                                  const float& y_min,
                                  const float& y_max,
                                  const float& z_min,
                                  const float& z_max) {
    range[0] = x_min;
    range[1] = x_max;
    range[2] = y_min;
    range[3] = y_max;
    range[4] = z_min;
    range[5] = z_max;
}

void PrimitiveExtractor::splitCloud() {
    for (int i = 0; i < N_SCAN; ++i) {
        Cloud::Ptr new_cloud(new Cloud);
        zSplits[i] = new_cloud;
    }
    Point p;
    size_t index;
    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            index = i + j * N_SCAN;
            p = fullCloud->points[index];

            if (p.x > range[0] && p.x < range[1] && p.y > range[2] && p.y < range[3]) {
                zSplits[i]->push_back(p);
            }
        }
        //        std::cout << zSplits[i]->points.size() << std::endl;
    }
}

void PrimitiveExtractor::getSplitCloudsImages(std::vector<cv::Mat>& layerProjections) {

    // Set image dimensions.
    int img_pixels_per_meter = 20;
    int img_height = std::ceil((range[1] - range[0]) * img_pixels_per_meter) + 1;
    int img_width = std::ceil((range[3] - range[2]) * img_pixels_per_meter) + 1;

    // Initialize a black image for each layer.
    int num_layers = zSplits.size(); // The number of lines behind is not required -40
    layerProjections.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        layerProjections[l] = cv::Mat::zeros(img_height, img_width, CV_8U);
    }

    // Set drawing dimensions.

    // Project points to images.
    // For each layer.
    for (int l = 0; l < num_layers; ++l) {
        // For each 3D point p in this layer.
        for (auto p : zSplits[l]->points) {
            cv::Point pos_in_img(std::round((img_width / 2 - p.y * (float)img_pixels_per_meter)),
                                 std::round((img_height / 2 - p.x * (float)img_pixels_per_meter)));
            if (pos_in_img.x < 0 || pos_in_img.x >= img_width || pos_in_img.y < 0 || pos_in_img.y >= img_height) {
                continue;
            }
            //      cv::circle(layer_projections[l], pos_in_img, radius, color);
            layerProjections[l].at<uchar>(pos_in_img.y, pos_in_img.x) = 255;
        }
    }
}

void PrimitiveExtractor::projectPointCloud() {
//    Time t("Projection PointCloud Time");
    pcl::PointXYZI thisPoint;
    float range_;
    size_t index;
    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {

            index = i + j * N_SCAN;
            if (index >= cloudIn->points.size()) {
                std::cerr << "Cloud size is less then 128 * 1800 = 230400" << std::endl;
                continue;
            }
            thisPoint.x = cloudIn->points[index].x;
            thisPoint.y = cloudIn->points[index].y;
            thisPoint.z = cloudIn->points[index].z;

            range_ = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            rangeMat.at<float>(i, j) = range_;
            thisPoint.intensity = (float)i + (float)j / 10000.0;
            // Filter out in regular points
            if (thisPoint.z < range[4]) continue;

            fullCloud->points[index] = thisPoint;
        }
    }
}

void PrimitiveExtractor::groundSegmentation() {
//    Time t("Ground Segmentation Time");
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    for (size_t j = 0; j < Horizon_SCAN; ++j) {
        for (size_t i = 0; i < groundScanInd; ++i) {

            lowerInd = i + j * N_SCAN;
            upperInd = (i + 1) + j * N_SCAN;

            if (fullCloud->points[lowerInd].intensity == -1 || fullCloud->points[upperInd].intensity == -1) {
                groundMat.at<int8_t>(i, j) = -1;
                continue;
            }

            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

            if (abs(angle) <= 10) {
                groundMat.at<int8_t>(i, j) = 1;
                groundMat.at<int8_t>(i + 1, j) = 1;
            }
        }
    }


    for (size_t i = 0; i <= groundScanInd; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            if (groundMat.at<int8_t>(i, j) == 1 && rangeMat.at<float>(i, j) > 2.2) {
                labelMat.at<int>(i, j) = -1;
                groundCloud->push_back(fullCloud->points[i + j * N_SCAN]);
            } else {
                nonGroundCloud->push_back(fullCloud->points[i + j * N_SCAN]);
            }
        }
    }
}
void PrimitiveExtractor::ground_segmentation(const std::vector<Cloud::Ptr>& z_splits,
                                             std::vector<std::vector<int>>& non_ground_indices) {
    float large_num = 99999;

    float ground_plane_cell_size_meter_ = 0.5;
    float img_pixels_per_meter = 1.0 / ground_plane_cell_size_meter_;
    int img_height = std::ceil((range[1] - range[0]) * img_pixels_per_meter) + 1;
    int img_width = std::ceil((range[3] - range[2]) * img_pixels_per_meter) + 1;

    //    std::cout << img_height << " " << img_width << std::endl;
    cv::Mat z_min_profile = cv::Mat::ones(img_height, img_width, CV_32FC1) * large_num;
    cv::Mat z_max_profile = cv::Mat::ones(img_height, img_width, CV_32FC1) * (-large_num);
    z_profile_ground_ = cv::Mat::ones(img_height, img_width, CV_32FC1) * large_num;
    cv::Mat support = cv::Mat::zeros(img_height, img_width, CV_32S);

    cv::Mat visu = cv::Mat::zeros(img_height, img_width, CV_8U);


    float height_support_tolerance = 0.2; // Points which are no more than this value heigher than the
    // z_min value count as supporters.
    int min_support = 5;

    int pos_in_img_row, pos_in_img_col;
    float x_min = range[0];
    float y_min = range[2];
    float* z_min_profile_row;
    float* z_max_profile_row;
    float* z_profile_row;
    // Go through all layers.
    for (auto& l : z_splits) {
        // Go through all points in this layer.
        for (auto& p : l->points) {
            // Get position of point in image.
            pos_in_img_row = std::round((p.x - x_min) * img_pixels_per_meter);
            pos_in_img_col = std::round((p.y - y_min) * img_pixels_per_meter);
            z_min_profile_row = z_min_profile.ptr<float>(pos_in_img_row); // 9999
            z_max_profile_row = z_max_profile.ptr<float>(pos_in_img_row); //-9999
            //      std::cout <<pos_in_img_row <<" ("<<img_height<<")  "
            //               <<pos_in_img_col <<" ("<<img_width<<")"<<std::endl;
            support.ptr<int>(pos_in_img_row)[pos_in_img_col]++;

            // Check if point is lower than the minimum of this cell.
            if (p.z < z_min_profile_row[pos_in_img_col]) {
                // Set new minimum.
                z_min_profile_row[pos_in_img_col] = p.z;
            }
            if (p.z > z_max_profile_row[pos_in_img_col]) {
                // Set new maximum.
                z_max_profile_row[pos_in_img_col] = p.z;
            }
        }
    }

    // Find plane cells.
    // Go through the min max profile images.
    for (int row_id = 0; row_id < img_height; ++row_id) {
        z_min_profile_row = z_min_profile.ptr<float>(row_id);
        z_max_profile_row = z_max_profile.ptr<float>(row_id);
        for (int col_id = 0; col_id < img_width; ++col_id) {
            // If the max and min are close then store as profile.
            if (z_min_profile_row[col_id] < large_num && z_max_profile_row[col_id] > -large_num) {
                if ((z_max_profile_row[col_id] - z_min_profile_row[col_id]) < height_support_tolerance &&
                    support.ptr<int>(row_id)[col_id] > min_support) {
                    z_profile_ground_.ptr<float>(row_id)[col_id] =
                        (z_max_profile_row[col_id] + z_min_profile_row[col_id]) * 0.5f;
                    //          visu.ptr<uchar>(row_id)[col_id] = 255;

                    //          std::cout << z_max_profile_row[col_id] - z_min_profile_row[col_id] <<std::endl;
                }
            }
        }
    }

    //    cv::namedWindow("ground_profile_1",cv::WINDOW_NORMAL);
    //    cv::imshow("ground_profile_1",visu);

    // Check each profile plane whether its in a larger area really still low.
    // Additionaly reset all profile cells which have very small support.
    float search_radius_meter = 2.5;
    int search_radius_pixel = std::round(search_radius_meter * img_pixels_per_meter);
    float max_profile_diff_per_meter_in_meter = 0.3;
    float max_profile_diff_per_pixel_in_meter = max_profile_diff_per_meter_in_meter * ground_plane_cell_size_meter_;

    float* z_profile_row_n;
    for (int row_id = 0; row_id < img_height; ++row_id) {
        z_profile_row = z_profile_ground_.ptr<float>(row_id);
        for (int col_id = 0; col_id < img_width; ++col_id) {
            // If the cell is a ground plane candidate.
            if (z_profile_row[col_id] < large_num) {
                // Check neighborhood for significantly lower profile.
                float seed_z = z_profile_row[col_id];
                bool reset = false;
                for (int row_id_n = -search_radius_pixel; row_id_n <= search_radius_pixel; ++row_id_n) {
                    if (row_id + row_id_n >= 0 && row_id + row_id_n < img_height) {
                        z_profile_row_n = z_profile_ground_.ptr<float>(row_id + row_id_n);
                        for (int col_id_n = -search_radius_pixel; col_id_n <= search_radius_pixel; ++col_id_n) {
                            if (col_id + col_id_n >= 0 && col_id + col_id_n < img_width) {
                                // If the neighboring profile is significantly lower then reset the seed.
                                if (z_profile_row_n[col_id + col_id_n] <
                                    seed_z - std::max(std::abs(row_id_n), std::abs(col_id_n)) *
                                                 max_profile_diff_per_pixel_in_meter) {
                                    z_profile_row[col_id] = large_num;
                                    //                  visu.ptr<uchar>(row_id)[col_id] = 200;

                                    reset = true;
                                }
                            }
                            if (reset)
                                break;
                        }
                    }
                    if (reset)
                        break;
                }
            }
        }
    }

    //  cv::namedWindow("ground_profile_2",cv::WINDOW_NORMAL);
    //  cv::imshow("ground_profile_2",visu);

    // For each cell in which there is a big diff between max and min (but not cells in which there are no points)
    // search through the neighborhood and check if there is a trustworthy ground cell which has comparable z_min.
    float similarity_distance_meter = 0.1;
    for (int row_id = 0; row_id < img_height; ++row_id) {
        z_profile_row = z_profile_ground_.ptr<float>(row_id);
        z_min_profile_row = z_min_profile.ptr<float>(row_id);
        z_max_profile_row = z_max_profile.ptr<float>(row_id);
        for (int col_id = 0; col_id < img_width; ++col_id) {
            // If the cell is a ground plane candidate.
            if (z_min_profile_row[col_id] < large_num) {
                // If the cell shows some kind of vertical structure.
                if (z_max_profile_row[col_id] - z_min_profile_row[col_id] >= height_support_tolerance &&
                    support.ptr<int>(row_id)[col_id] > min_support) {
                    // Check neighborhood for similar low profile.
                    float seed_z = z_min_profile_row[col_id];
                    bool reset = false;
                    for (int row_id_n = -search_radius_pixel; row_id_n <= search_radius_pixel; ++row_id_n) {
                        if (row_id + row_id_n >= 0 && row_id + row_id_n < img_height) {
                            z_profile_row_n = z_profile_ground_.ptr<float>(row_id + row_id_n);
                            for (int col_id_n = -search_radius_pixel; col_id_n <= search_radius_pixel; ++col_id_n) {
                                if (col_id + col_id_n >= 0 && col_id + col_id_n < img_height) {
                                    if (z_profile_row_n[col_id + col_id_n] < large_num) {
                                        // If the neighboring profile is similar.
                                        if (std::abs(z_profile_row_n[col_id + col_id_n] - seed_z) <
                                            similarity_distance_meter) {
                                            z_profile_row[col_id] = seed_z;
                                            //                      visu.ptr<uchar>(row_id)[col_id] = 70;
                                            reset = true;
                                        }
                                    }
                                }
                                if (reset)
                                    break;
                            }
                        }
                        if (reset)
                            break;
                    }
                }
            }
        }
    }

    // Fit plane to fill holes.

    // Sample points.
    CloudXYZ::Ptr sample_cloud(new CloudXYZ());
    for (int row_id = 0; row_id < img_height - 1; ++row_id) {
        z_profile_row = z_profile_ground_.ptr<float>(row_id);
        for (int col_id = 0; col_id < img_width - 1; ++col_id) {
            // If no height determined for this pixel.
            if (z_profile_row[col_id] != large_num) {
                PointXYZ p;
                p.z = z_profile_row[col_id];
                p.x = row_id / img_pixels_per_meter + x_min;
                p.y = col_id / img_pixels_per_meter + y_min;

                sample_cloud->points.push_back(p);
            }
        }
    }


    if (sample_cloud->points.size() > 10) {
        // Fit plane.
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.20);
        //    std::cout <<"SAMPLE CLOUD SIZE: "<<sample_cloud->points.size()<<std::endl;
        seg.setInputCloud(sample_cloud);
        seg.segment(*inliers, *coefficients); // ax+by+cz+d = 0
        Eigen::Vector3d n;
        n << coefficients->values[0], coefficients->values[1], coefficients->values[2];
        n.normalize();
        double d = coefficients->values[3];

        ground_plane_.d = d;
        ground_plane_.n = n.cast<float>();

        // If the plane is reasonably horizontal.
        if (n.dot(Eigen::Vector3d(0, 0, 1)) > cos(15 / 180.0 * M_PI)) {
            // Then replace all the very different z profile values.
            for (int row_id = 0; row_id < img_height; ++row_id) {
                z_profile_row = z_profile_ground_.ptr<float>(row_id);
                for (int col_id = 0; col_id < img_width; ++col_id) {
                    Eigen::Vector3d p;
                    p(0) = row_id / img_pixels_per_meter + x_min;
                    p(1) = col_id / img_pixels_per_meter + y_min;
                    p(2) = z_profile_row[col_id];

                    double dist = (p.dot(n) + d);
                    if (std::abs(dist) > 0.5) {
                        z_profile_row[col_id] = (-d - n(0) * p(0) - n(1) * p(1)) / n(2);
                    }
                }
            }
        }
    }


    // Segment the non_ground points.
    non_ground_indices.resize(z_splits.size());

    float ground_dist_max = height_support_tolerance * 0.5;
    float dist;
    float ground_average_z_temp = 0;
    int dividor = 0;
    // Go through all layers.
    int layer_id = 0;
    for (auto& l : z_splits) {
        // Go through all points in this layer.
        int point_id = 0;
        for (auto& p : l->points) {
            // Get position of point in image.
            pos_in_img_row = std::round((p.x - x_min) * img_pixels_per_meter);
            pos_in_img_col = std::round((p.y - y_min) * img_pixels_per_meter);

            // Check if current point is on ground.
            dist = std::abs(p.z - z_profile_ground_.ptr<float>(pos_in_img_row)[pos_in_img_col]);
            if (dist > ground_dist_max) {
                non_ground_indices[layer_id].push_back(point_id);
            } else {
                ground_average_z_temp += z_profile_ground_.ptr<float>(pos_in_img_row)[pos_in_img_col];
                ++dividor;
            }
            ++point_id;
        }
        ++layer_id;
    }
    ground_average_z_ = ground_average_z_temp / (float)dividor;
}
void PrimitiveExtractor::testGetGround() {
    projectPointCloud();
    splitCloud();
    std::vector<std::vector<int>> non_ground_indices;
    ground_segmentation(zSplits, non_ground_indices);
    Cloud::Ptr cloud(new Cloud());
    for (int i = 0; i < non_ground_indices.size(); ++i) {
        for (int j = 0; j < non_ground_indices[i].size(); ++j) {
            cloud->points.push_back(zSplits[i]->points[j]);
        }
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();
    pcl::io::savePCDFile(
        "/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/saveForlater/non_ground.pcd",
        *cloud);
}
void PrimitiveExtractor::facadeExtraction() {
//    Time t("Facade Rough Segmentation Time");
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    for (size_t j = 0; j < Horizon_SCAN; ++j) {
        for (size_t i = 0; i < N_SCAN - 1; ++i) {

            lowerInd = i + j * N_SCAN;
            upperInd = (i + 1) + j * N_SCAN;
            // in case points less than default number
            if (fullCloud->points[lowerInd].intensity == -1 || fullCloud->points[upperInd].intensity == -1) {
                continue;
            }

            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

            if (abs(angle) <= 10) {
                facadeMat.at<int8_t>(i, j) = 1;
                facadeMat.at<int8_t>(i + 1, j) = 1;
            }
        }
    }


    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            if (facadeMat.at<int8_t>(i, j) == 1 && groundMat.at<int8_t>(i, j) != 1) {
                if (fullCloud->points[i + j * N_SCAN].intensity != -1) {
                    nonFacadeCloudRough->push_back(fullCloud->points[i + j * N_SCAN]);
                }
            }
            if (rangeMat.at<float>(i, j) > 2.2) {
                if (facadeMat.at<int8_t>(i, j) != 1 && groundMat.at<int8_t>(i, j) != 1) {
                    // save nonSurface features for edge extraction
                    if (fullCloud->points[i + j * N_SCAN].intensity != -1) {
                        facadeCloudRough->push_back(fullCloud->points[i + j * N_SCAN]);
                    }
                }
            }
        }
    }
}
void PrimitiveExtractor::facadeRoughExtraction() {
//    Time t("Rough Facade Extraction Time");
    int i1, i2, i3, i4, i5, i6, i7, i8;
    Eigen::Vector3f vectorLR = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorDU = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorLR2 = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f vectorDU2 = Eigen::Vector3f(0, 0, 0);

    Eigen::Vector3f surfaceNormal = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f surfaceNormal2 = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f normalProduct = Eigen::Vector3f(0, 0, 0);

    float parallelThreshold = 0.01;
    float horizontalThreshold = 0.03;

    for (int i = 1; i < N_SCAN - 1; ++i) {
        for (int j = 2; j < Horizon_SCAN - 2; ++j) {
            i1 = (i + 1) + j * N_SCAN;
            i5 = (i + 1) + (j + 1) * N_SCAN; //     -  i1  - i5
            i2 = (i - 1) + j * N_SCAN;
            i6 = (i - 1) + (j + 1) * N_SCAN; // i3  -  i7  - i4  - i8
            i3 = i + (j - 1) * N_SCAN;
            i7 = i + (j - 1 + 1) * N_SCAN; //        -  i2  - i6
            i4 = i + (j + 1) * N_SCAN;
            i8 = i + (j + 2) * N_SCAN;

            Point topPoint = fullCloud->points[i1];
            Point downPoint = fullCloud->points[i2];
            Point leftPoint = fullCloud->points[i3];
            Point rightPoint = fullCloud->points[i4];

            Point topPoint2 = fullCloud->points[i5];
            Point downPoint2 = fullCloud->points[i6];
            Point leftPoint2 = fullCloud->points[i7];
            Point rightPoint2 = fullCloud->points[i8];


            vectorLR(0) = rightPoint.x - leftPoint.x;
            vectorLR(1) = rightPoint.y - leftPoint.y;
            vectorLR(2) = rightPoint.z - leftPoint.z;

            vectorDU(0) = topPoint.x - downPoint.x;
            vectorDU(1) = topPoint.y - downPoint.y;
            vectorDU(2) = topPoint.z - downPoint.z;
            surfaceNormal = vectorLR.cross(vectorDU);


            vectorLR2(0) = rightPoint2.x - leftPoint2.x;
            vectorLR2(1) = rightPoint2.y - leftPoint2.y;
            vectorLR2(2) = rightPoint2.z - leftPoint2.z;

            vectorDU2(0) = topPoint2.x - downPoint2.x;
            vectorDU2(1) = topPoint2.y - downPoint2.y;
            vectorDU2(2) = topPoint2.z - downPoint2.z;
            surfaceNormal2 = vectorLR2.cross(vectorDU2);

            normalProduct = surfaceNormal.cross(surfaceNormal2);


            if (normalProduct.norm() < parallelThreshold && abs(surfaceNormal[2]) < horizontalThreshold) {
                facadeMat.at<int8_t>(i, j) = 1;
                labelMat.at<int>(i, j) = 10;
            }
        }
    }

    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            //&& curbMat_.at<int8_t>(j,i) != 1
            if (facadeMat.at<int8_t>(i, j) == 1 && groundMat.at<int8_t>(i, j) != 1) {
                facadeCloudRough->push_back(fullCloud->points[i + j * N_SCAN]);
            }
            if (rangeMat.at<float>(i, j) > 2.2) {
                if (facadeMat.at<int8_t>(i, j) != 1 && groundMat.at<int8_t>(i, j) != 1) {
                    // save nonSurface features for edge extraction
                    nonFacadeCloudRough->push_back(fullCloud->points[i + j * N_SCAN]);
                }
            }
        }
    }
}

void PrimitiveExtractor::polesExtractionWithFullCloud() {
//    Time t(" Poles Extraction Without growing Time");
    length = (int)(range[1] - range[0]);
    width = (int)(range[3] - range[2]);
    int row = std::floor(width / meshSize);
    int column = std::floor(length / meshSize);

    std::vector<std::vector<mesh>> allMesh(column + 1, std::vector<mesh>(row + 1)); // column for x row for y
    for (int i = 0; i < fullCloud->points.size(); ++i) {
        if (fullCloud->points[i].intensity == -1) continue;
        float x = fullCloud->points[i].x;
        float y = fullCloud->points[i].y;
        float z = fullCloud->points[i].z;
        if (abs(x) < length / 2 && abs(y) < width / 2 && z < -0.1) { // range of selected area
            int c = std::floor((x - (-length / 2)) / meshSize);      // x
            int r = std::floor((y - (-width / 2)) / meshSize);       // y
            if (c > column || r > row) {
                std::cerr << "c and r: " << c << r << std::endl;
            }
            allMesh[c][r].pointsIndex.push_back(i);
            allMesh[c][r].density++;
        }
    }

    // label valid mesh
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].density > 60) { // 60
                allMesh[i][j].state = 1;
            }
        }
    }

    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].state == 1) {

                for (int k = 0; k < allMesh[i][j].pointsIndex.size(); k++) {
                    poleCloudRough->push_back(
                        fullCloud->points[allMesh[i][j].pointsIndex[k]]); // push in candidate for cylinder ransac
                }
            }
        }
    }
}
void PrimitiveExtractor::polesExtractionWithFullCloudWithPCA() {
//    Time t(" Poles Extraction With PCA");
    length = (int)(range[1] - range[0]);
    width = (int)(range[3] - range[2]);
    int row = std::floor(width / 1.5);
    int column = std::floor(length / 1.5);

    std::vector<std::vector<mesh>> allMesh(column + 1, std::vector<mesh>(row + 1)); // column for x row for y
    for (int i = 0; i < fullCloud->points.size(); ++i) {
        if (fullCloud->points[i].intensity == -1) continue;
        float x = fullCloud->points[i].x;
        float y = fullCloud->points[i].y;
        float z = fullCloud->points[i].z;
        if (abs(x) < length / 2 && abs(y) < width / 2 && z < -0.1) { // range of selected area
            int c = std::floor((x - (-length / 2)) / 1.5);      // x
            int r = std::floor((y - (-width / 2)) / 1.5);       // y
            if (c > column || r > row) {
                std::cerr << "c and r: " << c << r << std::endl;
            }
            allMesh[c][r].pointsIndex.push_back(i);
            allMesh[c][r].density++;
        }
    }

    // label valid mesh
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].density > 60) { // 60
                allMesh[i][j].state = 1;
            }
        }
    }

    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].state == 1) {

                // PCA to filter out pole like objects
                Eigen::Vector4f pcaCentroid;
                Eigen::Matrix3f covariance;
                int color = rand()%1000;
                Cloud::Ptr tmpCloud(new Cloud());
                for (int k = 0; k < allMesh[i][j].pointsIndex.size(); k++) {
                    fullCloud->points[allMesh[i][j].pointsIndex[k]].intensity = color;
                    tmpCloud->points.push_back(fullCloud->points[allMesh[i][j].pointsIndex[k]]);
                }
                pcl::compute3DCentroid(*tmpCloud, pcaCentroid);
                pcl::computeCovarianceMatrixNormalized(*tmpCloud, pcaCentroid, covariance);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance, Eigen::ComputeEigenvectors);
                Eigen::Matrix3f eigenVectorsPCA = eigenSolver.eigenvectors();
                Eigen::Vector3f eigenValuesPCA = eigenSolver.eigenvalues();
//                std::cout << "eigen value: " << eigenValuesPCA << std::endl;
                // If the biggest eigen value is much lager than the others -> pole like object.
                // And the biggest eigen vector should be vertical to z axis.
                Eigen::Vector3f bigEigenVector = eigenVectorsPCA.block<3, 1>(0, 2);
                Eigen::Vector3f smallEigenVector = eigenVectorsPCA.block<3, 1>(0, 0);

                Eigen::Vector3f smallEigenVectorNormalized = smallEigenVector.normalized();
                Eigen::Vector3f xAxis(1, 0, 0);
//                float horizontalDiff = (smallEigenVectorNormalized.cross(xAxis)).norm();
                float horizontalDiff = abs(smallEigenVector[2]);
                if (horizontalDiff > cos((float)70 / 180 * M_PI)) {
                    continue;
                }
                // cos between z axis should be less 0.03
                float cosAngle = abs(bigEigenVector[2]);
                float angleThreshold = (float)85 / 180 * M_PI; // angle dif threshold 5 deg.

                if (cosAngle > cos(angleThreshold) && eigenValuesPCA[2] > 3 * eigenValuesPCA[1]) {
                    *poleCloudRough += *tmpCloud;
                    poleCandidates.push_back(tmpCloud);
                }
            }
        }
    }
//    std::cout << "poleCloudRough size: " << poleCloudRough->points.size() << std::endl;
}


void PrimitiveExtractor::groundExtractionPCA() {
//    Time t(" Extraction With PCA");
    length = (int)(range[1] - range[0]);
    width = (int)(range[3] - range[2]);
    float blockSize = 1.5;
    int row = std::floor(width / blockSize);  // fround block 3 x 3 m2
    int column = std::floor(length / blockSize);

    std::vector<std::vector<mesh>> allMesh(column + 1, std::vector<mesh>(row + 1)); // column for x row for y
    for (int i = 0; i < nonFacadeCloudRough->points.size(); ++i) {
        if (nonFacadeCloudRough->points[i].intensity == -1) continue;
        float x = nonFacadeCloudRough->points[i].x;
        float y = nonFacadeCloudRough->points[i].y;
        float z = nonFacadeCloudRough->points[i].z;
        if (abs(x) < length / 2 && abs(y) < width / 2 && z < -0.1) { // range of selected area
            int c = std::floor((x - (-length / 2)) / blockSize);      // x
            int r = std::floor((y - (-width / 2)) / blockSize);       // y
            if (c > column || r > row) {
                std::cerr << "c and r: " << c << r << std::endl;
            }
            allMesh[c][r].pointsIndex.push_back(i);
            allMesh[c][r].density++;
        }
    }

    // label valid mesh
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].density > 30) { // 60
                allMesh[i][j].state = 1;
            }
        }
    }

    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].state == 1) {

                // PCA to filter out ground blocks
                Eigen::Vector4f pcaCentroid;
                Eigen::Matrix3f covariance;
                int color = rand()%1000;
                Cloud::Ptr tmpCloud(new Cloud());
                for (int k = 0; k < allMesh[i][j].pointsIndex.size(); k++) {
                    nonFacadeCloudRough->points[allMesh[i][j].pointsIndex[k]].intensity = color;
                    tmpCloud->points.push_back(nonFacadeCloudRough->points[allMesh[i][j].pointsIndex[k]]);
                }
                pcl::compute3DCentroid(*tmpCloud, pcaCentroid);
                pcl::computeCovarianceMatrixNormalized(*tmpCloud, pcaCentroid, covariance);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance, Eigen::ComputeEigenvectors);
                Eigen::Matrix3f eigenVectorsPCA = eigenSolver.eigenvectors();
                Eigen::Vector3f eigenValuesPCA = eigenSolver.eigenvalues();
//                std::cout << "eigen value: " << eigenValuesPCA << std::endl;
                // If the biggest eigen value is much lager than the others -> pole like object.
                // And the biggest eigen vector should be vertical to z axis.
                Eigen::Vector3f bigEigenVector = eigenVectorsPCA.block<3, 1>(0, 2);
                Eigen::Vector3f smallEigenVector = eigenVectorsPCA.block<3, 1>(0, 0);

                Eigen::Vector3f smallEigenVectorNormalized = smallEigenVector.normalized();
                Eigen::Vector3f zAxis(0, 0, 1);
//                float horizontalDiff = (smallEigenVectorNormalized.cross(xAxis)).norm();
                float verticalDiff = (smallEigenVectorNormalized.cross(zAxis)).norm();
//                std::cout << verticalDiff << std::endl;
                if (verticalDiff < 0.1 && eigenValuesPCA[0] < 100 * eigenValuesPCA[1]) {
//                    std::cout << "verticalDiff: " << verticalDiff << std::endl;
                    Cloud::Ptr filteredCloud(new Cloud());
                    for (auto& p : tmpCloud->points) {
                        if (p.x < 50 && p.x > -50 && p.y > -50 && p.y < 50) {
                            filteredCloud->points.push_back(p);
                        }
                    }
                    if (!(filteredCloud->points.empty())) {
                        *groundCloud += *filteredCloud;
                        groundCandidates.push_back(filteredCloud);
                    }

                }
            }
        }
    }
//    std::cout << "groundCandidates size: " << groundCandidates.size() << std::endl;
//    for (int i = 0; i < groundCandidates.size(); ++i) {
//        if (groundCandidates[i]->points.size() < 10) {
//            std::cout << "ground points: " << groundCandidates[i]->points.size() << std::endl;
//        }
//    }
}
void PrimitiveExtractor::polesClusteringWithFullCloud() {
//    Time t("Clustering Poles With Full Cloud Time");
    // Poles cluster and segment
    tree->setInputCloud(poleCloudRough);
    pcl::EuclideanClusterExtraction<Point> clustering;

    clustering.setClusterTolerance(0.5);
    clustering.setMinClusterSize(50);
    clustering.setMaxClusterSize(25000);
    clustering.setSearchMethod(tree);
    clustering.setInputCloud(poleCloudRough);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);
    //    std::cerr << "clusters number: " << clusters.size() << std::endl;
    //    poleCandidates.resize(clusters.size());
    // For every cluster...
    int currentClusterNum = 0;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i) {
        if (i->indices.size() > 1000) {
            continue;
        }
        pcl::PointCloud<Point>::Ptr cluster(new pcl::PointCloud<Point>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(poleCloudRough->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;


        if (cluster->points.size() <= 0)
            break;
        //        std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;

        //        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/"
        //                                      + std::to_string(currentClusterNum) + ".pcd", *cluster);

        poleCandidates.push_back(cluster);
        currentClusterNum++;
    }
}
void PrimitiveExtractor::testNewPoleExtractor() {
    Time t("New method time consumption");
    projectPointCloud();

    polesExtractionWithFullCloudWithPCA();
    estimatePolesModel();

    facadeExtraction();
    facadeSegmentationNew();
    estimateFacadesModel();

    groundExtractionPCA();
    estimateGroundModel();


}
void PrimitiveExtractor::runForMapping() {
//    Time t("Run for Facades And Poles Mapping Time");
    projectPointCloud();

//    polesExtractionWithFullCloudWithPCA();
//    estimatePolesModel();

    facadeExtraction();
    facadeSegmentationNew();
    estimateFacadesModel();

//    groundExtractionPCA();
//    estimateGroundModel();


}
void PrimitiveExtractor::getFusionFacade(Cloud::Ptr& fusionFacadeCloud, std::vector<Plane_Param>& mapFacadeParam) {
    std::cout << "Processing facades...." << std::endl;
    facadeCloudRough = fusionFacadeCloud;
    facadeSegmentationNew();
    estimateFacadesModel();
    for (auto& facade : facadesParam) {
        mapFacadeParam.push_back(facade);
    }
    std::cout << "Finish processing facades" << std::endl;
}
void PrimitiveExtractor::getFusionPole(Cloud::Ptr& fusionPoleCloud, std::vector<Cylinder_Fin_Param>& mapPoleParam) {
    std::cout << "Processing poles...." << std::endl;
    poleCloudRough = fusionPoleCloud;
    polesClusteringWithFullCloud();
    estimatePolesModel();
    for (auto& pole : polesParam) {
        mapPoleParam.push_back(pole);
    }
    std::cout << "Finish processing poles" << std::endl;
}


void PrimitiveExtractor::polesRoughExtraction() {
//    Time t("Rough Poles Extraction Time");
    length = (int)(range[1] - range[0]);
    width = (int)(range[3] - range[2]);
    int row = std::floor(width / meshSize);
    int column = std::floor(length / meshSize);

    std::vector<std::vector<mesh>> allMesh(column + 1, std::vector<mesh>(row + 1)); // column for x row for y
    for (int i = 0; i < nonFacadeCloudRough->points.size(); ++i) {
        float x = nonFacadeCloudRough->points[i].x;
        float y = nonFacadeCloudRough->points[i].y;
        float z = nonFacadeCloudRough->points[i].z;
        if (abs(x) < length / 2 && abs(y) < width / 2 && z < 0.7) { // range of selected area
            int c = std::floor((x - (-length / 2)) / meshSize);     // x
            int r = std::floor((y - (-width / 2)) / meshSize);      // y
            if (c > column || r > row) {
                std::cerr << "c and r: " << c << r << std::endl;
            }
            allMesh[c][r].pointsIndex.push_back(i);
            allMesh[c][r].density++;
        }
    }

    // label valid mesh
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].density > minNumPoint) {
                allMesh[i][j].state = 1;
            }
        }
    }

    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].state == 1) {

                for (int k = 0; k < allMesh[i][j].pointsIndex.size(); k++) {
                    poleCloudRough->push_back(
                        nonFacadeCloudRough
                            ->points[allMesh[i][j].pointsIndex[k]]); // push in candidate for cylinder ransac
                }
            }
        }
    }
}

void PrimitiveExtractor::polesGrowing() {

//    Time t("Growing Poles Time");
    for (int i = 0; i < N_SCAN; ++i) {
        for (int j = 0; j < Horizon_SCAN; ++j) {
            int index = i + j * N_SCAN;
            if (fullCloud->points[index].intensity == -1) {
                continue;
            }
            for (auto pPole : *poleCloudRough) {
                float distance =
                    std::sqrt((pPole.x - fullCloud->points[index].x) * (pPole.x - fullCloud->points[index].x) +
                              (pPole.y - fullCloud->points[index].y) * (pPole.y - fullCloud->points[index].y));
                if (distance < 0.5) {
                    poleCandidatesMat.at<int8_t>(i, j) = 1;
                }
            }
        }
    }
    for (int i = 0; i < N_SCAN; ++i) {
        for (int j = 0; j < Horizon_SCAN; ++j) {
            int index = i + j * N_SCAN;
            if (poleCandidatesMat.at<int8_t>(i, j) == 1) {
                poleCloudGrow->points.push_back(fullCloud->points[index]);
            }
        }
    }
}
void PrimitiveExtractor::cloudClustering() { // Old version with slow pole growing
//    Time t("Clustering Poles Time");
    // Poles cluster and segment
    tree->setInputCloud(poleCloudGrow);
    pcl::EuclideanClusterExtraction<Point> clustering;

    clustering.setClusterTolerance(0.2);
    clustering.setMinClusterSize(100);
    clustering.setMaxClusterSize(25000);
    clustering.setSearchMethod(tree);
    clustering.setInputCloud(poleCloudGrow);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);

    poleCandidates.resize(clusters.size());
    // For every cluster...
    int currentClusterNum = 0;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i) {

        pcl::PointCloud<Point>::Ptr cluster(new pcl::PointCloud<Point>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(poleCloudGrow->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;


        if (cluster->points.size() <= 0)
            break;
        //        std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;

        //        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/"
        //                                      + std::to_string(currentClusterNum) + ".pcd", *cluster);

        poleCandidates[currentClusterNum] = cluster;
        currentClusterNum++;
    }
}

bool PrimitiveExtractor::customRegionGrowing(const PointINormal& pointA,
                                             const PointINormal& pointB,
                                             float squaredDistance) {

    Eigen::Map<const Eigen::Vector3f> point_a_normal = pointA.getNormalVector3fMap(),
                                      point_b_normal = pointB.getNormalVector3fMap();
    if (squaredDistance < 10000) {
        if (std::abs(point_a_normal.dot(point_b_normal)) < 0.06)
            return (true);
    } else {
        if (std::abs(pointA.intensity - pointB.intensity) < 3.0f)
            return (true);
    }
    return (false);
}

void PrimitiveExtractor::facadeSegmentationCEC() {
    Cloud::Ptr cloud_out(new Cloud());
    pcl::PointCloud<PointINormal>::Ptr cloud_with_normals(new pcl::PointCloud<PointINormal>);
    pcl::IndicesClustersPtr clustersCEC(new pcl::IndicesClusters), small_clusters(new pcl::IndicesClusters),
        large_clusters(new pcl::IndicesClusters);
    pcl::search::KdTree<Point>::Ptr search_tree(new pcl::search::KdTree<Point>);

    pcl::VoxelGrid<Point> vg;
    vg.setInputCloud(facadeCloudRough);
    vg.setLeafSize(0.2, 0.2, 0.2);
    vg.setDownsampleAllData(true);
    vg.filter(*cloud_out);

    // Set up a Normal Estimation class and merge data in cloud_with_normals
    pcl::copyPointCloud(*cloud_out, *cloud_with_normals);
    pcl::NormalEstimation<Point, PointINormal> ne;
    ne.setInputCloud(cloud_out);
    ne.setSearchMethod(search_tree);
    ne.setRadiusSearch(300.0);
    ne.compute(*cloud_with_normals);

    // Set up a Conditional Euclidean Clustering class
    pcl::ConditionalEuclideanClustering<PointINormal> cec(true);
    cec.setInputCloud(cloud_with_normals);
    cec.setConditionFunction(&customRegionGrowing);
    cec.setClusterTolerance(500.0);
    // cec.setMinClusterSize (cloud_with_normals->size () / 1000);
    // cec.setMaxClusterSize (cloud_with_normals->size () / 5);
    cec.segment(*clustersCEC);

    // cec.getRemovedClusters (small_clusters, large_clusters);
    for (int clusterI = 0; clusterI < (*clustersCEC).size(); clusterI++) {
        Cloud::Ptr tempCloud(new Cloud());
        Point tempP;
        for (int i = 0; i < (*clustersCEC)[clusterI].indices.size(); ++i) {
            tempP = (*clustersCEC)[clusterI].indices[i];
            tempP.intensity = clusterI * 10;
            tempCloud->points.push_back(tempP);
        }
        *facadeCloudCandidates += *tempCloud;
    }
    //    pcl::io::savePCDFile("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/facadeClusters/clusters/cec_seg_3900.pcd",
    //                         *facadeCloudCandidates);
}

void PrimitiveExtractor::facadeSegmentationNew() {
//    Time t("Segmentation Facades Time");
    //// Region growing segmentation
    //// then store the got facades candidates in facadesCandidates for model estimation
    // Region Growing for facade segmentation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*facadeCloudRough, *cloud);
    pcl::search::Search<pcl::PointXYZ>::Ptr treeRG(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(treeRG);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(100);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(treeRG);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(5.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);
    std::vector<pcl::PointIndices> clustersRG;
    reg.extract(clustersRG);

    //    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
    //    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    //    viewer->setBackgroundColor(255,255,255);
    //    viewer->addPointCloud(colored_cloud);
    //    while (!viewer->wasStopped ())
    //    {
    //        viewer->spinOnce();
    //    }

    // For every cluster
    facadeCandidates.resize(clustersRG.size());
    int intensityIndex = 0;
    for (auto& cluster : clustersRG) {
        pcl::PointCloud<Point>::Ptr tmpCluster(new pcl::PointCloud<Point>);
        Point p;
        for (int i = 0; i < cluster.indices.size(); ++i) {
            p = facadeCloudRough->points[cluster.indices[i]];
            p.intensity = intensityIndex * 30;
            tmpCluster->points.push_back(p);
        }
        facadeCandidates[intensityIndex] = tmpCluster;
        intensityIndex++;
    }
}

void PrimitiveExtractor::groundSegmentationRG() {
    Time t("Segmentation Ground Time");
    //// Region growing segmentation
    //// then store the got ground candidates in groundCandidates for model estimation

    // Region Growing for ground segmentation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//    for (int i = 0; i < nonFacadeCloudRough; ++i) {
//
//    }


    pcl::copyPointCloud(*nonFacadeCloudRough, *cloud);
    pcl::search::Search<pcl::PointXYZ>::Ptr treeRG(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(treeRG);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(100);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(treeRG);
    reg.setNumberOfNeighbours(15);
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(5.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);
    std::vector<pcl::PointIndices> clustersRG;
    reg.extract(clustersRG);

    //    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
    //    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    //    viewer->setBackgroundColor(255,255,255);
    //    viewer->addPointCloud(colored_cloud);
    //    while (!viewer->wasStopped ())
    //    {
    //        viewer->spinOnce();
    //    }

    // For every cluster
//    groundCandidates.resize(clustersRG.size());
    int intensityIndex = 0;
    for (auto& cluster : clustersRG) {
        pcl::PointCloud<Point>::Ptr tmpCluster(new pcl::PointCloud<Point>);
        Point p;
        for (int i = 0; i < cluster.indices.size(); ++i) {
            p = nonFacadeCloudRough->points[cluster.indices[i]];
            p.intensity = intensityIndex * 30;
            tmpCluster->points.push_back(p);
        }

        groundCandidates.push_back(tmpCluster);
        intensityIndex++;
    }
    std::cout << "Ground Candidates: " << groundCandidates.size() << std::endl;
}
void PrimitiveExtractor::facadeSegmentation() {
//    Time t("Segmentation Facades Time");
    //// Region growing and Euclidean clustering and segmentation
    //// then store the got facades candidates in facadesCandidates for model estimation
    // Region Growing for facade segmentation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*facadeCloudRough, *cloud);
    pcl::search::Search<pcl::PointXYZ>::Ptr treeRG(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(treeRG);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(100);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(treeRG);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(1.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);
    std::vector<pcl::PointIndices> clustersRG;
    reg.extract(clustersRG);


    // Euclidean cluster to connect facade pieces;
    Cloud::Ptr facadesCloudRG(new Cloud());

    for (auto& cluster : clustersRG) {
        Cloud::Ptr tempCloud(new Cloud());
        Cloud::Ptr tempCloudZ(new Cloud());
        Point tempP;
        for (int i = 0; i < cluster.indices.size(); i++) {
            tempP = facadeCloudRough->points[cluster.indices[i]];
            tempCloud->points.push_back(tempP);
        }

        *facadesCloudRG += *tempCloud;
    }

    // facades cluster and segment
    treeEC->setInputCloud(facadesCloudRG);
    pcl::EuclideanClusterExtraction<Point> clustering;

    clustering.setClusterTolerance(0.5);
    clustering.setMinClusterSize(100);
    clustering.setMaxClusterSize(25000000);
    clustering.setSearchMethod(treeEC);

    clustering.setInputCloud(facadesCloudRG);
    std::vector<pcl::PointIndices> clustersEC;
    clustering.extract(clustersEC);
    facadeCandidates.resize(clustersEC.size());
    // For every cluster
    int currentClusterNum = 0;
    for (std::vector<pcl::PointIndices>::const_iterator i = clustersEC.begin(); i != clustersEC.end(); ++i) {

        pcl::PointCloud<Point>::Ptr cluster(new pcl::PointCloud<Point>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(facadesCloudRG->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;


        if (cluster->points.size() <= 0)
            break;

        facadeCandidates[currentClusterNum] = cluster;
        currentClusterNum++;
    }
}

void PrimitiveExtractor::estimateFacadesModel() {
//    Time t("Estimate Facades Model");
    // Process:
    // - filter out not suitable facades candidates
    // - facades plane estimation
    // - store facades plane parameters

    // filter out not suitable facades
    //#pragma omp parallel for
    for (int facadeCloudId = 0; facadeCloudId < facadeCandidates.size(); facadeCloudId++) {
        //        std::cout << "size " << facadeCandidates.size() << std::endl;
        Cloud::Ptr facadeCloud(facadeCandidates[facadeCloudId]);
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
        if ((x_max - x_min < 1.0) && (y_max - y_min < 1.0)) { // 2 2
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
        bool validFacadeFlag = findFacadesRANSAC(facadeCloud, normal, d);
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
        float plane_min_height = 1.5f; // 2.5
        float plane_long_width = 1;  // 5
        float plane_min_width = 1.0f; // 2.0
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

        //        cv::namedWindow("plane_projection",cv::WINDOW_NORMAL);
        //        cv::imshow("plane_projection",projection_image*100);
        //        cv::waitKey();

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
        float min_column_occupany_meter = 0.3; //0.3
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
            facadesParam.push_back(plane_param);
        }
    }
}

bool PrimitiveExtractor::findFacadesCeres(
    Cloud::Ptr& cloud, Eigen::Vector3f& initN, double& d, int& maxIteration, Plane_Param& pParam) {
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
    // build residual
    double n1 = initN[0];
    double n2 = initN[1];
    double n3 = initN[2];
    for (int i = 0; i < cloud->points.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<FacadeResidual, 1, 1, 1, 1, 1>(new FacadeResidual(cloud->points[i]));
        problem.AddResidualBlock(cost_function, loss_function, &n1, &n2, &n3, &d);
    }
    // solve.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = maxIteration;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //    std::cout << summary.BriefReport() << std::endl;
    pParam.d = d;
    pParam.n[0] = n1;
    pParam.n[1] = n2;
    pParam.n[2] = n3;
    return true;
}

bool PrimitiveExtractor::findFacadesRANSAC(Cloud::Ptr& cloud, Eigen::Vector3f& normal, float& d) {
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


// Same algorithm as facades
void PrimitiveExtractor::estimateGroundModel() {
//    Time t("Estimate Ground Model");
    // Process:
    // - filter out not suitable ground candidates
    // - ground plane estimation
    // - store ground plane parameters

    // filter out not suitable ground
    //#pragma omp parallel for
    for (int facadeCloudId = 0; facadeCloudId < groundCandidates.size(); facadeCloudId++) {
//                std::cout << "size " << groundCandidates.size() << std::endl;
//        if (groundCandidates[facadeCloudId]->points.empty()) continue;
        Cloud::Ptr facadeCloud(groundCandidates[facadeCloudId]);
        // Check if the cloud has reasonable large xy dimension.



        // Estimate facades model for each cluster
        Plane_Param facadeParam;
        // Accept initial norm from PCA and d for ceres
        Eigen::Vector3f normal;
        float d;
        bool validFacadeFlag = findFacadesRANSAC(facadeCloud, normal, d);
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

        //        cv::namedWindow("plane_projection",cv::WINDOW_NORMAL);
        //        cv::imshow("plane_projection",projection_image*100);
        //        cv::waitKey();

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
        float min_column_occupany_meter = 0.3; //0.3
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
            groundsParam.push_back(plane_param);
        }
    }
}

// Algorithm:
//  - split each cluster into multiple layers according to the laser layers
//  - fit each layer a circle
//  - sort the calculated circles only keep similar circles
//  - fit a cylinder model by averaging circles
//  - store the radius and axis this cylinder
void PrimitiveExtractor::estimatePolesModel() {
//    Time t1("Estimate poles time");

    struct CircleModel {
        CircleModel() : r(-1.0f), center(Eigen::Vector2f(0.0f, 0.0f)), z(0) {
        }
        float r;
        Eigen::Vector2f center;
        float z;
    };
    struct CylinderModel {
        CylinderModel() : r(-1.0f) {
        }
        float r;
        Eigen::VectorXf axis;
    };
    std::vector<CircleModel> circle_sum;
    std::vector<Cylinder_Fin_Param> poles_param_intern(poleCandidates.size());
    for (auto& pole : poles_param_intern) {
        pole.radius = -1;
    }

    //#pragma omp parallel for
    for (int poleCloudId = 0; poleCloudId < poleCandidates.size(); ++poleCloudId) {
        // Each cluster pole cloud
        Cloud::Ptr poleCloud(poleCandidates[poleCloudId]);

        // Define layers. Each layer shall only contain a single scanline.

        // divide point cloud according to vertical angular resolution
        // vector clouds to store each layer
        //// Sensor default parameter needed to change for different sensor
        //// ***************Split Each Pole********************
        std::vector<Cloud::Ptr> layerClouds;
        for (int i = 0; i < N_SCAN; ++i) {
            layerClouds.emplace_back(new Cloud());
        }
        float verticalRevolution = 0.3125;
        float minZValue = 99999; // minimal z value also is the ground intersection
        float maxZValue = -99999;
        for (auto& p : poleCloud->points) {
            if (p.z < minZValue) {
                minZValue = p.z;
            }
            if (p.z > maxZValue) {
                maxZValue = p.z;
            }
            int scanID;
            float angle = atan2(p.z, sqrt(p.x * p.x + p.y * p.y)) * 180 / M_PI;
            if (angle > -25.0 && angle < 15.0) { // -25.0 15.0
                if (angle > 0) {
                    scanID = (int)(((15.0 - angle) / verticalRevolution) * 10 + 5) / 10;
                } else {
                    scanID = (int)(((15.0 - angle) / verticalRevolution) * 10 - 5) / 10;
                }
            } else {
                continue;
            }
            p.intensity = scanID;
            layerClouds[scanID]->points.push_back(p);
        }


        for (int i = 0; i < layerClouds.size(); ++i) {
            int color = rand() % 5000;
            if (layerClouds[i]->points.empty()) continue;
            for (int j = 0; j < layerClouds[i]->points.size(); ++j) {
                layerClouds[i]->points[j].intensity = color;
                poleCloudRough->points.push_back(layerClouds[i]->points[j]);
            }
        }
        // calculate the average z value with the lowest layer
//        float averageZ = 0;
//        for (int i = 0; i < layerClouds[layerClouds.size() - 1]->points.size(); ++i) {
//            averageZ += layerClouds[layerClouds.size() - 1]->points[i].z;
//        }
//        averageZ = averageZ / layerClouds[layerClouds.size() - 1]->points.size();

        //// ***************for each layer cloud find a circle********************
        std::vector<CircleModel> circles;
        //        std::cout << "num of layers: " << layerClouds.size();
        for (int i = 0; i < layerClouds.size(); ++i) {
            auto& cloud = layerClouds[i];
            CircleModel circle;
            if (cloud->points.size() > 3) { // make sure each layer contain enough points//5
                bool validCircleFlag;
                float radius;
                Eigen::Vector2f centerOut;
                // Ceres test
                validCircleFlag = findCircleCeres(cloud, 5, 0.4, 0.05, radius, centerOut);
//                validCircleFlag = find_circle_RANSAC(cloud, 20, 0.06, 0.6, 0.5, 0.99, 0.5, 0.05, 0.4, radius, centerOut);
                if (validCircleFlag) {
                    circle.r = radius;
                    circle.center = centerOut;
                }
            }
            // Circle average z.
            float averageZ = 0;
            for (auto& p : cloud->points) {
                averageZ += p.z;
            }
            circle.z = averageZ / (float)cloud->points.size();
            if (circle.r != -1) { // Radius = -1 means invalid circle
                circles.push_back(circle);
            }
        }



        if (circles.size() < 3) {
            // too few valid circle
            continue;
        }

        // Check the number if valid circles.
        int numValidCircles = 0;
        int idValidCircles = 0;
        std::vector<float> radii;
        // Filter out circle center out of range
        float averX = 0;
        float averY = 0;
        int circleCount = 0;
        for (auto& circle : circles) {
            if (circles.empty())
                break;
            if (circle.r != -1.0f) {
                averX += circle.center[0];
                averY += circle.center[1];
                circleCount++;
            }
        }
        averX /= circleCount;
        averY /= circleCount;
        Eigen::Vector2f averCenter(averX, averY);
        // Check circles position validity
        for (auto& circle : circles) {
            if (circle.r != -1.0f) {
                Eigen::Vector2f center(circle.center);
                float dis = (center - averCenter).norm();
                if (dis > 0.07) {
                    circle.r = -1.0f;
                }
            }
        }
        for (auto& circle : circles) {
            if (circle.r != -1.0f) {
                numValidCircles++;
                radii.push_back(circle.r);
            }
        }
        // If there are too few good circles then no pole can be found.
        ////param need to be give
        int minNumValidCircles = 3;
        if (numValidCircles < minNumValidCircles) {
//            std::cerr << "valid circles: " << numValidCircles << std::endl;
            continue;
        }

        std::partial_sort(radii.begin(), radii.begin() + radii.size() / 2 + 1, radii.end());
        double medianRadius = radii[radii.size() / 2];


        //// ****************** RANSAC fit axis of pole ***************
        // input center to cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (int i = 0; i < circles.size(); ++i) {
            if (circles[i].r == -1.0f)
                continue;
            pcl::PointXYZ point;
            point.x = circles[i].center[0];
            point.y = circles[i].center[1];
            point.z = circles[i].z;
            tmpCloud->points.push_back(point);
        }
//        std::cerr << "Axis points Number: " << tmpCloud->points.size() <<  std::endl;
        pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr modelLine(
            new pcl::SampleConsensusModelLine<pcl::PointXYZ>(tmpCloud));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(modelLine);
        ransac.setDistanceThreshold(0.04);
        ransac.setMaxIterations(100);
        ransac.computeModel();
        // Store points on line.
        std::vector<int> inliers;
        ransac.getInliers(inliers);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudLine(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud<pcl::PointXYZ>(*tmpCloud, inliers, *cloudLine);
        // Store line model coefficients
        Eigen::VectorXf coef;
        ransac.getModelCoefficients(coef); // coef[0]、coef[1]、coef[2] point on the line
                                           //  coef[3]、coef[4]、coef[5] direction of this line
        // Check the direction of axis
        Eigen::Vector3f dirAxis(coef[3], coef[4], coef[5]);
        Eigen::Vector3f zAxis(0, 0, 1);
        if ((dirAxis.cross(zAxis)).norm() > 0.1) { // The axis should be parallel to zAxis
            //            std::cout << (dirAxis.cross(zAxis)).norm() << std::endl;
            continue;
        }
        // Find top and down point on axis
        pcl::PointXYZ topP{0, 0, -9999};
        pcl::PointXYZ downP{0, 0, 9999};
        for (auto& p : cloudLine->points) {
            if (p.z > topP.z) {
                topP = p;
            }
            if (p.z < downP.z) {
                downP = p;
            }
        }


        //// ***************Check estimated model whether fit the cloud enough**********
        float dis;
        int validCirclePoint = 0;
        int inValidCount = 0;
        int circleNum = 0;
        for (int i = 0; i < layerClouds.size(); ++i) {
            auto& cloud = layerClouds[i];
            if (cloud->points.size() < 3) {
                continue;
            }
            circleNum++;                    // layer number
            for (auto& p : cloud->points) { // check in each circle validity
                Eigen::Vector2f point2Center(p.x - coef[0], p.y - coef[1]);
                dis = point2Center.norm();
                if (std::abs(dis - medianRadius) < 0.05) { // 15 cm count as valid
                    validCirclePoint++;
                }
            }
            if (validCirclePoint < cloud->points.size() * 0.8) {
                // If valid circle point is less than 80% of whole points
                // -> invalid
                inValidCount++;
            }
        }
        //        std::cout << "inValidCount:" << inValidCount << std::endl;
        //        std::cout << "validCount: " << validCount << std::endl;

//        if (inValidCount > circleNum * 0.8) {
//            // If invalid circle number is more than 70% whole circle number
//            // -> invalid poles
//            continue;
//        }

        // Find final params.
        float radiusFinal = 0;
        int idCenters = 0;
        // Line fit chose the closet point to line
        Eigen::Vector3f pointOnLine(coef[0], coef[1], coef[2]);
        Eigen::Vector3f lineDirection(coef[3], coef[4], coef[5]);

//        for (auto& circle : circles) {
//            if (circle.r == -1.0f) continue;
////            Eigen::Vector3f lineDirection(0, 0, 1);
//            Eigen::Vector3f center(circle.center[0], circle.center[1], circle.z);
//            createTheoryCircle(lineDirection, center, circle.r, poleCloudRough, 50000);
//        }

        for (auto& circle : circles) {
            if (circle.r != -1.0f) {
                Eigen::Vector3f center(circle.center[0], circle.center[1], circle.z);
                float dis = (float)(((center - pointOnLine).cross(lineDirection)).norm() / lineDirection.norm());
                if (dis > 0.3) {
                    circle.r = -1.0f; // If center is invalid turn radius to -1
                }
            }
        }
        // After RANSAC chose closet point to axis as valid circle
        int numOpt = 0;
        for (auto& circle : circles) {
            if (circle.r == -1.0f)
                continue;
            radiusFinal += circle.r;
            numOpt++;
        }
        if (numOpt == 0) {
            radiusFinal = medianRadius;
        }
        radiusFinal /= (float)numOpt;


        //// *************** Save cylinder param ****************
        Cylinder_Fin_Param cylinder;
        cylinder.line_dir = lineDirection;
        cylinder.top_point = Eigen::Vector3f{topP.x, topP.y, topP.z};
        //        cylinder.down_point = Eigen::Vector3f{downP.x, downP.y, downP.z};
        cylinder.down_point = pointOnLine;
        // important para for localization
        cylinder.center_line.p_bottom =
            pointOnLine + lineDirection(2) / std::abs(lineDirection(2)) * (minZValue - pointOnLine(2)) * lineDirection;
        cylinder.center_line.p_top =
            pointOnLine + lineDirection(2) / std::abs(lineDirection(2)) * (maxZValue - pointOnLine(2)) * lineDirection;
        // Cut cylinder axis with ground plane.
        Eigen::Vector3f pos_2D = pointOnLine + (minZValue - pointOnLine(2) / lineDirection(2)) * lineDirection;
        cylinder.pos_2D = Eigen::Vector2f(pos_2D(0), pos_2D(1));
        cylinder.radius = radiusFinal;
        cylinder.ground_z = minZValue;
        poles_param_intern[poleCloudId] = cylinder;
    }
    for (auto& pole : poles_param_intern) {
        if (pole.radius != -1) {
            polesParam.push_back(pole);
        }
    }
}


bool PrimitiveExtractor::findCircleCeres(Cloud::Ptr& cloud,
                                         int numIterations,
                                         float maxDiameter,
                                         float thresholdDistance,
                                         float& radius,
                                         Eigen::Vector2f& centerOut) { // find circle model in each layer
    // param:
    //  - numIterations: maximal iteration times
    //  - minDiameter, maxDiameter: default border for radius
    //  - thresholdDistance: threshold for test got circle model
    //  - radius, centerOut: estimated circle model


    // Check if the cloud has enough points.
    if (cloud->points.size() < 2) {
        return false;
    }
    // Check if the cloud has reasonable large xy dimension.
    float x_min = cloud->points[0].x;
    float x_max = x_min;
    float y_min = cloud->points[0].y;
    float y_max = y_min;
    float small_radius = 0.025;
    for (auto& p : cloud->points) {
        if (p.x < x_min)
            x_min = p.x;
        else if (p.x > x_max)
            x_max = p.x;
        if (p.y < y_min)
            y_min = p.y;
        else if (p.y > y_max)
            y_max = p.y;
    }
    // If the points are very close together then fit a circle around.
    if (((x_max - x_min < small_radius * 2.0f) && (y_max - y_min < small_radius * 2.0f)) || cloud->points.size() < 3) {
        // Find center position by averaging.
        Eigen::Vector2f center_(0.0f, 0.0f);
        for (auto& p : cloud->points) {
            center_ += Eigen::Vector2f(p.x, p.y);
        }
        centerOut = center_ / cloud->points.size();

        // Find the average distance to center which will be the radius.
        radius = 0;
        for (auto& p : cloud->points) {
            radius += (centerOut - Eigen::Vector2f(p.x, p.y)).norm();
        }
        radius /= cloud->points.size();
        if (radius < small_radius) {
            radius = small_radius;
//            std::cout << "SMALL RADIUS CIRCLE:     RADIUS: " <<radius <<"  CENTER: "<< centerOut;
        }

        return true;
    }




    // Check if input cloud is small enough to possibly represent a circle with a diameter <= max_diameter.
    float cloud_min_x = cloud->points[0].x;
    float cloud_max_x = cloud->points[0].x;
    float cloud_min_y = cloud->points[0].y;
    float cloud_max_y = cloud->points[0].y;

    for (auto p : cloud->points) {
        if (p.x < cloud_min_x)
            cloud_min_x = p.x;
        else if (p.x > cloud_max_x)
            cloud_max_x = p.x;
        if (p.y < cloud_min_y)
            cloud_min_y = p.y;
        else if (p.y > cloud_max_y)
            cloud_max_y = p.y;
    }
    if (std::max(cloud_max_x - cloud_min_x, cloud_max_y - cloud_min_y) > maxDiameter * 1.5f) {
//        std::cerr << "Too big radius!" << std::endl;
        return false;
    }



    // ceres to estimate circle param
    int numCloudPoints = cloud->points.size();
    // Initial values for unknown circle parameters.
    // Init the center with components c_x and c_y by averaging over a few samples.
    int numCenterInit = cloud->points.size();
    double centerX = 0;
    double centerY = 0;
    for (int i = 0; i < numCenterInit; ++i) {
        centerX += cloud->points[i].x;
        centerY += cloud->points[i].y;
    }
    centerX /= numCenterInit;
    centerY /= numCenterInit;
    double r = std::max(cloud_max_x - cloud_min_x, cloud_max_y - cloud_min_y) / 2;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    // build residual
    for (int i = 0; i < cloud->points.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<CircleResidual, 1, 1, 1, 1>(new CircleResidual(cloud->points[i]));
        problem.AddResidualBlock(cost_function, loss_function, &centerX, &centerY, &r);
    }
    // solve.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = numIterations;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.BriefReport() << std::endl;



    if (2 * r > maxDiameter) {
//        std::cerr << "Got too large radius!" << std::endl;
        return false;
    }
    Eigen::Vector2f center(centerX, centerY);
    // check estimated circle whether default value
    int support_counter = 0;
//    Cloud::Ptr support_cloud(new Cloud);
//    Cloud::Ptr outlier_cloud(new Cloud);
    for (Cloud::const_iterator it = cloud->points.begin(); it < cloud->points.end(); it++) {
        // Calculate distance to circle .
        Eigen::Vector2f p(it->x, it->y);
        float dist = std::abs((p - center).norm() - r);
        if (dist < thresholdDistance) {
            support_counter++;
//            support_cloud->points.push_back(*it);
        } else {
//            outlier_cloud->points.push_back(*it);
        }
    }

    // Check if the circle is a good fit.
    if (support_counter >= (int)(numCloudPoints * 0.8)) {
        centerOut = center;
        radius = r;
        return true;
    } else {
//        std::cerr << "Bad fitting!" << std::endl;
        return false;
    }


}

bool PrimitiveExtractor::find_circle_RANSAC_fixed_radius(Cloud::Ptr& cloud,
                                                         const int& num_samples,
                                                         const float& support_dist,
                                                         const float& min_circle_support_percentage,
                                                         const float& neg_support_dist,
                                                         const float& min_circle_quality,
                                                         const float& small_radius,
                                                         const float& radius,
                                                         Eigen::Vector2f& center_out) {
    // Check if the cloud has enough points.
    if (cloud->points.size() < 2) {
        return false;
    }

    double r = radius;
    int cl_size = cloud->points.size();

    // If the input radius is also the small_radius then fit center based on averaging.
    if (std::abs(radius - small_radius) < 0.000001) {
        // Find center position by averaging.
        Eigen::Vector2f center(0.0f, 0.0f);
        for (auto& p : cloud->points) {
            center += Eigen::Vector2f(p.x, p.y);
        }
        center = center / cloud->points.size();

        // Check support.
        float neg_support_range_dist = neg_support_dist - support_dist;

        int support_counter = 0;
        int neg_support_counter = 0;
        double support_counter_weighted = 0;
        double neg_support_counter_weighted = 0;

        for (auto& p : cloud->points) {
            // Check dist to center.
            float dist = std::abs((center - Eigen::Vector2f(p.x, p.y)).norm() - r);

            if (dist < support_dist) {
                support_counter++;
                support_counter_weighted += 1.0001f - dist / support_dist;
            } else if (dist < neg_support_dist) {
                neg_support_counter++;
                neg_support_counter_weighted += 1.0f - 0.5f * dist / neg_support_range_dist;
            }
        }

        // If not enough support then ignore this circle estimation.
        if (support_counter < std::floor(min_circle_support_percentage * cl_size)) {
            return false;
        }

        // Check circle quality.
        float average_neg_score = 0;
        if (neg_support_counter != 0)
            average_neg_score = -neg_support_counter_weighted / (float)neg_support_counter;
        float average_pos_score = support_counter_weighted / (float)support_counter;
        float circle_quality =
            ((float)neg_support_counter * average_neg_score + (float)support_counter * average_pos_score) /
            ((float)neg_support_counter + (float)support_counter);

        if (circle_quality < min_circle_quality) {
            return false;
        }

        center_out = center;
        return true;
    } else {
        // Helper.
        struct Circle_Model {
            Circle_Model() : r(0.0f), center(Eigen::Vector2f(0.0f, 0.0f)) {
            }
            float r;
            Eigen::Vector2f center;
        };

        // Preparation.
        std::vector<Circle_Model> circles;
        circles.resize(2 * num_samples);
        std::vector<int> support_counter; // Points which have dist < support_dist.
        support_counter.resize(2 * num_samples);
        std::vector<int> neg_support_counter; // Points which have support _dist <= dist < neg_support_dist.
        neg_support_counter.resize(2 * num_samples);
        std::vector<float> support_counter_weighted; // Weighting: higher weight on points which are close to plane.
        support_counter_weighted.resize(2 * num_samples);
        std::vector<float>
            neg_support_counter_weighted; // Weighting: higher weight on points which are close to support_dist.
        neg_support_counter_weighted.resize(2 * num_samples);

        int best_idx = -1;

        // Multiple runs
        for (int i = 0; i < num_samples; i++) {

            // Sample two points.
            int rand_idx_1 = 0;
            int rand_idx_2 = 0;

            rand_idx_1 = std::rand() % cl_size;
            do {
                rand_idx_2 = std::rand() % cl_size;
            } while (rand_idx_1 == rand_idx_2);

            double x1 = cloud->points[rand_idx_1].x;
            double y1 = cloud->points[rand_idx_1].y;
            double x2 = cloud->points[rand_idx_2].x;
            double y2 = cloud->points[rand_idx_2].y;

            Eigen::Vector2f center_1, center_2;
            double q = std::sqrt(std::pow((x2 - x1), 2) + std::pow((y2 - y1), 2));
            double x3 = (x1 + x2) / 2.0;
            double y3 = (y1 + y2) / 2.0;
            double g = std::sqrt(r * r - q * q / 4.0) / q;

            center_1(0) = x3 + g * (y1 - y2);
            center_1(1) = y3 + g * (x2 - x1);
            center_2(0) = x3 - g * (y1 - y2);
            center_2(1) = y3 - g * (x2 - x1);

            // Set default values.
            support_counter[2 * i] = 0;
            neg_support_counter[2 * i] = 0;
            support_counter_weighted[2 * i] = 0;
            neg_support_counter_weighted[2 * i] = 0;
            support_counter[2 * i + 1] = 0;
            neg_support_counter[2 * i + 1] = 0;
            support_counter_weighted[2 * i + 1] = 0;
            neg_support_counter_weighted[2 * i + 1] = 0;

            // Check support.
            float neg_support_range_dist = neg_support_dist - support_dist;

            for (auto& p : cloud->points) {
                // Check dist to center.
                float dist_1 = std::abs((center_1 - Eigen::Vector2f(p.x, p.y)).norm() - r);
                float dist_2 = std::abs((center_2 - Eigen::Vector2f(p.x, p.y)).norm() - r);

                if (dist_1 < support_dist) {
                    support_counter[2 * i]++;
                    support_counter_weighted[2 * i] += 1.0001f - dist_1 / support_dist;
                } else if (dist_1 < neg_support_dist) {
                    neg_support_counter[2 * i]++;
                    neg_support_counter_weighted[2 * i] += 1.0f - 0.5f * dist_1 / neg_support_range_dist;
                }
                if (dist_2 < support_dist) {
                    support_counter[2 * i + 1]++;
                    support_counter_weighted[2 * i + 1] += 1.0001f - dist_2 / support_dist;
                } else if (dist_2 < neg_support_dist) {
                    neg_support_counter[2 * i + 1]++;
                    neg_support_counter_weighted[2 * i + 1] += 1.0f - 0.5f * dist_2 / neg_support_range_dist;
                }
            }

            // If not enough support then ignore this circle estimation.
            if (support_counter[2 * i] < std::round(min_circle_support_percentage * cl_size)) {
                support_counter[2 * i] = 0;
            }
            if (support_counter[2 * i + 1] < std::round(min_circle_support_percentage * cl_size)) {
                support_counter[2 * i + 1] = 0;
            }

            // Store params.
            circles[2 * i].r = r;
            circles[2 * i].center = center_1;
            circles[2 * i + 1].r = r;
            circles[2 * i + 1].center = center_2;
        }

        // Find best circle

        float best_circle_quality = min_circle_quality;
        for (int i = 0; i < num_samples; i++) {
            if (support_counter[i] != 0) {
                float average_neg_score = 0;
                if (neg_support_counter[i] != 0)
                    average_neg_score = -neg_support_counter_weighted[i] / (float)neg_support_counter[i];
                float average_pos_score = support_counter_weighted[i] / (float)support_counter[i];
                float circle_quality = ((float)neg_support_counter[i] * average_neg_score +
                                        (float)support_counter[i] * average_pos_score) /
                                       ((float)neg_support_counter[i] + (float)support_counter[i]);

                if (circle_quality > best_circle_quality) {
                    best_circle_quality = circle_quality;
                    best_idx = i;
                    //          std::cout <<"CIRCLE QUALITY: " << circle_quality <<std::endl;
                }
            }
        }

        //  std::cout<<"---------"<<std::endl;

        // If no good circle was found then set negative radius.
        if (best_idx == -1) {
            return false;
        }

        // Check if the support covers a angle range which is representative.
        int bin_num = 16;
        std::vector<bool> angle_histogram(bin_num, false);
        Eigen::Vector2f center_best = circles[best_idx].center;
        float radius_best = circles[best_idx].r;
        for (auto& p : cloud->points) {
            // Check dist to center.
            Eigen::Vector2f p_eigen(p.x, p.y);
            float dist = std::abs((center_best - p_eigen).norm() - radius_best);
            if (dist < support_dist) {
                Eigen::Vector2f vec = p_eigen - center_best;
                uint bin_id = std::floor((std::atan2(vec(1), vec(0)) + M_PI) / (2 * M_PI) * bin_num);
                angle_histogram[bin_id] = true;
            }
        }
        uint coverage = std::count(angle_histogram.begin(), angle_histogram.end(), true);
        if (coverage < 3) {
            return false;
        } else {
            center_out = circles[best_idx].center;
            return true;
        }
    }

}


bool PrimitiveExtractor::find_circle_RANSAC(Cloud::Ptr& cloud,
                                            const int& num_samples,
                                            const float& support_dist,
                                            const float& min_circle_support_percentage,
                                            const float& neg_support_dist,
                                            const float& early_stop_circle_quality,
                                            const float& min_circle_quality,
                                            const float& small_radius,
                                            const float& circle_max_radius, float& radius,
                                            Eigen::Vector2f& center_out) {
    // Check if the cloud has enough points.
    if (cloud->points.size() < 2) {
        return false;
    }

    // Check if the cloud has reasonable large xy dimension.
    float x_min = cloud->points[0].x;
    float x_max = x_min;
    float y_min = cloud->points[0].y;
    float y_max = y_min;
    for (auto& p : cloud->points) {
        if (p.x < x_min)
            x_min = p.x;
        else if (p.x > x_max)
            x_max = p.x;
        if (p.y < y_min)
            y_min = p.y;
        else if (p.y > y_max)
            y_max = p.y;
    }
    // If the points are very close together then fit a circle around.
    if (((x_max - x_min < small_radius * 2.0f) && (y_max - y_min < small_radius * 2.0f)) || cloud->points.size() < 3) {
        // Find center position by averaging.
        Eigen::Vector2f center(0.0f, 0.0f);
        for (auto& p : cloud->points) {
            center += Eigen::Vector2f(p.x, p.y);
        }
        center_out = center / cloud->points.size();

        // Find the average distance to center which will be the radius.
        radius = 0;
        for (auto& p : cloud->points) {
            radius += (center_out - Eigen::Vector2f(p.x, p.y)).norm();
        }
        radius /= cloud->points.size();
        if (radius < small_radius)
            radius = small_radius;
        //    std::cout << "SMALL RADIUS CIRCLE:     RADIUS: " <<radius <<"  CENTER: "<<center_out;
        return true;
    }

    // Helper.
    struct Circle_Model {
        Circle_Model() : r(0.0f), center(Eigen::Vector2f(0.0f, 0.0f)) {
        }
        float r;
        Eigen::Vector2f center;
    };
    int cl_size = cloud->points.size();

    // Preparation.
    std::vector<Circle_Model> circles;
    circles.resize(num_samples);
    std::vector<int> support_counter; // Points which have dist < support_dist.
    support_counter.resize(num_samples);
    std::vector<int> neg_support_counter; // Points which have support _dist <= dist < neg_support_dist.
    neg_support_counter.resize(num_samples);
    std::vector<float> support_counter_weighted; // Weighting: higher weight on points which are close to plane.
    support_counter_weighted.resize(num_samples);
    std::vector<float>
        neg_support_counter_weighted; // Weighting: higher weight on points which are close to support_dist.
    neg_support_counter_weighted.resize(num_samples);

    int best_idx = -1;

    // Multiple runs
    for (int i = 0; i < num_samples; i++) {

        // Sample three points.
        int rand_idx_1 = 0;
        int rand_idx_2 = 0;
        int rand_idx_3 = 0;

        rand_idx_1 = std::rand() % cl_size;
        do {
            rand_idx_2 = std::rand() % cl_size;
        } while (rand_idx_1 == rand_idx_2);
        do {
            rand_idx_3 = std::rand() % cl_size;
        } while (rand_idx_1 == rand_idx_3 || rand_idx_2 == rand_idx_3);


        Eigen::Vector2f p1(cloud->points[rand_idx_1].x, cloud->points[rand_idx_1].y);
        Eigen::Vector2f p2(cloud->points[rand_idx_2].x, cloud->points[rand_idx_2].y);
        Eigen::Vector2f p3(cloud->points[rand_idx_3].x, cloud->points[rand_idx_3].y);


        float yDelta_a = p2(1) - p1(1);
        float xDelta_a = p2(0) - p1(0);
        float yDelta_b = p3(1) - p2(1);
        float xDelta_b = p3(0) - p2(0);

        if (xDelta_a == 0 || xDelta_b == 0) {
            continue;
        }
        float aSlope = yDelta_a / xDelta_a;
        float bSlope = yDelta_b / xDelta_b;
        if (aSlope == bSlope) {
            continue;
        }
        Eigen::Vector2f center;
        center(0) = (aSlope * bSlope * (p1(1) - p3(1)) + bSlope * (p1(0) + p2(0)) - aSlope * (p2(0) + p3(0))) /
                    (2 * (bSlope - aSlope));
        center(1) = -1 * (center(0) - (p1(0) + p2(0)) / 2) / aSlope + (p1(1) + p2(1)) / 2;

        float r = (p1 - center).norm();

        // Set default values.
        support_counter[i] = 0;
        neg_support_counter[i] = 0;
        support_counter_weighted[i] = 0;
        neg_support_counter_weighted[i] = 0;

        // If the radius is not suitable then continue sampling.
        if (r > circle_max_radius) {
            continue;
        }

        // Check support.
        float neg_support_range_dist = neg_support_dist - support_dist;

        for (auto& p : cloud->points) {
            // Check dist to center.
            float dist = std::abs((center - Eigen::Vector2f(p.x, p.y)).norm() - r);
            if (dist < support_dist) {
                support_counter[i]++;
                support_counter_weighted[i] += 1.0001f - dist / support_dist;
            } else if (dist < neg_support_dist) {
                neg_support_counter[i]++;
                neg_support_counter_weighted[i] += 1.0f - 0.5f * dist / neg_support_range_dist;
            }
        }

        // If not enough support then ignore this circle estimation.
        if (support_counter[i] < std::round(min_circle_support_percentage * cl_size)) {
            support_counter[i] = 0;
            continue;
        }

        // Store params.
        if (r < small_radius)
            r = small_radius;
        circles[i].r = r;
        circles[i].center = center;

        // Check if circle is very hight quality -> early stop.
        if (support_counter[i] != 0) {
            float average_neg_score = 0;
            if (neg_support_counter[i] != 0)
                average_neg_score = -neg_support_counter_weighted[i] / (float)neg_support_counter[i];
            float average_pos_score = support_counter_weighted[i] / (float)support_counter[i];
            float circle_quality =
                ((float)neg_support_counter[i] * average_neg_score + (float)support_counter[i] * average_pos_score) /
                ((float)neg_support_counter[i] + (float)support_counter[i]);

            // If the circle quality is very high then stop the circle search.
            if (circle_quality > early_stop_circle_quality) {
                best_idx = i;
                //          std::cout <<"---Early Stop Circle Estimation---" <<std::endl;
                break;
            }
        }
    }

    // Find best circle
    if (best_idx == -1) {
        float best_circle_quality = min_circle_quality;
        for (int i = 0; i < num_samples; i++) {
            if (support_counter[i] != 0) {
                float average_neg_score = 0;
                if (neg_support_counter[i] != 0)
                    average_neg_score = -neg_support_counter_weighted[i] / (float)neg_support_counter[i];
                float average_pos_score = support_counter_weighted[i] / (float)support_counter[i];
                float circle_quality = ((float)neg_support_counter[i] * average_neg_score +
                                        (float)support_counter[i] * average_pos_score) /
                                       ((float)neg_support_counter[i] + (float)support_counter[i]);

                if (circle_quality > best_circle_quality) {
                    best_circle_quality = circle_quality;
                    best_idx = i;
                    //          std::cout <<"CIRCLE QUALITY: " << circle_quality <<std::endl;
                }
            }
        }
    }
    //  std::cout<<"---------"<<std::endl;

    // If no good circle was found then set negative radius.
    if (best_idx == -1) {
        radius = -1;
        return false;
    }

    // Check if the support covers a angle range which is representative.
    int bin_num = 16;
    std::vector<bool> angle_histogram(bin_num, false);
    Eigen::Vector2f center_best = circles[best_idx].center;
    float radius_best = circles[best_idx].r;
    for (auto& p : cloud->points) {
        // Check dist to center.
        Eigen::Vector2f p_eigen(p.x, p.y);
        float dist = std::abs((center_best - p_eigen).norm() - radius_best);
        if (dist < support_dist) {
            Eigen::Vector2f vec = p_eigen - center_best;
            uint bin_id = std::floor((std::atan2(vec(1), vec(0)) + M_PI) / (2 * M_PI) * bin_num);
            angle_histogram[bin_id] = true;
        }
    }
    uint coverage = std::count(angle_histogram.begin(), angle_histogram.end(), true);
    if (coverage < 3) {
        radius = -1;
        return false;
    } else {
        radius = circles[best_idx].r;
        center_out = circles[best_idx].center;
        return true;
    }


}



bool PrimitiveExtractor::run() {
    Time t("Feature extraction time consumption");
    if (cloudIn->points.size() == 0) {
        std::cout << "ERROR ----  NO POINTCLOUD SET!!!" << std::endl;
        return false; // No pointcloud set;
    }
    if (range[1] - range[0] <= 0 || range[3] - range[2] <= 0 || range[5] - range[4] <= 0) {
        std::cout << "ERROR ----  NO CORRECT RANGES SET!!!" << std::endl;
        return false; // No correct ranges set
    }
//    if (visual_tool_ == nullptr) {
//        visual_tool_.reset(new rviz_visual_tools::RvizVisualTools(cloudIn->header.frame_id, "/debugging_visu"));
//    }
//    visu_id_ = 0;

//    initColors();

    projectPointCloud();
    polesExtractionWithFullCloudWithPCA();
    estimatePolesModel();

    facadeExtraction();
    facadeSegmentationNew();
    estimateFacadesModel();

    groundExtractionPCA();
    estimateGroundModel();



    return true;
}

void PrimitiveExtractor::getPolesFinParam(std::vector<Cylinder_Fin_Param>& poles_param) {
    for (auto& pole : polesParam) {
        poles_param.push_back(pole);
    }
}
void PrimitiveExtractor::getFacadesFinParam(std::vector<Plane_Param>& facades_param) {
    for (auto& facade : facadesParam) {
        facades_param.push_back(facade);
    }
}
void PrimitiveExtractor::getGroundFinParam(std::vector<Plane_Param>& ground_param) {
    for (auto& ground : groundsParam) {
        ground_param.push_back(ground);
    }
}

void PrimitiveExtractor::getGroundCloud(Cloud::Ptr groundCloud_) {
//    std::cout << groundCloud->points.size() << std::endl;
    for (auto p : groundCloud->points) {
        Point point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.intensity = p.intensity;
        groundCloud_->points.push_back(point);
    }
    groundCloud_->width = 1;
    groundCloud_->height = groundCloud_->points.size();
}

void PrimitiveExtractor::getFacadeRoughCloud(Cloud::Ptr facadeCloud_) {
    for (auto p : facadeCloudRough->points) {
        Point point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.intensity = p.intensity;
        facadeCloud_->points.push_back(point);
    }
    facadeCloud_->width = 1;
    facadeCloud_->height = facadeCloud_->points.size();
}
void PrimitiveExtractor::getNonFacadeRoughCloud(Cloud::Ptr nonFacadeCloud_) {
    for (auto p : nonFacadeCloudRough->points) {
        Point point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.intensity = p.intensity;
        nonFacadeCloud_->points.push_back(point);
    }
    nonFacadeCloud_->width = 1;
    nonFacadeCloud_->height = nonFacadeCloud_->points.size();
}
void PrimitiveExtractor::getPoleRoughCloud(Cloud::Ptr poleCloud_) {
//    std::cout << poleCloudRough->points.size() << std::endl;
    for (auto p : poleCloudRough->points) {
        Point point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.intensity = p.intensity;
        poleCloud_->points.push_back(point);
    }
    poleCloud_->width = 1;
    poleCloud_->height = poleCloud_->points.size();
}
void PrimitiveExtractor::getSplitClouds(std::vector<Cloud::Ptr>& clouds) {
    if (zSplits.size() != 0) {
        clouds.resize(zSplits.size());
        for (size_t i = 0; i < zSplits.size(); i++) {
            clouds[i] = zSplits[i];
        }
    }
}
void PrimitiveExtractor::getBiggerPoleCloud(Cloud::Ptr biggerPole) {
    for(auto p : poleCloudGrow->points) {
        Point point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.intensity = p.intensity;
        biggerPole->push_back(point);
    }
    biggerPole->width = 1;
    biggerPole->height = biggerPole->points.size();
}

void PrimitiveExtractor::get3DPoleCandidateCloudsColored(std::vector<CloudColor::Ptr>& pole_candidates) {
    pole_candidates.clear();
    pole_candidates.resize(poleCandidates.size());
    std::cout << "Number of pole candidates: " << poleCandidates.size() << std::endl;
    for (size_t i = 0; i < poleCandidates.size(); i++) {
        pole_candidates[i].reset(new CloudColor());
        size_t num_points = poleCandidates[i]->points.size();
        pole_candidates[i]->points.resize(num_points);
        std::vector<uint8_t> color = colors[i % colors.size()];
        for (size_t p = 0; p < num_points; ++p) {
            pole_candidates[i]->points[p].x = poleCandidates[i]->points[p].x;
            pole_candidates[i]->points[p].y = poleCandidates[i]->points[p].y;
            pole_candidates[i]->points[p].z = poleCandidates[i]->points[p].z;
            pole_candidates[i]->points[p].r = color[0];
            pole_candidates[i]->points[p].g = color[1];
            pole_candidates[i]->points[p].b = color[2];
        }
    }
}
void PrimitiveExtractor::get3DFacadeCandidateCloudsColored(std::vector<CloudColor::Ptr>& facade_candidates) {
    facade_candidates.clear();
    facade_candidates.resize(facadeCandidates.size());
    std::cout << "Number of facades candidates: " << facadeCandidates.size() << std::endl;
    for (size_t i = 0; i < facadeCandidates.size(); i++) {
        facade_candidates[i].reset(new CloudColor());
        size_t num_points = facadeCandidates[i]->points.size();
        facade_candidates[i]->points.resize(num_points);
        std::vector<uint8_t> color = colors[i % colors.size()];
        for (size_t p = 0; p < num_points; ++p) {
            facade_candidates[i]->points[p].x = facadeCandidates[i]->points[p].x;
            facade_candidates[i]->points[p].y = facadeCandidates[i]->points[p].y;
            facade_candidates[i]->points[p].z = facadeCandidates[i]->points[p].z;
            facade_candidates[i]->points[p].r = color[0];
            facade_candidates[i]->points[p].g = color[1];
            facade_candidates[i]->points[p].b = color[2];
        }
    }
}
void PrimitiveExtractor::get3DGroundCandidateCloudsColored(std::vector<CloudColor::Ptr>& ground_candidates) {
    ground_candidates.clear();
    ground_candidates.resize(groundCandidates.size());
    std::cout << "Number of ground candidates: " << groundCandidates.size() << std::endl;
    for (size_t i = 0; i < groundCandidates.size(); i++) {
        ground_candidates[i].reset(new CloudColor());
        size_t num_points = groundCandidates[i]->points.size();
        ground_candidates[i]->points.resize(num_points);
        std::vector<uint8_t> color = colors[i % colors.size()];
        for (size_t p = 0; p < num_points; ++p) {
            ground_candidates[i]->points[p].x = groundCandidates[i]->points[p].x;
            ground_candidates[i]->points[p].y = groundCandidates[i]->points[p].y;
            ground_candidates[i]->points[p].z = groundCandidates[i]->points[p].z;
            ground_candidates[i]->points[p].r = color[0];
            ground_candidates[i]->points[p].g = color[1];
            ground_candidates[i]->points[p].b = color[2];
        }
    }
}
void PrimitiveExtractor::get3DFacadeCandidateClouds(Cloud::Ptr& facade_candidates) {
    facade_candidates->clear();
    for (int i = 0; i < facadesParam.size(); ++i) {
        Eigen::Vector3f tl = facadesParam[i].edge_poly[0];
        Eigen::Vector3f tr = facadesParam[i].edge_poly[1];
        Eigen::Vector3f dr = facadesParam[i].edge_poly[2];
        Eigen::Vector3f dl = facadesParam[i].edge_poly[3];
        Point pTl;
        pTl.x = tl[0];
        pTl.y = tl[1];
        pTl.z = tl[2];
        pTl.intensity = 1000;
        Point pTr;
        pTr.x = tr[0];
        pTr.y = tr[1];
        pTr.z = tr[2];
        pTr.intensity = 1000;
        Point pDr;
        pDr.x = dr[0];
        pDr.y = dr[1];
        pDr.z = dr[2];
        pDr.intensity = 1000;
        Point pDl;
        pDl.x = dl[0];
        pDl.y = dl[1];
        pDl.z = dl[2];
        pDl.intensity = 1000;

        Cloud::Ptr cloud(new Cloud());
        createTheoryLine(tl, tr, cloud);
        createTheoryLine(tr, dr, cloud);
        createTheoryLine(dr, dl, cloud);
        createTheoryLine(dl, tl, cloud);
        *facade_candidates += *cloud;
    }
}
void PrimitiveExtractor::get3DPoleCandidateClouds(Cloud::Ptr& pole_candidates) {
    pole_candidates->clear();
    for (int i = 0; i < polesParam.size(); ++i) {
        Eigen::Vector3f topPoint = polesParam[i].center_line.p_top;
        Eigen::Vector3f downPoint = polesParam[i].center_line.p_bottom;
        float R = polesParam[i].radius;

        Cloud::Ptr cloud(new Cloud());
        createTheoryCylinder(downPoint, topPoint, R, cloud);
        *pole_candidates += *cloud;
    }
}

void PrimitiveExtractor::get_3D_clusters(std::vector<CloudColor::Ptr>& clouds) {
    clouds.clear();
    clouds.resize(clusters_3D_.size());

    int i = 0;
    for (auto& ids_vector : clusters_3D_) {
        CloudColor::Ptr cloud(new CloudColor());
        std::vector<uint8_t> color = colors[i % colors.size()];
        for (auto& ids : ids_vector) {
            for (auto& p : pointClusters[ids.first][ids.second].cloud->points) {
                PointColor point;
                point.x = p.x;
                point.y = p.y;
                point.z = p.z;
                point.r = color[0];
                point.g = color[1];
                point.b = color[2];

                cloud->points.push_back(point);
            }
        }
        clouds[i] = cloud;
        ++i;
    }
}
void PrimitiveExtractor::testNonGround(std::vector<std::vector<int>>& non_ground_indices) {
    non_ground_indices.resize(N_SCAN);
    for (int i = 0; i < N_SCAN; ++i) {
        non_ground_indices[i].resize(0);
    }
    {
//        Time t("ground segmentation");
        for (int i = 0; i < N_SCAN; ++i) {
            for (int j = 0; j < Horizon_SCAN; ++j) {
                int index = i + j * N_SCAN;
                if (groundMat.at<int8_t>(i, j) == 1) {
                    continue;
                }
                non_ground_indices[i].push_back(index);
            }
        }
    }
//    Cloud::Ptr cloud(new Cloud);
    // Init black images for each layer.
    int img_pixels_per_meter = 1;
    int img_height = std::ceil((range[1] - range[0]) * img_pixels_per_meter) + 1;
    int img_width = std::ceil((range[3] - range[2]) * img_pixels_per_meter) + 1;
    std::vector<cv::Mat> layer_projection;
    std::vector<cv::Mat> layer_projection_labeled;
    int num_layers = N_SCAN;
    for (int i = 0; i < num_layers; ++i) {
        layer_projection.push_back(cv::Mat::zeros(img_height, img_width, CV_8UC1));
        layer_projection_labeled.push_back(cv::Mat::zeros(img_height, img_width, CV_32S));
    }
    std::cout << "size of layer projection: " << layer_projection.size() << std::endl;
    for (int i = 0; i < N_SCAN; ++i) {

        for (int j = 0; j < Horizon_SCAN; ++j) {
            int id = i + j * N_SCAN;
            Point p = fullCloud->points[id];
            if (poleCandidatesMat.at<int8_t>(i, j) != 1 || p.intensity == -1) {
                continue;
            }

            int pos_in_img_row = std::round((p.x - range[0]) * img_pixels_per_meter);
            int pos_in_img_col = std::round((p.y - range[2]) * img_pixels_per_meter);
            if (pos_in_img_row < 0 || pos_in_img_row >= img_height || pos_in_img_col < 0 ||
                pos_in_img_col >= img_width) {
                std::cout << "WRONG IMAGE IDX  "
                          << "ROW " << pos_in_img_row << " (max " << img_height << ")   "
                          << "COL " << pos_in_img_col << " (max " << img_width << ")" << std::endl;
                continue;
            }
            // Add point to image.
            if (layer_projection[i].ptr<uchar>(pos_in_img_row)[pos_in_img_col] == 0) // Mat::at(row,col).
            {
                layer_projection[i].ptr<uchar>(pos_in_img_row)[pos_in_img_col] = 255;
            }
        }
    }
    std::cout << "size of layer projection: " << layer_projection.size() << std::endl;
    for (int i = 0; i < layer_projection.size(); ++i) {
        cv::imwrite("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/images/test/" + std::to_string(i) + ".png",
                    layer_projection[i]);
    }

}

void PrimitiveExtractor::createTheoryCircle(Eigen::Vector3f& normal, Eigen::Vector3f& center, float R, Cloud::Ptr& cloud, int color) {

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
void PrimitiveExtractor::createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point,
                                          CloudColor::Ptr& cloud, int length) {
    Eigen::Vector3f lineDir = end_point - start_point;
    int index = 0;
    std::vector<Eigen::Vector3f> pointVector;
    Eigen::Vector3f tmpPoint;
    float step = lineDir.norm() / 50;
    while (index < length) {
        tmpPoint = start_point + step * index * lineDir / lineDir.norm();
        pointVector.push_back(tmpPoint);
        index++;
    }
    for (int i = 0; i < pointVector.size(); ++i) {
        PointColor p;
        p.x = pointVector[i][0];
        p.y = pointVector[i][1];
        p.z = pointVector[i][2];
        p.r = 0;
        p.g = 0;
        p.b = 0;
        cloud->points.push_back(p);
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();
}
void PrimitiveExtractor::createTheoryCylinder(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, const float R,
                                              Cloud::Ptr& cloud) {
    Eigen::Vector3f lineDir = end_point - start_point;
    int index = 0;
    Eigen::Vector3f axisPoint;
    float step = lineDir.norm() / 90;
    // Axis
    Point center;
    while (index < 5) {
        axisPoint = start_point + step * index * lineDir / lineDir.norm();
        center.x = axisPoint[0];
        center.y = axisPoint[1];
        center.z = axisPoint[2];
        center.intensity = 1000;
        createTheoryCircle(lineDir, axisPoint, R, cloud, 50);
        cloud->points.push_back(center);
        index++;
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();

}
void PrimitiveExtractor::createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point,
                                          Cloud::Ptr& cloud) {
    Eigen::Vector3f lineDir = end_point - start_point;
    int index = 0;
    std::vector<Eigen::Vector3f> pointVector;
    Eigen::Vector3f tmpPoint;
    int stepLength = 1;
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
        p.intensity = 1000;
        cloud->points.push_back(p);
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();
}
void PrimitiveExtractor::createTheoryLine(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point,
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
void PrimitiveExtractor::createTheoryCylinder(Eigen::Vector3f& start_point, Eigen::Vector3f& end_point, const float R,
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








} // end of namespace primitivesExtraction