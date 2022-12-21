//
// Created by zhao on 03.05.22.
//

#ifndef SRC_CURBDETECTION_H
#define SRC_CURBDETECTION_H
namespace llo {


using namespace std;


bool comp_up(const pcl::PointXYZ &a, const pcl::PointXYZ &b)
{
    return a.y < b.y;
}
// 降序。（针对y小于0的点）
bool comp_down(const pcl::PointXYZ &a, const pcl::PointXYZ &b)
{
    return a.y > b.y;
}


class curbDetector
// 此类用于检测curb。输入点云，返回检测到的curb点组成的点云。主执行函数为detector。
{
public:
    curbDetector(){}

    pcl::PointCloud<pcl::PointXYZ> detector(const std::vector<pcl::PointCloud<pcl::PointXYZ> > input)
    {
        pc_in = input;
        pcl::PointCloud<pcl::PointXYZ> look_test;

        for (int i = 0; i < 10; i++)
            // 对于每一环进行处理、检测。由于之前我们这里取了64线lidar的10线，所以这里循环次数为10.
        {
            pcl::PointCloud<pcl::PointXYZ> pointsInTheRing = pc_in[i]; // 储存此线上的点。
            pcl::PointCloud<pcl::PointXYZ> pc_left; // 储存y大于0的点（左侧点）。
            pcl::PointCloud<pcl::PointXYZ> pc_right; // 储存y小于0的点（右侧点）。

            pcl::PointCloud<pcl::PointXYZ> cleaned_pc_left; //储存经过算法1处理过后的左侧点。
            pcl::PointCloud<pcl::PointXYZ> cleaned_pc_right; //储存经过算法1处理过后的右侧点。

            pcl::PointXYZ point; // 点的一个载体。
            size_t numOfPointsInTheRing = pointsInTheRing.size(); // 此线上的点的数量。

            for (int idx = 0; idx < numOfPointsInTheRing; idx++)
                // 分开左、右侧点，分别储存到对应的点云。
            {
                point = pointsInTheRing[idx];
                if (point.y >= 0)
                {pc_left.push_back(point);}
                else
                {pc_right.push_back(point);}
            }

            // 排序。（按绝对值升序）
            sort(pc_left.begin(), pc_left.end(), comp_up);
            sort(pc_right.begin(), pc_right.end(), comp_down);

            // cleaned_pc_left = piao_rou(pc_left);
            // cleaned_pc_right = piao_rou(pc_right);

            // 滑动检测curb点。（对应算法2）
            slideForGettingPoints(pc_left, true);
            slideForGettingPoints(pc_right, false);

            // look_test += (cleaned_pc_left + cleaned_pc_right);
        }
        return curb_left + curb_right;

        // 注意检测结束执行reset，将此类参数置零。(zanshibuyong)
    }

    pcl::PointCloud<pcl::PointXYZ> piao_rou(const pcl::PointCloud<pcl::PointXYZ> hairs)
    {
        const float xy_treth = 0.4;
        const float z_treth = 0.1;
        const int n_N = 10;

        pcl::PointXYZ p0;
        pcl::PointXYZ p1;
        bool jump_flag = false;

        pcl::PointCloud<pcl::PointXYZ> cleanedHairs;
        pcl::PointCloud<pcl::PointXYZ> mayDropedHair;
        size_t numOfTheHairs = hairs.size();
        // cout << endl;

        for (int i = 1; i < numOfTheHairs; i++)
        {
            p0 = hairs[i-1];
            p1 = hairs[i];
            if (p0.y == p1.y)
            {continue;}
            // cout << p1.y << " ";
            float p_dist;
            float z_high;

            p_dist = sqrt(((p0.x - p1.x) * (p0.x - p1.x)) + ((p0.y - p1.y) * (p0.y - p1.y)));
            z_high = fabs(p0.z - p1.z);

            // cout << p_dist << " " << z_high << endl;

            if ((p_dist <= xy_treth) && (z_high <= z_treth))
            {
                if (jump_flag)
                {
                    mayDropedHair.push_back(p1);
                    if (mayDropedHair.size() >= n_N)
                    {
                        cleanedHairs = cleanedHairs + mayDropedHair;
                        jump_flag = false;
                        mayDropedHair.clear();
                    }
                }
                else
                {
                    cleanedHairs.push_back(p1);
                }
            }
            else
            {
                jump_flag = true;
                mayDropedHair.clear();
                mayDropedHair.push_back(p1);
            }
        }
        return cleanedHairs;
    }

    int slideForGettingPoints(pcl::PointCloud<pcl::PointXYZ> points, bool isLeftLine)
    {
        int w_0 = 10;
        int w_d = 30;
        int i = 0;

        // some important parameters influence the final performance.
        float xy_thresh = 0.1;
        float z_thresh = 0.06;

        int points_num = points.size();

        while((i + w_d) < points_num)
        {
            float z_max = points[i].z;
            float z_min = points[i].z;

            int idx_ = 0;
            float z_dis = 0;

            for (int i_ = 0; i_ < w_d; i_++)
            {
                float dis = fabs(points[i+i_].z - points[i+i_+1].z);
                if (dis > z_dis) {z_dis = dis; idx_ = i+i_;}
                if (points[i+i_].z < z_min){z_min = points[i+i_].z;}
                if (points[i+i_].z > z_max){z_max = points[i+i_].z;}
            }

            if (fabs(z_max - z_min) >= z_thresh)
            {
                for (int i_ = 0; i_ < (w_d - 1); i_++)
                {
                    float p_dist = sqrt(((points[i + i_].y - points[i + 1 + i_].y) * (points[i + i_].y - points[i + 1 + i_].y))
                                        + ((points[i + i_].x - points[i + 1 + i_].x) *(points[i + i_].x - points[i + 1 + i_].x)));
                    if (p_dist >= xy_thresh)
                    {
                        if (isLeftLine) {curb_left.push_back(points[i_ + i]);return 0;}
                        else {curb_right.push_back(points[i_ + i]);return 0;}
                    }
                }
                if (isLeftLine) {curb_left.push_back(points[idx_]);return 0;}
                else {curb_right.push_back(points[idx_]);return 0;}
            }
            i += w_0;
        }
    }

private:
    std::vector<pcl::PointCloud<pcl::PointXYZ> > pc_in;
    pcl::PointCloud<pcl::PointXYZ> curb_left;
    pcl::PointCloud<pcl::PointXYZ> curb_right;
};

}//end of namespace llo
#endif // SRC_CURBDETECTION_H
