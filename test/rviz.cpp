//
// Created by zhao on 16.09.22.
//
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
// %EndTag(INCLUDES)%

// %Tag(INIT)%
int main( int argc, char** argv )
{
    ros::init(argc, argv, "basic_shapes");
    ros::NodeHandle n;
    ros::Rate r(1);
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);
// %EndTag(INIT)%

    // Set our initial shape type to be a cube
// %Tag(SHAPE_INIT)%
    uint32_t shape = visualization_msgs::Marker::CUBE;

// %EndTag(SHAPE_INIT)%

// %Tag(MARKER_INIT)%
    while (ros::ok())
    {
        visualization_msgs::Marker marker;

        visualization_msgs::Marker plane_msg;
        plane_msg.header.frame_id = "/my_frame";
        plane_msg.header.stamp = ros::Time::now();
        plane_msg.id = 0;
        plane_msg.type = visualization_msgs::Marker::TRIANGLE_LIST;

        plane_msg.action = visualization_msgs::Marker::ADD;

        plane_msg.scale.x = plane_msg.scale.y = plane_msg.scale.z = 1.0;
        plane_msg.pose.position.x = 0.0;
        plane_msg.pose.position.y = 0.0;
        plane_msg.pose.position.z = 0.0;
        plane_msg.pose.orientation.x = 0.0;
        plane_msg.pose.orientation.y = 0.0;
        plane_msg.pose.orientation.z = 0.0;
        plane_msg.pose.orientation.w = 1.0;
        plane_msg.color.a = 1.0;

        int id = 0;

        // Set the frame ID and timestamp.  See the TF tutorials for information on these.
        marker.header.frame_id = "/my_frame";
        marker.header.stamp = ros::Time::now();
// %EndTag(MARKER_INIT)%

        // Set the namespace and id for this marker.  This serves to create a unique ID
        // Any marker sent with the same namespace and id will overwrite the old one
// %Tag(NS_ID)%
        marker.ns = "basic_shapes";
        marker.id = 0;
// %EndTag(NS_ID)%

        // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
// %Tag(TYPE)%
        marker.type = shape;
// %EndTag(TYPE)%

        // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
// %Tag(ACTION)%
        marker.action = visualization_msgs::Marker::ADD;
// %EndTag(ACTION)%

        // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
// %Tag(POSE)%
        marker.pose.position.x = 0;
        marker.pose.position.y = 0;
        marker.pose.position.z = 0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
// %EndTag(POSE)%

        // Set the scale of the marker -- 1x1x1 here means 1m on a side
// %Tag(SCALE)%
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
// %EndTag(SCALE)%

        // Set the color -- be sure to set alpha to something non-zero!
// %Tag(COLOR)%
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0;
// %EndTag(COLOR)%

// %Tag(LIFETIME)%
        marker.lifetime = ros::Duration();
// %EndTag(LIFETIME)%

        // Publish the marker
// %Tag(PUBLISH)%
        while (marker_pub.getNumSubscribers() < 1)
        {
            if (!ros::ok())
            {
                return 0;
            }
            ROS_WARN_ONCE("Please create a subscriber to the marker");
            sleep(1);
        }
        marker_pub.publish(marker);
// %EndTag(PUBLISH)%
// %EndTag(CYCLE_SHAPES)%

// %Tag(SLEEP_END)%
        r.sleep();
    }
// %EndTag(SLEEP_END)%
}
// %EndTag(FULLTEXT)%