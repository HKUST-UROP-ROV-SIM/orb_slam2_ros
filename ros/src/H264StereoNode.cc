/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "H264StereoNode.hpp"

#undef RUN_PERF
#ifdef RUN_PERF
#define START_PERF()\
auto __start__ = std::chrono::high_resolution_clock::now();

#define STOP_PERF(msg)\
auto __stop__ = std::chrono::high_resolution_clock::now();\
std::cout << msg << " " << std::chrono::duration_cast<std::chrono::microseconds>(__stop__ - __start__).count()\
  << " microseconds" << std::endl;
#else
#define START_PERF()
#define STOP_PERF(msg)
#endif

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto options = rclcpp::NodeOptions();

  // The base class (ORB SLAM Node, not rclcpp::Node) prefixes all topics with node_name. This doesn't match typical
  // ROS2 behavior, so it's a bit confusing. To facilitate re-use of launch and param files use the same name in
  // StereoNode and H264StereoNode.
  auto node = std::make_shared<H264StereoNode>("orb_slam2_stereo_node", options);

  node->init();

  rclcpp::spin(node);

  rclcpp::shutdown();

  return 0;
}

H264StereoNode::H264StereoNode(
  const std::string & node_name,
  const rclcpp::NodeOptions & node_options)
  : Node(node_name, node_options)
{
  declare_parameter("period_ms", rclcpp::ParameterValue(1000));
}

void H264StereoNode::init()
{
  (void) left_camera_info_sub_;
  (void) right_camera_info_sub_;
  (void) left_packet_sub_;
  (void) right_packet_sub_;
  (void) slam_thread_;

  Node::init(ORB_SLAM2::System::STEREO);

  rclcpp::QoS qos(10);
  if (subscribe_best_effort_param_) {
    qos.best_effort();
  } else {
    qos.reliable();
  }

  // Note that these topics are _not_ prefixed with node_name
  left_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>("stereo/left/camera_info", qos,
    [this](sensor_msgs::msg::CameraInfo::SharedPtr msg) // NOLINT
    {
      if (!left_model_.initialized()) {
        RCLCPP_INFO(get_logger(), "init left camera model"); // NOLINT
        left_model_.fromCameraInfo(msg);
      }
    });

  right_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>("stereo/right/camera_info", qos,
    [this](sensor_msgs::msg::CameraInfo::SharedPtr msg) // NOLINT
    {
      if (!isInitialized()) {
        RCLCPP_INFO(get_logger(), "init ORB params"); // NOLINT
        LoadOrbParameters(msg);
      }

      if (!right_model_.initialized()) {
        RCLCPP_INFO(get_logger(), "init right camera model"); // NOLINT
        right_model_.fromCameraInfo(msg);
      }
    });

  left_packet_sub_ = create_subscription<h264_msgs::msg::Packet>("stereo/left/image_raw/h264", qos,
    [this](h264_msgs::msg::Packet::SharedPtr msg) { decoder_.push_left(msg); }); // NOLINT

  right_packet_sub_ = create_subscription<h264_msgs::msg::Packet>("stereo/right/image_raw/h264", qos,
    [this](h264_msgs::msg::Packet::SharedPtr msg) { decoder_.push_right(msg); }); // NOLINT

  // Run the SLAM in it's own thread
  slam_thread_ = std::thread([this]()
  {
    // The cameras run at 20fps, the ORB SLAM algorithm runs at ~2.5fps on my desktop.
    // Provide a mechanism to slow down the SLAM algorithm to reduce CPU load.
    int period_ms;
    get_parameter("period_ms", period_ms);
    auto target_period = rclcpp::Duration(period_ms * 1000000);
    rclcpp::Time previous_time = now();

    while (true) {
      if (!isInitialized() || !left_model_.initialized() || !right_model_.initialized()) {
        RCLCPP_WARN(get_logger(), "missing camera info, sleep for 1s");  // NOLINT
        std::this_thread::sleep_for(1s);
        continue;
      }

      auto sleep_time = target_period - (now() - previous_time);
      if (sleep_time.nanoseconds() > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_time.nanoseconds()));
      }
      previous_time = now();

      // The stereo decoder will decode all H.264 packets on both streams, but only keep the last stereo pair.
      std::unique_ptr<sensor_msgs::msg::Image> rawImageLeftUnique;
      std::unique_ptr<sensor_msgs::msg::Image> rawImageRightUnique;
      if (!decoder_.pop_wait(rawImageLeftUnique, rawImageRightUnique)) {
        RCLCPP_INFO(get_logger(), "decoder shutting down");  // NOLINT
        break;
      }

      std::shared_ptr<sensor_msgs::msg::Image> msgLeft = std::move(rawImageLeftUnique);
      std::shared_ptr<sensor_msgs::msg::Image> msgRight = std::move(rawImageRightUnique);
      cv::Mat rawLeft, rawRight, rectLeft, rectRight;

      START_PERF()

      // This should pull out the cv::Mat w/o copying
      try {
        rawLeft = cv_bridge::toCvShare(msgLeft)->image;
      }
      catch (cv_bridge::Exception & e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());  // NOLINT
        continue;
      }

      try {
        rawRight = cv_bridge::toCvShare(msgRight)->image;
      }
      catch (cv_bridge::Exception & e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());  // NOLINT
        continue;
      }

      // Rectify, this will copy
      left_model_.rectifyImage(rawLeft, rectLeft);
      right_model_.rectifyImage(rawRight, rectRight);

      current_frame_time_ = msgLeft->header.stamp;
      orb_slam_->TrackStereo(rectLeft, rectRight, current_frame_time_.seconds());

      Update();

      STOP_PERF("SLAM time")
    }
  });

  RCLCPP_INFO(get_logger(), "SLAM created, but not initialized");
}
