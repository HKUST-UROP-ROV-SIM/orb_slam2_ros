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

#ifndef H264STEREONODE_HPP_
#define H264STEREONODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "image_geometry/pinhole_camera_model.h"
#include "h264_msgs/msg/packet.hpp"
#include "stereo_decoder/stereo_decoder.hpp"

#include "Node.hpp"

class H264StereoNode : public Node
{
public:
  H264StereoNode(
    const std::string & node_name,
    const rclcpp::NodeOptions & node_options);

  void init();

private:
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_camera_info_sub_;

  rclcpp::Subscription<h264_msgs::msg::Packet>::SharedPtr left_packet_sub_;
  rclcpp::Subscription<h264_msgs::msg::Packet>::SharedPtr right_packet_sub_;

  image_geometry::PinholeCameraModel left_model_;
  image_geometry::PinholeCameraModel right_model_;

  stereo_decoder::StereoDecoder decoder_;

  std::thread slam_thread_;
};

#endif  // H264STEREONODE_HPP_
