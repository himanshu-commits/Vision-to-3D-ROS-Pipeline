cmake_minimum_required(VERSION 3.0.2)
project(edge_detection)
catkin_python_setup()

## 1) Find packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  message_generation
  image_transport
  cv_bridge
)

## 2) Declare messages/services BEFORE generate_messages
add_message_files(
  FILES
  EdgePixel.msg
)

add_service_files(
  FILES
  DetectEdgesFromDir.srv
)

## 3) Generate message/service code
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## 4) Export package AFTER generate_messages
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs visualization_msgs
                 message_runtime image_transport cv_bridge
)

## 5) Includes
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 6) Install Python nodes (use your actual filenames)
# Make sure these files exist and are executable (chmod +x)
catkin_install_python(PROGRAMS
  scripts/edge_ros_node.py       
  scripts/edge_dir_service.py 
  scripts/edge_marker_node.py     
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

