cmake_minimum_required(VERSION 2.8.3)
project(mlda_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  visualization_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

add_message_files(
  FILES
  PointList.msg
  ImageList.msg
  StringList.msg
)

add_service_files(
  FILES
  mlda_data.srv
  mlda_human_input.srv
  mlda_image.srv
  mlda_learn.srv
  mlda_word.srv
  mlda_joint_load.srv
  mlda_hololens_coordinate.srv
  mlda_hololens_category_sample.srv
  SendImageMLDA.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  visualization_msgs
  geometry_msgs
  sensor_msgs
)
