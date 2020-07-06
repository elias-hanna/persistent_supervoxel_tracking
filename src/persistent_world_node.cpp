//cpp stdlib includes
#include <random>
#include <cmath>
// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <robot_arm_controllers/ActAtPosition.h>
#include <tf/transform_listener.h>
#include <gazebo_msgs/LinkStates.h>
// PCL basic includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <boost/make_shared.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/conversions.h>
// Local libs includes
#include "../libs/papon/supervoxel/sequential_supervoxel_clustering.h"
#include "../libs/pairwise_segmentation/pairwise_segmentation.h"

// Types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointXYZN;
typedef pcl::PointCloud<PointXYZN> PointCloudN;
typedef pcl::Normal Normal;
typedef pcl::PointCloud<Normal> NormalCloud;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::PointXYZRGBL PointRGBLT;
typedef pcl::PointCloud<PointRGBLT> PointRGBLCloudT;

static bool manual_mode = false;
static bool show_prev = false;
static bool show_curr = false;
static bool show_labels = false;
static bool show_segmentation = false;
static bool show_disappeared = false;
static bool show_transforms = false;
static bool show_unlabeled = false;
static bool show_numbers = false;
static uint64_t frame_count = 0;
static pcl::GlasbeyLUT colors;

PointCloudT::Ptr current_cloud(new PointCloudT);
bool moving_state = false;
std::vector<geometry_msgs::Pose> objects_poses;
std::vector<std::string> objects_names;

// Callback function to get the current pointcloud published by the kinect2 camera
void
kinectCallback(const sensor_msgs::PointCloud2ConstPtr& ros_pcl2)
{
  //  sensor_msgs::convertPointCloud2ToPointCloud (*pcl2, current_pcl);
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*ros_pcl2,pcl_pc2);
  //  PointCloudT::Ptr temp_cloud(new PointCloudT);
  current_cloud.reset (new PointCloudT);
  pcl::fromPCLPointCloud2(pcl_pc2,*current_cloud);
}

// Callback function to get the current moving state of the robot
// True if the robot is moving, false otherwise
void
movingStateCallback(const std_msgs::Bool state)
{
  moving_state = state.data;
}

// Callback function to get the current moving state of the robot
// True if the robot is moving, false otherwise
void
linkStatesCallback(const gazebo_msgs::LinkStates link_states)
{
  // Here I should get the poses of each object that interest me...
  // Then store their pose in some variable and then randomly chose over
  // these pose to interact with !
  std::vector<std::string> names(link_states.name);
  std::vector<geometry_msgs::Pose> poses(link_states.pose);

  std::string s = "scott>=tiger";
  std::string delimiter = "::";
  // We need to remove: pr2:: ; ground_plane:: ; full_base_exp:: ; table::
  //  std::cout << "#############################################\n";

  auto it_names = names.begin();
  auto it_poses = poses.begin();
  while (it_names != names.end())
  {
    // This only keeps the part before the first "::" in the string
    std::string token = it_names->substr(0, it_names->find(delimiter));
    // Remove the occurrence if it matches one of the env link
    if(token=="pr2" || token=="ground_plane" || token=="full_base_exp" || token=="table")
    {
      it_names = names.erase(it_names);
      it_poses = poses.erase(it_poses);
    }
    else
    {
      ++it_names; ++it_poses;
    }
  }
  objects_poses = poses;
  objects_names = names;
  //  for (int i=0; i<names.size(); ++i)
  //  {
  //    std::cout << "nom: " << names[i]
  //              << "\npose: " << poses[i] << "\n";
  //  }
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> &input)
{
  for (const auto& i: input)
    os << i << ";";
  return os;
}

void
keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event)
{
  //  std::cout << event.getKeySym() << "\n";
  if (event.getKeySym() == "KP_1" && event.keyDown())
  { show_prev = !show_prev; }
  if (event.getKeySym() == "KP_2" && event.keyDown())
  { show_curr = !show_curr; }
  if (event.getKeySym () == "KP_3" && event.keyDown ())
  { show_labels = !show_labels; }
  if (event.getKeySym () == "KP_4" && event.keyDown ())
  { show_transforms = !show_transforms; }
  if (event.getKeySym () == "KP_5" && event.keyDown ())
  { show_unlabeled = !show_unlabeled; }
  if (event.getKeySym () == "KP_6" && event.keyDown ())
  { show_disappeared = !show_disappeared; }
  if (event.getKeySym () == "KP_7" && event.keyDown ())
  { show_segmentation = !show_segmentation; }
  if (event.getKeySym () == "KP_8" && event.keyDown ())
  { show_numbers = !show_numbers; }
  if (event.getKeySym () == "Return" && event.keyDown ())
  { manual_mode = false; }
  if (event.getKeySym () == "Escape" && event.keyDown ())
  { exit(0); }
}

void
updateView (const pcl::visualization::PCLVisualizer::Ptr viewer,
            const PointCloudT::Ptr cloud,
            const pcl::SequentialSVClustering<PointT>& super,
            const pcl::SequentialSVClustering<PointT>::SequentialSVMapT supervoxel_clusters,
            const PairwiseSegmentation pw_seg)
{
  // Get the voxel centroid cloud
  PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
  // Get the colored voxel centroid cloud
  PointCloudT::Ptr colored_voxel_centroid_cloud = super.getColoredVoxelCloud ();
  // Get the labeled voxel cloud
  PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
  // Get the labeled voxel cloud
  PointRGBLCloudT::Ptr
      rgb_labeled_voxel_cloud = super.getLabeledRGBVoxelCloud ();
  // Get the voxel normal cloud
  NormalCloud::Ptr
      voxel_normal_cloud = super.getVoxelNormalCloud ();
  // Get the unlabeled voxel cloud
  PointCloudT::Ptr
      un_voxel_centroid_cloud = super.getUnlabeledVoxelCentroidCloud ();
  // Colors used for object colorization
  std::vector<uint32_t> label_colors = super.getLabelColors ();
  // Show the labeled observed pointcloud
  if (show_labels && !show_segmentation)
  {
    if (!viewer->updatePointCloud (labeled_voxel_cloud, "displayed cloud"))
    { viewer->addPointCloud (labeled_voxel_cloud, "displayed cloud"); }
    if (show_numbers)
    {
      for (const auto& cluster: supervoxel_clusters)
      {
        viewer->addText3D (std::to_string (cluster.first),
                           cluster.second->centroid_, 0.01, 1.0, 1.0, 1.0,
                           "text"+std::to_string (cluster.first));
      }
    }
  }
  // Show the observed pointcloud
  else if (!show_segmentation)
  {
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgba(cloud);
    if (!viewer->updatePointCloud<PointT> (cloud, rgba, "displayed cloud"))
    { viewer->addPointCloud<PointT> (cloud, rgba, "displayed cloud"); }
  }
  // Show the segmentation
  else
  {
    for (const auto& sv: supervoxel_clusters)
    {
      size_t obj_nb = 0;
      // Find to which object this sv corresponds
      for (const auto& obj: pw_seg.getCurrentSegmentation ())
      {
        std::vector<uint32_t>::const_iterator vec_it =
            std::find (obj.second.begin (), obj.second.end (), sv.first);
        if (vec_it != obj.second.end ())
        { obj_nb = obj.first; break; }
      }
      //      uint32_t color = label_colors[obj_nb];
      //      double r = static_cast<double> (static_cast<uint8_t> (color >> 16));
      //      double g = static_cast<double> (static_cast<uint8_t> (color >> 8));
      //      double b = static_cast<double> (static_cast<uint8_t> (color));
      //      pcl::visualization::PointCloudColorHandlerCustom<PointT>
      //          rgb (sv.second->voxels_, r, g, b);
      pcl::visualization::PointCloudColorHandlerCustom<PointT>
          rgb (sv.second->voxels_,
               colors.at(obj_nb).r, colors.at(obj_nb).g, colors.at(obj_nb).b);
      viewer->addPointCloud<PointT> (sv.second->voxels_, rgb,
                                     "sv_"+std::to_string (sv.first));
    }
  }
  if (show_unlabeled)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        rgb (un_voxel_centroid_cloud, 0, 255, 0); //This is blue
    if (!viewer->updatePointCloud<PointT> (cloud, rgb, "unlabeled cloud"))
    {
      viewer->addPointCloud<PointT>
          (un_voxel_centroid_cloud, rgb, "unlabeled cloud");
    }
    //    viewer->setPointCloudRenderingProperties
    //(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "unlabeled cloud");
  }
  if (!show_segmentation)
  {
    viewer->setPointCloudRenderingProperties
        (pcl::visualization::PCL_VISUALIZER_OPACITY,1., "displayed cloud");
  }
  viewer->addText ("press NUM1 to show/hide previous keypoints", 0, 0,
                   "t_prev");
  viewer->addText ("press NUM2 to show/hide current keypoints", 0, 10,
                   "t_curr");
  viewer->addText ("press NUM3 to switch between "
                   "labelized/normal view of cloud", 0, 20, "t_cloud");
  viewer->addText ("press NUM4 to show/hide the "
                   "computed transforms", 0, 30, "t_trans");
  viewer->addText ("press NUM5 to show/hide the "
                   "unlabeled voxel centroid cloud", 0, 40, "t_unlabeled");
  viewer->addText ("press NUM6 to show/hide points at the center "
                   "of disappeared/occluded supervoxels", 0, 50, "t_dis");
  viewer->addText ("press NUM7 to show the current segmentation "
                   "w/ pairwise approach", 0, 60, "t_seg");
  viewer->addText ("press NUM8 to show labels over each supervoxel of "
                   "the over-segmented scene", 0, 70, "t_lab");
  if (manual_mode)
  { viewer->addText ("press Enter to go to next frame", 700, 0, 20, 1., 1., 1.,
                     "t_manual"); }
  std::vector<int>
      current_keypoints_indices = super.current_keypoints_indices_;
  std::vector<int>
      previous_keypoints_indices = super.previous_keypoints_indices_;
  if (frame_count)
  {
    // Show current keypoint cloud
    if (show_curr)
    {
      for (const size_t& idx: super.current_keypoints_indices_)
      {
        if (!viewer->updateSphere((*voxel_centroid_cloud)[idx],
                                  0.002, 125, 125, 0,
                                  "current keypoint " + std::to_string (idx)))
        {
          viewer->addSphere((*voxel_centroid_cloud)[idx],
                            0.002, 125, 125, 0,
                            "current keypoint " + std::to_string (idx)); }
      }
    }
    // Show previous keypoint cloud
    if (show_prev)
    {
      // Get the previous voxel centroid cloud
      PointCloudT::Ptr
          prev_voxel_centroid_cloud = super.getPrevVoxelCentroidCloud ();
      for (const size_t& idx: super.previous_keypoints_indices_)
      {
        if (!viewer->updateSphere((*prev_voxel_centroid_cloud)[idx],
                                  0.002, 0, 0, 255,
                                  "previous keypoint " + std::to_string (idx)))
        {
          viewer->addSphere((*prev_voxel_centroid_cloud)[idx],
                            0.002, 0, 0, 255,
                            "previous keypoint " + std::to_string (idx)); }
      }
    }
    // Show the transforms that were computed
    if (show_transforms)
    {
      std::unordered_map<uint32_t, std::pair<Eigen::Vector4f, Eigen::Vector4f>>
          lines = super.lines_;
      std::vector<uint32_t> label_color = super.getLabelColors ();
      for (const auto& line: super.lines_)
      {
        uint32_t color = label_color[line.first];
        double r = static_cast<double> (static_cast<uint8_t> (color >> 16));
        double g = static_cast<double> (static_cast<uint8_t> (color >> 8));
        double b = static_cast<double> (static_cast<uint8_t> (color));
        PointT pt1; PointT pt2;
        pt1.x = line.second.first[0]; pt2.x = line.second.second[0];
        pt1.y = line.second.first[1]; pt2.y = line.second.second[1];
        pt1.z = line.second.first[2]; pt2.z = line.second.second[2];

        viewer->addLine (pt1, pt2, r/255., g/255., b/255.,
                         std::to_string (line.first));

        viewer->addSphere(pt1, 0.005, 0, 255, 0, "start "
                          + std::to_string (line.first));
        viewer->addSphere(pt2, 0.005, 255, 0, 0, "end "
                          + std::to_string (line.first));
      }
    }
    // Show white sphere over disappeared/occluded supervoxels
    if (show_disappeared)
    {
      int count = 0;
      for (const auto& centroid: super.centroid_of_dynamic_svs_)
      {
        if (!viewer->updateSphere(centroid, 0.005, 255/255., 192/255., 203/255., "to_track_"
                                  + std::to_string (count)))
        { viewer->addSphere(centroid, 0.005, 255/255., 192/255., 203/255., "to_track_"
                            + std::to_string (count)); ++count; }
      }
    }
  }
}

void
transformPointFromPclToBaseLink (PointT &pt, const tf::StampedTransform &transform_to_base_link, bool verbose=false)
{
  if(verbose)
  {std::cout << "Point before transform: " << pt.x << " " << pt.y << " " << pt.z << "\n";}
  // First step:
  // transform the point from pcl kinect frame to gazebo kinect frame
  // Firstly rotate around y-axis of -90 degree
  //  double angle_in_rad_1 = -90.*M_PI/180.;
  //  tf::Matrix3x3 r_y(cos(angle_in_rad_1),  0, sin(angle_in_rad_1),
  //                    0,                    1, 0,
  //                    -sin(angle_in_rad_1), 0, cos(angle_in_rad_1));

  //  // Secondly rotate around x-axis of +90 degree
  //  double angle_in_rad_2 = 90.*M_PI/180.;
  //  tf::Matrix3x3 r_x(1, 0,                   0,
  //                    0, cos(angle_in_rad_2), -sin(angle_in_rad_2),
  //                    0, sin(angle_in_rad_2), cos(angle_in_rad_2));
  //  // Compute the total rotation matrix
  tf::Point tf_pt;//(pt.x, pt.y, pt.z);
  //  tf::Matrix3x3 r_total = r_y*r_x;
  //  tf_pt = r_total*tf_pt;

  tf_pt.setX(pt.z); tf_pt.setY(-pt.x); tf_pt.setZ(-pt.y);

  if(verbose)
  {std::cout << "Point after rotation from pcl to gazebo head_mount_kinect2_link: " << tf_pt.getX() << " " << tf_pt.getY() << " " << tf_pt.getZ() << "\n";}

  //  pt.x = tf_pt.getX(); pt.y = tf_pt.getY(); pt.z = tf_pt.getZ();


  //  double tmp_x = cos(angle_in_rad)*pt.x + sin(angle_in_rad)*pt.z;
  //  double tmp_z = -sin(angle_in_rad)*pt.x + cos(angle_in_rad)*pt.z;
  //  pt.x = tmp_x;
  //  pt.z = tmp_z;
  //  // Remember that now the x-axis is on the z-axis...
  //  // I could have multiplied the two rotation matrix might have been clearer
  //  tmp_x = cos(angle_in_rad)*pt.x - sin(angle_in_rad)*pt.y;
  //  double tmp_y = sin(angle_in_rad)*pt.x + cos(angle_in_rad)*pt.y;
  //  pt.x = tmp_x;
  //  pt.y = tmp_y;

  // Second step:
  // transform the point from pcl kinect frame to base_link frame


  //  double yaw, pitch, roll;
  //  transform_to_base_link.getBasis().getRPY(roll, pitch, yaw);
  //  tf::Quaternion q = transform_to_base_link.getRotation();
  tf::Vector3 v = transform_to_base_link.getOrigin();
  //  std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
  //  std::cout << "- Rotation: in RPY (radian) [" <<  roll << ", " << pitch << ", " << yaw << "]" << std::endl
  //            << "            in RPY (degree) [" <<  roll*180.0/M_PI << ", " << pitch*180.0/M_PI << ", " << yaw*180.0/M_PI << "]" << std::endl;

  //  tf_pt = transform_to_base_link.getBasis()*tf_pt;
  //  std::cout << "Point before translation: " << tf_pt.getX() << " " << tf_pt.getY() << " " << tf_pt.getZ() << "\n";
  //  std::cout << "Translation vector : " << v.getX() << " " << v.getY() << " " << v.getZ() << "\n";

  //  std::cout << "Point before rotation: " << tf_pt.getX() << " " << tf_pt.getY() << " " << tf_pt.getZ() << "\n";
  tf_pt = transform_to_base_link.getBasis()*tf_pt;
  if(verbose)
  {std::cout << "Point after rotation from head_moun_kinect2_link to base_footprint: " << tf_pt.getX() << " " << tf_pt.getY() << " " << tf_pt.getZ() << "\n";}

  tf_pt += v;
  if(verbose)
  {std::cout << "Point after translation: " << tf_pt.getX() << " " << tf_pt.getY() << " " << tf_pt.getZ() << "\n";}

  pt.x = tf_pt.getX(); pt.y = tf_pt.getY(); pt.z = tf_pt.getZ();
  if(verbose)
  {std::cout << "Point after transform: " << pt.x << " " << pt.y << " " << pt.z << "\n";}
}

int
main( int argc, char** argv )
{
  bool help_asked = pcl::console::find_switch (argc, argv, "-h")
      || pcl::console::find_switch (argc, argv, "--help");

  if(help_asked)
  {
    pcl::console::print_error ("Usage: %s \n "
                               "-v <voxel resolution>\n"
                               "-s <seed resolution>\n"
                               "-c <color weight> \n"
                               "-z <spatial weight> \n"
                               "-n <normal_weight>\n"
                               "-t <time pause between frames>\n"
                               "--noise <min number of neighbours in radius "
                               "of each point to not be considered as noise>\n"
                               "--manual-mode", argv[0]);
    return 1;
  }
  // ROS initializations and subscribers
  ros::init (argc, argv, "persistent_world");
  ros::NodeHandle nh;
  ros::Subscriber kinect_sub = nh.subscribe ("/head_mount_kinect2/depth/points", 1, kinectCallback);
  ros::Subscriber moving_state_sub = nh.subscribe ("/moving_state", 1, movingStateCallback);
  ros::Subscriber link_states_sub = nh.subscribe ("/gazebo/link_states", 1, linkStatesCallback);
  ros::ServiceClient client_act_at_position = nh.serviceClient<robot_arm_controllers::ActAtPosition>("act_at_position");
  robot_arm_controllers::ActAtPosition srv;

  tf::TransformListener listener;
  tf::StampedTransform transform;

  // Variables for supervoxels generation
  float voxel_resolution = 0.008f;
  bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
  if (voxel_res_specified)
    pcl::console::parse (argc, argv, "-v", voxel_resolution);

  float seed_resolution = 0.08f;
  bool seed_res_specified = pcl::console::find_switch (argc, argv, "-s");
  if (seed_res_specified)
    pcl::console::parse (argc, argv, "-s", seed_resolution);

  float color_importance = 0.2f;
  if (pcl::console::find_switch (argc, argv, "-c"))
    pcl::console::parse (argc, argv, "-c", color_importance);

  float spatial_importance = 0.4f;
  if (pcl::console::find_switch (argc, argv, "-z"))
    pcl::console::parse (argc, argv, "-z", spatial_importance);

  float normal_importance = 1.0f;
  if (pcl::console::find_switch (argc, argv, "-n"))
    pcl::console::parse (argc, argv, "-n", normal_importance);

  int time_pause_in_ms = 100;
  if (pcl::console::find_switch(argc, argv, "-t"))
    pcl::console::parse (argc, argv, "-t", time_pause_in_ms);

  uint64_t min_number_in_radius_for_noise_reduction = 20;
  if (pcl::console::find_switch(argc, argv, "--noise"))
    pcl::console::parse (argc, argv, "--noise",
                         min_number_in_radius_for_noise_reduction);

  if (pcl::console::find_switch (argc, argv, "--manual-mode"))
  { manual_mode = true; }

  // Pointcloud, used to store the cloud from the rgbd camera
  PointCloudT::Ptr cloud (new PointCloudT);
  // Tmp Pointcloud, used to store the cloud from the rgbd camera
  PointCloudT::Ptr tmp_cloud (new PointCloudT);

  // Vector of Pointclouds, used to store clouds from datasets
  std::vector<PointCloudT::Ptr> clouds;

  // Create a supervoxel clustering instance
  pcl::SequentialSVClustering<PointT> super (voxel_resolution, seed_resolution);

  // Setting the importance of each parameter in feature space
  super.setColorImportance(color_importance);
  super.setSpatialImportance(spatial_importance);
  super.setNormalImportance(normal_importance);

  // Create supervoxel clusters
  pcl::SequentialSVClustering<PointT>::SequentialSVMapT supervoxel_clusters;
  //  std::map <uint32_t, pcl::SequentialSV<PointT>::Ptr > supervoxel_clusters;

  // Create a visualizer instance
  pcl::visualization::PCLVisualizer::Ptr viewer
      (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0/255., 0/255., 0/255.);
  viewer->initCameraParameters ();
  viewer->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);
  viewer->registerKeyboardCallback(keyboardEventOccurred);

  // Create a pairwise segmentation instance
  PairwiseSegmentation pw_seg;

  // Save data
  std::ofstream csv_file;
  std::string path_home(std::getenv("HOME"));
  std::string path_to_file("/data/data_trunk.csv");
  csv_file.open ((path_home+path_to_file).c_str());
  //  csv_file << "lost_labels,total,\n";
  csv_file << "total,\n";

  //Random generator
  std::random_device generator;
  int random_sv_ind = -1;
  int random_obj_ind = -1;
  int number_of_interactions = 15;
  while(!viewer->wasStopped () && ros::ok() && number_of_interactions > 0)
  {
    ros::spinOnce ();
    try
    {
      listener.lookupTransform("/base_footprint", "/head_mount_kinect2_rgb_link",
                               ros::Time(0), transform);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s",ex.what());
      continue;
    }

    if (supervoxel_clusters.size() != 0 && !moving_state)
    {
      //      std::cout << "size: " << objects_names.size() << "\n";
      //      for (int i = 0; i<objects_names.size(); ++i)
      //      {
      //        std::cout << "name: " << objects_names[i] << "\npose: " << objects_poses[i] << "\n";
      //      }

      // Select a random SV to try to interact with
      //      std::uniform_int_distribution<int> distribution(0,supervoxel_clusters.size() - 1);
      //      random_sv_ind = distribution(generator);
      //      PointT centroid;
      //      centroid.x = supervoxel_clusters[random_sv_ind]->centroid_.x;
      //      centroid.y = supervoxel_clusters[random_sv_ind]->centroid_.y;
      //      centroid.z = supervoxel_clusters[random_sv_ind]->centroid_.z;
      //      transformPointFromPclToBaseLink(centroid, transform);
//      if (centroid.x > 0.8) {continue;}
//      srv.request.x = centroid.x;
//      srv.request.y = centroid.y;
//      srv.request.z = centroid.z;
//      if (client_act_at_position.call(srv))
//      {
//        ROS_INFO("Planning and execution of push primitive at supervoxel with label %d !", random_sv_ind + 1);
//      }
      // Select a random object to interact with
      std::uniform_int_distribution<int> distribution(0,objects_names.size() -1);
      random_obj_ind = distribution(generator);
      if (objects_poses[random_obj_ind].position.x > 0.8) {continue;}
      srv.request.x = objects_poses[random_obj_ind].position.x;
      srv.request.y = objects_poses[random_obj_ind].position.y;
      srv.request.z = objects_poses[random_obj_ind].position.z;
      if (client_act_at_position.call(srv))
      {
        ROS_INFO("Planning and execution of push primitive at object with name %s !", objects_names[random_obj_ind].c_str());
        number_of_interactions--;
      }
      else
      {
        ROS_ERROR("Failed to call service act_at_position");
      }
    }



    cloud.reset (new PointCloudT);
    tmp_cloud.reset (new PointCloudT);
    // Get the cloud
    copyPointCloud(*current_cloud, *tmp_cloud);
    //    viewer->addCoordinateSystem();
    //    viewer->addSphere(pt1, 0.005, 0, 255, 0, "start_test ");
    //    viewer->addSphere(pt2, 0.005, 255, 0, 0, "end_test ");
    //    float minX = -0.4; float minY = -0.35; float minZ = 0.95;
    //    float maxX = 0.4; float maxY = 0.25; float maxZ = 1.65;
    float minX = -0.6; float minY = -0.6; float minZ = 0.5;
    float maxX = 0.6; float maxY = 0.55; float maxZ = 1.5;
    pcl::CropBox<PointT> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(tmp_cloud);
    boxFilter.filter(*cloud);

    // If a cloud got captured from the device
    if(!cloud->empty())
    {
      if (!moving_state)
      {
        // Filter the noise from the input pointcloud
        pcl::RadiusOutlierRemoval<PointT> rorfilter;
        rorfilter.setInputCloud (cloud);
        rorfilter.setRadiusSearch (2*voxel_resolution);
        rorfilter.setMinNeighborsInRadius
            (min_number_in_radius_for_noise_reduction);
        rorfilter.filter (*cloud);

        super.setInputCloud(boost::make_shared<PointCloudT>(*cloud));

        pcl::console::print_highlight ("Extracting supervoxels!\n");

        super.extract (supervoxel_clusters);

        // Interactive segmentation
        std::vector <uint32_t> to_reset_parts = super.getToResetParts ();
        //      csv_file << to_reset_parts << "," << supervoxel_clusters.size () << ",\n";
        csv_file << supervoxel_clusters.size () << ",\n";
        pw_seg.resetParts (to_reset_parts);
        std::vector <uint32_t> moving_parts = super.getMovingParts ();
        pw_seg.update (moving_parts, frame_count);
        PairwiseSegmentation::Segmentation curr_seg =
            pw_seg.getCurrentSegmentation ();

        std::cout << "------------CURRENT CLUSTERING------------\n";
        for (const auto& pair: curr_seg)
        {
          std::cout << "[";
          for (const uint32_t label: pair.second)
          { std::cout << label << ", "; }
          std::cout << "\b\b] ";
        }
        std::cout << "\n";
        std::cout << "Clusters size:\n";
        int tot = 0;
        for (const auto& pair: curr_seg)
        {
          tot += pair.second.size ();
          std::cout << pair.second.size () << " ";
        }
        std::cout << "total: " << tot << "\n";

        pcl::console::print_info ("Found %d supervoxels\n",
                                  supervoxel_clusters.size ());
      }
      updateView (viewer, cloud, super, supervoxel_clusters, pw_seg);

      viewer->spinOnce (time_pause_in_ms);
      while (manual_mode && !viewer->wasStopped ())
      {
        viewer->removeAllShapes ();
        viewer->removeAllPointClouds ();
        updateView (viewer, cloud, super, supervoxel_clusters, pw_seg);
        PointT pt1, pt2, pt3, pt4;
        pt1.x = 0; pt1.y = 0; pt1.z = 0;
        pt2.x = 1; pt2.y = 0; pt2.z = 0;
        pt3.x = 0; pt3.y = 1; pt3.z = 0;
        pt4.x = 0; pt4.y = 0; pt4.z = 1;
        //        transformPointFromPclToBaseLink(pt1, transform);
        //        transformPointFromPclToBaseLink(pt2, transform);
        //        transformPointFromPclToBaseLink(pt3, transform);
        //        transformPointFromPclToBaseLink(pt4, transform);
        viewer->addLine (pt1, pt2,255./255.,0.,0., "test_line_x");
        viewer->addLine (pt1, pt3,0.,255./255., 0., "test_line_y");
        viewer->addLine (pt1, pt4,0.,0., 255./255., "test_line_z");
        viewer->spinOnce (time_pause_in_ms);
      }
      viewer->removeAllShapes ();
      viewer->removeAllPointClouds ();
      ++frame_count;
    }
  }
  csv_file.close ();
  return (0);
}
