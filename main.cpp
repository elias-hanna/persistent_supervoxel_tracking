// PCL basic includes
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <boost/make_shared.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
// Local libs includes
#include "libs/papon/supervoxel/sequential_supervoxel_clustering.h"
#include "libs/getter/getter.hpp"
#include "libs/supervoxel_tracker/supervoxel_tracker.h"
// Macros
#define N_DATA 2
// Types
typedef pcl::tracking::ParticleXYZRPY StateT;
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

POINT_CLOUD_REGISTER_POINT_STRUCT (Histogram<32>,
                                   (float[32], histogram, histogram)
)

static bool manual_mode = false;
static bool show_prev = false;
static bool show_curr = false;
static bool show_labels = false;
static bool show_disappeared = false;
static bool show_transforms = false;
static uint64_t frame_count = 0;

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
  { show_disappeared = !show_disappeared; }
  if (event.getKeySym () == "KP_5" && event.keyDown ())
  { show_transforms = !show_transforms; }
  if (event.getKeySym () == "Return" && event.keyDown ())
  { manual_mode = false; }
}

void
updateView (const pcl::visualization::PCLVisualizer::Ptr viewer, const PointCloudT::Ptr cloud, const pcl::SequentialSVClustering<PointT>& super)
{
  // Get the voxel centroid cloud
  PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
  // Get the labeled voxel cloud
  PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
  // Get the labeled voxel cloud
  PointRGBLCloudT::Ptr rgb_labeled_voxel_cloud = super.getLabeledRGBVoxelCloud ();
  // Get the voxel normal cloud
  NormalCloud::Ptr voxel_normal_cloud = super.getVoxelNormalCloud ();
  // Show the labeled observed pointcloud
  if (show_labels)
  {
    if (!viewer->updatePointCloud (labeled_voxel_cloud, "displayed cloud"))
    { viewer->addPointCloud (labeled_voxel_cloud, "displayed cloud"); }
  }
  // Show the observed pointcloud
  else
  {
    pcl::visualization::PointCloudColorHandlerRGBAField<PointT> rgba(cloud);
    if (!viewer->updatePointCloud<PointT> (cloud, rgba, "displayed cloud"))
    { viewer->addPointCloud<PointT> (cloud, rgba, "displayed cloud"); }
  }
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,1., "displayed cloud");

  viewer->addText ("press NUM1 to show/hide previous keypoints",0, 0, "t_prev");
  viewer->addText ("press NUM2 to show/hide current keypoints",0, 10, "t_curr");
  viewer->addText ("press NUM3 to switch between labelized/normal view of cloud",0, 20, "t_cloud");
  viewer->addText ("press NUM4 to show/hide text over disappeared/occluded supervoxels",0, 30, "t_dis");
  viewer->addText ("press NUM5 to show/hide the computed transforms",0, 40, "t_trans");
  if (manual_mode)
  { viewer->addText ("press Enter to go to next frame",700, 0, 20, 1., 1., 1., "t_manual"); }
  std::vector<int> current_keypoints_indices = super.current_keypoints_indices_;
  std::vector<int> previous_keypoints_indices = super.previous_keypoints_indices_;
  if (frame_count)
  {
    // Show current keypoint cloud
    if (show_curr)
    {
      for (const auto& idx: super.current_keypoints_indices_)
      {
        if (!viewer->updateSphere((*voxel_centroid_cloud)[idx], 0.002, 125, 125, 0, "current keypoint " + std::to_string (idx)))
        { viewer->addSphere((*voxel_centroid_cloud)[idx], 0.002, 125, 125, 0, "current keypoint " + std::to_string (idx)); }
      }
    }
    // Show previous keypoint cloud
    if (show_prev)
    {
      // Get the previous voxel centroid cloud
      PointCloudT::Ptr prev_voxel_centroid_cloud = super.getPrevVoxelCentroidCloud ();
      for (const auto& idx: super.previous_keypoints_indices_)
      {
        if (!viewer->updateSphere((*prev_voxel_centroid_cloud)[idx], 0.002, 0, 0, 255, "previous keypoint " + std::to_string (idx)))
        { viewer->addSphere((*prev_voxel_centroid_cloud)[idx], 0.002, 0, 0, 255, "previous keypoint " + std::to_string (idx)); }
      }
    }
    // Show the transforms that were computed
    if (show_transforms)
    {
      std::unordered_map<uint32_t, std::pair<Eigen::Vector4f, Eigen::Vector4f>> lines = super.lines_;
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

        viewer->addLine (pt1, pt2, r/255, g/255, b/255, std::to_string (line.first));

        viewer->addSphere(pt1, 0.005, 0, 255, 0, "start " + std::to_string (line.first));
        viewer->addSphere(pt2, 0.005, 255, 0, 0, "end " + std::to_string (line.first));
      }
    }
    // Show white sphere over disappeared/occluded supervoxels
    if (show_disappeared)
    {
      int count = 0;
      for (const auto& centroid: super.centroid_of_dynamic_svs_)
      {
        if (!viewer->updateSphere(centroid, 0.002, 1., 1., 1., "to_track_" + std::to_string (count)))
        { viewer->addSphere(centroid, 0.002, 1., 1., 1., "to_track_" + std::to_string (count)); ++count; }
      }
    }
  }
}

int 
main( int argc, char** argv )
{
  bool help_asked = pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help");

  if(help_asked)
  {
    pcl::console::print_error ("Usage: %s \n "
                               "-v <voxel resolution>\n"
                               "-s <seed resolution>\n"
                               "-c <color weight> \n"
                               "-z <spatial weight> \n"
                               "-n <normal_weight>\n"
                               "-t <time pause between frames>\n"
                               "--manual-mode", argv[0]);
    return 1;
  }

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

  // OpenNIGrabber, used to capture pointclouds from various rgbd cameras
  //  boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::OpenNIGrabber>();

  // Pointcloud, used to store the cloud from the rgbd camera
  PointCloudT::Ptr cloud (new PointCloudT);
  // Tmp Pointcloud, used to store the cloud from the rgbd camera
  PointCloudT::Ptr tmp_cloud (new PointCloudT);

  // Vector of Pointclouds, used to store clouds from datasets
  std::vector<PointCloudT::Ptr> clouds;

  // SupervoxelTracker instantiation
  pcl::SupervoxelTracker<PointT, StateT> tracker;

  //.pcd Files
  // This is where clouds is filled
  for(int i = 0 ; i < N_DATA ; i++)
  {
    PointCloudT::Ptr cloud(new PointCloudT);
    //    std::string path = "../data/test" + std::to_string(i) + ".pcd";
    std::string path = "../data/example_" + std::to_string(i) + ".pcd";
    pcl::console::print_highlight (("Loading point cloud" + std::to_string(i) + "...\n").c_str());
    if (pcl::io::loadPCDFile<PointT> (path, *cloud))
    {
      pcl::console::print_error (("Error loading cloud" + std::to_string(i) + " file!\n").c_str());
      return (1);
    }
    clouds.push_back(cloud);
  }

  // Getter Class to get the point cloud from various capturing devices
  //  Getter<pcl::PointXYZRGBA> getter( *grabber);

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
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->initCameraParameters ();
  viewer->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);
  viewer->registerKeyboardCallback(keyboardEventOccurred);

  while(!viewer->wasStopped ())
  {
    if (pcl::console::find_switch (argc, argv, "--manual-mode"))
    { manual_mode = true; }

    cloud.reset (new PointCloudT);
    tmp_cloud.reset (new PointCloudT);
    // Get the cloud
    //    copyPointCloud(getter.getCloud(), cloud);

    copyPointCloud(*clouds[frame_count%N_DATA], *tmp_cloud);//cloud = clouds[i%N_DATA];
    //    PointT pt1;
    //    pt1.x = -0.35;
    //    pt1.y = 0.5;
    //    pt1.z = 1 - 0.1;
    //    PointT pt2;
    //    pt2.x = 0.35;
    //    pt2.y = -0.25;
    //    pt2.z = 1.25 - 0.02;
    //    viewer->addLine (pt1, pt2, "test_line");
    //    viewer->addSphere(pt1, 0.005, 0, 255, 0, "start_test ");
    //    viewer->addSphere(pt2, 0.005, 255, 0, 0, "end_test ");
    //    float minX = -0.25; float minY = -0.25; float minZ = 1.1;
    //    float maxX = 0.45; float maxY = 0.5; float maxZ = 1.5;
    //    pcl::CropBox<PointT> boxFilter;
    //    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    //    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    //    boxFilter.setInputCloud(tmp_cloud);
    //    boxFilter.filter(*cloud);
    copyPointCloud(*tmp_cloud, *cloud);//cloud = clouds[i%N_DATA];

    // If a cloud got captured from the device
    if(!cloud->empty())
    {

      super.setInputCloud(boost::make_shared<PointCloudT>(*cloud));

      pcl::console::print_highlight ("Extracting supervoxels!\n");

      super.extract (supervoxel_clusters);

      pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());
      updateView (viewer, cloud, super);

      viewer->spinOnce (time_pause_in_ms);
      while (manual_mode && !viewer->wasStopped ())
      {
        viewer->removeAllShapes ();
        viewer->removeAllPointClouds ();
        updateView (viewer, cloud, super);
        viewer->spinOnce (time_pause_in_ms);
      }
      viewer->removeAllShapes ();
      viewer->removeAllPointClouds ();
      ++frame_count;
    }
  }

  return 0;
}
