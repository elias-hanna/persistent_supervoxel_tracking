// PCL basic includes
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <boost/make_shared.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
// Local libs includes
#include "libs/papon/supervoxel/sequential_supervoxel_clustering.h"
#include "libs/supervoxel_tracker/supervoxel_tracker.h"
#include "libs/pairwise_segmentation/pairwise_segmentation.h"

#define USE_KINECT 1
#define KINECT_V2 1

#if USE_KINECT == 1
#if KINECT_V2 == 1
// Kinect2 includes
#define WITH_PCL
#include "libs/libfreenect2pclgrabber/include/k2g.h"
#else
#include "libs/getter/getter.hpp"
#endif
#endif
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

  /* Point clouds from Kinect */
  // /!\ Empty constructor does nothing, need to call setGrabber
#if USE_KINECT == 1
#if KINECT_V2 == 1
  K2G k2g;
  Processor freenectprocessor = OPENGL;
  k2g.setProcessorAndStartRecording (freenectprocessor);
  k2g.disableLog ();
#else
  Getter<pcl::PointXYZRGBA> getter;
  // OpenNIGrabber, used to capture pointclouds from various rgbd cameras
  boost::shared_ptr<pcl::Grabber>
      grabber = boost::make_shared<pcl::OpenNIGrabber>();
  // Getter Class to get the point cloud from various capturing devices
  getter.setGrabber (grabber);
#endif
#endif
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
#if USE_KINECT == 0
  for(int i = 0 ; i < N_DATA ; i++)
  {
    PointCloudT::Ptr cloud(new PointCloudT);
    //    std::string path = "../data/test" + std::to_string(i) + ".pcd";
    std::string path = "../data/example_" + std::to_string(i) + ".pcd";
    pcl::console::print_highlight (("Loading point cloud" + std::to_string(i) +
                                    "...\n").c_str());
    if (pcl::io::loadPCDFile<PointT> (path, *cloud))
    {
      pcl::console::print_error (("Error loading cloud" + std::to_string(i) +
                                  " file!\n").c_str());
      return (1);
    }
    clouds.push_back(cloud);
  }
#endif
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
  csv_file.open ("../data/data_trunk.csv");
  csv_file << "lost_labels,total,\n";
  while(!viewer->wasStopped ())
  {
    if (pcl::console::find_switch (argc, argv, "--manual-mode"))
    { manual_mode = true; }

    cloud.reset (new PointCloudT);
    tmp_cloud.reset (new PointCloudT);
    // Get the cloud
#if USE_KINECT == 1
#if KINECT_V2 == 1
    { copyPointCloud(*k2g.getCloud(), *tmp_cloud); }
#else
    { copyPointCloud(getter.getCloud(), *tmp_cloud); }
#endif
#else
    copyPointCloud(*clouds[frame_count%N_DATA], *tmp_cloud);
#endif
    //    PointT pt1, pt2;
    //    pt1.x = -0.35; pt1.y = 0.5; pt1.z = 1 - 0.1;
    //    pt2.x = 0.35; pt2.y = -0.25; pt2.z = 1.25 - 0.02;
    //    viewer->addLine (pt1, pt2, "test_line");
    //    viewer->addSphere(pt1, 0.005, 0, 255, 0, "start_test ");
    //    viewer->addSphere(pt2, 0.005, 255, 0, 0, "end_test ");
    float minX = -0.4; float minY = -0.35; float minZ = 0.8;
    float maxX = 0.4; float maxY = 0.25; float maxZ = 1.3;
    pcl::CropBox<PointT> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(tmp_cloud);
    boxFilter.filter(*cloud);

    // If a cloud got captured from the device
    if(!cloud->empty())
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
      csv_file << to_reset_parts << "," << supervoxel_clusters.size () << ",\n";
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
      updateView (viewer, cloud, super, supervoxel_clusters, pw_seg);

      viewer->spinOnce (time_pause_in_ms);
      while (manual_mode && !viewer->wasStopped ())
      {
        viewer->removeAllShapes ();
        viewer->removeAllPointClouds ();
        updateView (viewer, cloud, super, supervoxel_clusters, pw_seg);
        viewer->spinOnce (time_pause_in_ms);
      }
      viewer->removeAllShapes ();
      viewer->removeAllPointClouds ();
      ++frame_count;
    }
  }
#if USE_KINECT == 1
#if KINECT_V2 == 1
  k2g.shutDown ();
#endif
#endif
  csv_file.close ();
  return (0);
}
