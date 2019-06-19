// PCL basic includes
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/make_shared.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
// PCL tracking includes
//#include <pcl/tracking/tracking.h>
//#include <pcl/tracking/particle_filter.h>
//#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
//#include <pcl/tracking/particle_filter_omp.h>
//#include <pcl/tracking/coherence.h>
//#include <pcl/tracking/distance_coherence.h>
//#include <pcl/tracking/hsv_color_coherence.h>
//#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
//#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>
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

typedef pcl::tracking::ParticleFilterTracker<PointT, StateT> ParticleFilter;

POINT_CLOUD_REGISTER_POINT_STRUCT (Histogram<32>,
                                   (float[32], histogram, histogram)
)

static bool show_prev = false;
static bool show_curr = false;
static bool show_labels = false;

void
keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event)
{
  //  std::cout << "key: " << event.getKeySym() << "\n";
  if (event.getKeySym() == "KP_1" && event.keyDown())
  { show_prev = !show_prev; }
  if (event.getKeySym() == "KP_2" && event.keyDown())
  { show_curr = !show_curr; }
  if (event.getKeySym () == "KP_3" && event.keyDown ())
  { show_labels = !show_labels; }
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
                               "-t <time pause between frames>\n", argv[0]);
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
  boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::OpenNIGrabber>();

  // Pointcloud, used to store the cloud from the rgbd camera
  PointCloudT::Ptr cloud (new PointCloudT);
  // Tmp Pointcloud, used to store the cloud from the rgbd camera
  PointCloudT::Ptr tmp_cloud (new PointCloudT);

  // Vector of Pointclouds, used to store clouds from datasets
  std::vector<PointCloudT::Ptr> clouds;

  // SupervoxelTracker instantiation
  pcl::SupervoxelTracker<PointT, StateT> tracker;

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
  Getter<pcl::PointXYZRGBA> getter( *grabber);

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

  int i = 0;
  while(!viewer->wasStopped ())
  {
    cloud.reset (new PointCloudT);
    tmp_cloud.reset (new PointCloudT);
    // Get the cloud
    //    copyPointCloud(getter.getCloud(), cloud);

    copyPointCloud(*clouds[(i++)%N_DATA], *tmp_cloud);//cloud = clouds[i%N_DATA];
    //    PointT pt1;
    //    pt1.x = -0.35;
    //    pt1.y = 0.5;
    //    pt1.z = 1;
    //    PointT pt2;
    //    pt2.x = 0.35;
    //    pt2.y = -0.25;
    //    pt2.z = 1.25;
    float minX = -0.35; float minY = -0.25; float minZ = 1.;
    float maxX = 0.35; float maxY = 0.5; float maxZ = 1.4;
    pcl::CropBox<PointT> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(tmp_cloud);
    boxFilter.filter(*cloud);

    // If a cloud got captured from the device
    if(!cloud->empty())
    {

      super.setInputCloud(boost::make_shared<PointCloudT>(*cloud));

      pcl::console::print_highlight ("Extracting supervoxels!\n");

      super.extract (supervoxel_clusters);

      pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

      // Get the voxel centroid cloud
      PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
      // Get the labeled voxel cloud
      PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
      // Get the labeled voxel cloud
      PointRGBLCloudT::Ptr rgb_labeled_voxel_cloud = super.getLabeledRGBVoxelCloud ();
      // Get the voxel normal cloud
      NormalCloud::Ptr voxel_normal_cloud = super.getVoxelNormalCloud ();

      // Show the unlabeled voxel cloud in green
      //      PointCloudT::Ptr unlabeled_voxel_centroid_cloud = super.getUnlabeledVoxelCentroidCloud ();
      //      pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color (unlabeled_voxel_centroid_cloud, 0, 255, 0);
      //      if(!viewer->updatePointCloud<PointT>(unlabeled_voxel_centroid_cloud, single_color, "voxel centroids"))
      //        viewer->addPointCloud<PointT> (unlabeled_voxel_centroid_cloud, single_color, "voxel centroids");
      // Without color
      //      if (!viewer->updatePointCloud (voxel_centroid_cloud, "voxel centroids"))
      //        viewer->addPointCloud (voxel_centroid_cloud, "voxel centroids");
      // With color
      //      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(voxel_centroid_cloud);
      //      if (!viewer->updatePointCloud<PointT> (voxel_centroid_cloud, rgb, "voxel centroids"))
      //        viewer->addPointCloud<PointT> (voxel_centroid_cloud, rgb, "voxel centroids");



      if (show_labels) // Show the labeled observed pointcloud
      {
        if (!viewer->updatePointCloud (labeled_voxel_cloud, "displayed cloud"))
        { viewer->addPointCloud (labeled_voxel_cloud, "displayed cloud"); }
      }
      else // Show the observed pointcloud
      {
        pcl::visualization::PointCloudColorHandlerRGBAField<PointT> rgba(cloud);
        if (!viewer->updatePointCloud<PointT> (cloud, rgba, "displayed cloud"))
        { viewer->addPointCloud<PointT> (cloud, rgba, "displayed cloud"); }
      }
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,1., "displayed cloud");

      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////TEST////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////

      viewer->addText ("press NUM1 to show/hide previous keypoints",0, 0, "t_prev");
      viewer->addText ("press NUM2 to show/hide current keypoints",0, 10, "t_curr");
      viewer->addText ("press NUM3 to switch between labelized/normal view of cloud",0, 20, "t_cloud");
      std::vector<int> current_keypoints_indices = super.current_keypoints_indices_;
      std::vector<int> previous_keypoints_indices = super.previous_keypoints_indices_;

      if(show_curr)
      {
        for(auto idx: current_keypoints_indices)
        {
          PointT pt = (*super.getUnlabeledVoxelCentroidCloud())[idx];
          viewer->addSphere(pt, 0.002, 125, 125, 0, "current keypoint " + std::to_string (idx));
        }
      }
      if(show_prev)
      {
        for(auto idx: previous_keypoints_indices)
        {
          PointT pt = (*super.getPrevVoxelCentroidCloud())[idx];
          viewer->addSphere(pt, 0.002, 0, 0, 255, "previous keypoint " + std::to_string (idx));
        }
      }
      std::unordered_map<uint32_t, std::pair<Eigen::Vector4f, Eigen::Vector4f>> lines = super.lines_;
      std::vector<uint32_t> label_color = super.getLabelColors ();
      for (auto line: lines)
      {
        uint32_t color = label_color[line.first];
        double r = static_cast<double> (static_cast<uint8_t> (color >> 16));
        double g = static_cast<double> (static_cast<uint8_t> (color >> 8));
        double b = static_cast<double> (static_cast<uint8_t> (color));
        //        std::cout << "r: " << r << " g: " << g << " b: " << b << "\n";
        PointT pt1;
        pt1.x = line.second.first[0];
        pt1.y = line.second.first[1];
        pt1.z = line.second.first[2];
        PointT pt2;
        pt2.x = line.second.second[0];
        pt2.y = line.second.second[1];
        pt2.z = line.second.second[2];
        //        pt2.x = pt1.x + seed_resolution;
        //        pt2.y = pt1.y;
        //        pt2.z = pt1.z;
        viewer->addLine (pt1, pt2, r/255, g/255, b/255, std::to_string (line.first));

        viewer->addSphere(pt1, 0.005, 0, 255, 0, "start " + std::to_string (line.first));
        viewer->addSphere(pt2, 0.005, 255, 0, 0, "end " + std::to_string (line.first));
      }

      // TEST
      // Useful types
      //      typedef pcl::Histogram<32> FeatureT;
      ////      typedef flann::L1<float> DistanceT;
      ////      typedef flann::L2<float> DistanceT;
      //      typedef flann::KL_Divergence<float> DistanceT;
      //      typedef pcl::search::FlannSearch<FeatureT, DistanceT> SearchT;
      //      typedef typename SearchT::FlannIndexCreatorPtr CreatorPtrT;
      //      typedef typename SearchT::KdTreeMultiIndexCreator IndexT;
      //      typedef typename SearchT::PointRepresentationPtr RepresentationPtrT;
      //      // Instantiate search object with 4 randomized trees and 128 checks
      //      SearchT search (true, CreatorPtrT (new IndexT (4)));
      //      search.setPointRepresentation (RepresentationPtrT (new pcl::DefaultFeatureRepresentation<FeatureT>));
      //      search.setChecks (128); // The more checks the more precise the solution
      //      FeatureT hist;
      //      for (int k = 0; k < 32; ++k) { hist.histogram[i] = k/10.f; }
      //      pcl::PointCloud<FeatureT>::Ptr search_cloud(new pcl::PointCloud<FeatureT>);
      //      search_cloud->push_back (hist);
      //      // search_cloud is filled with the keypoints to match
      //      search.setInputCloud (search_cloud);
      //      std::vector<int> indices;
      //      std::vector<float> distances;
      //      search.nearestKSearch(hist, 1, indices, distances);
      //      std::cout << "distance: " << distances[0] << "\n";
      // TEST
      //      PointT pt1;
      //      pt1.x = -0.35;
      //      pt1.y = 0.5;
      //      pt1.z = 1;
      //      PointT pt2;
      //      pt2.x = 0.35;
      //      pt2.y = -0.25;
      //      pt2.z = 1.25;
      //      viewer->addLine (pt1, pt2, "test");

      //      viewer->addSphere(pt1, 0.005, 0, 255, 0, "start test");
      //      viewer->addSphere(pt2, 0.005, 255, 0, 0, "end test");

      viewer->spinOnce (time_pause_in_ms);

      viewer->removeAllShapes ();
      viewer->removeAllPointClouds ();
    }
  }

  return 0;
}
