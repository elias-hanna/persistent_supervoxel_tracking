// PCL basic includes
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/make_shared.hpp>
#include <pcl/visualization/pcl_visualizer.h>
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
  PointCloudT cloud;

  // Vector of Pointclouds, used to store clouds from datasets
  std::vector<PointCloudT::Ptr> clouds;

  // SupervoxelTracker instantiation
  pcl::SupervoxelTracker<PointT, StateT> tracker;

  // This is where clouds is filled
  for(int i = 0 ; i < N_DATA ; i++)
  {
    PointCloudT::Ptr cloud(new PointCloudT);
    std::string path = "../data/test" + std::to_string(i) + ".pcd";
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

  int i = 0;
  while(!viewer->wasStopped ())
  {
    // Get the cloud
//    copyPointCloud(getter.getCloud(), cloud);
    copyPointCloud(*clouds[i%N_DATA], cloud);//cloud = clouds[i%N_DATA];

    // If a cloud got captured from the device
    if(!cloud.empty())
    {

      super.setInputCloud(boost::make_shared<PointCloudT>(cloud));

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


      //      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "voxel centroids");
      //      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,1, "voxel centroids");

      if (!viewer->updatePointCloud (labeled_voxel_cloud, "labeled voxels"))
        viewer->addPointCloud (labeled_voxel_cloud, "labeled voxels");

      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,1, "labeled voxels");


      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////TEST////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      //      uint32_t label = 124;
      //      if(!viewer->updateSphere(supervoxel_clusters[label]->centroid_ , 0.005, 255, 0, 0))
      //        viewer->addSphere(supervoxel_clusters[label]->centroid_ , 0.005, 255, 0, 0);

      //      PointT pt1;
      //      pt1.x = 0.011487;
      //      pt1.y = 0.131805;
      //      pt1.z = 0.649465;
      //      PointT pt2;
      //      pt2.x = 0.011487;
      //      pt2.y = 0.131805;
      //      pt2.z = 0.649465;

      //      PointT pt1;
      //      pt1.x = 0;
      //      pt1.y = 0;
      //      pt1.z = 0;
      //      PointT pt2;
      //      pt2.x = 1;
      //      pt2.y = 0;
      //      pt2.z = 0;

      //      viewer->addLine(pt1, pt2, 255, 0, 0);

      std::unordered_map<uint32_t, std::pair<Eigen::Vector3f, Eigen::Vector3f>> lines = super.lines_;
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
        viewer->addLine (pt1, pt2, r/255, g/255, b/255, std::to_string (line.first));

        viewer->addSphere(pt1, 0.005, 0, 255, 0, "start " + std::to_string (line.first));
        viewer->addSphere(pt2, 0.005, 255, 0, 0, "end " + std::to_string (line.first));
      }

      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      /////////////////////TEST SIFT//////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////

//      pcl::StopWatch timer;
//      double begin = timer.getTime ();
//      // Parameters for sift computation
//      const float min_scale = 0.01f;
//      const int n_octaves = 8;
//      const int n_scales_per_octave = 8;
//      const float min_contrast = 0.3f;

//      // Copy pointcloud
//      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
//      copyPointCloud(cloud, *cloud2);
//      // Estimate the sift interest points using Intensity values from RGB values
//      pcl::SIFTKeypoint<pcl::PointXYZRGBA, pcl::PointWithScale> sift;
//      pcl::PointCloud<pcl::PointWithScale> result;
//      pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA> ());
//      sift.setSearchMethod(tree);
//      sift.setScales(min_scale, n_octaves, n_scales_per_octave);
//      sift.setMinimumContrast(min_contrast);
//      sift.setInputCloud(cloud2);
//      sift.compute(result);

//      double end = timer.getTime ();

//      std::cout << "Time elapsed computing sift keypoints: " << end - begin << " ms\n";
//      // Copying the pointwithscale to pointxyz so as visualize the cloud
//      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZ>);
//      copyPointCloud(result, *cloud_temp);

////       Saving the resultant cloud
//      std::cout << "Resulting sift points are of size: " << cloud_temp->points.size () <<std::endl;
//      pcl::io::savePCDFileASCII("sift_points.pcd", *cloud_temp);

////       Visualization of keypoints along with the original cloud
//       pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (cloud_temp, 0, 255, 0);

//       if (!viewer->updatePointCloud (cloud_temp, keypoints_color_handler, "keypoints"))
//         viewer->addPointCloud (cloud_temp, keypoints_color_handler, "keypoints");

//       viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");

      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////TEST////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      //      if(false)//i >= 10)
      //      {
      //        pcl::StopWatch timer_;

      //        double init = timer_.getTime ();
      //        tracker.setReferenceClouds(supervoxel_clusters);
      //        //      }
      //        //      else if (i>10)
      //        //      {
      //        std::map< uint32_t, StateT> predicted_states(tracker.track(rgb_labeled_voxel_cloud));
      //        double end = timer_.getTime ();
      //        std::cout << "Temps de calcul filtre particulaire: " << end - init << " ms\n";
      //        for (auto i: predicted_states)
      //        {
      //          uint32_t label = i.first;
      ////          std::cout << "label: " << i.first << " etat: " << i.second << " centroid of sv: " << supervoxel_clusters[i.first]->centroid_<< "\n";
      //          pcl::tracking::ParticleFilterTracker< PointT, StateT>* trackerAt = tracker.getTrackerAt(label);
      //          ParticleFilter::PointCloudStatePtr particles = trackerAt->getParticles ();

      //          //Set pointCloud with particle's points
      //          pcl::PointCloud<pcl::PointXYZ>::Ptr particle_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
      //          for (size_t i = 0; i < particles->points.size (); i++)
      //          {
      //            pcl::PointXYZ point;

      //            point.x = particles->points[i].x;
      //            point.y = particles->points[i].y;
      //            point.z = particles->points[i].z;
      //            particle_cloud->points.push_back (point);
      //            particle_cloud->points.push_back (pcl::PointXYZ(particles->points[i].x, particles->points[i].y, particles->points[i].z));
      //          }

      //          //Draw red particles
      //          {
      //            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color (particle_cloud, 250, 99, 71);
      //            std::string str = "particle cloud " + std::to_string(label);
      //            if (!viewer->updatePointCloud (particle_cloud, red_color, str))
      //              viewer->addPointCloud (particle_cloud, red_color, str);
      //          }
      //        }
      //      }
      if(i!=0)
        viewer->spinOnce (time_pause_in_ms);
      else
        viewer->spinOnce (100);
      for (auto line: lines)
      {
        viewer->removeShape(std::to_string (line.first));
        viewer->removeShape("start " + std::to_string (line.first));
        viewer->removeShape("end " + std::to_string (line.first));
      }
      ++i;
    }
  }

  return 0;
}
