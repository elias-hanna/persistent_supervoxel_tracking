#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/filters/statistical_outlier_removal.h>
//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>

// Types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

void addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                       PointCloudT &adjacent_supervoxel_centers,
                                       std::string supervoxel_name,
                                       pcl::visualization::PCLVisualizer::Ptr & viewer);

int
main (int argc, char ** argv)
{
  if (argc < 2)
  {
    pcl::console::print_error ("Syntax is: %s <pcd-file> \n "
                                "--NT Dsables the single cloud transform \n"
                                "-v <voxel resolution>\n-s <seed resolution>\n"
                                "-c <color weight> \n-z <spatial weight> \n"
                                "-n <normal_weight>\n", argv[0]);
    return (1);
  }


  PointCloudT::Ptr cloud1 (new PointCloudT);
  pcl::console::print_highlight ("Loading point cloud1...\n");
  if (pcl::io::loadPCDFile<PointT> (argv[1], *cloud1))
  {
    pcl::console::print_error ("Error loading cloud1 file!\n");
    return (1);
  }

  PointCloudT::Ptr cloud2 (new PointCloudT);
  pcl::console::print_highlight ("Loading point cloud2...\n");
  if (pcl::io::loadPCDFile<PointT> ("../data/test2.pcd", *cloud2))
  {
    pcl::console::print_error ("Error loading cloud2 file!\n");
    return (1);
  }

  bool disable_transform = pcl::console::find_switch (argc, argv, "--NT");

  float voxel_resolution = 0.008f;
  bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
  if (voxel_res_specified)
    pcl::console::parse (argc, argv, "-v", voxel_resolution);

  float seed_resolution = 0.1f;
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

  //////////////////////////////  //////////////////////////////
  ////// This is how to use supervoxels
  //////////////////////////////  //////////////////////////////

  pcl::octree::OctreePointCloudChangeDetector<PointT> diff_octree(voxel_resolution);
  diff_octree.setInputCloud(cloud1);
  diff_octree.addPointsFromInputCloud();
  diff_octree.switchBuffers();
  diff_octree.setInputCloud(cloud2);
  diff_octree.addPointsFromInputCloud();

  std::vector<int> newPointIdxVector;

  // Get vector of point indices from octree voxels which did not exist in previous buffer
  diff_octree.getPointIndicesFromNewVoxels (newPointIdxVector);

  // Output points
  PointCloudT::Ptr diff_cloud(new PointCloudT);
  diff_cloud->width = newPointIdxVector.size();
  diff_cloud->height = 1;
  diff_cloud->points.resize (diff_cloud->width * diff_cloud->height);

  for (size_t i = 0; i < diff_cloud->points.size (); ++i)
  {
    diff_cloud->points[i].x = cloud2->points[newPointIdxVector[i]].x;
    diff_cloud->points[i].y = cloud2->points[newPointIdxVector[i]].y;
    diff_cloud->points[i].z = cloud2->points[newPointIdxVector[i]].z;
    diff_cloud->points[i].r = cloud2->points[newPointIdxVector[i]].r;
    diff_cloud->points[i].g = cloud2->points[newPointIdxVector[i]].g;
    diff_cloud->points[i].b = cloud2->points[newPointIdxVector[i]].b;
    diff_cloud->points[i].a = cloud2->points[newPointIdxVector[i]].a;
  } 

  // Create the filtering object
  PointCloudT::Ptr diff_filtered(new PointCloudT);
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud (diff_cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*diff_filtered);


  // Get the new octree

  *cloud1 += *diff_filtered;

  
  // std::cout << "Output from getPointIndicesFromNewVoxels:" << std::endl;
  // for (size_t i = 0; i < newPointIdxVector.size (); ++i)
  //   std::cout << i << "# Index:" << newPointIdxVector[i]
  //             << "  Point:" << cloud2->points[newPointIdxVector[i]].x << " "
  //             << cloud2->points[newPointIdxVector[i]].y << " "
  //             << cloud2->points[newPointIdxVector[i]].z << std::endl;

  //////////////////////////////////////////

  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
  if (disable_transform)
    super.setUseSingleCameraTransform (false);
  super.setInputCloud (cloud1);
  super.setColorImportance (color_importance);
  super.setSpatialImportance (spatial_importance);
  super.setNormalImportance (normal_importance);

  std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;

  pcl::console::print_highlight ("Extracting supervoxels!\n");
  super.extract (supervoxel_clusters);
  pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->initCameraParameters ();

  // View the initial point cloud
  // *cloud1 += *diff_filtered;

  pcl::visualization::PCLVisualizer::Ptr viewerb (new pcl::visualization::PCLVisualizer ("Initial Image"));
  viewerb->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgba(cloud1);
  viewerb->addPointCloud<PointT> (cloud1, rgba, "point cloud");
  viewerb->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
  viewerb->initCameraParameters ();

  // View the differential point cloud
  pcl::visualization::PCLVisualizer::Ptr viewerc (new pcl::visualization::PCLVisualizer ("Differential Image"));
  viewerc->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgba2(diff_cloud);
  viewerc->addPointCloud<PointT> (diff_cloud, rgba2, "diff point cloud");
  viewerc->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "diff point cloud");
  viewerc->initCameraParameters ();

  // View the differential point cloud
  pcl::visualization::PCLVisualizer::Ptr viewerd (new pcl::visualization::PCLVisualizer ("Filtered Differential Image"));
  viewerd->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgba3(diff_filtered);
  viewerd->addPointCloud<PointT> (diff_filtered, rgba3, "diff filtered point cloud");
  viewerd->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "diff filtered point cloud");
  viewerd->initCameraParameters ();


  

  // PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
  // viewer->addPointCloud (voxel_centroid_cloud, "voxel centroids");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "voxel centroids");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.95, "voxel centroids");

  PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
  viewer->addPointCloud (labeled_voxel_cloud, "labeled voxels");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

  PointNCloudT::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);


  //Save the position of the camera
  // std::vector<pcl::visualization::Camera> cam;           
  // viewer->getCameras(cam); 

  //Print recorded points on the screen: 
  // cout << "Cam: " << endl 
  //              << " - positionned at: (" << cam[0].pos[0] << ", "    << cam[0].pos[1] << ", "    << cam[0].pos[2] << ")" << endl 
  //              << " - looking at: ("   << cam[0].focal[0] << ", "  << cam[0].focal[1] << ", "  << cam[0].focal[2] << ")" << endl 
  //              << " - up vector: ("    << cam[0].view[0] << ", "   << cam[0].view[1] << ", "   << cam[0].view[2] << ")"  << endl;

  // Set the camera position
  viewer->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);
  viewerb->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);
  viewerc->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);
  viewerd->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);

  // Everything above acounts for one time step (an over segmentation of the scene)
  // We will now update the previous octree with the new input cloud according to the rules given in the article
  // Then we need to show it again (update the viewer)
  while (!viewer->wasStopped () && !viewerb->wasStopped () && !viewerc->wasStopped ())
  {
    viewer->spinOnce (100);
    viewerb->spinOnce (100);
    viewerc->spinOnce (100);
    viewerd->spinOnce (100);
  }
  return (0);
}