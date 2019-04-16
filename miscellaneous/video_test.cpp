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
// Macros
#define N_DATA 3
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

int main (int argc, char ** argv)
{
  std::vector<PointCloudT::Ptr> clouds;

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

  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));






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

  // Create a supervoxel clustering instance
  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
  super.setColorImportance(color_importance);
  super.setSpatialImportance(spatial_importance);
  super.setNormalImportance(normal_importance);



  int i = 0;
  while (!viewer->wasStopped ())
  {

    super.setInputCloud(clouds[i%N_DATA]);

    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;

    pcl::console::print_highlight ("Extracting supervoxels!\n");
    super.extract (supervoxel_clusters);
    pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

    PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
    viewer->addPointCloud (voxel_centroid_cloud, "voxel centroids");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "voxel centroids");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.95, "voxel centroids");

    PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
    viewer->addPointCloud (labeled_voxel_cloud, "labeled voxels");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

    viewer->spinOnce (100);

    // viewer->removePointCloud("point cloud");
    viewer->removePointCloud("voxel centroids");
    viewer->removePointCloud("labeled voxels");



    // viewer->setBackgroundColor(0, 0, 0);
    // pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgba(clouds[i%N_DATA]);
    // viewer->addPointCloud<PointT> (clouds[i%N_DATA], rgba, "point cloud");
    // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
    // viewer->initCameraParameters ();

    // // Set the camera position
    // viewer->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);

    // // Spin once
    // viewer->spinOnce (500);
    // viewer->removePointCloud("point cloud");

    i++;
  }
  return (0);
}