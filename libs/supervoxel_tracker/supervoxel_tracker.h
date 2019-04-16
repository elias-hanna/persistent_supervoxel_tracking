#ifndef SUPERVOXEL_TRACKER_H_
#define SUPERVOXEL_TRACKER_H_

// STL Includes
#include <map>
#include <utility>
// PCL Common Includes
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
// PCL Tracker Includes
#include <pcl/tracking/tracker.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/passthrough.h>
// Local libairies Includes
#include "../papon/supervoxel/sequential_supervoxel_clustering.h"


//template <typename TrackerT, typename PointT, typename StateT>
//template <typename PointT, typename StateT>
template <typename StateT>
class SupervoxelTracker
{
    public:
        typedef pcl::PointXYZRGBNormal PointT;
        typedef pcl::PointCloud<PointT> PointCloudT;
        typedef PointCloudT::Ptr PointCloudPtrT;
        typedef PointCloudT::ConstPtr PointCloudConstPtrT;
        typedef pcl::tracking::ParticleFilterTracker< PointT, StateT> TrackerT;
        typedef std::map< uint32_t, TrackerT* > TrackerMapT;

    private:
        TrackerMapT trackers_;

    public:
        /** \brief Default constructor
         */
        SupervoxelTracker ()
        {

        }
        /** \brief Constructor that initialize the trackers with supervoxels clusters
         */
        SupervoxelTracker (std::map < uint32_t, pcl_papon::SequentialSV::Ptr> supervoxel_clusters)
        {
            setReferenceClouds(supervoxel_clusters);
        }
        /** \brief This method is used to replace all the trackers to follow new reference supervoxels
         * \note Should be use wether at initialisation or to reset the trackers with new reference supervoxels
         */
        void setReferenceClouds (std::map < uint32_t, pcl_papon::SequentialSV::Ptr> supervoxel_clusters);
        /** \brief This method is used to add/replace the reference cloud of a supervoxel label
         * \note this method creates a new tracker for the concerned label
         */
        void addReferenceCloud(uint32_t label, PointCloudPtrT target_cloud);
        /** \brief This method is used to delete a tracker of a specified supervoxel by label
         */
        void deleteTrackerAt(uint32_t label);
        /** \brief This method is used to get all the predicted states from the particle filters for each supervoxel
         */
        std::map< uint32_t, StateT> track(PointCloudConstPtrT cloud);


    private:
        void gridSampleApprox (const PointCloudConstPtrT &cloud, PointCloudT &result, double leaf_size);
        //Filter along a specified dimension
        void filterPassThrough (const PointCloudConstPtrT& cloud, PointCloudT& result);
};

#endif
