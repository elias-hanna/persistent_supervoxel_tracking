
/*
  * Software License Agreement (BSD License)
  *
  *  Point Cloud Library (PCL) - www.pointclouds.org
  *
  *  All rights reserved.
  *
  *  Redistribution and use in source and binary forms, with or without
  *  modification, are permitted provided that the following conditions
  *  are met:
  *
  *   * Redistributions of source code must retain the above copyright
  *     notice, this list of conditions and the following disclaimer.
  *   * Redistributions in binary form must reproduce the above
  *     copyright notice, this list of conditions and the following
  *     disclaimer in the documentation and/or other materials provided
  *     with the distribution.
  *   * Neither the name of Willow Garage, Inc. nor the names of its
  *     contributors may be used to endorse or promote products derived
  *     from this software without specific prior written permission.
  *
  *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  *  POSSIBILITY OF SUCH DAMAGE.
  *
  * Author : jpapon@gmail.com
  * Email  : jpapon@gmail.com
  *
  */

#ifndef PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_
#define PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_

//#include "supervoxel_clustering.h"
#include <pcl/segmentation/supervoxel_clustering.h>
#include "../octree/octree_pointcloud_sequential.h"
#include <tbb/tbb.h>
//#include <boost/make_shared.hpp>

namespace pcl
{
  /** \brief Supervoxel container class - stores a cluster extracted using supervoxel clustering
   */
  template <typename PointT>
  class SequentialSV : public Supervoxel<PointT>
  {
    public:
      using Supervoxel<PointT>::Supervoxel;

      typedef boost::shared_ptr<SequentialSV> Ptr;
      typedef boost::shared_ptr<const SequentialSV> ConstPtr;

      /** \brief The normal calculated for the voxels contained in the supervoxel */
      //            pcl::Normal normal_;
      using Supervoxel<PointT>::normal_;
      /** \brief The centroid of the supervoxel - average voxel */
      //            pcl::PointXYZRGBA centroid_;
      using Supervoxel<PointT>::centroid_;
      /** \brief A Pointcloud of the voxels in the supervoxel */
      //            typename pcl::PointCloud<PointT>::Ptr voxels_;
      using Supervoxel<PointT>::voxels_;
      /** \brief A Pointcloud of the normals for the points in the supervoxel */
      //            typename pcl::PointCloud<Normal>::Ptr normals_;
      using Supervoxel<PointT>::normals_;

      //            typedef pcl::PointXYZRGBNormal CentroidT;
      //            typedef pcl::PointXYZRGBNormal VoxelT;


      //            /** \brief The centroid of the supervoxel */
      //            using Supervoxel::centroid_;
      //            /** \brief The label ID of this supervoxel */
      //            using Supervoxel::label_;
      //            /** \brief A Pointcloud of the voxels in the supervoxel */
      //            using Supervoxel::voxels_;

      //            SequentialSV (uint32_t label = 0) :
      //                Supervoxel (label)
      //            {  }

      //            //! \brief Maps voxel index to measured weight - used by tracking
      //            std::map <size_t, float> voxel_weight_map_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  /** \brief NEW MESSAGE
   *  \author Jeremie Papon (jpapon@gmail.com)
   *  \ingroup segmentation
   */
  template <typename PointT>
  class PCL_EXPORTS SequentialSVClustering : public pcl::PCLBase<PointT>
  {
      class SequentialSupervoxelHelper;
      friend class SequentialSupervoxelHelper;
    public:
      /** \brief VoxelData is a structure used for storing data within a pcl::octree::OctreePointCloudAdjacencyContainer
       *  \note It stores xyz, rgb, normal, distance, an index, and an owner.
       */
      class SequentialVoxelData : public SupervoxelClustering<PointT>::VoxelData
      {
        public:
          SequentialVoxelData ():
            new_leaf_ (true),
            has_changed_ (false),
            frame_occluded_ (0),
            label_ (-1)
          {
            idx_ = -1;
            // Initialize previous state of the voxel
            previous_xyz_ = xyz_;
            previous_rgb_ = rgb_;
            previous_normal_ = normal_;
          }

          bool
          isNew () const { return new_leaf_; }

          void
          setNew (bool new_arg) { new_leaf_ = new_arg; }

          bool
          isChanged () const { return has_changed_; }

          void
          setChanged (bool new_val) { has_changed_ = new_val; }

          void
          prepareForNewFrame ()
          {
            new_leaf_ = false;
            has_changed_ = false;
//            idx_ = -1;
            // Update the previous state of the voxel
            previous_xyz_ = xyz_;
            previous_rgb_ = rgb_;
            previous_normal_ = normal_;
            xyz_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            rgb_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            normal_ = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
            // owner_ = 0;
          }

          void
          revertToLastPoint ()
          {
            xyz_ = previous_xyz_;
            rgb_ = previous_rgb_;
            normal_ = previous_normal_;
          }

          void
          initLastPoint ()
          {
            previous_xyz_ = xyz_;
            previous_rgb_ = rgb_;
            previous_normal_ = normal_;
          }

          // Use the methods from VoxelData
          using SupervoxelClustering<PointT>::VoxelData::getPoint;
          using SupervoxelClustering<PointT>::VoxelData::getNormal;

          // Use the attributes from VoxelData
          using SupervoxelClustering<PointT>::VoxelData::idx_;
          using SupervoxelClustering<PointT>::VoxelData::xyz_;
          using SupervoxelClustering<PointT>::VoxelData::rgb_;
          using SupervoxelClustering<PointT>::VoxelData::normal_;
          using SupervoxelClustering<PointT>::VoxelData::curvature_;
          using SupervoxelClustering<PointT>::VoxelData::distance_;

          // Used by the difference function
          Eigen::Vector3f previous_xyz_;
          Eigen::Vector3f previous_rgb_;
          Eigen::Vector4f previous_normal_;

          // New attributes
          bool has_changed_, new_leaf_;
          int frame_occluded_;
          int label_;
          SequentialSupervoxelHelper* owner_;

        public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      };

      typedef pcl::octree::OctreePointCloudSequentialContainer<PointT, SequentialVoxelData> LeafContainerT;
      typedef std::vector <LeafContainerT*> LeafVectorT;
      typedef std::map<uint32_t,typename Supervoxel<PointT>::Ptr> SupervoxelMapT;
      typedef std::map<uint32_t,typename SequentialSV<PointT>::Ptr> SequentialSVMapT;

      typedef typename pcl::PointCloud<PointT> PointCloudT;
      typedef typename pcl::PointCloud<Normal> NormalCloudT;
      typedef typename pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT> OctreeSequentialT;
      typedef typename pcl::octree::OctreePointCloudSearch <PointT> OctreeSearchT;
      typedef typename pcl::search::KdTree<PointT> KdTreeT;
      typedef boost::shared_ptr<std::vector<int> > IndicesPtr;

      using pcl::PCLBase <PointT>::initCompute;
      using pcl::PCLBase <PointT>::deinitCompute;
      using pcl::PCLBase <PointT>::input_;

      typedef boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, uint32_t, float> VoxelAdjacencyList;
      typedef VoxelAdjacencyList::vertex_descriptor VoxelID;
      typedef VoxelAdjacencyList::edge_descriptor EdgeID;

    public:
      /** \brief Constructor that sets default values for member variables.
       *  \param[in] voxel_resolution The resolution (in meters) of voxels used
       *  \param[in] seed_resolution The average size (in meters) of resulting supervoxels
       *  \param[in] use_single_camera_transform Set to true if point density in cloud falls off with distance from origin (such as with a cloud coming from one stationary camera), set false if input cloud is from multiple captures from multiple locations.
       */
      SequentialSVClustering (float voxel_resolution, float seed_resolution, bool use_single_camera_transform = true, bool prune_close_seeds=true);

      /** \brief This destructor destroys the cloud, normals and search method used for
       * finding neighbors. In other words it frees memory.
       */
      virtual
      ~SequentialSVClustering ();

      /** \brief Set the resolution of the octree voxels */
      void
      setVoxelResolution (float resolution);

      /** \brief Get the resolution of the octree voxels */
      float
      getVoxelResolution () const;

      /** \brief Set the resolution of the octree seed voxels */
      void
      setSeedResolution (float seed_resolution);

      /** \brief Get the resolution of the octree seed voxels */
      float
      getSeedResolution () const;

      /** \brief Set the importance of color for supervoxels */
      void
      setColorImportance (float val);

      /** \brief Set the importance of spatial distance for supervoxels */
      void
      setSpatialImportance (float val);

      /** \brief Set the importance of scalar normal product for supervoxels */
      void
      setNormalImportance (float val);

      /** \brief Set whether or not to use the single camera transform
       *  \note By default it will be used for organized clouds, but not for unorganized - this parameter will override that behavior
       *  The single camera transform scales bin size so that it increases exponentially with depth (z dimension).
       *  This is done to account for the decreasing point density found with depth when using an RGB-D camera.
       *  Without the transform, beyond a certain depth adjacency of voxels breaks down unless the voxel size is set to a large value.
       *  Using the transform allows preserving detail up close, while allowing adjacency at distance.
       *  The specific transform used here is:
       *  x /= z; y /= z; z = ln(z);
       *  This transform is applied when calculating the octree bins in OctreePointCloudAdjacency
       */
      void
      setUseSingleCameraTransform (bool val);

      /** \brief Set to ignore input normals and calculate normals internally
       *  \note Default is False - ie, SupervoxelClustering will use normals provided in PointT if there are any
       *  \note You should only need to set this if eg PointT=PointXYZRGBNormal but you don't want to use the normals it contains
       */
      void
      setIgnoreInputNormals (bool val);

      /** \brief Returns the current maximum (highest) label */
      int
      getMaxLabel () const;

      /** \brief This method launches the segmentation algorithm and returns the supervoxels that were
       * obtained during the segmentation.
       * \param[out] supervoxel_clusters A map of labels to pointers to supervoxel structures
       */
      virtual void
      extract (std::map<uint32_t,typename SequentialSV<PointT>::Ptr > &supervoxel_clusters);

      /** \brief This method sets the cloud to be supervoxelized
       * \param[in] cloud The cloud to be supervoxelize
       */
      virtual void
      setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr& cloud);

      /** \brief This method sets the normals to be used for supervoxels (should be same size as input cloud)
      * \param[in] normal_cloud The input normals
      */
      virtual void
      setNormalCloud (typename NormalCloudT::ConstPtr normal_cloud);

      pcl::PointCloud<pcl::PointXYZL>::Ptr
      getLabeledCloud () const;

      pcl::PointCloud<pcl::PointXYZL>::Ptr
      getLabeledVoxelCloud () const;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getColoredVoxelCloud () const;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getColoredCloud () const;

      /** \brief Returns a deep copy of the voxel centroid cloud */
      typename pcl::PointCloud<PointT>::Ptr
      getVoxelCentroidCloud () const;

      /** \brief Returns a deep copy of the voxel centroid cloud */
      typename pcl::PointCloud<PointT>::Ptr
      getUnlabeledVoxelCentroidCloud () const;

      /** \brief Gets the adjacency list (Boost Graph library) which gives connections between supervoxels
       *  \param[out] adjacency_list_arg BGL graph where supervoxel labels are vertices, edges are touching relationships
       */
      void
      getSupervoxelAdjacency (std::multimap<uint32_t, uint32_t> &label_adjacency) const;

    private:
      /** \brief This method initializes the label_colors_ vector (assigns random colors to labels)
       * \note Checks to see if it is already big enough - if so, does not reinitialize it
       */
      void
      initializeLabelColors ();

      /** \brief Init the computation, update the sequential octree, perform global check to see wether supervoxel have changed
       * more than their half and finally compute the voxel data to be used to determine the supervoxel seeds
       */
      void
      buildVoxelCloud ();

      /** \brief Update the sequential octree, perform global check to see wether supervoxel have changed
       * more than their half and finally compute the voxel data to be used to determine the supervoxel seeds
       */
      bool
      prepareForSegmentation ();

      /** \brief This method unlabels changed voxels between two frames and also unlabel more than half
       * changing supervoxels
       */
      void
      globalCheck ();

      /** \brief Compute the voxel data (index of each voxel in the octree and normal of each voxel) */
      void
      computeVoxelData ();

      /** \brief This method compute the normal of each leaf belonging to the sequential octree
       */
      void
      parallelComputeNormals ();

      /** \brief Distance function used for comparing voxelDatas */
      float
      sequentialVoxelDataDistance (const SequentialVoxelData &v1, const SequentialVoxelData &v2) const;

      /** \brief Transform function used to normalize voxel density versus distance from camera */
      void
      transformFunction (PointT &p);

      /** \brief This selects points to use as initial supervoxel centroids
       *  \param[out] seed_indices The selected leaf indices
       */
      void
      selectInitialSupervoxelSeeds (std::vector<int> &seed_indices);

      /** \brief This roughly founds the same seeding points as those from the previous frame
       *  \param[out] existing_seed_indices The selected leaf indices
       */
      void
      getPreviousSeedingPoints (SequentialSVMapT &supervoxel_clusters, std::vector<int> &existing_seed_indices);

      /** \brief This method finds seeding points, then prune the seeds that are too close to existing ones
       * and stores the resulting seeds in seed_indices
       */
      void
      pruneSeeds (std::vector<int> &existing_seed_indices, std::vector<int> &seed_indices);

      void
      clearOwnersSetCentroids ();

      /** \brief This performs the superpixel evolution */
      void
      expandSupervoxels ( int depth );

      /** \brief This method appends internal supervoxel helpers to the list based on the provided seed points
       *  \param[in] seed_indices Indices of the leaves to use as seeds
       */
      void
      appendHelpersFromSeedIndices (std::vector<int> &seed_indices);

      /** \brief Constructs the map of supervoxel clusters from the internal supervoxel helpers */
      void
      makeSupervoxels (std::map<uint32_t,typename SequentialSV<PointT>::Ptr > &supervoxel_clusters);

      void
      createHelpersFromSeedIndices (std::vector<int> &seed_indices);

      void
      addHelpersFromUnlabeledSeedIndices(std::vector<int> &seed_indices);

      std::vector<int>
      getAvailableLabels ();

      /** \brief Stores the resolution used in the octree */
      float resolution_;

      /** \brief Stores the resolution used to seed the superpixels */
      float seed_resolution_;

      /** \brief Contains a KDtree for the voxelized cloud */
      typename pcl::search::KdTree<PointT>::Ptr voxel_kdtree_;

      /** \brief Stores the colors used for the superpixel labels*/
      std::vector<uint32_t> label_colors_;

      /** \brief Octree Sequential structure with leaves at voxel resolution */
      typename OctreeSequentialT::Ptr sequential_octree_;

      /** \brief Contains the Voxelized centroid cloud of the unlabeled voxels */
      typename PointCloudT::Ptr unlabeled_voxel_centroid_cloud_;

      /** \brief Contains the Voxelized centroid Cloud */
      typename PointCloudT::Ptr voxel_centroid_cloud_;

      /** \brief Contains the Normals of the input Cloud */
      typename NormalCloudT::ConstPtr input_normals_;

      /** \brief Importance of color in clustering */
      float color_importance_;

      /** \brief Importance of distance from seed center in clustering */
      float spatial_importance_;

      /** \brief Importance of similarity in normals for clustering */
      float normal_importance_;

      /** \brief Option to ignore normals in input Pointcloud. Defaults to false */
      bool ignore_input_normals_;

      /** \brief Whether or not to use the transform compressing depth in Z
       *  This is only checked if it has been manually set by the user.
       *  The default behavior is to use the transform for organized, and not for unorganized.
       */
      bool use_single_camera_transform_;

      /** \brief Whether to use default transform behavior or not */
      bool use_default_transform_behaviour_;



      bool prune_close_seeds_;

      pcl::StopWatch timer_;
      boost::mutex mutex_normals_;

      int nb_of_unlabeled_voxels_;

      /** \brief Internal storage class for supervoxels
       * \note Stores pointers to leaves of clustering internal octree,
       * \note so should not be used outside of clustering class
       */
      class SequentialSupervoxelHelper
      {
        public:

          /** \brief Comparator for LeafContainerT pointers - used for sorting set of leaves
         * \note Compares by index in the overall leaf_vector. Order isn't important, so long as it is fixed.
         */
          struct compareLeaves
          {
              bool operator() (LeafContainerT* const &left, LeafContainerT* const &right) const
              {
                const SequentialVoxelData& leaf_data_left = left->getData ();
                const SequentialVoxelData& leaf_data_right = right->getData ();
                return leaf_data_left.idx_ < leaf_data_right.idx_;
              }
          };
          typedef std::set<LeafContainerT*, typename SequentialSupervoxelHelper::compareLeaves> LeafSetT;
          typedef typename LeafSetT::iterator iterator;
          typedef typename LeafSetT::const_iterator const_iterator;

          SequentialSupervoxelHelper (uint32_t label, SequentialSVClustering* parent_arg):
            label_ (label),
            parent_ (parent_arg)
          { }

          void
          addLeaf (LeafContainerT* leaf_arg);

          void
          removeLeaf (LeafContainerT* leaf_arg);

          void
          removeAllLeaves ();

          void
          expand ();

          void
          updateCentroid ();

          void
          getVoxels (typename pcl::PointCloud<PointT>::Ptr &voxels) const;

          void
          getNormals (typename pcl::PointCloud<Normal>::Ptr &normals) const;

          typedef float (SequentialSVClustering::*DistFuncPtr)(const SequentialVoxelData &v1, const SequentialVoxelData &v2);

          uint32_t
          getLabel () const
          { return label_; }

          Eigen::Vector4f
          getNormal () const
          { return centroid_.normal_; }

          Eigen::Vector3f
          getRGB () const
          { return centroid_.rgb_; }

          Eigen::Vector3f
          getXYZ () const
          { return centroid_.xyz_;}

          void
          getXYZ (float &x, float &y, float &z) const
          { x=centroid_.xyz_[0]; y=centroid_.xyz_[1]; z=centroid_.xyz_[2]; }

          void
          getRGB (uint32_t &rgba) const
          {
            rgba = static_cast<uint32_t>( centroid_.rgb_[0]) << 16 |
                                                                static_cast<uint32_t>(centroid_.rgb_[1]) << 8 |
                                                                                                            static_cast<uint32_t>(centroid_.rgb_[2]);
          }

          void
          getNormal (pcl::Normal &normal_arg) const
          {
            normal_arg.normal_x = centroid_.normal_[0];
            normal_arg.normal_y = centroid_.normal_[1];
            normal_arg.normal_z = centroid_.normal_[2];
            normal_arg.curvature = centroid_.curvature_;
          }

          void
          getNeighborLabels (std::set<uint32_t> &neighbor_labels) const;

          SequentialVoxelData
          getCentroid () const
          {
            return centroid_;
          }

          size_t
          size () const { return leaves_.size (); }
        private:
          //Stores leaves
          LeafSetT leaves_;
          uint32_t label_;
          SequentialVoxelData centroid_;
          SequentialSVClustering* parent_;
        public:
          //Type VoxelData may have fixed-size Eigen objects inside
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      };

      //Make boost::ptr_list can access the private class SupervoxelHelper
      friend void boost::checked_delete<> (const typename pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper *);

      typedef boost::ptr_list<SequentialSupervoxelHelper> HelperListT;
      HelperListT supervoxel_helpers_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

}

#ifdef PCL_NO_PRECOMPILE
#include "impl/sequential_supervoxel_clustering.hpp"
#endif

#endif //PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_
