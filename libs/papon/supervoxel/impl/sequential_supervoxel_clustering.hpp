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

#ifndef PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_HPP_
#define PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_HPP_

#include "../sequential_supervoxel_clustering.h"
#include <boost/interprocess/sync/scoped_lock.hpp>

#define NUM_THREADS 1

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::SequentialSVClustering<PointT>::SequentialSVClustering (float voxel_resolution, float seed_resolution, bool use_single_camera_transform, bool prune_close_seeds) :
  resolution_ (voxel_resolution),
  seed_resolution_ (seed_resolution),
  voxel_centroid_cloud_ (),
  color_importance_ (0.1f),
  spatial_importance_ (0.4f),
  normal_importance_ (1.0f),
  ignore_input_normals_ (false),
  prune_close_seeds_ (prune_close_seeds),
  label_colors_ (0),
  use_single_camera_transform_ (use_single_camera_transform),
  use_default_transform_behaviour_ (true),
  nb_of_unlabeled_voxels_ (0)
{
  sequential_octree_.reset (new OctreeSequentialT (resolution_));
  if (use_single_camera_transform_)
    sequential_octree_->setTransformFunction (boost::bind (&SequentialSVClustering::transformFunction, this, _1));
  initializeLabelColors ();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr& cloud)
{
  if ( cloud->size () == 0 )
  {
    PCL_ERROR ("[pcl::SequentialSVClustering::setInputCloud] Empty cloud set, doing nothing \n");
    return;
  }

  input_ = cloud;
  if (sequential_octree_->size() == 0)
  {
    sequential_octree_.reset (new OctreeSequentialT (resolution_));
    if ( (use_default_transform_behaviour_ && input_->isOrganized ())
         || (!use_default_transform_behaviour_ && use_single_camera_transform_))
    {
      sequential_octree_->setTransformFunction (boost::bind (&SequentialSVClustering::transformFunction, this, _1));
    }

    sequential_octree_->setDifferenceFunction (boost::bind (&OctreeSequentialT::SeqVoxelDataDiff, _1));
    sequential_octree_->setDifferenceThreshold (0.2);
    sequential_octree_->setNumberOfThreads (NUM_THREADS);
    sequential_octree_->setOcclusionTestInterval (0.25f);
    sequential_octree_->setInputCloud (cloud);
    sequential_octree_->defineBoundingBox();//OnInputCloud ();
  }
  else
  {
    sequential_octree_->setInputCloud (cloud);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setNormalCloud (typename NormalCloudT::ConstPtr normal_cloud)
{
  if ( normal_cloud->size () == 0 )
  {
    PCL_ERROR ("[pcl::SequentialSVClustering::setNormalCloud] Empty cloud set, doing nothing \n");
    return;
  }

  input_normals_ = normal_cloud;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::SequentialSVClustering<PointT>::~SequentialSVClustering ()
{

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::extract (std::map<uint32_t,typename SequentialSV<PointT>::Ptr > &supervoxel_clusters)
{
  timer_.reset ();
  double t_start = timer_.getTime ();

  unsigned long nb_previous_supervoxel_clusters = supervoxel_clusters.size();

  buildVoxelCloud();

  double t_update = timer_.getTime ();

  std::vector<int> seed_indices, existing_seed_indices;

  if(false)//nb_previous_supervoxel_clusters == 0)
  {
    selectInitialSupervoxelSeeds (seed_indices);
    createHelpersFromSeedIndices (seed_indices);
  }
  else
  {
    getPreviousSeedingPoints(supervoxel_clusters, existing_seed_indices);
    pruneSeeds(existing_seed_indices, seed_indices);
    addHelpersFromUnlabeledSeedIndices(seed_indices);
  }
  double t_seeds = timer_.getTime ();

  int max_depth = static_cast<int> (sqrt(2)*seed_resolution_/resolution_);
  expandSupervoxelsFast (max_depth);

  double t_iterate = timer_.getTime ();

  makeSupervoxels (supervoxel_clusters);

  deinitCompute ();

  // Time computation
  double t_supervoxels = timer_.getTime ();

  std::cout << "--------------------------------- Timing Report --------------------------------- \n";
  std::cout << "Time to update octree                          ="<<t_update-t_start<<" ms\n";
  std::cout << "Time to seed clusters                          ="<<t_seeds-t_update<<" ms\n";
  std::cout << "Time to expand clusters                        ="<<t_iterate-t_seeds<<" ms\n";
  std::cout << "Time to create supervoxel structures           ="<<t_supervoxels-t_iterate<<" ms\n";
  std::cout << "Total run time                                 ="<<t_supervoxels-t_start<<" ms\n";
  std::cout << "--------------------------------------------------------------------------------- \n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::buildVoxelCloud ()
{
  //    std::cout << "Building voxel cloud..." << std::endl;
  //    std::cout << "Initializing computation..." << std::endl;
  bool segmentation_is_possible = initCompute ();
  if ( !segmentation_is_possible )
  {
    PCL_ERROR ("[pcl::SequentialSVClustering::initCompute] Init failed.\n");
    deinitCompute ();
    return;
  }
  //    std::cout << "Initialization complete...\nPreparing for segmentation..." << std::endl;
  //std::cout << "Preparing for segmentation \n";
  segmentation_is_possible = prepareForSegmentation ();
  if ( !segmentation_is_possible )
  {
    PCL_ERROR ("[pcl::SequentialSVClustering::prepareForSegmentation] Building of voxel cloud failed.\n");
    deinitCompute ();
    return;
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PointT> bool
pcl::SequentialSVClustering<PointT>::prepareForSegmentation ()
{

  // if user forgot to pass point cloud or if it is empty
  if ( input_->points.size () == 0 )
    return (false);

  //Add the new cloud of data to the octree
  //std::cout << "Populating adjacency octree with new cloud \n";
  //double prep_start = timer_.getTime ();
  double update_start = timer_.getTime ();
  sequential_octree_->addPointsFromInputCloud ();
  double update_end = timer_.getTime ();
  std::cout << "Time elapsed adding points to octree= "<<update_end-update_start<<" ms\n";
  // GlobalCheck unlabels the voxels that changed and update nb_of_unlabeled_voxels_
  globalCheck();
  //double prep_end = timer_.getTime ();
  //std::cout<<"Time elapsed populating octree with next frame ="<<prep_end-prep_start<<" ms\n";
  //Compute normals and insert data for centroids into data field of octree
  double normals_start = timer_.getTime ();

  computeVoxelData ();
  double normals_end = timer_.getTime ();
  std::cout << "Time elapsed finding normals= "<<normals_end-normals_start<<" ms\n";

  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::computeVoxelData ()
{
  // Updating the unlabeled voxel centroid cloud (used for seeding)
  voxel_centroid_cloud_.reset (new PointCloudT);
  voxel_centroid_cloud_->resize (sequential_octree_->getLeafCount ());
  unlabeled_voxel_centroid_cloud_.reset (new PointCloudT);
  unlabeled_voxel_centroid_cloud_->resize (nb_of_unlabeled_voxels_);
  typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin ();
  typename PointCloudT::iterator cent_cloud_itr = voxel_centroid_cloud_->begin ();
  typename PointCloudT::iterator un_cent_cloud_itr = unlabeled_voxel_centroid_cloud_->begin ();
  std::cout << "START" << std::endl;
  for (int idx = 0 ; leaf_itr != sequential_octree_->end (); ++leaf_itr, ++cent_cloud_itr, ++idx)
  {
    SequentialVoxelData& new_voxel_data = (*leaf_itr)->getData ();
//    std::cout << new_voxel_data.xyz_ << std::endl;
    // Add the point to the centroid cloud
    new_voxel_data.getPoint (*cent_cloud_itr);
    // Push correct index in
    new_voxel_data.idx_ = idx;
    if(new_voxel_data.label_ == -1)
    {
      // Add the point to unlabelized the centroid cloud
      new_voxel_data.getPoint (*un_cent_cloud_itr);
      ++un_cent_cloud_itr;
    }
  }
  std::cout << "END" << std::endl;
  // If input point type does not have normals, we need to compute them
  if (!pcl::traits::has_normal<PointT>::value || ignore_input_normals_)
  {
    cent_cloud_itr = voxel_centroid_cloud_->begin ();
    un_cent_cloud_itr = unlabeled_voxel_centroid_cloud_->begin ();
    parallelComputeNormals ();

    // Fill the unlabeled voxel cloud
    for (leaf_itr = sequential_octree_->begin () ; leaf_itr != sequential_octree_->end () ; ++leaf_itr, ++cent_cloud_itr)
    {
      SequentialVoxelData& new_voxel_data = (*leaf_itr)->getData ();
      //Add the point to the centroid cloud
      if(new_voxel_data.label_ == -1)
      {
        //Add the point to unlabelized the centroid cloud
        new_voxel_data.getPoint (*un_cent_cloud_itr);
        ++un_cent_cloud_itr;
      }
//      std::cout << new_voxel_data.normal_ << std::endl;
    }
  }
  //Update kdtree now that we have updated centroid cloud
  voxel_kdtree_.reset (new pcl::search::KdTree<PointT>);
  voxel_kdtree_ ->setInputCloud (voxel_centroid_cloud_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::parallelComputeNormals ()
{
  tbb::parallel_for(tbb::blocked_range<int>(0, sequential_octree_->getLeafCount ()),
                    [=](const tbb::blocked_range<int>& r)
  {
    for (int idx = r.begin () ; idx != r.end (); ++idx)
    {
      LeafContainerT* leaf = sequential_octree_->at (idx);
      SequentialVoxelData& new_voxel_data = leaf->getData();
      //For every point, get its neighbors, build an index vector, compute normal
      std::vector<int> indices;
      indices.reserve (81);
      //Push this point
      indices.push_back (new_voxel_data.idx_);//or just idx ?
      for (typename LeafContainerT::const_iterator neighb_itr= leaf->cbegin (); neighb_itr!= leaf->cend (); ++neighb_itr)
      {
        SequentialVoxelData& neighb_voxel_data = (*neighb_itr)->getData ();
        //Push neighbor index
        indices.push_back (neighb_voxel_data.idx_);
        //Get neighbors neighbors, push onto cloud
        for (typename LeafContainerT::const_iterator neighb_neighb_itr=(*neighb_itr)->cbegin (); neighb_neighb_itr!=(*neighb_itr)->cend (); ++neighb_neighb_itr)
        {
          SequentialVoxelData& neighb2_voxel_data = (*neighb_neighb_itr)->getData ();
          indices.push_back (neighb2_voxel_data.idx_);
        }
      }
      //Compute normal
      //Is this temp vector really the only way to do this? Why can't I substitute vectormap for vector?
      pcl::computePointNormal (*voxel_centroid_cloud_, indices, new_voxel_data.normal_, new_voxel_data.curvature_);
      pcl::flipNormalTowardsViewpoint (voxel_centroid_cloud_->points[new_voxel_data.idx_], 0.0f,0.0f,0.0f, new_voxel_data.normal_);
      new_voxel_data.normal_[3] = 0.0f;
      new_voxel_data.normal_.normalize ();
      //      new_voxel_data.owner_ = 0;
      //      new_voxel_data.distance_ = std::numeric_limits<float>::max ();
      {
        //        boost::interprocess::scoped_lock<boost::mutex> lock(mutex_normals_);
        //        new_voxel_data.getPoint (voxel_centroid_cloud_->at (idx));
      }
    }
  }
  );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::selectInitialSupervoxelSeeds (std::vector<int> &seed_indices)
{
  //TODO THIS IS BAD - SEEDING SHOULD BE BETTER
  //TODO Switch to assigning leaves! Don't use Octree!

  // std::cout << "Size of centroid cloud="<<voxel_centroid_cloud_->size ()<<", seeding resolution="<<seed_resolution_<<"\n";
  //Initialize octree with voxel centroids
  pcl::octree::OctreePointCloudSearch <PointT> seed_octree (seed_resolution_);
  seed_octree.setInputCloud (voxel_centroid_cloud_);
  seed_octree.addPointsFromInputCloud ();
  // std::cout << "Size of octree ="<<seed_octree.getLeafCount ()<<"\n";
  std::vector<PointT, Eigen::aligned_allocator<PointT> > voxel_centers;
  int num_seeds = seed_octree.getOccupiedVoxelCenters(voxel_centers);
  //std::cout << "Number of seed points before filtering="<<voxel_centers.size ()<<std::endl;

  std::vector<int> seed_indices_orig;
  seed_indices_orig.resize (num_seeds, 0);
  seed_indices.clear ();
  std::vector<int> closest_index;
  std::vector<float> distance;
  closest_index.resize(1,0);
  distance.resize(1,0);
  if (voxel_kdtree_ == 0)
  {
    std::cout << "Taille cloud initial: " << voxel_centroid_cloud_->size() << std::endl;
    voxel_kdtree_.reset (new pcl::search::KdTree<PointT>);
    voxel_kdtree_ ->setInputCloud (voxel_centroid_cloud_);
  }

  for (int i = 0; i < num_seeds; ++i)
  {
    // Search for the nearest neighbour to voxel center[i], stores its index in closest_index and distance in distance
    voxel_kdtree_->nearestKSearch (voxel_centers[i], 1, closest_index, distance);
    seed_indices_orig[i] = closest_index[0];
  }

  std::vector<int> neighbors;
  std::vector<float> sqr_distances;
  seed_indices.reserve (seed_indices_orig.size ());
  float search_radius = 0.5f*seed_resolution_;
  // This is 1/20th of the number of voxels which fit in a planar slice through search volume
  // Area of planar slice / area of voxel side. (Note: This is smaller than the value mentioned in the original paper)
  float min_points = 0.05f * (search_radius)*(search_radius) * 3.1415926536f  / (resolution_*resolution_);
  for (size_t i = 0; i < seed_indices_orig.size (); ++i)
  {
    int num = voxel_kdtree_->radiusSearch (seed_indices_orig[i], search_radius , neighbors, sqr_distances);
    int min_index = seed_indices_orig[i];
    if ( num > min_points)
    {
      seed_indices.push_back (min_index);
    }

  }
  // std::cout << "Number of seed points after filtering="<<seed_points.size ()<<std::endl;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::createHelpersFromSeedIndices (std::vector<int> &seed_indices)
{
  supervoxel_helpers_.clear ();
  for (size_t i = 0; i < seed_indices.size (); ++i)
  {
    // Append to the vector of supervoxel helpers a new sv helper corresponding to the considered seed point
    supervoxel_helpers_.push_back (new SequentialSupervoxelHelper(i+1,this));
    // Find which leaf corresponds to this seed index
    LeafContainerT* seed_leaf = sequential_octree_->at(seed_indices[i]);
    if (seed_leaf)
    {
      // Add the seed leaf to the most recent sv helper added (the one that has just been pushed back)
      supervoxel_helpers_.back ().addLeaf (seed_leaf);
    }
    else
    {
      PCL_WARN ("Could not find leaf in pcl::SequentialSVClustering<PointT>::createHelpersFromSeedIndices - supervoxel will be deleted \n");
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::addHelpersFromUnlabeledSeedIndices (std::vector<int> &seed_indices)
{
  std::vector<int> available_labels = getAvailableLabels ();
  int max_label = getMaxLabel();
  for (int i = 0; i < seed_indices.size (); ++i)
  {
    if(!available_labels.empty())
    {
      // Append to the vector of supervoxel helpers a new sv helper corresponding to the considered seed point
      supervoxel_helpers_.push_back (new SequentialSupervoxelHelper(available_labels.back(),this));
      available_labels.pop_back();
    }
    else
    {
      supervoxel_helpers_.push_back (new SequentialSupervoxelHelper(++max_label,this));
    }
    // Find which leaf corresponds to this seed index
    LeafContainerT* seed_leaf = sequential_octree_->at(seed_indices[i]);
    if (seed_leaf)
    {
      // Add the seed leaf to the most recent sv helper added (the one that has just been pushed back)
      supervoxel_helpers_.back ().addLeaf (seed_leaf);
    }
    else
    {
      PCL_WARN ("Could not find leaf in pcl::SequentialSVClustering<PointT>::createHelpersFromSeedIndices - supervoxel will be deleted \n");
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::vector<int>
pcl::SequentialSVClustering<PointT>::getAvailableLabels ()
{
  std::vector<int> available_labels;
  // Fill the vector with 1, 2, ..., max label
  for(int i = 1 ; i < getMaxLabel() ; ++i) { available_labels.push_back(i); }
  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
  {
    available_labels.erase(std::remove(available_labels.begin(), available_labels.end(), sv_itr->getLabel()), available_labels.end());
  }
  return available_labels;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::expandSupervoxelsFast ( int depth )
{
  for (int i = 0; i < depth; ++i)
  {
    //Expand the the supervoxels one iteration each
    for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
    {
      sv_itr->expand ();
    }

    //Update the centers to reflect new centers
    for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); )
    {
      if (sv_itr->size () == 0)
      {
        sv_itr = supervoxel_helpers_.erase (sv_itr);
      }
      else
      {
        sv_itr->updateCentroid ();
        ++sv_itr;
      }
    }

    if ( i < depth - 1 ) //If not on last iteration clear all leaves of ownership
    {
      typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin ();
      for (leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
      {
        SequentialVoxelData& voxel = (*leaf_itr)->getData ();
        voxel.owner_ = 0;
        voxel.label_ = -1;
        voxel.distance_ = std::numeric_limits<float>::max ();
      }
    }

  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::makeSupervoxels (SequentialSVMapT &supervoxel_clusters)
{
  //  supervoxel_clusters.clear ();
  //  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
  //  {
  //    uint32_t label = sv_itr->getLabel ();
  //    std::pair<typename std::map<uint32_t,typename SequentialSV<PointT>::Ptr>::iterator,bool> ret;
  //    ret = supervoxel_clusters.insert (std::pair<uint32_t,typename SequentialSV<PointT>::Ptr> (label, boost::make_shared<SequentialSV<PointT>> (label)));
  //    typename std::map<uint32_t,typename SequentialSV<PointT>::Ptr>::iterator new_supervoxel_itr = ret.first;
  //    new_supervoxel_itr->second->centroid_ = sv_itr->getCentroid ();
  //    sv_itr->getVoxels (new_supervoxel_itr->second->voxels_);
  //  }

  supervoxel_clusters.clear ();
  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
  {
    uint32_t label = sv_itr->getLabel ();
    supervoxel_clusters[label].reset (new SequentialSV<PointT>);
    sv_itr->getXYZ (supervoxel_clusters[label]->centroid_.x,supervoxel_clusters[label]->centroid_.y,supervoxel_clusters[label]->centroid_.z);
    sv_itr->getRGB (supervoxel_clusters[label]->centroid_.rgba);
    sv_itr->getNormal (supervoxel_clusters[label]->normal_);
    sv_itr->getVoxels (supervoxel_clusters[label]->voxels_);
    sv_itr->getNormals (supervoxel_clusters[label]->normals_);
  }
}

template <typename PointT> void
pcl::SequentialSVClustering<PointT>::globalCheck()
{
  nb_of_unlabeled_voxels_ = 0;
  if(getMaxLabel() > 0)
  {
    int nb_voxels_by_labels[getMaxLabel()] = {0};

    for(typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
    {
      SequentialVoxelData& voxel = (*leaf_itr)->getData ();
      if( voxel.label_ == -1)
      {
        ++nb_of_unlabeled_voxels_;
      }
      else if(voxel.isChanged())
      {
        // Minus 1 because labels start at 1"
        --nb_voxels_by_labels[voxel.label_ - 1];
        voxel.label_ = -1;
        ++nb_of_unlabeled_voxels_;
      }
      else
      {
        ++nb_voxels_by_labels[voxel.label_ - 1];
      }
    }

    for(typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
    {
      SequentialVoxelData& voxel = (*leaf_itr)->getData ();
      if(voxel.label_ != -1)
      {
        if(nb_voxels_by_labels[voxel.label_ - 1] < 3)
        {
          voxel.label_ = -1;
          ++nb_of_unlabeled_voxels_;
        }
      }
    }
  }
  else
  {
    nb_of_unlabeled_voxels_ = sequential_octree_->getLeafCount();
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::getPreviousSeedingPoints(SequentialSVMapT &supervoxel_clusters, std::vector<int>& existing_seed_indices)
{
  existing_seed_indices.clear ();
  supervoxel_helpers_.clear ();
  typename SequentialSVMapT::iterator sv_itr;
  // Iterate over all previous supervoxel clusters
  for(sv_itr = supervoxel_clusters.begin (); sv_itr != supervoxel_clusters.end (); ++sv_itr)
  {
    uint32_t label = sv_itr->first;
    // Push back a new supervoxel helper with an already existing label
    supervoxel_helpers_.push_back (new SequentialSupervoxelHelper(label,this));
    // Count the number of points belonging to that supervoxel and compute its centroid
    int sum = 0;
    PointT centroid;
    centroid.getVector3fMap ().setZero ();
    for (typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
    {
      SequentialVoxelData& voxel = (*leaf_itr)->getData ();
      if(voxel.label_ == label)
      {
        centroid.getVector3fMap () += voxel.xyz_;
        sum += 1;
      }
    }
    // If there was points in it, add the closest point in kdtree as the seed point for this supervoxel
    if(sum != 0)
    {
      centroid.getVector3fMap () /= (double)sum;
      std::vector<int> closest_index;
      std::vector<float> distance;
      voxel_kdtree_->nearestKSearch (centroid, 1, closest_index, distance);
      LeafContainerT* seed_leaf = sequential_octree_->at (closest_index[0]);
      if (seed_leaf)
      {
        existing_seed_indices.push_back (closest_index[0]);
        (supervoxel_helpers_.back()).addLeaf(seed_leaf);
      }
      else
      {
        PCL_WARN ("Could not find leaf in pcl::SequentialSVClustering<PointT>::createHelpersFromWeightMaps - supervoxel will be deleted \n");
      }
    }
    // If there was no point in this supervoxel, just remove it
    else
    {
      supervoxel_helpers_.pop_back();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::pruneSeeds(std::vector<int> &existing_seed_indices, std::vector<int> &seed_indices)
{
  //Initialize octree with voxel centroids
  pcl::octree::OctreePointCloudAdjacency<PointT> seed_octree (seed_resolution_);
  //  if (use_single_camera_transform_)
  //    seed_octree.setTransformFunction (boost::bind (&SequentialSVClustering::transformFunction, this, _1));
  seed_octree.setInputCloud (voxel_centroid_cloud_);
  seed_octree.addPointsFromInputCloud ();

  std::vector<PointT, Eigen::aligned_allocator<PointT> > voxel_centers;
  int num_seeds = seed_octree.getOccupiedVoxelCenters(voxel_centers);

  std::vector<int> seed_indices_orig;
  seed_indices_orig.resize (num_seeds, 0);
  seed_indices.clear ();
  std::vector<int> closest_index(1,0);
  std::vector<float> distance(1,0);

  if (voxel_kdtree_ == 0)
  {
    voxel_kdtree_.reset (new pcl::search::KdTree<PointT>);
    voxel_kdtree_ ->setInputCloud (voxel_centroid_cloud_);
  }

  for (int i = 0; i < num_seeds; ++i)
  {
    if (use_single_camera_transform_)
    {
      //Inverse transform the point.
      voxel_centers[i].z = std::exp (voxel_centers[i].z);
      voxel_centers[i].x *= voxel_centers[i].z;
      voxel_centers[i].y *= voxel_centers[i].z;
    }
    voxel_kdtree_->nearestKSearch (voxel_centers[i], 1, closest_index, distance);
    seed_indices_orig[i] = closest_index[0];
  }

  std::vector<int> neighbors;
  std::vector<float> sqr_distances;
  seed_indices.reserve (seed_indices_orig.size ());
  float search_radius = seed_resolution_;
  // This is 1/20th of the number of voxels which fit in a planar slice through search volume
  // Area of planar slice / area of voxel side. (Note: This is smaller than the value mentioned in the original paper)
  float min_points = 0.05f * (search_radius)*(search_radius) * 3.1415926536f  / (resolution_*resolution_);
  for (size_t i = 0; i < seed_indices_orig.size (); ++i)
  {
    int num = voxel_kdtree_->radiusSearch (seed_indices_orig[i], search_radius , neighbors, sqr_distances);
    int min_index = seed_indices_orig[i];
    bool not_too_close = true;
    // For all neighbours
    for(int j = 0 ; j < neighbors.size() ; ++j )
    {
      if(not_too_close)
      {
        // For all existing seed indices
        for(int k = 0 ; k < existing_seed_indices.size() ; ++k)
        {
          if(neighbors[j] == existing_seed_indices[k])
          {
            not_too_close = false;
          }
        }
      }
      else
      {
        break;
      }
    }
    if ( num > min_points && not_too_close)
    {
      seed_indices.push_back (min_index);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::clearOwnersSetCentroids ()
{
  int num_erased = 0;
  //Update centroids and delete empties
  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); )
  {
    if (sv_itr->size () == 0)
    {
      sv_itr = supervoxel_helpers_.erase (sv_itr);
      ++num_erased;
    }
    else
    {
      sv_itr->updateCentroid ();
      ++sv_itr;
    }
  }
  std::cout <<"Erased "<<num_erased<<" Supervoxels!\n";
  //Clear all leaves of ownership
  typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin ();
  for (leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
  {
    SequentialVoxelData& voxel = (*leaf_itr)->getData ();
    voxel.owner_ = 0;
    voxel.distance_ = std::numeric_limits<float>::max ();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::appendHelpersFromSeedIndices (std::vector<int> &seed_indices)
{
  //TODO REMOVE
  int num_helpers_before = supervoxel_helpers_.size ();
  int max_label_before = SequentialSVClustering<PointT>::getMaxLabel ();
  //////////////////////////////////
  int num_not_added = 0;
  uint32_t next_label = 1;
  typename HelperListT::iterator help_itr = supervoxel_helpers_.begin ();
  if (help_itr != supervoxel_helpers_.end ())
    ++help_itr;
  typename HelperListT::iterator help_itr2 = supervoxel_helpers_.begin ();
  for (size_t i = 0; i < seed_indices.size (); ++i)
  {
    //First check if the leaf is already owned - if so, don't add a new seed here.
    LeafContainerT* seed_leaf = sequential_octree_->at(seed_indices[i]);//adjacency_octree_->getLeafContainerAtPoint
    if (seed_leaf)
    {
      SequentialVoxelData& voxel = seed_leaf->getData ();
      if (voxel.owner_ != 0)
      {
        ++num_not_added;
        continue;
      }
    }else
    {
      PCL_WARN ("Could not find leaf in pcl::SequentialSVClustering<PointT>::appendHelpersFromSeedIndices - supervoxel will be deleted \n");
      continue;
    }
    //If we want to add this, need to find a free label
    //The first iterator always stays one ahead - when we find a gap in labels we break
    while (help_itr != supervoxel_helpers_.end () && help_itr->getLabel () == help_itr2->getLabel () + 1)
    {
      ++help_itr; ++help_itr2;
    }
    next_label = help_itr2->getLabel () + 1;
    supervoxel_helpers_.insert (help_itr, new SequentialSupervoxelHelper(next_label,this));
    // this makes itr2 point to the new element
    ++help_itr2;
    help_itr2->addLeaf (seed_leaf);
  }
  std::cout <<"Did not add "<<num_not_added<<" due to already owned!\n";
  //TODO Can remove this - just for debugging
  int num_helpers_after = supervoxel_helpers_.size ();
  int max_label_after = SequentialSVClustering<PointT>::getMaxLabel ();
  std::cout <<"Num SV Before ="<<num_helpers_before<<"  maxlabel="<<max_label_before<<"     Num SV After ="<<num_helpers_after<<"  maxlabel="<<max_label_after<<"   num new seeds="<<seed_indices.size ()<<"\n";
  int empty_slots = max_label_before - num_helpers_before;
  //Either the max label hasn't changed (we only filled in holes), or the max label has increased and there are no holes
  assert ( (max_label_before == max_label_after) || (max_label_after  == (max_label_before + seed_indices.size () - empty_slots - num_not_added)));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::addLeaf (LeafContainerT* leaf_arg)
{
  leaves_.insert (leaf_arg);
  SequentialVoxelData& voxel_data = leaf_arg->getData ();
  voxel_data.owner_ = this;
  voxel_data.label_ = this->getLabel();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::removeLeaf (LeafContainerT* leaf_arg)
{

  leaves_.erase (leaf_arg);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::removeAllLeaves ()
{
  typename SequentialSupervoxelHelper::iterator leaf_itr;
  for (leaf_itr = leaves_.begin (); leaf_itr != leaves_.end (); ++leaf_itr)
  {
    SequentialVoxelData& voxel = ((*leaf_itr)->getData ());
    voxel.owner_ = 0;
    voxel.distance_ = std::numeric_limits<float>::max ();
  }
  leaves_.clear ();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::expand ()
{
  //Buffer of new neighbors - initial size is just a guess of most possible
  std::vector<LeafContainerT*> new_owned;
  new_owned.reserve (leaves_.size () * 9);
  //For each leaf belonging to this supervoxel
  pcl::StopWatch timer_;
  typename SequentialSupervoxelHelper::iterator leaf_itr;
  for (leaf_itr = leaves_.begin (); leaf_itr != leaves_.end (); ++leaf_itr)
  {
    //for each neighbor of the leaf
    for (typename LeafContainerT::const_iterator neighb_itr=(*leaf_itr)->cbegin (); neighb_itr!=(*leaf_itr)->cend (); ++neighb_itr)
    {
      //Get a reference to the data contained in the leaf
      SequentialVoxelData& neighbor_voxel = ((*neighb_itr)->getData ());
      //TODO this is a shortcut, really we should always recompute distance
      if(neighbor_voxel.owner_ == this)
      {
        continue;
      }

      //Compute distance to the neighbor
      float dist = parent_->sequentialVoxelDataDistance (centroid_, neighbor_voxel);
      //If distance is less than previous, we remove it from its owner's list
      //and change the owner to this and distance (we *steal* it!)
      if (dist < neighbor_voxel.distance_)
      {
        neighbor_voxel.distance_ = dist;
        if (neighbor_voxel.owner_ != this)
        {
          if (neighbor_voxel.owner_)
          {
            (neighbor_voxel.owner_)->removeLeaf(*neighb_itr);
          }
          neighbor_voxel.owner_ = this;
          neighbor_voxel.label_ = this->getLabel();
          new_owned.push_back (*neighb_itr);
        }
      }
    }
  }
  //Push all new owned onto the owned leaf set
  typename std::vector<LeafContainerT*>::iterator new_owned_itr;
  for (new_owned_itr=new_owned.begin (); new_owned_itr!=new_owned.end (); ++new_owned_itr)
  {
    leaves_.insert (*new_owned_itr);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::updateCentroid ()
{
  centroid_.normal_ = Eigen::Vector4f::Zero ();
  centroid_.xyz_ = Eigen::Vector3f::Zero ();
  centroid_.rgb_ = Eigen::Vector3f::Zero ();
  typename SequentialSupervoxelHelper::iterator leaf_itr = leaves_.begin ();
  for ( ; leaf_itr!= leaves_.end (); ++leaf_itr)
  {
    const SequentialVoxelData& leaf_data = (*leaf_itr)->getData ();
    centroid_.normal_ += leaf_data.normal_;
    centroid_.xyz_ += leaf_data.xyz_;
    centroid_.rgb_ += leaf_data.rgb_;
  }
  centroid_.normal_.normalize ();
  centroid_.xyz_ /= static_cast<float> (leaves_.size ());
  centroid_.rgb_ /= static_cast<float> (leaves_.size ());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::getVoxels (typename pcl::PointCloud<PointT>::Ptr &voxels) const
{
  voxels.reset (new pcl::PointCloud<PointT>);
  voxels->clear ();
  voxels->resize (leaves_.size ());
  typename pcl::PointCloud<PointT>::iterator voxel_itr = voxels->begin ();
  typename SequentialSupervoxelHelper::const_iterator leaf_itr;
  for ( leaf_itr = leaves_.begin (); leaf_itr != leaves_.end (); ++leaf_itr, ++voxel_itr)
  {
    const SequentialVoxelData& leaf_data = (*leaf_itr)->getData ();
    leaf_data.getPoint (*voxel_itr);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::getNormals (typename pcl::PointCloud<Normal>::Ptr &normals) const
{
  normals.reset (new pcl::PointCloud<Normal>);
  normals->clear ();
  normals->resize (leaves_.size ());
  typename SequentialSupervoxelHelper::const_iterator leaf_itr;
  typename pcl::PointCloud<Normal>::iterator normal_itr = normals->begin ();
  for (leaf_itr = leaves_.begin (); leaf_itr != leaves_.end (); ++leaf_itr, ++normal_itr)
  {
    const SequentialVoxelData& leaf_data = (*leaf_itr)->getData ();
    leaf_data.getNormal (*normal_itr);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper::getNeighborLabels (std::set<uint32_t> &neighbor_labels) const
{
  neighbor_labels.clear ();
  //For each leaf belonging to this supervoxel
  typename SequentialSupervoxelHelper::const_iterator leaf_itr;
  for (leaf_itr = leaves_.begin (); leaf_itr != leaves_.end (); ++leaf_itr)
  {
    //for each neighbor of the leaf
    for (typename LeafContainerT::const_iterator neighb_itr=(*leaf_itr)->cbegin (); neighb_itr!=(*leaf_itr)->cend (); ++neighb_itr)
    {
      //Get a reference to the data contained in the leaf
      SequentialVoxelData& neighbor_voxel = ((*neighb_itr)->getData ());
      //If it has an owner, and it's not us - get it's owner's label insert into set
      if (neighbor_voxel.owner_ != this && neighbor_voxel.owner_)
      {
        neighbor_labels.insert (neighbor_voxel.owner_->getLabel ());
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::getSupervoxelAdjacency (std::multimap<uint32_t, uint32_t> &label_adjacency) const
{
  label_adjacency.clear ();
  for (typename HelperListT::const_iterator sv_itr = supervoxel_helpers_.cbegin (); sv_itr != supervoxel_helpers_.cend (); ++sv_itr)
  {
    uint32_t label = sv_itr->getLabel ();
    std::set<uint32_t> neighbor_labels;
    sv_itr->getNeighborLabels (neighbor_labels);
    for (std::set<uint32_t>::iterator label_itr = neighbor_labels.begin (); label_itr != neighbor_labels.end (); ++label_itr)
      label_adjacency.insert (std::pair<uint32_t,uint32_t> (label, *label_itr) );
    //if (neighbor_labels.size () == 0)
    //  std::cout << label<<"(size="<<sv_itr->size () << ") has "<<neighbor_labels.size () << "\n";
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> typename pcl::PointCloud<PointT>::Ptr
pcl::SequentialSVClustering<PointT>::getVoxelCentroidCloud () const
{
  typename PointCloudT::Ptr centroid_copy (new PointCloudT);
  copyPointCloud (*voxel_centroid_cloud_, *centroid_copy);
  return centroid_copy;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> typename pcl::PointCloud<PointT>::Ptr
pcl::SequentialSVClustering<PointT>::getUnlabeledVoxelCentroidCloud () const
{
  typename PointCloudT::Ptr centroid_copy (new PointCloudT);
  copyPointCloud (*unlabeled_voxel_centroid_cloud_, *centroid_copy);
  return centroid_copy;
}

template <typename PointT> pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
pcl::SequentialSVClustering<PointT>::getColoredVoxelCloud () const
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  for (typename HelperListT::const_iterator sv_itr = supervoxel_helpers_.cbegin (); sv_itr != supervoxel_helpers_.cend (); ++sv_itr)
  {
    typename pcl::PointCloud<PointT>::Ptr voxels;
    sv_itr->getVoxels (voxels);
    pcl::PointCloud<pcl::PointXYZRGBA> rgb_copy;
    copyPointCloud (*voxels, rgb_copy);

    pcl::PointCloud<pcl::PointXYZRGBA>::iterator rgb_copy_itr = rgb_copy.begin ();
    for ( ; rgb_copy_itr != rgb_copy.end (); ++rgb_copy_itr)
      rgb_copy_itr->rgba = label_colors_ [sv_itr->getLabel ()];

    *colored_cloud += rgb_copy;
  }

  return colored_cloud;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
pcl::SequentialSVClustering<PointT>::getColoredCloud () const
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::copyPointCloud (*input_,*colored_cloud);

  pcl::PointCloud <pcl::PointXYZRGBA>::iterator i_colored;
  typename pcl::PointCloud <PointT>::const_iterator i_input = input_->begin ();
  std::vector <int> indices;
  std::vector <float> sqr_distances;
  for (i_colored = colored_cloud->begin (); i_colored != colored_cloud->end (); ++i_colored,++i_input)
  {
    if ( !pcl::isFinite<PointT> (*i_input))
      i_colored->rgb = 0;
    else
    {
      i_colored->rgb = 0;
      LeafContainerT *leaf = sequential_octree_->getLeafContainerAtPoint (*i_input);
      if (leaf)
      {
        SequentialVoxelData& voxel_data = leaf->getData ();
        if (voxel_data.owner_)
          i_colored->rgba = label_colors_[voxel_data.owner_->getLabel ()];
      }
      else
        std::cout <<"Could not find point in getColoredCloud!!!\n";

    }

  }

  return (colored_cloud);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> pcl::PointCloud<pcl::PointXYZL>::Ptr
pcl::SequentialSVClustering<PointT>::getLabeledVoxelCloud () const
{
  pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_voxel_cloud (new pcl::PointCloud<pcl::PointXYZL>);
  for (typename HelperListT::const_iterator sv_itr = supervoxel_helpers_.cbegin (); sv_itr != supervoxel_helpers_.cend (); ++sv_itr)
  {
    typename pcl::PointCloud<PointT>::Ptr voxels;
    sv_itr->getVoxels (voxels);
    pcl::PointCloud<pcl::PointXYZL> xyzl_copy;
    copyPointCloud (*voxels, xyzl_copy);

    pcl::PointCloud<pcl::PointXYZL>::iterator xyzl_copy_itr = xyzl_copy.begin ();
    for ( ; xyzl_copy_itr != xyzl_copy.end (); ++xyzl_copy_itr)
      xyzl_copy_itr->label = sv_itr->getLabel ();

    *labeled_voxel_cloud += xyzl_copy;
  }

  return labeled_voxel_cloud;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> pcl::PointCloud<pcl::PointXYZL>::Ptr
pcl::SequentialSVClustering<PointT>::getLabeledCloud () const
{
  pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud (new pcl::PointCloud<pcl::PointXYZL>);
  pcl::copyPointCloud (*input_,*labeled_cloud);

  pcl::PointCloud <pcl::PointXYZL>::iterator i_labeled;
  typename pcl::PointCloud <PointT>::const_iterator i_input = input_->begin ();
  std::vector <int> indices;
  std::vector <float> sqr_distances;
  for (i_labeled = labeled_cloud->begin (); i_labeled != labeled_cloud->end (); ++i_labeled,++i_input)
  {
    if ( !pcl::isFinite<PointT> (*i_input))
      i_labeled->label = 0;
    else
    {
      i_labeled->label = 0;
      LeafContainerT *leaf = sequential_octree_->getLeafContainerAtPoint (*i_input);
      SequentialVoxelData& voxel_data = leaf->getData ();
      if (voxel_data.owner_)
        i_labeled->label = voxel_data.owner_->getLabel ();

    }

  }

  return (labeled_cloud);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> float
pcl::SequentialSVClustering<PointT>::getVoxelResolution () const
{
  return (resolution_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setVoxelResolution (float resolution)
{
  resolution_ = resolution;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> float
pcl::SequentialSVClustering<PointT>::getSeedResolution () const
{
  return (resolution_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setSeedResolution (float seed_resolution)
{
  seed_resolution_ = seed_resolution;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setColorImportance (float val)
{
  color_importance_ = val;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setSpatialImportance (float val)
{
  spatial_importance_ = val;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setNormalImportance (float val)
{
  normal_importance_ = val;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setUseSingleCameraTransform (bool val)
{
  use_default_transform_behaviour_ = false;
  use_single_camera_transform_ = val;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setIgnoreInputNormals (bool val)
{
  ignore_input_normals_ = val;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::initializeLabelColors ()
{
  uint32_t max_label = static_cast<uint32_t> (10000); //TODO Should be getMaxLabel
  //If we already have enough colors, return
  if (label_colors_.size () >= max_label)
    return;

  //Otherwise, generate new colors until we have enough
  label_colors_.reserve (max_label + 1);
  srand (static_cast<unsigned int> (0));
  while (label_colors_.size () <= max_label )
  {
    uint8_t r = static_cast<uint8_t>( (rand () % 256));
    uint8_t g = static_cast<uint8_t>( (rand () % 256));
    uint8_t b = static_cast<uint8_t>( (rand () % 256));
    label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> int
pcl::SequentialSVClustering<PointT>::getMaxLabel () const
{
  int max_label = 0;
  if(supervoxel_helpers_.size() > 0)
  {
    for (typename HelperListT::const_iterator sv_itr = supervoxel_helpers_.cbegin (); sv_itr != supervoxel_helpers_.cend (); ++sv_itr)
    {
      int temp = sv_itr->getLabel ();
      if (temp > max_label)
        max_label = temp;
    }
  }
  return max_label;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::transformFunction (PointT &p)
{
  p.x /= p.z;
  p.y /= p.z;
  p.z = std::log (p.z);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> float
pcl::SequentialSVClustering<PointT>::sequentialVoxelDataDistance (const SequentialVoxelData &v1, const SequentialVoxelData &v2) const
{

  float spatial_dist = (v1.xyz_ - v2.xyz_).norm () / seed_resolution_;
  float color_dist =  (v1.rgb_ - v2.rgb_).norm () / 255.0f;
  float cos_angle_normal = 1.0f - std::abs (v1.normal_.dot (v2.normal_));
  // std::cout << "s="<<spatial_dist<<"  c="<<color_dist<<"   an="<<cos_angle_normal<<"\n";
  return  cos_angle_normal * normal_importance_ + color_dist * color_importance_+ spatial_dist * spatial_importance_;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{ 
  namespace octree
  {
    //Explicit overloads for RGB types
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB,pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGB &new_point);

    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA,pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGBA &new_point);

//    template<>
//    void
//    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal,pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGBNormal &new_point);

    //Explicit overloads for RGB types
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB,pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData>::computeData ();

    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA,pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData>::computeData ();

//    template<> void
//    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal,pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData>::computeData ();
  }
}



#define PCL_INSTANTIATE_SequentialSVClustering(T) template class PCL_EXPORTS pcl::SequentialSVClustering<T>;

#endif    // PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_HPP_

