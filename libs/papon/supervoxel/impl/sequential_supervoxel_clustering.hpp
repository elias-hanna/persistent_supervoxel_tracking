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
 * Author : h.elias@hotmail.fr
 * Email  : h.elias@hotmail.fr
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
  prev_sequential_octree_.reset (new OctreeSequentialT (resolution_));
  if (use_single_camera_transform_)
  {
    sequential_octree_->setTransformFunction (boost::bind (&SequentialSVClustering::transformFunction, this, _1));
    prev_sequential_octree_->setTransformFunction (boost::bind (&SequentialSVClustering::transformFunction, this, _1));
  }
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
    sequential_octree_->defineBoundingBoxOnInputCloud ();
    /////////////////////////////////////////////////////////////////////////
    if ( (use_default_transform_behaviour_ && input_->isOrganized ())
         || (!use_default_transform_behaviour_ && use_single_camera_transform_))
    {
      prev_sequential_octree_->setTransformFunction (boost::bind (&SequentialSVClustering::transformFunction, this, _1));
    }

    prev_sequential_octree_->setDifferenceFunction (boost::bind (&OctreeSequentialT::SeqVoxelDataDiff, _1));
    prev_sequential_octree_->setDifferenceThreshold (0.2);
    prev_sequential_octree_->setNumberOfThreads (NUM_THREADS);
    prev_sequential_octree_->setOcclusionTestInterval (0.25f);
    prev_sequential_octree_->setInputCloud (cloud);
    prev_sequential_octree_->defineBoundingBoxOnInputCloud ();
  }
  else
  {
    prev_sequential_octree_ = sequential_octree_;
    sequential_octree_->setInputCloud (cloud);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::setNormalCloud (typename NormalCloud::ConstPtr normal_cloud)
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
pcl::SequentialSVClustering<PointT>::extract (SequentialSVMapT &supervoxel_clusters)
{
  timer_.reset ();
  double t_start = timer_.getTime ();

  //  unsigned long nb_previous_supervoxel_clusters = supervoxel_clusters.size();

  // Maybe try to do something where this method is either called here or when clouds are matched to
  // restrain from computing two times the normals (even if they are parallelized)
  buildVoxelCloud();

  double t_update = timer_.getTime ();

  std::vector<int> seed_indices, existing_seed_indices;

  getPreviousSeedingPoints(supervoxel_clusters, existing_seed_indices);
  /** Need a function that at each time step:
   * - check if a label has disappeared or is occluded
   * - look into the unlabeled voxel cloud if its in it with recognize from ObjRecRANSAC
   * - if it is in it with sufficient certainty (look into object output) get the transformation found and translate the previous centroid
   * in order to get a new seeding point (don't forget to keep the same label !!!)
   * - Add all object model for next step
   * THIS FUNCTION SHOULD BE BETWEEN getPreviousSeedingPoints AND pruneSeeds !
   */
  pruneSeeds(existing_seed_indices, seed_indices);
  addHelpersFromUnlabeledSeedIndices(seed_indices);

  double t_seeds = timer_.getTime ();

  int max_depth = static_cast<int> (1.8f*seed_resolution_/resolution_);
  expandSupervoxels (max_depth);

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
  bool segmentation_is_possible = initCompute ();
  if ( !segmentation_is_possible )
  {
    PCL_ERROR ("[pcl::SequentialSVClustering::initCompute] Init failed.\n");
    deinitCompute ();
    return;
  }
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

  // Update previous clouds
  updatePrevClouds ();
  //Add the new cloud of data to the octree
  sequential_octree_->addPointsFromInputCloud ();
  // GlobalCheck unlabels the voxels that changed and update nb_of_unlabeled_voxels_
  globalCheck();
  //Compute normals and insert data for centroids into data field of octree
  computeVoxelData ();

  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::updatePrevClouds ()
{
  // Updating the previous voxel centroid cloud and normal cloud
  if(voxel_centroid_cloud_)
  {
    prev_voxel_centroid_cloud_.reset (new PointCloudT);
    prev_voxel_centroid_cloud_->resize (voxel_centroid_cloud_->size ());
    prev_voxel_centroid_normal_cloud_.reset (new NormalCloud);
    prev_voxel_centroid_normal_cloud_->resize (voxel_centroid_cloud_->size ());
    typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin ();
    typename PointCloudT::iterator prev_cent_cloud_itr = prev_voxel_centroid_cloud_->begin ();
    NormalCloud::iterator prev_cent_norm_cloud_itr = prev_voxel_centroid_normal_cloud_->begin ();
    for (; leaf_itr != sequential_octree_->end (); ++leaf_itr, ++prev_cent_norm_cloud_itr, ++prev_cent_cloud_itr)
    {
      SequentialVoxelData& prev_voxel_data = (*leaf_itr)->getData ();
      prev_voxel_data.getPoint (*prev_cent_cloud_itr);
      prev_voxel_data.getNormal (*prev_cent_norm_cloud_itr);
    }
  }
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
  for (int idx = 0 ; leaf_itr != sequential_octree_->end (); ++leaf_itr, ++cent_cloud_itr, ++idx)
  {
    SequentialVoxelData& new_voxel_data = (*leaf_itr)->getData ();
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
  //If normals were provided
  if (input_normals_)
  {
    //Verify that input normal cloud size is same as input cloud size
    assert (input_normals_->size () == input_->size ());
    //For every point in the input cloud, find its corresponding leaf
    typename NormalCloud::const_iterator normal_itr = input_normals_->begin ();
    for (typename PointCloudT::const_iterator input_itr = input_->begin (); input_itr != input_->end (); ++input_itr, ++normal_itr)
    {
      //If the point is not finite we ignore it
      if ( !pcl::isFinite<PointT> (*input_itr))
        continue;
      //Otherwise look up its leaf container
      LeafContainerT* leaf = sequential_octree_->getLeafContainerAtPoint (*input_itr);

      //Get the voxel data object
      SequentialVoxelData& voxel_data = leaf->getData ();
      //Add this normal in (we will normalize at the end)
      voxel_data.normal_ += normal_itr->getNormalVector4fMap ();
      voxel_data.curvature_ += normal_itr->curvature;
    }
    //Now iterate through the leaves and normalize
    for (leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
    {
      SequentialVoxelData& voxel_data = (*leaf_itr)->getData ();
      voxel_data.normal_.normalize ();
      voxel_data.owner_ = 0;
      voxel_data.distance_ = std::numeric_limits<float>::max ();
      //Get the number of points in this leaf
      int num_points = (*leaf_itr)->getPointCounter ();
      voxel_data.curvature_ /= num_points;
    }
  }
  // Otherwise compute the normals
  else
  {
    parallelComputeNormals ();
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
      pcl::computePointNormal (*voxel_centroid_cloud_, indices, new_voxel_data.normal_, new_voxel_data.curvature_);
      pcl::flipNormalTowardsViewpoint (voxel_centroid_cloud_->points[new_voxel_data.idx_], 0.0f,0.0f,0.0f, new_voxel_data.normal_);
      new_voxel_data.normal_[3] = 0.0f;
      new_voxel_data.normal_.normalize ();
      new_voxel_data.owner_ = 0;
      new_voxel_data.distance_ = std::numeric_limits<float>::max ();
    }
  }
  );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::vector<int>
pcl::SequentialSVClustering<PointT>::getAvailableLabels ()
{
  std::vector<int> available_labels;
  // Fill the vector with 1, 2, ..., max label
  for(int i = 1 ; i < getMaxLabel () ; ++i) { available_labels.push_back (i); }
  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
  {
    available_labels.erase (std::remove(available_labels.begin(), available_labels.end(), sv_itr->getLabel ()), available_labels.end());
  }
  return available_labels;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::expandSupervoxels ( int depth )
{
  for (int i = 1; i < depth; ++i)
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
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::makeSupervoxels (SequentialSVMapT &supervoxel_clusters)
{
  supervoxel_clusters.clear ();
  for (typename HelperListT::iterator sv_itr = supervoxel_helpers_.begin (); sv_itr != supervoxel_helpers_.end (); ++sv_itr)
  {
    uint32_t label = sv_itr->getLabel ();
    supervoxel_clusters[label].reset (new SequentialSV<PointT>(sv_itr->isNew()));
    sv_itr->getXYZ (supervoxel_clusters[label]->centroid_.x,supervoxel_clusters[label]->centroid_.y,supervoxel_clusters[label]->centroid_.z);
    sv_itr->getRGB (supervoxel_clusters[label]->centroid_.rgba);
    sv_itr->getNormal (supervoxel_clusters[label]->normal_);
    sv_itr->getVoxels (supervoxel_clusters[label]->voxels_);
    sv_itr->getNormals (supervoxel_clusters[label]->normals_);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
      // Handling new voxels
      if( voxel.label_ == -1)
      {
        ++nb_of_unlabeled_voxels_;
      }
      // Handling existing voxels that have changed between two frames
      else if(voxel.isChanged())
      {
        // Minus 1 because labels start at 1"
        --nb_voxels_by_labels[voxel.label_ - 1];
        voxel.label_ = -1;
        ++nb_of_unlabeled_voxels_;
      }
      // Handling unchanged voxels
      else
      {
        ++nb_voxels_by_labels[voxel.label_ - 1];
      }
    }
    // Unlabel all the voxels whom supervoxel has changed by more than a half (a little less than a half in reality)
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
template <typename PointT> std::unordered_map <uint32_t, pcl::recognition::ObjRecRANSAC::Output>
pcl::SequentialSVClustering<PointT>::getMatches (SequentialSVMapT supervoxel_clusters)
{
  std::unordered_map <uint32_t, pcl::recognition::ObjRecRANSAC::Output> valid_outputs;

  // Get the labels of the disappeared and fully occluded supervoxels
  if(getMaxLabel() > 0)
  {
    std::vector<uint32_t> labels_to_track = getLabelsOfDynamicSV (supervoxel_clusters);

    // Try to find the disappeared / occluded supervoxels in the new voxel cloud
    pcl::recognition::ObjRecRANSAC tracker_ransac(seed_resolution_*0.5, resolution_);
    tracker_ransac.icpHypothesesRefinementOff ();
    // Macro turn degre to rad, here 3 deg is the default value (the one in the constructor)
    //  tracker_ransac.setMaxCoplanarityAngleDegrees(3.0f*AUX_DEG_TO_RADIANS);

    for(auto label: labels_to_track)
    {
      pcl::PointCloud<pcl::PointXYZ> voxels_copy;
      copyPointCloud(*(supervoxel_clusters[label]->voxels_), voxels_copy);
      tracker_ransac.addModel(voxels_copy, *(supervoxel_clusters[label]->normals_), std::to_string(label));
    }
    std::list<pcl::recognition::ObjRecRANSAC::Output> recognized_objects;
    double success_probability = 0.99;
    pcl::PointCloud<pcl::PointXYZ> voxel_cloud_copy;
    pcl::PointCloud<pcl::Normal> voxel_normal_cloud_copy;
    copyPointCloud(*(unlabeled_voxel_centroid_cloud_), voxel_cloud_copy);
    copyPointCloud(*getUnlabeledVoxelNormalCloud (), voxel_normal_cloud_copy);

    tracker_ransac.recognize(voxel_cloud_copy, voxel_normal_cloud_copy, recognized_objects, success_probability);

    float threshold = 0.0f;
    for(auto output: recognized_objects)
    {
      uint32_t label = std::stoi(output.object_name_);

      // If this match exceeds threshold
      if(output.match_confidence_ > threshold)
      {
        auto it_at = valid_outputs.find(label);

        pcl::PointXYZRGBA prev_centroid_tmp = supervoxel_clusters[label]->centroid_;
        PointT prev_centroid;
        prev_centroid.x = prev_centroid_tmp.x;
        prev_centroid.y = prev_centroid_tmp.y;
        prev_centroid.z = prev_centroid_tmp.z;

        float x = output.rigid_transform_[9] // Translation part
            // Rotational part
            + output.rigid_transform_[0]*prev_centroid.x
            + output.rigid_transform_[1]*prev_centroid.y
            + output.rigid_transform_[2]*prev_centroid.z;
        float y = output.rigid_transform_[10] // Translation part
            // Rotational part
            + output.rigid_transform_[3]*prev_centroid.x
            + output.rigid_transform_[4]*prev_centroid.y
            + output.rigid_transform_[5]*prev_centroid.z;
        float z = output.rigid_transform_[11] // Translation part
            // Rotational part
            + output.rigid_transform_[6]*prev_centroid.x
            + output.rigid_transform_[7]*prev_centroid.y
            + output.rigid_transform_[8]*prev_centroid.z;

        PointT new_centroid;
        new_centroid.x = x;
        new_centroid.y = y;
        new_centroid.z = z;

        LeafContainerT* prev_centroid_leaf = prev_sequential_octree_->getLeafContainerAtPoint(prev_centroid);
        LeafContainerT* new_centroid_leaf = sequential_octree_->getLeafContainerAtPoint(new_centroid);

        if(prev_centroid_leaf && new_centroid_leaf)
        {
          float color_dist =  (prev_centroid_leaf->getData ().rgb_
                               - new_centroid_leaf->getData ().rgb_).norm () / 255.0f;
          std::cout << "Color dist: " << color_dist  << " Match confidence: " << output.match_confidence_ << "\n";
        }
        else
        {
          std::cout << "Problem:\n";
          if (!prev_centroid_leaf)
          { std::cout << "No prev\n"; }
          if (!new_centroid_leaf)
          { std::cout << "No new\n"; }
        }
        // There was already a valid match
        if(it_at != valid_outputs.end())
        {
          // Check if this valid match is better
          if(it_at->second.match_confidence_ < output.match_confidence_)
          {
            it_at->second = output;
          }
        }
        // There was no previous valid match
        else
        {
          valid_outputs.insert(std::pair<uint32_t, pcl::recognition::ObjRecRANSAC::Output> (label, output));
        }
      }
    }
    // Update the octree and the voxel clouds
    std::vector<uint32_t> labels;
    for(auto output: valid_outputs)
    {
      labels.push_back(static_cast<uint32_t> (std::stoi(output.second.object_name_)));
    }
    sequential_octree_->updateOctreeFromMatchedClouds(labels);

    computeVoxelData (); // COULD DO BETTER DON'T NEED TO RECALCULATE THE NORMALS
  }
  return valid_outputs;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::computeIntensityGradientCloud (PointCloudIG::Ptr cloud_ig, const PointCloudI::Ptr xyzi_total_cloud, const NormalCloud::Ptr total_normal_cloud) const
{
  pcl::IntensityGradientEstimation<pcl::PointXYZI, pcl::Normal, pcl::IntensityGradient> gradient_est;
  typename pcl::search::KdTree<pcl::PointXYZI>::Ptr treept2 (new pcl::search::KdTree<pcl::PointXYZI> (false));
  gradient_est.setSearchMethod(treept2);
  gradient_est.setRadiusSearch(3*resolution_);
  gradient_est.setInputCloud(xyzi_total_cloud);
  gradient_est.setInputNormals(total_normal_cloud);
  gradient_est.compute(*cloud_ig);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::pair< pcl::IndicesPtr, pcl::PointCloud< pcl::Histogram<32> >::Ptr >
pcl::SequentialSVClustering<PointT>::computeRIFTDescriptors (const PointCloudScale sift_result, const PointCloudIG::Ptr cloud_ig, const PointCloudI::Ptr xyzi_total_cloud) const
{
  // Get a pointcloud of keypoints in the type PointXYZI (we are just interested in the XYZ position of those points)
  PointCloudI::Ptr input_keypoints_cloud_i(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_keypoints_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  copyPointCloud(sift_result, *input_keypoints_cloud_rgb);
  PointCloudXYZRGBtoXYZI(*input_keypoints_cloud_rgb, *input_keypoints_cloud_i);

  // Estimate the RIFT feature
  pcl::RIFTEstimation<pcl::PointXYZI, pcl::IntensityGradient, FeatureT > rift_est;
  typename pcl::search::KdTree<pcl::PointXYZI>::Ptr treept3 (new pcl::search::KdTree<pcl::PointXYZI> (false));
  rift_est.setSearchMethod(treept3);
  rift_est.setRadiusSearch(seed_resolution_);
  rift_est.setNrDistanceBins (4);
  rift_est.setNrGradientBins (8);
  // Compute only RIFT descriptor for SIFT keypoints
  rift_est.setInputCloud(input_keypoints_cloud_i);
  rift_est.setInputGradient(cloud_ig);
  // Use all the points in the supervoxel cloud to compute the RIFT descriptor
  rift_est.setSearchSurface(xyzi_total_cloud);

  pcl::IndicesPtr rift_indices = boost::make_shared<std::vector<int>> ();
  rift_indices->reserve (sift_result.size ());

  // We only search the indice of each keypoint in the main PointCloud so we are just interested in their XYZ position
  pcl::search::KdTree<pcl::PointXYZ> point_indices_kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(*xyzi_total_cloud, *xyz_cloud);
  point_indices_kdtree.setInputCloud (xyz_cloud);

  for(auto point_scale: sift_result)
  {
    std::vector<int> indices;
    std::vector<float> sqr_distances;

    pcl::PointXYZ pt(point_scale.x, point_scale.y, point_scale.z);
    point_indices_kdtree.nearestKSearch (pt, 1, indices, sqr_distances);
    int keypoint_indice = indices[0];
    rift_indices->push_back(keypoint_indice);
  }
  // Remove doublons
  sort( rift_indices->begin(), rift_indices->end() );
  rift_indices->erase( unique( rift_indices->begin(), rift_indices->end() ), rift_indices->end() );

  rift_est.setIndices(rift_indices);
  PointCloudFeatureT::Ptr rift_output(new PointCloudFeatureT);
  rift_est.compute (*rift_output);


  return KeypointFeatureT (rift_indices, rift_output);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::filterKeypoints(const std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr > to_filter_keypoints, std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr >& filtered_keypoints) const
{
  for (size_t idx = 0; idx < to_filter_keypoints.second->size (); ++idx)
  {
    bool to_remove = false;
    int nb_of_0s = 0;
    FeatureT descriptor = (*to_filter_keypoints.second)[idx];
    // 32 = size of descriptor
    for (int i = 0; i < 32; ++i)
    {
      if(std::isnan(descriptor.histogram[i]))
      { to_remove = true; break; }
      if(descriptor.histogram[i] == 0.0)
      { ++nb_of_0s; }
    }
    if (nb_of_0s > 15)
    { to_remove = true; }
    if(!to_remove)
    {
      filtered_keypoints.first->push_back((*to_filter_keypoints.first)[idx]);
      filtered_keypoints.second->push_back(descriptor);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::vector<int>
pcl::SequentialSVClustering<PointT>::computeKeypointsMatches(const std::vector<int> to_match_indices, const PointCloudFeatureT to_match_feature_cloud,
                                                             const std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr > indices_point_pair)
{
  // Search the nearest previous keypoint in feature space for each of the maybe inliers
  // in order to compute a transform using singular value decomposition
  // Instantiate search object with 4 randomized trees and 256 checks
  SearchT search (true, CreatorPtrT (new IndexT (4)));
  search.setPointRepresentation (RepresentationPtrT (new DefaultFeatureRepresentation<FeatureT>));
  search.setChecks (256);
  search.setInputCloud (indices_point_pair.second);
  // Create a vector that contains -1
  std::vector<int> matches_of_to_match(to_match_indices.size (), -1);
  std::vector<int> k_indices;
  std::vector<float> k_sqr_distances;
  // Do search
  std::unordered_map<int, std::pair<std::vector<int>, std::vector<float>>> neighbours_of_inliers;

  for(size_t idx = 0; idx < to_match_feature_cloud.size (); ++idx)
  {
    // We look up for k neighbours with k equal to the number of min
    // inliers
    size_t k = to_match_feature_cloud.size ();
    search.nearestKSearch(to_match_feature_cloud[idx], static_cast<int> (k), k_indices, k_sqr_distances);
    neighbours_of_inliers.insert(std::pair<int, std::pair<std::vector<int>, std::vector<float>>>
                                 (idx,
                                  std::pair<std::vector<int>, std::vector<float>>
                                  (k_indices, k_sqr_distances))
                                 );
  }
  // Associate each maybe inlier to its best match in the previous
  // keypoints (priority for order of importance for each neigbour)
  std::vector<int> unmatched_idx(to_match_indices.size ());
  std::iota(unmatched_idx.begin (), unmatched_idx.end (), 1);
  for (size_t depth = 0; depth < to_match_indices.size (); ++depth)
  {
    std::unordered_map<int, std::pair<int, float>> min;
    for (auto idx: unmatched_idx)
    {
      // If the distance equals to 0, it's a bug because we search too far
      // around the point when building neighbours_of_inliers
      if(neighbours_of_inliers[idx-1].second[depth] != 0.0)
      {
        std::unordered_map<int, std::pair<int, float>>::iterator map_it = min.find (neighbours_of_inliers[idx-1].first[depth]);
        // If another point is competing for this match, compare errors
        if (map_it != min.end ())
        {
          if (map_it->second.second > neighbours_of_inliers[idx-1].second[depth])
          {
            map_it->second.first = idx-1;
            map_it->second.second = neighbours_of_inliers[idx-1].second[depth];
          }
        }
        // Otherwise just insert a new element
        else
        {
          min.insert(std::pair<int, std::pair<int, float>>
                     (neighbours_of_inliers[idx-1].first[depth],
                     std::pair<int, float>
                     (idx-1, neighbours_of_inliers[idx-1].second[depth])));
        }
      }
    }
    // Now allocate each match to each point and remove the allocated
    // point index from unmatched index vector
    for(std::unordered_map<int, std::pair<int, float>>::iterator map_it = min.begin (); map_it != min.end (); ++map_it)
    {
      matches_of_to_match[map_it->second.first] = (*indices_point_pair.first)[map_it->first];
      unmatched_idx.erase (std::remove (unmatched_idx.begin (), unmatched_idx.end (), map_it->second.first+1), unmatched_idx.end ());
    }
  }
  return matches_of_to_match;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::vector<uint32_t>
pcl::SequentialSVClustering<PointT>::getLabelsOfDynamicSV (SequentialSVMapT &supervoxel_clusters)
{
  std::vector<uint32_t> labels_to_track;
  // Get the labels of the disappeared and fully occluded supervoxels
  int nb_occluded_voxels_by_labels[getMaxLabel ()] = {0};
  int nb_voxels_by_labels[getMaxLabel ()] = {0};
  for(typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
  {
    SequentialVoxelData& voxel = (*leaf_itr)->getData ();
    if(voxel.frame_occluded_ != 0) // It is currently occluded
    { ++nb_occluded_voxels_by_labels[voxel.label_ - 1]; }
    ++nb_voxels_by_labels[voxel.label_ - 1];
  }
  for(auto cluster: supervoxel_clusters)
  {
    uint32_t i = cluster.first;
    // If the sv has disappeared or is fully occluded
    if(nb_voxels_by_labels[i - 1] == 0 ||
       ( (nb_voxels_by_labels[i - 1] == nb_occluded_voxels_by_labels[i - 1]) && (supervoxel_clusters.find(i) != supervoxel_clusters.end()) ) )
    { labels_to_track.push_back(i); }
  }
  return labels_to_track;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template < typename PointT> void
pcl::SequentialSVClustering<PointT>::initializeSIFT (pcl::SIFTKeypoint<PointT, pcl::PointWithScale> &sift,
                                                     float min_scale, float min_contrast, int n_octaves, int n_scales_per_octave)
{
  typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT> ());
  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::computeSIFTRequiredClouds(PointCloudIG::Ptr cloud_ig, PointCloudI::Ptr xyzi_total_cloud,
                                                               typename PointCloudT::Ptr cloud, NormalCloud::Ptr normal_cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  // Convert previous voxel centroid cloud from XYZRGB/XYZRGBA to XYZI
  copyPointCloud(*cloud, *xyzrgb_cloud);
  PointCloudXYZRGBtoXYZI(*xyzrgb_cloud, *xyzi_total_cloud);

  // Estimate the Intensity Gradient for this cloud
  computeIntensityGradientCloud (cloud_ig, xyzi_total_cloud, normal_cloud);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::computeSIFTKeypointsAndRIFTDescriptors(SequentialSVMapT &supervoxel_clusters,
                                                                            KeypointMapFeatureT &previous_keypoints,
                                                                            int min_nb_of_keypoints,
                                                                            float min_scale, float min_contrast,
                                                                            int n_octaves, int n_scales_per_octave)
{
  // Get the labels that need to be tracked
  std::vector<uint32_t> labels_to_track = getLabelsOfDynamicSV (supervoxel_clusters);
  // Initiliaze the SIFT keypoints structure
  pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift;
  initializeSIFT (sift, min_scale, min_contrast, n_octaves, n_scales_per_octave);
  PointCloudScale sift_result;
  // Compute clouds
  PointCloudI::Ptr xyzi_total_cloud (new PointCloudI);
  PointCloudIG::Ptr cloud_ig (new PointCloudIG);
  computeSIFTRequiredClouds(cloud_ig, xyzi_total_cloud,
                            prev_voxel_centroid_cloud_, prev_voxel_centroid_normal_cloud_);
  // Get SIFT keypoints and RIFT descriptors for each keypoint from precedent cloud for each label
  for(auto label: labels_to_track)
  {
    if(supervoxel_clusters[label]->voxels_->size () > 20)// Need to get a good metric
    {
      // Estimate the sift interest points using Intensity values from RGB values
      sift_result.clear ();
      sift.setInputCloud(supervoxel_clusters[label]->voxels_);
      sift.compute(sift_result);
      // Compute the RIFT descriptors
      KeypointFeatureT rift_output = computeRIFTDescriptors (sift_result, cloud_ig, xyzi_total_cloud);
      if(rift_output.second->size () > min_nb_of_keypoints)// Min number of keypoints in order to track the supervoxel
      {
        previous_keypoints.insert (std::pair<uint32_t, KeypointFeatureT> (label, rift_output));
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::computeSIFTKeypointsAndRIFTDescriptors(KeypointFeatureT &current_keypoints,
                                                                            float min_scale, float min_contrast,
                                                                            int n_octaves, int n_scales_per_octave)
{
  // Initiliaze the SIFT keypoints structure
  pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift;
  initializeSIFT (sift, min_scale, min_contrast, n_octaves, n_scales_per_octave);
  PointCloudScale sift_result;
  // Compute clouds
  PointCloudI::Ptr xyzi_total_cloud (new PointCloudI);
  PointCloudIG::Ptr cloud_ig (new PointCloudIG);
  computeSIFTRequiredClouds(cloud_ig, xyzi_total_cloud,
                            getUnlabeledVoxelCentroidCloud (), getUnlabeledVoxelNormalCloud ());
  // Estimate the sift interest points using Intensity values from RGB values
  sift.setInputCloud(getUnlabeledVoxelCentroidCloud ());
  sift.compute(sift_result);

  // Compute the RIFT descriptors
  current_keypoints = computeRIFTDescriptors (sift_result, cloud_ig, xyzi_total_cloud);

  std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr > filtered_keypoints
      (boost::make_shared<std::vector<int>> (), boost::make_shared<PointCloudFeatureT> ());

  filterKeypoints(current_keypoints, filtered_keypoints);

  current_keypoints = filtered_keypoints;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::samplePotentialInliers(std::vector<int> &indices,
                                                            PointCloudFeatureT::Ptr &cloud,
                                                            std::vector<int> &potential_inliers,
                                                            PointCloudFeatureT::Ptr &potential_inliers_feature_cloud,
                                                            int nb_to_sample)
{
  // Select min_number_of_inliers values from data (the SIFT keypoints from current frame)
  for (int j = 0; j < nb_to_sample; ++j)
  {
    // True random generator
    std::random_device generator;
    std::uniform_int_distribution<int> distribution(0, indices.size () - 1);
    int rand_indice = distribution (generator);
    potential_inliers.push_back (indices[rand_indice]);
    potential_inliers_feature_cloud->push_back (cloud->at(rand_indice));
    indices.erase(indices.begin () + rand_indice);
    cloud->erase (cloud->begin () + rand_indice);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::unordered_map<uint32_t, Eigen::Matrix<float, 4, 4>>
pcl::SequentialSVClustering<PointT>::getMatchesRANSAC (SequentialSVMapT &supervoxel_clusters)
{
  // This will store the transforms that are valid
  std::unordered_map<uint32_t, Eigen::Matrix<float, 4, 4>> found_transforms;
  // previous_keypoints is a map where the key is the label of the concerned supervoxel, and the value is a pair
  // consisting as first element of the Indices of the keypoints in the previous voxel centroid cloud and as second
  // element of the RIFT descriptor matching this indice.
  KeypointMapFeatureT previous_keypoints;
  KeypointFeatureT current_keypoints;

  // Parameters for sift computation
  float min_scale = 0.005f;
  float min_contrast = 0.03f;
  int n_octaves = 4;
  int n_scales_per_octave = 8;

  // RANSAC variables
  int min_number_of_inliers = 4;
  float proba_of_pure_inlier = 0.99f;
  int num_max_iter = 100; // Same as in Van Hoof paper
  std::vector<uint32_t> labels;

  if(getMaxLabel() > 0)
  {
    ///////////////////////////////////////////////////////////////////////////////////////////
    /////////////////COMPUTE KEYPOINTS AND DESCRIPTORS FOR PREVIOUS CLOUD//////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    computeSIFTKeypointsAndRIFTDescriptors(supervoxel_clusters, previous_keypoints,
                                           min_number_of_inliers, min_scale, min_contrast);

    if(previous_keypoints.size () > 0)
    {
      ///////////////////////////////////////////////////////////////////////////////////////////
      //////////////////COMPUTE KEYPOINTS AND DESCRIPTORS FOR CURRENT CLOUD//////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////
      computeSIFTKeypointsAndRIFTDescriptors(current_keypoints, min_scale, min_contrast);

      current_keypoints_indices_ = *(current_keypoints.first);
      previous_keypoints_indices_.clear();
      for (auto pair: previous_keypoints)
      {
        previous_keypoints_indices_.insert(std::end(previous_keypoints_indices_),
                                           std::begin(*pair.second.first), std::end(*pair.second.first));
      }

      for(auto pair: previous_keypoints)
      {
        uint32_t label = pair.first;
        Eigen::Matrix<float, 4, 4> best_fit;
        int max_nb_of_inliers = 0;
        float best_err = std::numeric_limits<float>::max ();

        // RANSAC implementation to find the transform that best explain
        // new keypoints observations
        for(size_t i = 0; i < num_max_iter; ++i)
        {
          // Temporary structures that store copies from current keypoints data
          std::vector<int> tmp_indices = *current_keypoints.first;
          PointCloudFeatureT::Ptr tmp_cloud (new PointCloudFeatureT);
          copyPointCloud (*current_keypoints.second, *tmp_cloud);
          // Structures that store potential inliers data
          std::vector<int> maybe_inliers;
          maybe_inliers.reserve (min_number_of_inliers);
          PointCloudFeatureT::Ptr maybe_inliers_feature_cloud (new PointCloud<FeatureT>);
          //          // Select min_number_of_inliers values from data (the SIFT keypoints from current frame)
          //          for (int j = 0; j < min_number_of_inliers; ++j)
          //          {
          //            // True random generator
          //            std::random_device generator;
          //            std::uniform_int_distribution<int> distribution(0, tmp_indices.size () - 1);
          //            int rand_indice = distribution (generator);
          //            maybe_inliers.push_back (tmp_indices[rand_indice]);
          //            maybe_inliers_feature_cloud->push_back (tmp_cloud->at(rand_indice));
          //            tmp_indices.erase(tmp_indices.begin () + rand_indice);
          //            tmp_cloud->erase (tmp_cloud->begin () + rand_indice);
          //          }
          samplePotentialInliers(tmp_indices, tmp_cloud, maybe_inliers, maybe_inliers_feature_cloud, min_number_of_inliers);

          // Compute keypoints matches with maybe inliers
          std::vector<int> matches_of_maybe_inliers = computeKeypointsMatches(maybe_inliers, *maybe_inliers_feature_cloud,
                                                                              pair.second);

          // Compute the transform between the points sampled from data and the previous keypoints
          boost::shared_ptr< pcl::registration::TransformationEstimation< PointT, PointT > > estimator; // Generic container for estimators
          estimator.reset ( new pcl::registration::TransformationEstimationSVD < PointT, PointT > () ); // SVD transform estimator
          Eigen::Matrix<float, 4, 4> transformation_est;
          typename PointCloudT::Ptr cloud_src, cloud_tgt;
          cloud_src.reset(new PointCloudT);
          cloud_tgt.reset(new PointCloudT);
          for (size_t idx = 0; idx < matches_of_maybe_inliers.size (); ++idx)
          {
            cloud_src->push_back ((*prev_voxel_centroid_cloud_)[matches_of_maybe_inliers[idx]]);
            cloud_tgt->push_back ((*getUnlabeledVoxelCentroidCloud ())[maybe_inliers[idx]]);
          }
          estimator->estimateRigidTransformation (*cloud_src, *cloud_tgt, transformation_est);

          // Compute the new transformed centroid of the current supervoxel being matched
          pcl::PointXYZRGBA prev_centroid_tmp = supervoxel_clusters[label]->centroid_;
          Eigen::Vector4f new_centroid, prev_centroid(prev_centroid_tmp.x,
                                                      prev_centroid_tmp.y,
                                                      prev_centroid_tmp.z,
                                                      1);

          new_centroid = transformation_est*prev_centroid;

          pcl::PointXYZ cent(new_centroid[0], new_centroid[1], new_centroid[2]);

          // Do radius search around new estimated centroid
          // This search is only based on spatiality in a radius of
          // seed_resolution (we are looking for the previous supervoxel)
          std::vector<int> k_indices;
          std::vector<float> k_sqr_distances;
          typename pcl::search::KdTree<pcl::PointXYZ>::Ptr unlabeled_voxel_cloud_search;
          unlabeled_voxel_cloud_search.reset (new pcl::search::KdTree<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
          copyPointCloud(*getUnlabeledVoxelCentroidCloud (), *xyz_cloud);
          unlabeled_voxel_cloud_search->setInputCloud (xyz_cloud);
          unlabeled_voxel_cloud_search->radiusSearch (cent, seed_resolution_, k_indices, k_sqr_distances);
          //          std::cout << "number of potential inliers: " << k_indices.size () <<"\n";
          // Find the current keypoints that are in this potential supervoxel
          float threshold = 0.1f;
          // Search the nearest previous keypoint in feature space for each of the maybe inliers
          // in order to compute a transform using singular value decomposition
          // Instantiate search object with 4 randomized trees and 256 checks
          SearchT search (true, CreatorPtrT (new IndexT (4)));
          search.setPointRepresentation (RepresentationPtrT (new DefaultFeatureRepresentation<FeatureT>));
          search.setChecks (256);
          search.setInputCloud (pair.second.second);
          for(auto neigh_ind: k_indices)
          {
            std::vector<int>::iterator vec_it = std::find (tmp_indices.begin (), tmp_indices.end (), neigh_ind);
            // If indice in the current keypoints (without maybe inliers)
            if( vec_it != tmp_indices.end ())
            {
              std::vector<int> indices;
              std::vector<float> distances;
              // Search for the nearest neighbour in feature space
              // between corresponding keypoint and keypoints in previous
              // supervoxel
              search.nearestKSearch((*tmp_cloud)[vec_it-tmp_indices.begin ()], 1, indices, distances);
              // If the NN of this point is close enough it's an inlier
              if(distances[0] < threshold)
              {
                maybe_inliers.push_back (neigh_ind);
                maybe_inliers_feature_cloud->push_back (tmp_cloud->at(vec_it-tmp_indices.begin ()));
              }
            }
          }
          //          std::cout << "Number of inliers found with first transform: " << maybe_inliers.size () << "\n";
          // Condition: there is more than a quarter of possible matches
          if(maybe_inliers.size () >= pair.second.second->size ()/4)
            //            && max_nb_of_inliers < maybe_inliers.size () )
          {
            // Compute keypoints matches with maybe inliers
            std::vector<int> matches_of_maybe_inliers = computeKeypointsMatches(maybe_inliers, *maybe_inliers_feature_cloud,
                                                                                pair.second);
            cloud_src.reset(new PointCloudT);
            cloud_tgt.reset(new PointCloudT);
            for (size_t idx = 0; idx < matches_of_maybe_inliers.size (); ++idx)
            {
              cloud_src->push_back ((*prev_voxel_centroid_cloud_)[matches_of_maybe_inliers[idx]]);
              cloud_tgt->push_back ((*getUnlabeledVoxelCentroidCloud ())[maybe_inliers[idx]]);
            }
            estimator->estimateRigidTransformation (*cloud_src, *cloud_tgt, transformation_est);

            // Compute the new transformed centroid of the current supervoxel being matched
            new_centroid = transformation_est*prev_centroid;

            pcl::PointXYZ cent_refit(new_centroid[0], new_centroid[1], new_centroid[2]);
            // Do radius search around new estimated centroid
            // This search is only based on spatiality in a radius of
            // seed_resolution (we are looking for the previous supervoxel)
            unlabeled_voxel_cloud_search->radiusSearch (cent_refit, seed_resolution_, k_indices, k_sqr_distances);
            //            std::cout << "number of potential inliers refit: " << k_indices.size () <<"\n";
            // Find the current keypoints that are in this potential supervoxel
            std::vector<int> inliers;
            PointCloudFeatureT::Ptr inliers_feature_cloud (new PointCloudFeatureT);
            std::vector<int> indices_copy = *current_keypoints.first;
            PointCloudFeatureT::Ptr cloud_copy (new PointCloudFeatureT);
            copyPointCloud (*current_keypoints.second, *cloud_copy);

            float err = 0.0f;
            for(auto neigh_ind: k_indices)
            {
              std::vector<int>::iterator vec_it = std::find (indices_copy.begin (), indices_copy.end (), neigh_ind);
              // If indice in the current keypoints (without maybe inliers)
              if( vec_it != indices_copy.end ())
              {
                std::vector<int> indices;
                std::vector<float> distances;
                // Search for the nearest neighbour in feature space
                // between corresponding keypoint and keypoints in previous
                // supervoxel
                search.nearestKSearch((*cloud_copy)[vec_it-indices_copy.begin ()], 1, indices, distances);
                // If the NN of this point is close enough it's an inlier
                if(distances[0] < threshold)
                {
                  inliers.push_back (neigh_ind);
                  inliers_feature_cloud->push_back (cloud_copy->at(vec_it-indices_copy.begin ()));
                  err += distances[0];
                }
              }
            }
            if(inliers.size () > maybe_inliers.size ()
               && max_nb_of_inliers < inliers.size () )
              //               && err < best_err)//
            {
              best_err = err;
              best_fit = transformation_est;
              max_nb_of_inliers = inliers.size ();
            }
          }
          // (This need a GOOD transform)
          // Transform each point of supervoxel
          // Get NN distance, if below threshold add as inlier
        }
        std::cout << "best fit has " << max_nb_of_inliers << " inliers\n";
        if(max_nb_of_inliers>0)
        {
          pcl::PointXYZRGBA prev_centroid_tmp = supervoxel_clusters[label]->centroid_;
          Eigen::Vector4f prev_centroid(prev_centroid_tmp.x,
                                        prev_centroid_tmp.y,
                                        prev_centroid_tmp.z,
                                        1);
          Eigen::Vector4f new_centroid = best_fit*prev_centroid;
          found_transforms.insert (std::pair<uint32_t, Eigen::Matrix<float, 4, 4>>
                                   (label, best_fit));
          labels.push_back(label);
        }
        //    return bestFit
      }
    }
  }
  // Update the octree to remove the matched SVs
  sequential_octree_->updateOctreeFromMatchedClouds(labels);
  computeVoxelData (); // COULD DO BETTER DON'T NEED TO RECALCULATE THE NORMALS
  return found_transforms;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::getPreviousSeedingPoints(SequentialSVMapT &supervoxel_clusters, std::vector<int>& existing_seed_indices)
{
  //  std::unordered_map <uint32_t, pcl::recognition::ObjRecRANSAC::Output> matches;// = getMatches(supervoxel_clusters);
  std::unordered_map<uint32_t, Eigen::Matrix<float, 4, 4>> matches = getMatchesRANSAC (supervoxel_clusters);
  lines_.clear ();

  existing_seed_indices.clear ();
  supervoxel_helpers_.clear ();
  typename SequentialSVMapT::iterator sv_itr;
  // Iterate over all previous supervoxel clusters
  for (sv_itr = supervoxel_clusters.begin (); sv_itr != supervoxel_clusters.end (); ++sv_itr)
  {
    uint32_t label = sv_itr->first;
    // Push back a new supervoxel helper with an already existing label
    supervoxel_helpers_.push_back (new SequentialSupervoxelHelper(label,this));
    // Count the number of points belonging to that supervoxel and compute its centroid
    int nb_of_points = 0;
    PointT centroid;
    centroid.getVector3fMap ().setZero ();
    for (typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
    {
      SequentialVoxelData& voxel = (*leaf_itr)->getData ();
      if(voxel.label_ == label)
      {
        centroid.getVector3fMap () += voxel.xyz_;
        nb_of_points += 1;
      }
    }
    std::unordered_map<uint32_t, Eigen::Matrix<float, 4, 4>>::iterator unmap_matches_it = matches.find (label);
    // If the supervoxel was found using the matching method
    if (unmap_matches_it != matches.end ())
    {
      pcl::PointXYZRGBA prev_centroid_tmp = supervoxel_clusters[label]->centroid_;
      Eigen::Vector4f prev_centroid(prev_centroid_tmp.x,
                                    prev_centroid_tmp.y,
                                    prev_centroid_tmp.z,
                                    1);
      Eigen::Vector4f new_centroid = unmap_matches_it->second*prev_centroid;
      lines_.insert (std::pair<uint32_t, std::pair<Eigen::Vector4f, Eigen::Vector4f>>
                     (label,
                      std::pair<Eigen::Vector4f, Eigen::Vector4f>
                      (prev_centroid, new_centroid)));
      // Compute new centroid from matching method transform
      centroid.getVector3fMap () = new_centroid.head<3> ();

      /*
      pcl::PointXYZRGBA prev_centroid_tmp = supervoxel_clusters[label]->centroid_;
      PointT prev_centroid;
      prev_centroid.x = prev_centroid_tmp.x;
      prev_centroid.y = prev_centroid_tmp.y;
      prev_centroid.z = prev_centroid_tmp.z;

      float prev_x = prev_centroid.x;
      float prev_y = prev_centroid.y;
      float prev_z = prev_centroid.z;
      float x = matches.find(label)->second.rigid_transform_[9] // Translation part
          // Rotational part
          + matches.find(label)->second.rigid_transform_[0]*prev_x
          + matches.find(label)->second.rigid_transform_[1]*prev_y
          + matches.find(label)->second.rigid_transform_[2]*prev_z;
      float y = matches.find(label)->second.rigid_transform_[10] // Translation part
          // Rotational part
          + matches.find(label)->second.rigid_transform_[3]*prev_x
          + matches.find(label)->second.rigid_transform_[4]*prev_y
          + matches.find(label)->second.rigid_transform_[5]*prev_z;
      float z = matches.find(label)->second.rigid_transform_[11] // Translation part
          // Rotational part
          + matches.find(label)->second.rigid_transform_[6]*prev_x
          + matches.find(label)->second.rigid_transform_[7]*prev_y
          + matches.find(label)->second.rigid_transform_[8]*prev_z;

      Eigen::Vector3f new_centroid (x, y, z);

      //      lines_.insert (std::pair<uint32_t, std::pair<Eigen::Vector3f, Eigen::Vector3f>> (label, std::pair<Eigen::Vector3f, Eigen::Vector3f> (Eigen::Vector3f(prev_x, prev_y, prev_z), new_centroid)));
      // Compute new centroid from matching method transform
      centroid.getVector3fMap () = new_centroid;
      */
    }
    // If there was points in it, add the closest point in kdtree as the seed point for this supervoxel
    else if (nb_of_points != 0)
    {
      centroid.getVector3fMap () /= static_cast<double> (nb_of_points);
    }
    // If there was no point in this supervoxel at this frame, just remove it
    else
    {
      supervoxel_helpers_.pop_back();
      continue;
    }
    std::vector<int> closest_index;
    std::vector<float> distance;
    voxel_kdtree_->nearestKSearch (centroid, 1, closest_index, distance);
    LeafContainerT* seed_leaf = sequential_octree_->at (closest_index[0]);
    if (seed_leaf)
    {
      existing_seed_indices.push_back (closest_index[0]);
      (supervoxel_helpers_.back()).addLeaf(seed_leaf);
      (supervoxel_helpers_.back()).setNew(false);
    }
    else
    {
      PCL_WARN ("Could not find leaf in pcl::SequentialSVClustering<PointT>::createHelpersFromWeightMaps - supervoxel will be deleted \n");
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SequentialSVClustering<PointT>::pruneSeeds(std::vector<int> &existing_seed_indices, std::vector<int> &seed_indices)
{
  //TODO THIS IS BAD - SEEDING SHOULD BE BETTER
  //TODO Switch to assigning leaves! Don't use Octree!

  // std::cout << "Size of centroid cloud="<<voxel_centroid_cloud_->size ()<<", seeding resolution="<<seed_resolution_<<"\n";
  //Initialize octree with voxel centroids
  pcl::octree::OctreePointCloudSearch <PointT> seed_octree (seed_resolution_);
  seed_octree.setInputCloud (unlabeled_voxel_centroid_cloud_);
  seed_octree.addPointsFromInputCloud ();
  // std::cout << "Size of octree ="<<seed_octree.getLeafCount ()<<"\n";
  std::vector<PointT, Eigen::aligned_allocator<PointT> > voxel_centers;
  int num_seeds = seed_octree.getOccupiedVoxelCenters(voxel_centers);
  //std::cout << "Number of seed points before filtering="<<voxel_centers.size ()<<"\n";

  std::vector<int> seed_indices_orig;
  seed_indices_orig.resize (static_cast<size_t> (num_seeds), 0);
  seed_indices.clear ();
  std::vector<int> closest_index;
  std::vector<float> distance;
  closest_index.resize(1,0);
  distance.resize(1,0);

  for (size_t i = 0; i < num_seeds; ++i)
  {
    // Search for the nearest neighbour to voxel center[i], stores its index in closest_index and distance in distance
    voxel_kdtree_->nearestKSearch (voxel_centers[i], 1, closest_index, distance);
    seed_indices_orig[i] = closest_index[0];
  }

  std::vector<int> neighbors;
  std::vector<float> sqr_distances;
  seed_indices.reserve (seed_indices_orig.size ());
  float search_radius = 0.51f*seed_resolution_;
  // This is 1/20th of the number of voxels which fit in a planar slice through search volume
  // Area of planar slice / area of voxel side. (Note: This is smaller than the value mentioned in the original paper)
  float min_points = 0.05f * (search_radius)*(search_radius) * 3.1415926536f  / (resolution_*resolution_);
  for (size_t i = 0; i < seed_indices_orig.size (); ++i)
  {
    int num = voxel_kdtree_->radiusSearch (seed_indices_orig[i], search_radius , neighbors, sqr_distances);

    int min_index = seed_indices_orig[i];
    bool not_too_close = true;
    // For all neighbours
    for(size_t j = 0 ; j < neighbors.size() ; ++j )
    {
      if(not_too_close)
      {
        // For all existing seed indices
        for(size_t k = 0 ; k < existing_seed_indices.size() ; ++k)
        {
          if(neighbors[j] == existing_seed_indices[k])
          {
            not_too_close = false;
            break;
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
      PCL_WARN ("Could not find leaf in pcl::SequentialSVClustering<PointT>::addHelpersFromUnlabeledSeedIndices - supervoxel will be deleted \n");
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{

  template<> void
  pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData::getPoint (pcl::PointXYZRGB &point_arg) const;

  template<> void
  pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData::getPoint (pcl::PointXYZRGBA &point_arg ) const;

  template<typename PointT> void
  pcl::SequentialSVClustering<PointT>::SequentialVoxelData::getPoint (PointT &point_arg ) const
  {
    //XYZ is required or this doesn't make much sense...
    point_arg.x = xyz_[0];
    point_arg.y = xyz_[1];
    point_arg.z = xyz_[2];
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename PointT> void
  pcl::SequentialSVClustering<PointT>::SequentialVoxelData::getNormal (Normal &normal_arg) const
  {
    normal_arg.normal_x = normal_[0];
    normal_arg.normal_y = normal_[1];
    normal_arg.normal_z = normal_[2];
    normal_arg.curvature = curvature_;
  }
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
pcl::SequentialSVClustering<PointT>::getPrevVoxelCentroidCloud () const
{
  typename PointCloudT::Ptr centroid_copy (new PointCloudT);
  copyPointCloud (*prev_voxel_centroid_cloud_, *centroid_copy);
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> pcl::PointCloud<pcl::Normal>::Ptr
pcl::SequentialSVClustering<PointT>::getVoxelNormalCloud () const
{
  NormalCloud::Ptr normal_cloud (new NormalCloud);
  normal_cloud->resize(sequential_octree_->getLeafCount ());
  NormalCloud::iterator normal_cloud_itr = normal_cloud->begin();
  typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin ();
  for (; leaf_itr != sequential_octree_->end (); ++leaf_itr, ++normal_cloud_itr)
  {
    SequentialVoxelData& new_voxel_data = (*leaf_itr)->getData ();
    // Add the point to the normal cloud
    new_voxel_data.getNormal (*normal_cloud_itr);
  }
  return normal_cloud;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> pcl::PointCloud<pcl::Normal>::Ptr
pcl::SequentialSVClustering<PointT>::getUnlabeledVoxelNormalCloud () const
{
  NormalCloud::Ptr normal_cloud (new NormalCloud);
  normal_cloud->resize(getUnlabeledVoxelCentroidCloud()->size ());
  NormalCloud::iterator normal_cloud_itr = normal_cloud->begin();
  typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin ();
  for (; leaf_itr != sequential_octree_->end (); ++leaf_itr)
  {
    SequentialVoxelData& new_voxel_data = (*leaf_itr)->getData ();
    if(new_voxel_data.label_ == -1)
    {
      // Add the point to the normal cloud
      new_voxel_data.getNormal (*normal_cloud_itr);
      ++normal_cloud_itr;
    }
  }
  return normal_cloud;
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
template <typename PointT> pcl::PointCloud<pcl::PointXYZRGBL>::Ptr
pcl::SequentialSVClustering<PointT>::getLabeledRGBVoxelCloud () const
{
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr labeled_voxel_cloud (new pcl::PointCloud<pcl::PointXYZRGBL>);
  for (typename HelperListT::const_iterator sv_itr = supervoxel_helpers_.cbegin (); sv_itr != supervoxel_helpers_.cend (); ++sv_itr)
  {
    typename pcl::PointCloud<PointT>::Ptr voxels;
    sv_itr->getVoxels (voxels);
    pcl::PointCloud<pcl::PointXYZRGBL> xyzrgbl_copy;
    copyPointCloud (*voxels, xyzrgbl_copy);

    pcl::PointCloud<pcl::PointXYZRGBL>::iterator xyzrgbl_copy_itr = xyzrgbl_copy.begin ();
    for ( ; xyzrgbl_copy_itr != xyzrgbl_copy.end (); ++xyzrgbl_copy_itr)
      xyzrgbl_copy_itr->label = sv_itr->getLabel ();

    *labeled_voxel_cloud += xyzrgbl_copy;
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
      /** \bug Sometimes points don't have a leaf container (maximum 5 points on more than 200 000) so we need the if statement
       */
      if(leaf)
      {
        SequentialVoxelData& voxel_data = leaf->getData ();
        if (voxel_data.owner_)
          i_labeled->label = voxel_data.owner_->getLabel ();
      }
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
template <typename PointT> std::vector<uint32_t>
pcl::SequentialSVClustering<PointT>::getLabelColors ()
{
  return label_colors_;
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

