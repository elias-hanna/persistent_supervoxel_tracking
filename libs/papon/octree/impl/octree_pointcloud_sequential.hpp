/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012, Jeremie Papon
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 */

#ifndef PCL_OCTREE_POINTCLOUD_SEQUENTIAL_HPP_

#define PCL_OCTREE_POINTCLOUD_SEQUENTIAL_HPP_

#include "../octree_pointcloud_sequential.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> 
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::OctreePointCloudSequential (const double resolution_arg)
: OctreePointCloud<PointT, LeafContainerT, BranchContainerT
  , OctreeBase<LeafContainerT, BranchContainerT> > (resolution_arg),
  diff_func_ (0),
  difference_threshold_ (0.1f),
  occlusion_test_interval_ (0.5f),
  threads_ (1),
  stored_keys_valid_ (false),
  frame_counter_ (0)
{
 
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::addPointsFromInputCloud ()
{
  ++frame_counter_;
  //If we're empty, just use plain version 
  if (leaf_vector_.size () == 0 )
  {
    OctreePointCloud<PointT, LeafContainerT, BranchContainerT>::addPointsFromInputCloud ();
    LeafContainerT *leaf_container;
    typename OctreeSequentialT::LeafNodeIterator leaf_itr;
    leaf_vector_.reserve (this->getLeafCount ());
    key_vector_.reserve (this->getLeafCount ());
    for ( leaf_itr = this->leaf_begin () ; leaf_itr != this->leaf_end (); ++leaf_itr)
    {
      boost::shared_ptr<OctreeKey> leaf_key (new OctreeKey);
      *leaf_key = leaf_itr.getCurrentOctreeKey ();
      //std::cout << "Key\nx: " << (*leaf_key).x << " y: " << (*leaf_key).y << " z: " << (*leaf_key).z << std::endl;
      //OctreeKey leaf_key = leaf_itr.getCurrentOctreeKey ();
      leaf_container = &(leaf_itr.getLeafContainer ());
      
      //Run the leaf's compute function
      leaf_container->computeData ();
      this->computeNeighbors (*leaf_key, leaf_container);
      
      leaf_vector_.push_back (leaf_container);
      key_vector_.push_back (leaf_key);
    }
    return;
  }
  //Otherwise, start sequential update
  //Go through and reset all the containers for the new frame.
  typename LeafVectorT::iterator leaf_itr;
  for (leaf_itr = leaf_vector_.begin () ; leaf_itr != leaf_vector_.end (); ++leaf_itr)
  {
    (*leaf_itr)->getData ().prepareForNewFrame ();
    (*leaf_itr)->resetPointCount ();
  }

  // Clear and Alloc memory for the new vectors containing leaves and keys. 
  // Alloc at least same size as current leaf_vector
  new_leaves_.clear ();
  new_leaves_.reserve (this->getLeafCount ());
  new_keys_.clear ();
  new_keys_.reserve (this->getLeafCount ());

  //Adapt the octree to fit all input points
  /*
  int depth_before_addition = this->octree_depth_;
  for (size_t i = 0; i < input_->points.size (); i++)
  {
    adoptBoundingBoxToPoint (input_->points[i]);
  }
  //If depth has changed all stored keys are not valid
  if (this->octree_depth_ != depth_before_addition)
  {
    stored_keys_valid_ = false;
  }
  */

  //Now go through and add all points to the octree
  // #ifdef _OPENMP
  // #pragma omp parallel for schedule(dynamic, 1024) num_threads(threads_)
  // #endif
  for (size_t i = 0; i < input_->points.size (); i++)
  {
    // Add points to the octree. If point fall within an existing leaf, add it to the leaf, otherwise create a new leaf
    // The new points (those that created new leaves) are stored within new_leaves_
    addPointSequential (static_cast<unsigned int> (i));
  }
  // parallelAddPoint();
  //If Geometrically new - need to reciprocally update neighbors and compute Data
  //New frame leaf/key vectors now contain only the geometric new leaves
  for (size_t i = 0; i < new_leaves_.size (); ++i)
  {
    this->computeNeighbors (*new_keys_[i],new_leaves_[i]);
    (new_leaves_[i])->computeData ();
    //(new_frame_pairs_[i].first)->getData ().setNew (true);
  }
  // This will store old leaves that will need to be deleted
  LeafVectorT delete_leaves;
  std::vector<OctreeKey> delete_keys;
  delete_leaves.reserve (this->getLeafCount () / 4); //Probably a reasonable upper bound
  delete_keys.reserve (this->getLeafCount () / 4);


  // parallelUpdate (&leaf_vector_, &key_vector_, &delete_leaves, &delete_keys, &new_leaves_, &new_keys_);
  // std::cout << "I finished to update in parallel" << std::endl;

  // for (size_t i = 0 ; i < leaf_vector_.size () ; ++i)
  // {
  //   LeafContainerT* leaf_container = leaf_vector_[i];
  //   // std::cout << leaf_container->getData ().idx_ << std::endl;
  // }

  // std::cout << "Passed" << std::endl;
  //Now we need to iterate through all leaves from previous frame - 
  // #ifdef _OPENMP
  // #pragma omp parallel for schedule(dynamic, 1024) shared (delete_leaves, delete_keys) //num_threads(threads_)
  // #endif
  for (size_t i = 0; i < leaf_vector_.size (); ++i)
  {
    LeafContainerT *leaf_container = leaf_vector_[i];
    boost::shared_ptr<OctreeKey> key = key_vector_[i];
    //If no neighbors probably noise - delete 
    if (leaf_container->getNumNeighbors () <= 1)
    {
      #ifdef _OPENMP
      #pragma omp critical (delete_leaves)
      #endif
      {
        delete_leaves.push_back (leaf_container); 
        delete_keys.push_back (*key);
      }
    }
    //Check if the leaf had no points observed this frame
    else if (leaf_container->getPointCounter () == 0)
    {
      std::pair<float, LeafContainerT*> occluder_pair = testForOcclusion (key, leaf_container);
      //If occluded (distance to occluder != 0)
      if (occluder_pair.first != 0.0f)
      {
        
        //This is basically a test to remove extra voxels caused by objects moving towards the camera
        if (occluder_pair.first <= 4.0f || (testForNewNeighbors(leaf_container)
                                              && leaf_container->getData().frame_occluded_ != occluder_pair.second->getData().frame_occluded_))
        {
          // #ifdef _OPENMP
          // #pragma omp critical (delete_leaves)
          // #endif
          {  
            delete_leaves.push_back (leaf_container); 
            delete_keys.push_back (*key);
          }
        }
        else //otherwise add it to the current leaves and revert it to last timestep (since current has nothing in it)
        { //TODO Maybe maintain a separate list of occluded leaves?
          // #ifdef _OPENMP
          // #pragma omp critical (new_leaves_)
          // #endif
          {   
            new_leaves_.push_back (leaf_container); 
            new_keys_.push_back (key);
          }
          if (leaf_container->getData ().frame_occluded_ == 0)
            leaf_container->getData ().frame_occluded_ = frame_counter_;
          //We don't need to do this anymore since we're using the accumulator
          //leaf_container->getData ().revertToLastPoint ();
        }
      }
      else //not occluded & not observed safe to delete
      { 
        // #ifdef _OPENMP
        // #pragma omp critical (delete_leaves)
        // #endif
        {  
          delete_leaves.push_back (leaf_container); 
          delete_keys.push_back (*key);
        }
      }
    }
    else //Existed in previous frame and observed so just update data
    {
      // #ifdef _OPENMP
      // #pragma omp critical (new_leaves_)
      // #endif
      {  
        new_leaves_.push_back (leaf_container);
        new_keys_.push_back (key);
      }
      //Compute the data from the points added to the voxel container
      leaf_container->computeData ();
      //Not occluded
      leaf_container->getData ().frame_occluded_ = 0;
      //Use the difference function to check if the leaf has changed
      if ( diff_func_ && diff_func_ (leaf_container) > difference_threshold_)
      {
        leaf_container->getData ().setChanged (true);
        // leaf_container->getData ().label_ = -1;
      }
      else
      {
        leaf_container->getData ().setChanged (false);
      }
    }
    
  }
  //Swap new leaf_key vector (which now contains old and new combined) for old (which is not needed anymore)
  leaf_vector_.swap (new_leaves_);
  key_vector_.swap (new_keys_);

  //All keys which were stored in new_frame_pairs_ are valid, so this is now true for leaf_key_vec_
  stored_keys_valid_ = true;
 //Go through and delete voxels scheduled
  for (typename LeafVectorT::iterator delete_itr = delete_leaves.begin (); delete_itr != delete_leaves.end (); ++delete_itr)
  {
    LeafContainerT *leaf_container = *delete_itr;
    //Remove pointer to it from all neighbors
    typename std::set<LeafContainerT*>::iterator neighbor_itr = leaf_container->begin ();
    typename std::set<LeafContainerT*>::iterator neighbor_end = leaf_container->end ();   
    for ( ; neighbor_itr != neighbor_end; ++neighbor_itr)
    {
      (*neighbor_itr)->removeNeighbor (leaf_container);
    }
  }

  typename std::vector<OctreeKey>::iterator delete_key_itr = delete_keys.begin ();
  for ( ; delete_key_itr != delete_keys.end (); ++delete_key_itr)
  {
    this->removeLeaf ( *delete_key_itr );
  }
  //Final check to make sure they match the leaf_key_vector is correct size after deletion
  assert (leaf_vector_.size () == this->getLeafCount ());
  assert (key_vector_.size () == leaf_vector_.size ());

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> LeafContainerT*
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::getLeafContainerAtPoint (const PointT& point_arg) const
{
  OctreeKey key;
  LeafContainerT* leaf = 0;
  // generate key
//  OctreeAdjacencyT::genOctreeKeyforPoint (point_arg, key);
  OctreePointCloudT::genOctreeKeyforPoint (point_arg, key);

  leaf = this->findLeaf (key);
  
  return leaf;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::addPointSequential (const int point_index_arg)
{
  const PointT& point = this->input_->points[point_index_arg];
  
  if (!isFinite (point))
    return;
  
  boost::shared_ptr<OctreeKey> key (new OctreeKey);
  
  // generate key - use adjacency function since it possibly has a transform
//  OctreeAdjacencyT::genOctreeKeyforPoint (point, *key);
  OctreePointCloudT::genOctreeKeyforPoint (point, *key);

  // Check if leaf exists in octree at key
  LeafContainerT* container = this->findLeaf(*key);
  // std::cout << "In function addPointSequential owner_: " << container->getData().owner_ << std::endl; 
  if (container == 0) //If not, do a lock and add the leaf
  {
    boost::mutex::scoped_lock (create_mutex_);
    //Check again, since another thread might have created between the first find and now
    container = this->findLeaf(*key);
    if (container == 0)
    {
      container = this->createLeaf(*key); //This is fine if the leaf has already been created by another
      if (container == 0)
      {
        PCL_ERROR ("FAILED TO CREATE LEAF in OctreePointCloudSequential::addPointSequential");
        return;
      }
      new_leaves_.push_back (container);
      new_keys_.push_back (key);
    }
  }
  //Add the point to the leaf
  container->addPoint (point);
  
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::computeNeighbors (OctreeKey &key_arg, LeafContainerT* leaf_container)
{ 
  //Make sure requested key is valid
  if (key_arg.x > this->max_key_.x || key_arg.y > this->max_key_.y || key_arg.z > this->max_key_.z)
  {
    PCL_ERROR ("OctreePointCloudSequential::computeNeighbors Requested neighbors for invalid octree key\n");
    return;
  }

  OctreeKey neighbor_key;
  int dx_min = (key_arg.x > 0) ? -1 : 0;
  int dy_min = (key_arg.y > 0) ? -1 : 0;
  int dz_min = (key_arg.z > 0) ? -1 : 0;
  int dx_max = (key_arg.x == this->max_key_.x) ? 0 : 1;
  int dy_max = (key_arg.y == this->max_key_.y) ? 0 : 1;
  int dz_max = (key_arg.z == this->max_key_.z) ? 0 : 1;
  for (int dx = dx_min; dx <= dx_max; ++dx)
  {
    for (int dy = dy_min; dy <= dy_max; ++dy)
    {
      for (int dz = dz_min; dz <= dz_max; ++dz)
      {
        neighbor_key.x = static_cast<uint32_t> (key_arg.x + dx);
        neighbor_key.y = static_cast<uint32_t> (key_arg.y + dy);
        neighbor_key.z = static_cast<uint32_t> (key_arg.z + dz);
        LeafContainerT *neighbor = this->findLeaf (neighbor_key);
        if (neighbor && neighbor != leaf_container)
        {
          leaf_container->addNeighbor (neighbor);
          neighbor->addNeighbor (leaf_container);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> bool
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::testForNewNeighbors (const LeafContainerT* leaf_container) const
{
  typename LeafContainerT::const_iterator neighb_itr = leaf_container->cbegin ();
  for ( ; neighb_itr != leaf_container->cend (); ++neighb_itr)
  {
    if ( (*neighb_itr)->getData ().isNew () )
      return true;
  }
  return false;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename LeafContainerT, typename BranchContainerT> std::pair<float, LeafContainerT*>
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::testForOcclusion (boost::shared_ptr<OctreeKey> &key, LeafContainerT *leaf_container) const
{
  OctreeKey kinect_min_z;
  PointT min_z_pt;
  min_z_pt.getVector3fMap ().setZero ();
  min_z_pt.z = 0.3;
//  OctreeAdjacencyT::genOctreeKeyforPoint (min_z_pt, kinect_min_z);
  OctreePointCloudT::genOctreeKeyforPoint (min_z_pt, kinect_min_z);

  OctreeKey current_key = *key;
  OctreeKey camera_key;
  PointT temp;
  temp.getVector3fMap ().setZero ();
  if (transform_func_)
    temp.z = 0.03f; //Can't set to zero if using a transform.
//  OctreeAdjacencyT::genOctreeKeyforPoint (temp, camera_key);
  OctreePointCloudT::genOctreeKeyforPoint (temp, camera_key);
  Eigen::Vector3f camera_key_vals (camera_key.x, camera_key.y, camera_key.z);
  Eigen::Vector3f leaf_key_vals (current_key.x,current_key.y,current_key.z);
  
  Eigen::Vector3f direction = (camera_key_vals - leaf_key_vals);
  float norm = direction.norm ();
  direction.normalize ();
  
  const int nsteps = std::max (1, static_cast<int> (norm / occlusion_test_interval_));
  leaf_key_vals += (direction); //* occlusion_test_interval_);
  OctreeKey test_key;
  LeafContainerT* occluder = 0;
  // Walk along the line segment with small steps.
  for (int i = 0; i < nsteps; ++i)
  {
    //Start at the leaf voxel, and move back towards sensor.
    leaf_key_vals += (direction * occlusion_test_interval_);
    //This is a shortcut check - if we're outside of the bounding box of the 
    //octree there's no possible occluders. It might be worth it to check all, but < min_z_ is probably sufficient.
    if (leaf_key_vals.z () < kinect_min_z.z) 
      return std::make_pair (0,occluder);
    //Now we need to round the key
    test_key.x = ::round(leaf_key_vals.x ());
    test_key.y = ::round(leaf_key_vals.y ());
    test_key.z = ::round(leaf_key_vals.z ());
    
    if (test_key == current_key)
      continue;
    
    current_key = test_key;
    
    occluder = this->findLeaf (test_key);
    //If the voxel is occupied, there is a possible occlusion
    if (occluder)
    {
      float voxels_to_occluder= 1 + i *occlusion_test_interval_;
      if (voxels_to_occluder <= 2.5f && !occluder->getData().isNew ())
        continue;
      return std::make_pair (voxels_to_occluder, occluder); 
    }
  }
  //If we didn't run into a leaf on the way to this camera, it can't be occluded.
  return std::make_pair (0,occluder);
}

template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::parallelUpdate ( LeafVectorT* leaves, KeyVectorT* keys, LeafVectorT* delete_leaves, std::vector<OctreeKey>* delete_keys, LeafVectorT* new_leaves, KeyVectorT* new_keys)//typename OctreeSequentialT::Ptr sequential_octree, typename VoxelCloudT::Ptr unlabeled_voxel_centroid_cloud, typename VoxelCloudT::Ptr voxel_centroid_cloud)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaf_vector_.size()),
                      [=](const tbb::blocked_range<size_t>& r)
    {
        for (size_t i = r.begin() ; i < r.end(); ++i)
        {
          LeafContainerT *leaf_container = (*leaves)[i];
          boost::shared_ptr<OctreeKey> key = (*keys)[i];
          //If no neighbors probably noise - delete
          if (leaf_container->getNumNeighbors () <= 1)
          {
              boost::mutex::scoped_lock (delete_mutex_);
              delete_leaves->push_back (leaf_container);
              delete_keys->push_back (*key);
          }
          //Check if the leaf had no points observed this frame
          else if (leaf_container->getPointCounter () == 0)
          {
            std::pair<float, LeafContainerT*> occluder_pair = testForOcclusion (key, leaf_container);
            //If occluded (distance to occluder != 0)
            if (occluder_pair.first != 0.0f)
            {
              //This is basically a test to remove extra voxels caused by objects moving towards the camera
              if (occluder_pair.first <= 4.0f || (testForNewNeighbors(leaf_container)
                                                    && leaf_container->getData().frame_occluded_ != occluder_pair.second->getData().frame_occluded_))
              {
                boost::mutex::scoped_lock (delete_mutex_);
                delete_leaves->push_back (leaf_container);
                delete_keys->push_back (*key);
              }
              else //otherwise add it to the current leaves and revert it to last timestep (since current has nothing in it)
              { //TODO Maybe maintain a separate list of occluded leaves?
                {
                  boost::mutex::scoped_lock (new_mutex_);
                  new_leaves->push_back (leaf_container);
                  new_keys->push_back (key);
                }
                if (leaf_container->getData ().frame_occluded_ == 0)
                  leaf_container->getData ().frame_occluded_ = frame_counter_;
                //We don't need to do this anymore since we're using the accumulator
                //leaf_container->getData ().revertToLastPoint ();
              }
            }
            else //not occluded & not observed safe to delete
            {
              boost::mutex::scoped_lock (delete_mutex_);
              delete_leaves->push_back (leaf_container);
              delete_keys->push_back (*key);
            }
          }
          else //Existed in previous frame and observed so just update data
          {
            {
              boost::mutex::scoped_lock (new_mutex_);
              new_leaves->push_back (leaf_container);
              new_keys->push_back (key);
            }
            //Compute the data from the points added to the voxel container
            leaf_container->computeData ();
            //Not occluded
            leaf_container->getData ().frame_occluded_ = 0;
            //Use the difference function to check if the leaf has changed
            if ( diff_func_ && diff_func_ (leaf_container) > difference_threshold_)
            {
              leaf_container->getData ().setChanged (true);
              // leaf_container->getData ().label_ = -1;
            }
            else
            {
              leaf_container->getData ().setChanged (false);
            }
          }
        }
    }
    );
}

template<typename PointT, typename LeafContainerT, typename BranchContainerT> void
pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT, BranchContainerT>::parallelAddPoint ()//typename OctreeSequentialT::Ptr sequential_octree, typename VoxelCloudT::Ptr unlabeled_voxel_centroid_cloud, typename VoxelCloudT::Ptr voxel_centroid_cloud)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, input_->points.size()),
                      [=](const tbb::blocked_range<size_t>& r)
    {
        for (size_t i = r.begin () ; i != r.end (); ++i)
        {
            addPointSequential (static_cast<unsigned int> (i));
        }
    }
    );
}

/*
////////////////////////////////////////////////////////////////////////////////
// The rest are container explicit instantiations for XYZ, XYZRGB, and XYZRGBA  point types ///
////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  namespace octree
  {
    /// XYZRGBA  ////////////////////////////////////////
    template<>
    float
    OctreePointCloudSequential<PointXYZRGBA,
    OctreePointCloudSequentialContainer <PointXYZRGBA, SequentialVoxelData<PointXYZRGBA> > > 
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <PointXYZRGBA, SequentialVoxelData<PointXYZRGBA> >* leaf)
    {
      float temp1 = leaf->getData ().rgb_.norm () ;
      float temp2 = leaf->getData ().rgb_old_.norm ();
      
      return 1.0f - (leaf->getData ().rgb_ / temp1).dot ((leaf->getData ().rgb_old_ / temp2));
    }
    
    template<>
    void
    OctreePointCloudSequentialContainer<PointXYZRGBA,
    SequentialVoxelData<PointXYZRGBA> >::addPoint (const pcl::PointXYZRGBA &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      num_points_++;     
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[0] += new_point.x;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[1] += new_point.y;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[2] += new_point.z;
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[0] += static_cast<float> (new_point.r); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[1] += static_cast<float> (new_point.g); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[2] += static_cast<float> (new_point.b); 
    }
    
    template<> void
    OctreePointCloudSequentialContainer<PointXYZRGBA,
    SequentialVoxelData<PointXYZRGBA> >::computeData ()
    {
      data_.rgb_ /= static_cast<float> (num_points_ + num_prev_);
      data_.xyz_ /= static_cast<float> (num_points_ + num_prev_);
    }
    
    template<> void
    SequentialVoxelData<PointXYZRGBA>::getPoint (pcl::PointXYZRGBA &point_arg ) const
    {
      point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 | 
      static_cast<uint32_t>(rgb_[1]) << 8 | 
      static_cast<uint32_t>(rgb_[2]);  
      point_arg.x = xyz_[0];
      point_arg.y = xyz_[1];
      point_arg.z = xyz_[2];
    }
    
    /// XYZRGBNormal  ////////////////////////////////////////
    template<>
    float
    OctreePointCloudSequential<PointXYZRGBNormal,
    OctreePointCloudSequentialContainer <PointXYZRGBNormal, SequentialVoxelData<PointXYZRGBNormal> > > 
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <PointXYZRGBNormal, SequentialVoxelData<PointXYZRGBNormal> >* leaf)
    {
      float temp1 = leaf->getData ().rgb_.norm () ;
      float temp2 = leaf->getData ().rgb_old_.norm ();
      
      return 1.0f - (leaf->getData ().rgb_ / temp1).dot ((leaf->getData ().rgb_old_ / temp2));
    }
    
    template<>
    void
    OctreePointCloudSequentialContainer<PointXYZRGBNormal,
    SequentialVoxelData<PointXYZRGBNormal> >::addPoint (const pcl::PointXYZRGBNormal &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      num_points_++;     
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[0] += new_point.x;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[1] += new_point.y;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[2] += new_point.z;
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[0] += static_cast<float> (new_point.r); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[1] += static_cast<float> (new_point.g); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[2] += static_cast<float> (new_point.b); 
    }
    
    template<> void
    OctreePointCloudSequentialContainer<PointXYZRGBNormal,
    SequentialVoxelData<PointXYZRGBNormal> >::computeData ()
    {
      data_.rgb_ /= static_cast<float> (num_points_ + num_prev_);
      data_.xyz_ /= static_cast<float> (num_points_ + num_prev_);
    }
    
    template<> void
    SequentialVoxelData<PointXYZRGBNormal>::getPoint (pcl::PointXYZRGBNormal &point_arg ) const
    {
      point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 | 
      static_cast<uint32_t>(rgb_[1]) << 8 | 
      static_cast<uint32_t>(rgb_[2]);  
      point_arg.x = xyz_[0];
      point_arg.y = xyz_[1];
      point_arg.z = xyz_[2];
    }
    // XYZRGB ///////////////////////////////
    template<>
    float
    OctreePointCloudSequential<PointXYZRGB,
    OctreePointCloudAdjacencyContainer <PointXYZRGB, SequentialVoxelData<PointXYZRGB> > > 
    ::SeqVoxelDataDiff (const OctreePointCloudAdjacencyContainer <PointXYZRGB, SequentialVoxelData<PointXYZRGB> >* leaf)
    {
      return (leaf->getData ().rgb_ - leaf->getData ().rgb_old_).norm () / 255.0f;
    }
    
    template<>
    void
    OctreePointCloudAdjacencyContainer<PointXYZRGB,
    SequentialVoxelData<PointXYZRGB> >::addPoint (const pcl::PointXYZRGB &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ++num_points_;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[0] += new_point.x;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[1] += new_point.y;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[2] += new_point.z;
      
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[0] += static_cast<float> (new_point.r); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[1] += static_cast<float> (new_point.g); 
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.rgb_[2] += static_cast<float> (new_point.b); 
    }
    
    template<> void
    OctreePointCloudAdjacencyContainer<PointXYZRGB,
    SequentialVoxelData<PointXYZRGB> >::computeData ()
    {
      data_.rgb_ /= static_cast<float> (num_points_+num_prev_);
      data_.xyz_ /= static_cast<float> (num_points_+num_prev_);
    }
    
    template<> void
    SequentialVoxelData<PointXYZRGB>::getPoint (pcl::PointXYZRGB &point_arg ) const
    {
      // In XYZRGB you need to do this nonsense
      uint32_t temp_rgb = static_cast<uint32_t>(rgb_[0]) << 16 | 
      static_cast<uint32_t>(rgb_[1]) << 8 | 
      static_cast<uint32_t>(rgb_[2]);
      point_arg.rgb = *reinterpret_cast<float*> (&temp_rgb);  
      point_arg.x = xyz_[0];
      point_arg.y = xyz_[1];
      point_arg.z = xyz_[2];
    }
    
    // XYZ /////////////////////////////////////////
    template<>
    void
    OctreePointCloudAdjacencyContainer<PointXYZ,
    SequentialVoxelData<PointXYZ> >::addPoint (const pcl::PointXYZ &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ++num_points_;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[0] += new_point.x;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[1] += new_point.y;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      data_.xyz_[2] += new_point.z;
    }
    
    template<> void
    OctreePointCloudAdjacencyContainer<PointXYZ,
    SequentialVoxelData<PointXYZ> >::computeData ()
    {
      data_.xyz_ /= static_cast<float> (num_points_ + num_prev_);
    }
    
    template<> void
    SequentialVoxelData<PointXYZ>::getPoint (pcl::PointXYZ &point_arg ) const
    {
      point_arg.x = xyz_[0];
      point_arg.y = xyz_[1];
      point_arg.z = xyz_[2];
    }
    
  }
}
*/

#define PCL_INSTANTIATE_OctreePointCloudSequential(T) template class PCL_EXPORTS pcl::octree::OctreePointCloudSequential<T>;

#endif
