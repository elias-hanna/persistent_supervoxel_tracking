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

#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include "../impl/sequential_supervoxel_clustering.hpp"
#include "../../octree/impl/octree_pointcloud_sequential.hpp"

namespace pcl
{ 
  namespace octree
  {
    //Explicit overloads for RGB types
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB,pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGB &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ++num_points_;
      
      #ifdef _OPENMP
      #pragma omp critical (add_point)
      #endif
      data_.point_accumulator_.add (new_point);
    }
    
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA,pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGBA &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ++num_points_;
      
      #ifdef _OPENMP
      #pragma omp critical (add_point)
      #endif
      data_.point_accumulator_.add (new_point);
    }
    
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal,pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGBNormal &new_point)
    {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ++num_points_;
      
      #ifdef _OPENMP
      #pragma omp critical (add_point)
      #endif
      data_.point_accumulator_.add (new_point);
    }
    
    //Explicit overloads for RGB types
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB,pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData>::computeData ()
    {
      data_.point_accumulator_.get (data_.voxel_centroid_);
    }
    
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA,pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData>::computeData ()
    {
      data_.point_accumulator_.get (data_.voxel_centroid_);
    }
    
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal,pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData>::computeData ()
    {
      data_.point_accumulator_.get (data_.voxel_centroid_);
    }
    
    
    template<> float
    OctreePointCloudSequential<pcl::PointXYZRGBA,
    OctreePointCloudSequentialContainer <pcl::PointXYZRGBA, SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData > > 
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZRGBA, SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData >* leaf)
    {
      float color_dist =  (leaf->getData ().voxel_centroid_.getRGBVector3i () 
                           - leaf->getData ().previous_centroid_.getRGBVector3i ()).norm () / 255.0f;
      
      return color_dist;
    }
    
    template<> float
    OctreePointCloudSequential<pcl::PointXYZRGB,
    OctreePointCloudSequentialContainer <pcl::PointXYZRGB, SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData > > 
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZRGB, SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData >* leaf)
    {
      float color_dist =  (leaf->getData ().voxel_centroid_.getRGBVector3i () 
      - leaf->getData ().previous_centroid_.getRGBVector3i ()).norm () / 255.0f;
      
      return color_dist;
    }
    
    template<> float
    OctreePointCloudSequential<pcl::PointXYZRGBNormal,
    OctreePointCloudSequentialContainer <pcl::PointXYZRGBNormal, SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData > > 
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZRGBNormal, SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData >* leaf)
    {
      float color_dist =  (leaf->getData ().voxel_centroid_.getRGBVector3i () 
      - leaf->getData ().previous_centroid_.getRGBVector3i ()).norm () / 255.0f;
      
  //    float cos_angle_normal = 1.0f - (leaf->getData ().voxel_centroid_.getNormalVector4fMap ().dot (leaf->getData ().previous_centroid_.getNormalVector4fMap ()));
      
      return color_dist;
    }
  }
}


PCL_INSTANTIATE(SequentialSVClustering, (pcl::PointXYZRGB)(pcl::PointXYZRGBNormal)(pcl::PointXYZRGBA))

#include "../../octree/impl/octree_pointcloud_sequential.hpp"
typedef pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData SeqVoxelDataRGBT;
typedef pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData SeqVoxelDataRGBAT;
typedef pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData SeqVoxelDataRGBNormalT;

typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB, SeqVoxelDataRGBT> SequentialContainerRGB;
typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA, SeqVoxelDataRGBAT> SequentialContainerRGBA;
typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal, SeqVoxelDataRGBNormalT> SequentialContainerRGBNormal;

template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB, SeqVoxelDataRGBT>;
template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA, SeqVoxelDataRGBAT>;
template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal, SeqVoxelDataRGBNormalT>;

template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGB, SequentialContainerRGB>;
template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGBA, SequentialContainerRGBA>;
template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGBNormal, SequentialContainerRGBNormal>;

