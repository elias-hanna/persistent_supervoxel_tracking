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

/*
 * Do not use pre-compiled versions in this compilation unit (cpp-file),
 * especially for the octree classes. This way the OctreePointCloudAdjacency
 * class is instantiated with the custom leaf container SupervoxelClustering.
 */
namespace pcl
{ 
  namespace octree
  {
    // Explicit overloads for XYZ types
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZ,pcl::SequentialSVClustering<pcl::PointXYZ>::SequentialVoxelData>::addPoint (const pcl::PointXYZ &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
    }

    // Explicit overloads for XYZRGB types
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB,pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGB &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
      //Separate sums for r,g,b since we can't sum in uchars
      data_.rgb_[0] += static_cast<float> (new_point.r);
      data_.rgb_[1] += static_cast<float> (new_point.g);
      data_.rgb_[2] += static_cast<float> (new_point.b);
    }

    // Explicit overloads for XYZRGBA types
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA,pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGBA &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
      //Separate sums for r,g,b since we can't sum in uchars
      data_.rgb_[0] += static_cast<float> (new_point.r);
      data_.rgb_[1] += static_cast<float> (new_point.g);
      data_.rgb_[2] += static_cast<float> (new_point.b);
    }

    // Explicit overloads for XYZRGBL types
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBL,pcl::SequentialSVClustering<pcl::PointXYZRGBL>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGBL &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
      //Separate sums for r,g,b since we can't sum in uchars
      data_.rgb_[0] += static_cast<float> (new_point.r);
      data_.rgb_[1] += static_cast<float> (new_point.g);
      data_.rgb_[2] += static_cast<float> (new_point.b);
    }

    // Explicit overloads for XYZRGBNormal types
    template<>
    void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal,pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData>::addPoint (const pcl::PointXYZRGBNormal &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
      //Separate sums for r,g,b since we can't sum in uchars
      data_.rgb_[0] += static_cast<float> (new_point.r);
      data_.rgb_[1] += static_cast<float> (new_point.g);
      data_.rgb_[2] += static_cast<float> (new_point.b);
      //Normal handling
      data_.normal_[0] += new_point.normal[0];
      data_.normal_[1] += new_point.normal[1];
      data_.normal_[2] += new_point.normal[2];
      data_.curvature_ += new_point.curvature;
    }

    //Explicit overloads for XYZ types
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZ,pcl::SequentialSVClustering<pcl::PointXYZ>::SequentialVoxelData>::computeData ()
    {
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
    }

    //Explicit overloads for XYZRGB types
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB,pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData>::computeData ()
    {
      data_.rgb_[0] /= (static_cast<float> (num_points_));
      data_.rgb_[1] /= (static_cast<float> (num_points_));
      data_.rgb_[2] /= (static_cast<float> (num_points_));
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
    }

    //Explicit overloads for XYZRGBL types
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBL,pcl::SequentialSVClustering<pcl::PointXYZRGBL>::SequentialVoxelData>::computeData ()
    {
      data_.rgb_[0] /= (static_cast<float> (num_points_));
      data_.rgb_[1] /= (static_cast<float> (num_points_));
      data_.rgb_[2] /= (static_cast<float> (num_points_));
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
    }

    //Explicit overloads for XYZRGBA types
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA,pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData>::computeData ()
    {
      data_.rgb_[0] /= (static_cast<float> (num_points_));
      data_.rgb_[1] /= (static_cast<float> (num_points_));
      data_.rgb_[2] /= (static_cast<float> (num_points_));
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
    }

    //Explicit overloads for XYZRGBANormal types
    template<> void
    pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal,pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData>::computeData ()
    {
      data_.rgb_[0] /= (static_cast<float> (num_points_));
      data_.rgb_[1] /= (static_cast<float> (num_points_));
      data_.rgb_[2] /= (static_cast<float> (num_points_));
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
      data_.normal_[0] /= (static_cast<float> (num_points_));
      data_.normal_[1] /= (static_cast<float> (num_points_));
      data_.normal_[2] /= (static_cast<float> (num_points_));
      data_.curvature_ /= (static_cast<float> (num_points_));
    }
    
    //Explicit overloads for XYZ types
    template<> float
    OctreePointCloudSequential<pcl::PointXYZ,
    OctreePointCloudSequentialContainer <pcl::PointXYZ, SequentialSVClustering<pcl::PointXYZ>::SequentialVoxelData > >
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZ, SequentialSVClustering<pcl::PointXYZ>::SequentialVoxelData >* leaf)
    {
      return 0;
    }

    //Explicit overloads for XYZRGB types
    template<> float
    OctreePointCloudSequential<pcl::PointXYZRGB,
    OctreePointCloudSequentialContainer <pcl::PointXYZRGB, SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData > >
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZRGB, SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData >* leaf)
    {
      float color_dist =  (leaf->getData ().rgb_
                           - leaf->getData ().previous_rgb_).norm () / 255.0f;

      return color_dist;
    }

    //Explicit overloads for XYZRGBA types
    template<> float
    OctreePointCloudSequential<pcl::PointXYZRGBA,
    OctreePointCloudSequentialContainer <pcl::PointXYZRGBA, SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData > >
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZRGBA, SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData >* leaf)
    {
      float color_dist =  (leaf->getData ().rgb_
                           - leaf->getData ().previous_rgb_).norm () / 255.0f;
      
      return color_dist;
    }

    //Explicit overloads for XYZRGBL types
    template<> float
    OctreePointCloudSequential<pcl::PointXYZRGBL,
    OctreePointCloudSequentialContainer <pcl::PointXYZRGBL, SequentialSVClustering<pcl::PointXYZRGBL>::SequentialVoxelData > >
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZRGBL, SequentialSVClustering<pcl::PointXYZRGBL>::SequentialVoxelData >* leaf)
    {
      float color_dist =  (leaf->getData ().rgb_
                           - leaf->getData ().previous_rgb_).norm () / 255.0f;

      return color_dist;
    }

    //Explicit overloads for XYZRGBNormal types
    template<> float
    OctreePointCloudSequential<pcl::PointXYZRGBNormal,
    OctreePointCloudSequentialContainer <pcl::PointXYZRGBNormal, SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData > >
    ::SeqVoxelDataDiff (const OctreePointCloudSequentialContainer <pcl::PointXYZRGBNormal, SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData >* leaf)
    {
      float color_dist =  (leaf->getData ().rgb_
                           - leaf->getData ().previous_rgb_).norm () / 255.0f;
      
      //    float cos_angle_normal = 1.0f - (leaf->getData ().voxel_centroid_.getNormalVector4fMap ().dot (leaf->getData ().previous_centroid_.getNormalVector4fMap ()));
      
      return color_dist;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  //Explicit overloads for XYZRGB types
  template<> void
  pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData::getPoint (pcl::PointXYZRGB &point_arg) const
  {
    point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 |
    static_cast<uint32_t>(rgb_[1]) << 8 |
    static_cast<uint32_t>(rgb_[2]);
    point_arg.x = xyz_[0];
    point_arg.y = xyz_[1];
    point_arg.z = xyz_[2];
  }

  //Explicit overloads for XYZRGBA types
  template<> void
  pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData::getPoint (pcl::PointXYZRGBA &point_arg ) const
  {
    point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 |
    static_cast<uint32_t>(rgb_[1]) << 8 |
    static_cast<uint32_t>(rgb_[2]);
    point_arg.x = xyz_[0];
    point_arg.y = xyz_[1];
    point_arg.z = xyz_[2];
  }

  //Explicit overloads for XYZRGBL types
  template<> void
  pcl::SequentialSVClustering<pcl::PointXYZRGBL>::SequentialVoxelData::getPoint (pcl::PointXYZRGBL &point_arg ) const
  {
    point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 |
    static_cast<uint32_t>(rgb_[1]) << 8 |
    static_cast<uint32_t>(rgb_[2]);
    point_arg.x = xyz_[0];
    point_arg.y = xyz_[1];
    point_arg.z = xyz_[2];
  }

  //Explicit overloads for XYZRGBNormal types
  template<> void
  pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData::getPoint (pcl::PointXYZRGBNormal &point_arg ) const
  {
    point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 |
                                                       static_cast<uint32_t>(rgb_[1]) << 8 |
                                                                                         static_cast<uint32_t>(rgb_[2]);
    point_arg.x = xyz_[0];
    point_arg.y = xyz_[1];
    point_arg.z = xyz_[2];
    point_arg.normal[0] = normal_[0];
    point_arg.normal[1] = normal_[1];
    point_arg.normal[2] = normal_[2];
  }

  // Explicit overload of getVoxels from SequentialSvClustering::SequentialSupervoxelHelper for XYZRGBL type
  template <> void
  pcl::SequentialSVClustering<pcl::PointXYZRGBL>::SequentialSupervoxelHelper::getVoxels (typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &voxels) const
  {
    voxels.reset (new pcl::PointCloud<pcl::PointXYZRGBL>);
    voxels->clear ();
    voxels->resize (leaves_.size ());
    typename pcl::PointCloud<pcl::PointXYZRGBL>::iterator voxel_itr = voxels->begin ();
    typename SequentialSupervoxelHelper::const_iterator leaf_itr;
    for ( leaf_itr = leaves_.begin (); leaf_itr != leaves_.end (); ++leaf_itr, ++voxel_itr)
    {
      const SequentialVoxelData& leaf_data = (*leaf_itr)->getData ();
      leaf_data.getPoint (*voxel_itr);
      (*voxel_itr).label = this->getLabel ();
    }
  }
}

typedef pcl::SequentialSVClustering<pcl::PointXYZ>::SequentialVoxelData VoxelDataT;
typedef pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData VoxelDataRGBT;
typedef pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData VoxelDataRGBAT;
typedef pcl::SequentialSVClustering<pcl::PointXYZRGBL>::SequentialVoxelData VoxelDataRGBLT;
typedef pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData VoxelDataRGBNT;

typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZ, VoxelDataT> AdjacencyContainerT;
typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB, VoxelDataRGBT> AdjacencyContainerRGBT;
typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA, VoxelDataRGBAT> AdjacencyContainerRGBAT;
typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBL, VoxelDataRGBLT> AdjacencyContainerRGBLT;
typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal, VoxelDataRGBNT> AdjacencyContainerRGBNT;

//template class pcl::SequentialSVClustering<pcl::PointXYZ>;
template class pcl::SequentialSVClustering<pcl::PointXYZRGB>;
template class pcl::SequentialSVClustering<pcl::PointXYZRGBA>;
//template class pcl::SequentialSVClustering<pcl::PointXYZRGBL>;
//template class pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>;

template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZ, VoxelDataT>;
template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB, VoxelDataRGBT>;
template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA, VoxelDataRGBAT>;
template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBL, VoxelDataRGBLT>;
template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal, VoxelDataRGBNT>;

template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZ, AdjacencyContainerT>;
template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGB, AdjacencyContainerRGBT>;
template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGBA, AdjacencyContainerRGBAT>;
template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGBL, AdjacencyContainerRGBLT>;
template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGBNormal, AdjacencyContainerRGBNT>;


/////////////////////////////
///////////////////////////////
/// /////////////////////////////
/*
PCL_INSTANTIATE(SequentialSVClustering, (pcl::PointXYZRGB)(pcl::PointXYZRGBA))
//PCL_INSTANTIATE(SequentialSVClustering, (pcl::PointXYZRGB)(pcl::PointXYZRGBNormal)(pcl::PointXYZRGBA))
//template class pcl::SequentialSVClustering<pcl::PointXYZ>;
//template class pcl::SequentialSVClustering<pcl::PointXYZRGB>;
//template class pcl::SequentialSVClustering<pcl::PointXYZRGBA>;

#include "../../octree/impl/octree_pointcloud_sequential.hpp"
typedef pcl::SequentialSVClustering<pcl::PointXYZRGB>::SequentialVoxelData SeqVoxelDataRGBT;
typedef pcl::SequentialSVClustering<pcl::PointXYZRGBA>::SequentialVoxelData SeqVoxelDataRGBAT;
//typedef pcl::SequentialSVClustering<pcl::PointXYZRGBNormal>::SequentialVoxelData SeqVoxelDataRGBNormalT;

typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB, SeqVoxelDataRGBT> SequentialContainerRGB;
typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA, SeqVoxelDataRGBAT> SequentialContainerRGBA;
//typedef pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal, SeqVoxelDataRGBNormalT> SequentialContainerRGBNormal;

template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGB, SeqVoxelDataRGBT>;
template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBA, SeqVoxelDataRGBAT>;
//template class pcl::octree::OctreePointCloudSequentialContainer<pcl::PointXYZRGBNormal, SeqVoxelDataRGBNormalT>;

template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGB, SequentialContainerRGB>;
template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGBA, SequentialContainerRGBA>;
//template class pcl::octree::OctreePointCloudSequential<pcl::PointXYZRGBNormal, SequentialContainerRGBNormal>;

*/
