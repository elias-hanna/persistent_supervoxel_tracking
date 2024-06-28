# Persistent Supervoxel Tracking
This repository is an implementation of the paper "[Point cloud video object segmentation using a persistent supervoxel world-model](https://ieeexplore.ieee.org/abstract/document/6696886)" by Papon *et al.*

## Requirements
Requires [Point Cloud Library](https://github.com/PointCloudLibrary/pcl) 1.9 version. Requires (libfreenect2)[https://github.com/OpenKinect/libfreenect2] if used with a Kinect 2 3D camera. Other 3D camera could be used but a suited library should be installed and setup with the rest of the code in order to run it. Requires cmake (can be installed simply using your package manager).

## Installation
From the root directory of this repository, create a build directory and cd into it:
```
mkdir build && cd build
```
Create the Makefile using cmake:
```
cmake ..
```
Build the executable (change the ${nproc} variable to a suited number of processors to be used for compilation):
```
make -j${nproc}
```
And now finally run the obtained executable file.

## Running a segmentation
When running the algorithm, the camera will over-segment the environment using supervoxels. Each supervoxels are maintained persistent between two frames using geometric descriptors and a RANdom SAmple Consensus algorithm. Supervoxels color change depending on a comovement matrix counting the number of frames two supervoxels were seen moving at the same time. That way, the more supervoxels move together, the more likely they are to belong to the same object, enabling an agent interacting with its environment to obtain a full scene segmentation after enough interactions. 

## To do
To do.
