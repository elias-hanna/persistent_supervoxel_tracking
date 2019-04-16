# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.1)

# Notre projet est étiqueté libhello
project(libs)

# Flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Local variables
set(USE_TRACKER FALSE)

# Include required directories (also directories that contain only implemented template headers and no source files)
include_directories(viewer
                    getter
                    )

########################### SequentialSupervoxelClustering Library ##########################
# Create variables with the files to compile
if(USE_TRACKER)
  set(SRCS_TRACKER
      supervoxel_tracker/src/supervoxel_tracker.cpp
      )
  set(HEADERS_TRACKER
      supervoxel_tracker/impl/supervoxel_tracker.hpp
      )
  set(IMPL_HEADERS_TRACKER
      supervoxel_tracker/supervoxel_tracker.h
      )
  # Add libraries (directories with sources files)
  add_library(lib_tracker ${SRCS_TRACKER} ${HEADERS_TRACKER} ${IMPL_HEADERS_TRACKER})
endif()
########################### SequentialSupervoxelClustering Library ##########################
# Create variables with the files to compile
set(SRCS_PAPON
    # Supervoxel
    papon/supervoxel/src/sequential_supervoxel_clustering.cpp
    )

set(HEADERS_PAPON
    # Supervoxel
    papon/supervoxel/sequential_supervoxel_clustering.h
    # Octree
    papon/octree/octree_pointcloud_sequential.h
    papon/octree/octree_pointcloud_sequential_container.h
    )

set(IMPL_HEADERS_PAPON
    # Supervoxel
    papon/supervoxel/impl/sequential_supervoxel_clustering.hpp
    # Octree
    papon/octree/impl/octree_pointcloud_sequential.hpp
    )

# Add libraries (directories with sources files)
add_library(lib_papon ${SRCS_PAPON} ${HEADERS_PAPON} ${IMPL_HEADERS_PAPON})

# Check for extern dependecies
find_package (PCL 1.8 REQUIRED)

if (PCL_FOUND)
  # Une fois la dépendance trouvée, nous l'incluons au projet
  target_include_directories(lib_papon PUBLIC ${PCL_INCLUDE_DIRS})
  target_link_libraries (lib_papon ${PCL_LIBRARIES})
  if(USE_TRACKER)
    target_include_directories(lib_tracker PUBLIC ${PCL_INCLUDE_DIRS})
    target_link_libraries (lib_tracker ${PCL_LIBRARIES})
  endif()
else ()
  # Sinon, nous affichons un message
  message(FATAL_ERROR "PCL not found")
endif ()

find_package(OpenMP)
if (OpenMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  # Une fois la dépendance trouvée, nous l'incluons au projet
#  target_include_directories(lib_papon PUBLIC ${OpenMP_INCLUDE_DIRS})
#  target_link_libraries (lib_papon ${OpenMP_LIBRARIES})
else ()
  # Sinon, nous affichons un message
  message(FATAL_ERROR "OpenMP not found")
endif()
