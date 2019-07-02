# Lets get a recet version of CMake
cmake_minimum_required(VERSION 3.1)

# Project name
project(persistent_supervoxel_tracking)

# Flags
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -ltbb")

# On inclut notre bibliothèque dans le processus de CMake
add_subdirectory(libs)

# Crée des variables avec les fichiers à compiler
set(SRCS
  main.cpp
  # miscellaneous/main.cpp
  # miscellaneous/video_test.cpp
  # miscellaneous/persistent_world.cpp
  )

# Notre exécutable
add_executable(main ${SRCS})

# Link local libraries
target_link_libraries(main  #lib_tracker
                            lib_papon)

########################### PCL Library ##########################
find_package (PCL 1.9 REQUIRED)

if (PCL_FOUND)
  # Une fois la dépendance trouvée, nous l'incluons au projet
  target_include_directories(main PUBLIC ${PCL_INCLUDE_DIRS})
  target_link_libraries (main ${PCL_LIBRARIES})
else ()
  # Sinon, nous affichons un message
  message(FATAL_ERROR "PCL not found")
endif ()

########################### TBB Library ##########################
target_link_libraries(main tbb)

######################### OpenCV Library #########################

set(OpenCV_DIR "/home/elias-hanna/Documents/projects/opencv.OSID_ubuntu-16.04.installtree/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})

########################### Kinect2 ##############################

FIND_LIBRARY(freenect2_LIBRARY freenect2
    PATHS /home/elias-hanna/freenect2/lib
    NO_DEFAULT_PATH
)
SET(freenect2_LIBRARIES ${freenect2_LIBRARY} pthread)
FIND_PATH(freenect2_INCLUDE_DIR libfreenect2/libfreenect2.hpp
    PATHS /home/elias-hanna/freenect2/include
    NO_DEFAULT_PATH
)
SET(freenect2_INCLUDE_DIRS ${freenect2_INCLUDE_DIR})
target_include_directories(main PUBLIC ${freenect2_INCLUDE_DIRS})
target_link_libraries (main ${freenect2_LIBRARIES})

########################### OpenMP Library ##########################
# Recherche de la dépendance externe
#find_package(OpenMP)

#if (OPENMP_FOUND)
#  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#else()
#  #    Sinon, nous affichons un message
#  message(FATAL_ERROR "OpenMP not found")
#endif()
