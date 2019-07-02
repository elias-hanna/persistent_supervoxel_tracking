#ifndef GETTER_HPP
#define GETTER_HPP

#include "getter.h"

template <typename PointType>
Getter<PointType>::Getter ()
{
}

template <typename PointType>
Getter<PointType>::Getter (boost::shared_ptr<pcl::Grabber>): grabber(grabber)
{
  boost::function<void (const ConstPtr&)> callback = boost::bind(&Getter::cloud_callback, this, _1);
  connection = grabber->registerCallback(callback);
  grabber->start();
}

template <typename PointType>
Getter<PointType>::~Getter ()
{
  grabber->stop();

  if(connection.connected())
  {
    connection.disconnect();
  }
}

template <typename PointType> pcl::PointCloud<PointType>
Getter<PointType>::setGrabber (boost::shared_ptr<pcl::Grabber>)
{
  boost::function<void (const ConstPtr&)> callback = boost::bind(&Getter::cloud_callback, this, _1);
  connection = grabber->registerCallback(callback);
  grabber->start();
}

template <typename PointType> pcl::PointCloud<PointType>
Getter<PointType>::getCloud ()
{
  return buffer;
}

template <typename PointType> void
Getter<PointType>::cloud_callback (const ConstPtr& cloud)
{  
  buffer = *cloud;
}

#endif
