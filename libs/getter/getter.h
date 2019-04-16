#ifndef GETTER_H
#define GETTER_H

#include <pcl/io/openni_grabber.h>

template <typename PointType>
class Getter
{
  typedef pcl::PointCloud<PointType> PointCloud;
  typedef typename PointCloud::ConstPtr ConstPtr;

  private:
    pcl::Grabber& grabber;
    boost::signals2::connection connection;
    boost::mutex mutex;
    pcl::PointCloud<PointType> buffer;

  public:
    Getter(pcl::Grabber& grabber);
    ~Getter();
    pcl::PointCloud<PointType> getCloud();

  private:
    void cloud_callback(const ConstPtr& cloud);
};

#endif