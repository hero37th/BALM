#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>
#include "bavoxel.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>

using namespace std;

template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

ros::Publisher pub_path, pub_show, pub_cute;

Eigen::Matrix3d world_R = Eigen::Matrix3d::Identity();
Eigen::Vector3d world_p = Eigen::Vector3d::Zero();

void pub_odom_func(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  Eigen::Quaterniond q_curr(R);
  transform.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
  q.setW(q_curr.w());
  q.setX(q_curr.x());
  q.setY(q_curr.y());
  q.setZ(q_curr.z());
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                                        "/camera_init", "/aft_mapped"));
}

vector<IMUST> x_buf;
vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;
int frame_num = 20;
bool process_flag = false;

void data_show(vector<IMUST> x_in, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls)
{
  IMUST es0 = x_in[0];
  for(uint i=0; i<x_in.size(); i++)
  {
    x_in[i].p = es0.R.transpose() * (x_in[i].p - es0.p);
    x_in[i].R = es0.R.transpose() * x_in[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;
  int winsize = x_in.size();
  for(int i=0; i<winsize; i++)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_in[i]);
    pl_send += pl_tem;

    if((i%200==0 && i!=0) || i == winsize-1)
    {
      pub_pl_func(pl_send, pub_show);
      pl_send.clear();
      ros::Duration(0.5).sleep();
    }

    PointType ap;
    ap.x = x_in[i].p.x();
    ap.y = x_in[i].p.y();
    ap.z = x_in[i].p.z();
    ap.curvature = i;
    pl_path.push_back(ap);
  }

  pub_pl_func(pl_path, pub_path);
}

void process_data()
{
  if(process_flag) return;
  process_flag = true;

  IMUST es0 = x_buf[0];
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  win_size = x_buf.size();
  ROS_INFO("The size of poses: %d", win_size);

  data_show(x_buf, pl_fulls);

  unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

  eigen_value_array[0] = 1.0 / 16;
  eigen_value_array[1] = 1.0 / 16;
  eigen_value_array[2] = 1.0 / 9;

  for(int i=0; i<win_size; i++)
    cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);

  pcl::PointCloud<PointType> pl_send;
  pub_pl_func(pl_send, pub_show);

  VOX_HESS voxhess;
  for(auto iter=surf_map.begin(); iter!=surf_map.end() && ros::ok(); iter++)
  {
    iter->second->recut(win_size);
    iter->second->tras_opt(voxhess, win_size);
    iter->second->tras_display(pl_send, win_size);
  }

  pub_pl_func(pl_send, pub_cute);

  if(voxhess.plvec_voxels.size() < 3 * x_buf.size())
  {
    ROS_WARN("Initial error too large, optimization terminated.");
    process_flag = false;
    return;
  }

  BALM2 opt_lsv;
  opt_lsv.damping_iter(x_buf, voxhess);

  for(auto iter=surf_map.begin(); iter!=surf_map.end();)
  {
    delete iter->second;
    surf_map.erase(iter++);
  }
  surf_map.clear();

  malloc_trim(0);

  data_show(x_buf, pl_fulls);

  IMUST curr = x_buf.back();
  pub_odom_func(world_R * curr.R, world_R * curr.p + world_p);
  process_flag = false;
}

void lidar_cb(const sensor_msgs::PointCloud2ConstPtr &msg)
{

  pcl::PointCloud<pcl::PointXYZI> pl_tmp;
  pcl::fromROSMsg(*msg, pl_tmp);
  pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
  for(const auto &pp: pl_tmp.points)
  {
    PointType ap;
    ap.x = pp.x; ap.y = pp.y; ap.z = pp.z;
    ap.intensity = pp.intensity;
    pl_ptr->push_back(ap);
  }

  pl_fulls.push_back(pl_ptr);

  IMUST curr;
  curr.R = Eigen::Matrix3d::Identity();
  curr.p = Eigen::Vector3d::Zero();
  curr.t = msg->header.stamp.toSec();
  x_buf.push_back(curr);

  if(x_buf.size() > static_cast<size_t>(frame_num))
  {
    world_p += world_R * x_buf.front().p;
    world_R = world_R * x_buf.front().R;
    pl_fulls.erase(pl_fulls.begin());
    x_buf.erase(x_buf.begin());
  }

  if(!process_flag && x_buf.size() >= static_cast<size_t>(frame_num))
    process_data();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "benchmark_lidar");
  ros::NodeHandle n;

  pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_show = n.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
  pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

  string topic;
  n.param<string>("lidar_topic", topic, "/lslidar_point_cloud");
  n.param<int>("frame_num", frame_num, 20);
  n.param<double>("voxel_size", voxel_size, 1);

  ros::Subscriber sub = n.subscribe(topic, 100, lidar_cb);

  ros::spin();
  return 0;
}
