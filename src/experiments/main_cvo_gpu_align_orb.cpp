#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include "dataset_handler/KittiHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/VoxelMap.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace boost::filesystem;

Eigen::Vector3f get_pc_mean(const cvo::CvoPointCloud & pc) {
  Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
  for (int k = 0; k < pc.num_points(); k++)
    p_mean_tmp = (p_mean_tmp + pc.positions()[k]).eval();
  p_mean_tmp = (p_mean_tmp) / pc.num_points();    
  return p_mean_tmp;
}


int main(int argc, char *argv[]) {

  std::string img1_file(argv[1]);
  std::string img2_file(argv[2]);
//  string cvo_param_file(argv[3]);
//  float ell = -1;
//  if (argc > 4)
//	  ell = std::stof(argv[4]);

  cv::Mat img_1 = cv::imread(img1_file, CV_LOAD_IMAGE_COLOR);
  cv::Mat img_2 = cv::imread(img2_file, CV_LOAD_IMAGE_COLOR);
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  cv::Mat descriptor_1, descriptor_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  detector->detect ( img_1,keypoints_1 );
  detector->detect ( img_2,keypoints_2 );
  descriptor->compute ( img_1, keypoints_1, descriptor_1 );
  descriptor->compute ( img_2, keypoints_2, descriptor_2 );
  cout << "# Keypoints: " << keypoints_1.size() << " " << keypoints_2.size() << "\n";

//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
//  pcl::io::loadPCDFile(source_file, *source_pcd);
//  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));
//  Eigen::Vector3f source_mean = get_pc_mean(*source);
//
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
//  pcl::io::loadPCDFile(target_file, *target_pcd);
//  std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*target_pcd));
//  Eigen::Vector3f target_mean = get_pc_mean(*target);
//
//  float dist = (source_mean - target_mean).norm();
//  std::cout<<"source mean is "<<source_mean<<", target mean is "<<target_mean<<", dist is "<<dist<<std::endl;
//  cvo::CvoGPU cvo_align(cvo_param_file );
//  cvo::CvoParams & init_param = cvo_align.get_params();
//  init_param.ell_init = dist; //init_param.ell_init_first_frame;
//
//  if (argc > 4)
//	  init_param.ell_init = ell;
//  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
//  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
//  cvo_align.write_params(&init_param);
//
//  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;
//
//  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
//
//
//  Eigen::Matrix4f result, init_guess_inv;
//  Eigen::Matrix4f identity_init = Eigen::Matrix4f::Identity();
//  init_guess_inv = init_guess.inverse();
//
//  printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
//  std::cout<<std::flush;
//
//  double this_time = 0;
//  cvo_align.align(*source, *target, init_guess_inv, result, nullptr,&this_time);
//
//
//
//  //cvo_align.align(*source, *target, init_guess, result);
//
//  std::cout<<"Transform is "<<result <<"\n\n";
//  pcl::PointCloud<pcl::PointXYZRGB> pcd_old, pcd_new;
//  cvo::CvoPointCloud new_pc(3, 19), old_pc(3, 19);
//  cvo::CvoPointCloud::transform(init_guess, * target, old_pc);
//  cvo::CvoPointCloud::transform(result, *target, new_pc);
//  std::cout<<"Just finished transform\n";
//  cvo::CvoPointCloud sum_old = old_pc + *source;
//  cvo::CvoPointCloud sum_new = new_pc  + *source ;
//  std::cout<<"Just finished CvoPointCloud concatenation\n";
//  std::cout<<"num of points before and after alignment is "<<sum_old.num_points()<<", "<<sum_new.num_points()<<"\n";
//  sum_old.export_to_pcd(pcd_old);
//  sum_new.export_to_pcd(pcd_new);
//  std::cout<<"Just export to pcd\n";
//  std::string fname("before_align.pcd");
//  pcl::io::savePCDFileASCII(fname, pcd_old);
//  fname= "after_align.pcd";
//  pcl::io::savePCDFileASCII(fname, pcd_new);
//  // append accum_tf_list for future initialization
//  std::cout<<"Average registration time is "<<this_time<<std::endl;


  return 0;
}
