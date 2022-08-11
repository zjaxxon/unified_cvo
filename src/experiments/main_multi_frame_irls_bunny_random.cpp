#include <Eigen/Geometry>
#include <iostream>
#include <list>
#include <cmath>
#include <fstream>
#include <filesystem>
#include "utils/def_assert.hpp"
#include "utils/GassianMixture.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <vector>
#include <utility>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname) {
  pcl::PointCloud<pcl::PointXYZ> pc_all;
  for (auto ptr : frames) {
    cvo::CvoPointCloud new_pc(0,0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZ> pc_curr;
    new_pc.export_to_pcd(pc_curr);

    pc_all += pc_curr;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}



template <typename PointT>
void add_gaussian_mixture_noise(pcl::PointCloud<pcl::PointNormal> & input,
                                pcl::PointCloud<PointT> & output,
                                float ratio,
                                float sigma,
                                float uniform_range,
                                bool is_using_viewpoint) {

  cvo::GaussianMixtureDepthGenerator gaussion_mixture(ratio, sigma, uniform_range);

  output.resize(input.size());
  for (int i = 0; i < input.size(); i++) {
    if (is_using_viewpoint) {
      // using a far away view point
    } else {
      // using normal direction
      auto pt = input[i];
      Eigen::Vector3f normal_dir;
      normal_dir << pt.normal_x, pt.normal_y, pt.normal_z;
      Eigen::Vector3f center_pt = pt.getVector3fMap();
      Eigen::Vector3f result = gaussion_mixture.sample(center_pt, normal_dir);
      pcl::PointXYZ new_pt;
      output[i].getVector3fMap() = result;
      if (i == 0) {
        std::cout<<"transform "<<pt.getVector3fMap().transpose()<<" to "<<result.transpose()<<std::endl;
      }
    }
  }

  
}

void gen_random_poses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> & poses, int num_poses,
                      float max_angle_axis=1.0 // max: [0, 1)
                      ) {
  poses.resize(num_poses);
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, max_angle_axis);
  std::uniform_real_distribution<> dist_trans(0,1);
  for (int i = 0; i < num_poses; i++) {
    if (i != 0) {
      Eigen::Matrix3f rot;
      Eigen::Vector3f axis;
      axis << (float)dist_trans(e2), (float)dist_trans(e2), (float)dist_trans(e2);
      axis = axis.normalized().eval();
      
      //rot = Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitX())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI,  Eigen::Vector3f::UnitY())
      //  * Eigen::AngleAxisf(dist(e2)*M_PI, Eigen::Vector3f::UnitZ());
      rot = Eigen::AngleAxisf(max_angle_axis * M_PI, axis);
      poses[i] = Eigen::Matrix4f::Identity();
      poses[i].block<3,3>(0,0) = rot;
      poses[i](0,3) = dist_trans(e2);
      poses[i](1,3) = dist_trans(e2);
      poses[i](2,3) = dist_trans(e2);
    } else {
      poses[i] = Eigen::Matrix4f::Identity();
    }
    //rot = Eigen::AngleAxisf(dist(e2)*M_PI, axis );


    std::cout<<"random pose "<<i<<" is \n"<<poses[i]<<"\n";
  }
}
template <typename T>
void pcd_rescale(typename pcl::PointCloud<T>::Ptr pcd ){
  Eigen::MatrixXf pcd_eigen(3, pcd->size());
  for (int j = 0; j < pcd->size(); j++) {
    pcd_eigen.col(j) = pcd->at(j).getVector3fMap();
  }

  float scale = (pcd_eigen.rowwise().maxCoeff() - pcd_eigen.rowwise().minCoeff()).norm();
  std::cout << "scale = " << scale << std::endl;
  pcd_eigen /= scale;

  for (int j = 0; j < pcd->size(); j++)
    pcd->at(j).getVector3fMap() = pcd_eigen.col(j);
}

void eval_poses(std::vector<Sophus::SE3f> & estimates,
                std::vector<Sophus::SE3f> & gt,
                std::string & fname
                ){
  assert(std::filesystem::exists(fname));
  std::ofstream err_f(fname,std::fstream::out |   std::ios::app);
  float total_err = 0;
  for (int i = 0; i < gt.size(); i ++) {
    float err = (estimates[i].inverse() * gt[i]).log().norm();
    total_err += err;
  }
  err_f << total_err<<"\n";
  err_f.close();
  //std::cout<<"Total: "<<counter<<" success out of "<<gt.size()<<"\n";
}

int main(int argc, char** argv) {

  omp_set_num_threads(24);
  std::string in_pcd_fname(argv[1]);
  std::string cvo_param_file(argv[2]);
  int num_frames = std::stoi(argv[3]);
  float max_angle_per_axis = std::stof(argv[4]);
  int num_runs = std::stoi(argv[5]);
  int is_adding_outliers = std::stoi(argv[6]);
  float ratio = 0.8;
  float sigma = 0.01;
  float uniform_range = 0.5;
  if (is_adding_outliers) {
    ratio = std::stof(argv[7]);
    sigma = std::stof(argv[8]);
    uniform_range = std::stof(argv[9]);
  }

  cvo::CvoGPU cvo_align(cvo_param_file);

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr raw_pcd_normal(new pcl::PointCloud<pcl::PointNormal>);
  pcl::io::loadPCDFile<pcl::PointNormal> (in_pcd_fname, *raw_pcd_normal);
  
  std::string fname("err_bunny.txt");
  std::ofstream err_f(fname);
  err_f.close();
  for (int k = 0; k < num_runs; k++) {
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> tracking_poses;
    gen_random_poses(tracking_poses, num_frames, 0.2);    

    std::vector<Sophus::SE3f> poses_gt;
    std::transform(tracking_poses.begin(), tracking_poses.end(), std::back_inserter(poses_gt),
                   [&](const Eigen::Matrix4f & in){
                     Sophus::SE3f pose(in);
                     return pose;
                   });


    // read point cloud
    std::vector<cvo::CvoFrame::Ptr> frames;
    std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
    for (int i = 0; i<num_frames; i++) {
      // Create the filtering object

      pcl::PointCloud<pcl::PointNormal>::Ptr raw_pcd_curr;//(new pcl::PointCloud<pcl::PointXYZ>);
      if (is_adding_outliers) {
        raw_pcd_curr.reset(new pcl::PointCloud<pcl::PointNormal>);
        add_gaussian_mixture_noise(*raw_pcd_normal,
                                   *raw_pcd_curr,
                                   ratio,
                                   sigma,
                                   uniform_range,
                                   false
                                   );
      } else {
        raw_pcd_curr = raw_pcd_normal;
      }

      copyPointCloud(*raw_pcd_curr, *raw_pcd);
      pcl::io::savePCDFileASCII(std::to_string(i)+"normal.pcd", *raw_pcd);      
      pcd_rescale<pcl::PointXYZ>(raw_pcd);
      
    
      
      pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pcd_transformed(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::transformPointCloud (*raw_pcd, *raw_pcd_transformed, tracking_poses[i]);
    
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud (raw_pcd_transformed);
      sor.setLeafSize (0.025f, 0.025f, 0.025f);
      sor.filter (*raw_pcd_transformed);
      std::cout<<"after downsampling, num of points is "<<raw_pcd_transformed->size()<<std::endl;

      std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(*raw_pcd_transformed));  
      //std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(0,0));
    
      // cvo::CvoPointCloud::transform(tracking_poses[i],
      //                               *raw,
      //                               *pc);
    
      cvo::Mat34d_row pose;
      Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose44 = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
      pose = pose44.block<3,4>(0,0);
    

      cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), pose.data()));
      frames.push_back(new_frame);
      pcs.push_back(pc);
    }
  
  

    //std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
    std::cout<<"write to before_BA.pcd\n";
    std::string f_name("before_BA_bunny.pcd");
    write_transformed_pc(frames, f_name);

    // std::list<std::pair<std::shared_ptr<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
    std::cout<<"Start constructing cvo edges\n";
    std::list<cvo::BinaryState::Ptr> edge_states;            
    for (int i = 0; i < num_frames; i++) {
      for (int j = i+1; j < num_frames; j++ ) {
        //std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[i], frames[j]);
        //edges.push_back(p);
        const cvo::CvoParams & params = cvo_align.get_params();
        cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[i]),
                                                                    std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[j]),
                                                                    &params,
                                                                    cvo_align.get_params_gpu(),
                                                                    params.multiframe_num_neighbors,
                                                                    params.multiframe_ell_init
                                                                    //dist / 3
                                                                    ));
        edge_states.push_back((edge_state));
      
      }
    }

    std::vector<bool> const_flags(frames.size(), false);
    const_flags[0] = true;

    std::cout<<"start align\n";
    cvo_align.align(frames, const_flags,
                    edge_states,  nullptr);

    //std::cout<<"Align ends. Total time is "<<time<<std::endl;
    f_name="after_BA_bunny_" + std::to_string(k)+".pcd";
    write_transformed_pc(frames, f_name);
    
    std::vector<Sophus::SE3f> estimates;
    std::transform(frames.begin(), frames.end(), std::back_inserter(estimates),
                   [&](auto & frame){
                     Eigen::Matrix<double, 3, 4, Eigen::RowMajor> pose_row = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frame->pose_vec);
                     Eigen::Matrix<double , 3,4> pose_col = pose_row;
                     Eigen::Matrix<float, 4, 4> pose_eigen = Eigen::Matrix4f::Identity();
                     pose_eigen.block<3,4>(0,0) = pose_col.cast<float>();
                     Sophus::SE3f pose(pose_eigen);
                     return pose;
                   });


    eval_poses(estimates,
               poses_gt,
               fname
               );


  }
  return 0;
}
