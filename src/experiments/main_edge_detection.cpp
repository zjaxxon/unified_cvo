//
// Created by Jaxxon Zhang on 2/5/2024.
//

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "utils/CvoPointCloud.hpp"
#include "utils/LidarPointDownsampler.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/CvoGPU.hpp"

using namespace std;

std::shared_ptr<cvo::CvoPointCloud> downsample_lidar_points(bool is_edge_only,
                                                            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                            float leaf_size) {

  int expected_points = 20000;
  double intensity_bound = 9.0;
  double depth_bound = 5.0;
  double distance_bound = 60.0;
  int kitti_beam_num = 128;
  cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);

  if (is_edge_only) {
    cvo::VoxelMap<pcl::PointXYZI> full_voxel(leaf_size);
    for (int k = 0; k < pc_in->size(); k++) {
      full_voxel.insert_point(&pc_in->points[k]);
    }
    std::vector<pcl::PointXYZI*> downsampled_results = full_voxel.sample_points();
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    for (int k = 0; k < downsampled_results.size(); k++)
      downsampled->push_back(*downsampled_results[k]);
    std::shared_ptr<cvo::CvoPointCloud>  ret(new cvo::CvoPointCloud(downsampled, 20000, 128, cvo::CvoPointCloud::PointSelectionMethod::FULL));
    return ret;
  } else {

    /// edge points
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> selected_edge_inds;
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
    std::unordered_set<int> edge_inds;
    for (auto && j : selected_edge_inds) edge_inds.insert(j);

    /// surface points
    std::vector<float> edge_or_surface;
    std::vector<int> selected_loam_inds;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);
    lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, selected_loam_inds);

    /// declare voxel map
    cvo::VoxelMap<pcl::PointXYZI> edge_voxel(leaf_size);
    cvo::VoxelMap<pcl::PointXYZI> surface_voxel(leaf_size / 10);

    /// edge and surface downsample
    for (int k = 0; k < pc_out_edge->size(); k++)
      edge_voxel.insert_point(&pc_out_edge->points[k]);
    std::vector<pcl::PointXYZI*> edge_results = edge_voxel.sample_points();
    for (int k = 0; k < pc_out_loam->size(); k++)  {
      if (edge_or_surface[k] > 0 &&
          edge_inds.find(selected_loam_inds[k]) == edge_inds.end())
        surface_voxel.insert_point(&pc_out_loam->points[k]);
    }
    std::vector<pcl::PointXYZI*> surface_results = surface_voxel.sample_points();
    int total_selected_pts_num = edge_results.size() + surface_results.size();
    std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
    ret->reserve(total_selected_pts_num, 1, NUM_CLASSES);
    std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
    std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;

    /// push
    for (int k = 0; k < edge_results.size(); k++) {
      Eigen::VectorXf feat(1);
      feat(0) = edge_results[k]->intensity;
      Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
      Eigen::VectorXf geo_t(2);
      geo_t << 1.0, 0;
      ret->add_point(k, edge_results[k]->getVector3fMap(),  feat, semantics, geo_t);
    }
    /// surface downsample
    for (int k = 0; k < surface_results.size(); k++) {
      // surface_pcl.push_back(*surface_results[k]);
      Eigen::VectorXf feat(1);
      feat(0) = surface_results[k]->intensity;
      Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
      Eigen::VectorXf geo_t(2);
      geo_t << 0, 1.0;
      ret->add_point(k+edge_results.size(), surface_results[k]->getVector3fMap(), feat,
                     semantics, geo_t);
    }
    return ret;

  }

}

int main(int argc, char *argv[]) {
  string data_path = argv[1];
  int align_option = std::stoi(argv[2]);
  if (align_option == 0) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(data_path, *pc_pcl);
    cout << "Read in finished\n";
    const vector<int> semantics_vec;
//  std::shared_ptr<cvo::CvoPointCloud> ret = downsample_lidar_points(false, pc_pcl, 0.1);  // use voxel size 0.1
    std::shared_ptr<cvo::CvoPointCloud> ret = downsample_lidar_points(false, pc_pcl, 0.1);

//  pcl::PointCloud<pcl::PointXYZIL>::Ptr pc_edge(new pcl::PointCloud<pcl::PointXYZIL>);
//  pcl::PointCloud<pcl::PointXYZIL>::Ptr pc_surface(new pcl::PointCloud<pcl::PointXYZIL>);
//  cvo::select_pc_inds_edge_surface(pc_pcl, semantics_vec, *pc_edge, *pc_surface, 20000, 9.0, 5, 60.0);
//  std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(FEATURE_DIMENSIONS, NUM_CLASSES));
//  for (int k = 0; k < pc_surface->size(); k++) {
//    pcl::PointXYZIL p = (*pc_surface)[k];
//    cvo::CvoPoint t(p.x, p.y, p.z);s
//    t.features[0] = 1;
//    ret->push_back(t);
//  }
    ret->write_to_intensity_pcd("test.pcd");
    return 0;
  }

  string in_pcd_folder = data_path;
  string cvo_param_file(argv[3]);
  std::ofstream accum_output(argv[4]);
  int start_frame = std::stoi(argv[5]);
  int last_frame = std::stoi(argv[6]);
  int is_stacking_results = std::stoi(argv[7]);

  int num_frames = last_frame - start_frame + 1;

  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_decay_rate = init_param.ell_decay_rate;
  int ell_decay_start = init_param.ell_decay_start;
  init_param.ell_init = init_param.ell_init_first_frame;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  cvo_align.write_params(&init_param);

  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

  //cvo::cvo cvo_align_cpu("/home/rayzhang/outdoor_cvo/cvo_params/cvo_params.txt");
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  Eigen::Quaternionf q(accum_mat.block<3,3>(0,0));
  accum_output<<accum_mat(0,3)<<" "<<accum_mat(1,3)<<" "<<accum_mat(2,3)<<" ";
  accum_output<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";
  accum_output.flush();
  // start the iteration


  std::string fname = in_pcd_folder + "/" + std::to_string(start_frame) + ".pcd";
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile<pcl::PointXYZI> (fname, *raw_pcd);
  std::shared_ptr<cvo::CvoPointCloud> source = downsample_lidar_points(false, raw_pcd, 0.1);
  std::cout<<"read source cvo point cloud\n";
  std::cout<<"source point cloud size is "<<source->size()<<std::endl;

  cvo::CvoPointCloud pc_all(FEATURE_DIMENSIONS, NUM_CLASSES);
  pc_all += *source;

  for (int i = start_frame+1; i< last_frame ; i++) {

    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i-1<<" and "<<i<<" with GPU "<<std::endl;

    fname = in_pcd_folder + "/" + std::to_string(i)+ ".pcd";
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcd_target(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI> (fname, *raw_pcd_target);
    std::shared_ptr<cvo::CvoPointCloud> target = downsample_lidar_points(false, raw_pcd_target, 0.1);


    if (i == start_frame+1){
      std::cout<<"Write first pcd\n";
      //target->write_to_color_pcd(std::to_string(i+1)+".pcd");
    }
    //std::cout<<"First point is "<<target->at(0).transpose()<<std::endl;

    // std::cout<<"reading "<<files[cur_kf]<<std::endl;

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
    std::cout<<std::flush;
    cvo_align.align(*source, *target, init_guess_inv, result);

    // get tf and inner product from cvo getter
    //double in_product = cvo_align.inner_product_cpu(*source, *target, result, ell_init);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    //std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result <<"\n\n";

    // append accum_tf_list for future initialization
    init_guess = result;
    accum_mat = accum_mat * result;
    std::cout<<"accum tf: \n"<<accum_mat<<std::endl;


    // log accumulated pose

    // accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
    //             <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
    //             <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    // accum_output<<"\n";
    // accum_output<<std::flush;

    Eigen::Quaternionf q(accum_mat.block<3,3>(0,0));
    //accum_output<<vstrRGBName[i]<<" ";
    accum_output<<accum_mat(0,3)<<" "<<accum_mat(1,3)<<" "<<accum_mat(2,3)<<" ";
    accum_output<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";
    accum_output.flush();

    if (is_stacking_results) {
      cvo::CvoPointCloud target_transformed(FEATURE_DIMENSIONS, NUM_CLASSES);
      cvo::CvoPointCloud::transform(result, *target, target_transformed);
      pc_all += target_transformed;
    }

    std::cout<<"\n\n===========next frame=============\n\n";

    source = target;
    if (i == start_frame) {
      init_param.ell_init = ell_init;
      init_param.ell_decay_rate = ell_decay_rate;
      init_param.ell_decay_start = ell_decay_start;

      cvo_align.write_params(&init_param);

    }


  }

  if (is_stacking_results) {
    //pc_all.write_to_color_pcd("stacked_tracking.pcd");
  }

  accum_output.close();


  return 0;
}