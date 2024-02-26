#pragma once
#include <iostream>
#include <unordered_set>
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/data_type.hpp"
#include "dataset_handler/DataHandler.hpp"
#include "utils/PointXYZIL.hpp"

namespace cvo {

  void select_pc_inds_edge_surface(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                   const std::vector<int> & semantics_vec,
                                   /// output
                                   //std::vector<int> & selected_edge_inds,
                                   pcl::PointCloud<pcl::PointXYZIL> & pc_edge,
                                   pcl::PointCloud<pcl::PointXYZIL> & pc_surface
                                   //std::vector<int> & selected_surface_inds
                                   ) {

    const int expected_points = 10000;
    const double intensity_bound = 0.4;
    const double depth_bound = 4.0;
    const double distance_bound = 40.0;
    const int kitti_beam_num = 64;
    cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);
    
    /// edge points
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    std::vector<int> selected_edge_inds;
    lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
    std::unordered_set<int> edge_inds;
    for (int l = 0; l < selected_edge_inds.size(); l++){
      int j = selected_edge_inds[l];
      edge_inds.insert(j);

      pcl::PointXYZIL pt;      
      if (semantics_vec.size()) {
        if (semantics_vec[j] == -1)
          continue;
        else
          pt.label = semantics_vec[j];        
      }
      pt.getVector3fMap() = pc_out_edge->points[l].getVector3fMap();
      pt.intensity = pc_out_edge->points[l].intensity;
      pc_edge.push_back(pt);
      
    }

    /// surface points
    std::vector<float> edge_or_surface;
    std::vector<int> loam_inds;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);
    lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, loam_inds);
    for (int l = 0; l < loam_inds.size(); l++) {
      int j = loam_inds[l];
      if (edge_or_surface[l] > 0 && edge_inds.find(j) == edge_inds.end()) {
        pcl::PointXYZIL pt;      
        if (semantics_vec.size()) {
          if (semantics_vec[j] == -1)
            continue;
          pt.label = semantics_vec[j];        
        }
        pt.getVector3fMap() = pc_out_loam->points[l].getVector3fMap();
        pt.intensity = pc_out_loam->points[l].intensity;
        pc_surface.push_back(pt);
        
      }
    }      
    
  }

  
  CvoPoint PointXYZIL_to_CvoPoint(const pcl::PointXYZIL & p_pcl,
                                  bool is_edge) {
    cvo::CvoPoint pt;
    pt.getVector3fMap() = p_pcl.getVector3fMap();
    pt.features[0] = p_pcl.intensity;
    pt.label_distribution[p_pcl.label] = 1; 
    pt.geometric_type[0] = is_edge?  1.0 : 0.0 ;
    pt.geometric_type[1] = is_edge?  0.0 : 1.0;
    return pt;
  }


  std::shared_ptr<cvo::CvoPointCloud> downsample_edge_surface_with_voxel(pcl::PointCloud<pcl::PointXYZIL> & pc_edge,
                                                                         pcl::PointCloud<pcl::PointXYZIL> & pc_surface,
                                                                         float edge_leaf_size,
                                                                         float surface_leaf_size
                                                                         ) {
    /// declare voxel map
    cvo::VoxelMap<pcl::PointXYZIL> edge_voxel(edge_leaf_size); 
    cvo::VoxelMap<pcl::PointXYZIL> surface_voxel(surface_leaf_size);
    //std::unordered_map<pcl::PointXYZIL*, int> edge_pt_to_ind, surface_pt_to_ind;

    /// edge and surface downsample
    for (int k = 0; k < pc_edge.size(); k++) 
      edge_voxel.insert_point(&pc_edge[k]);
    std::vector<pcl::PointXYZIL*> edge_results = edge_voxel.sample_points();
    for (int k = 0; k < pc_surface.size(); k++)  
      surface_voxel.insert_point(&pc_surface[k]);
    std::vector<pcl::PointXYZIL*> surface_results = surface_voxel.sample_points();
    int total_selected_pts_num = edge_results.size() + surface_results.size();
    std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
    std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
    std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;    

    /// push
    for (int k = 0; k < edge_results.size(); k++) {
      ret->push_back(PointXYZIL_to_CvoPoint(*edge_results[k], true));
    }
    /// surface downsample
    for (int k = 0; k < surface_results.size(); k++) {
      ret->push_back(PointXYZIL_to_CvoPoint(*surface_results[k], false));
    }
    return ret;
    
  }
  


  void select_pc_inds_edge_surface(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                   const std::vector<int> & semantics_vec,
                                   /// output
                                   std::vector<int> & selected_edge_inds,
                                   std::vector<int> & selected_surface_inds
                                   ) {

    const int expected_points = 10000;
    const double intensity_bound = 0.4;
    const double depth_bound = 4.0;
    const double distance_bound = 40.0;
    const int kitti_beam_num = 64;
    cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);
    
    /// edge points
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector <double> output_depth_grad;
    std::vector <double> output_intenstity_grad;
    lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
    std::unordered_set<int> edge_inds;
    for (int l = 0; l < selected_edge_inds.size(); l++){
      int j = selected_edge_inds[l];
      edge_inds.insert(j);
    }

    /// surface points
    std::vector<float> edge_or_surface;
    std::vector<int> loam_inds;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);
    lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, loam_inds);
    for (int l = 0; l < loam_inds.size(); l++) {
      if (edge_or_surface[l] > 0 && edge_inds.find(loam_inds[l]) == edge_inds.end()) {
        int j = loam_inds[l];
        selected_surface_inds.push_back( loam_inds[l]);
      }
    }      
    
  }
  

  std::shared_ptr<cvo::CvoPointCloud> downsample_edge_surface_with_voxel(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                                         const std::vector<int> & semantics_vec,
                                                                         float leaf_size,
                                                                         std::vector<int> & selected_edge_inds,
                                                                         std::vector<int> & selected_surface_inds
                                                                         ) {
    /// declare voxel map
    cvo::VoxelMap<pcl::PointXYZI> edge_voxel(leaf_size / 4); 
    cvo::VoxelMap<pcl::PointXYZI> surface_voxel(leaf_size);
    std::unordered_map<pcl::PointXYZI*, int> edge_pt_to_ind, surface_pt_to_ind;

    /// edge and surface downsample
    for (int k = 0; k < selected_edge_inds.size(); k++) {
      edge_voxel.insert_point(&pc_in->points[selected_edge_inds[k]]);
      edge_pt_to_ind.insert(std::make_pair(&pc_in->points[selected_edge_inds[k]], selected_edge_inds[k]));
    }
    std::vector<pcl::PointXYZI*> edge_results = edge_voxel.sample_points();
    for (int k = 0; k < selected_surface_inds.size(); k++)  {
      //if (edt_.find(selected_surface_inds[k]) == selected_edge_inds.end()) {
        surface_voxel.insert_point(&pc_in->points[selected_surface_inds[k]]);
        surface_pt_to_ind.insert(std::make_pair(&pc_in->points[selected_surface_inds[k]], selected_surface_inds[k]));        
        // }
    }
    std::vector<pcl::PointXYZI*> surface_results = surface_voxel.sample_points();
    
    int total_selected_pts_num = edge_results.size() + surface_results.size();    
    std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
    std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
    std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;    

    /// push
    for (int k = 0; k < edge_results.size(); k++) {
      cvo::CvoPoint pt;
      pt.getVector3fMap() = edge_results[k]->getVector3fMap();
      pt.features[0] = edge_results[k]->intensity;
      if (semantics_vec.size()){
        int index = semantics_vec[edge_pt_to_ind[edge_results[k]]];
        if (index == -1)
          continue;
        pt.label_distribution[ index] = 1; 
      }
      pt.geometric_type[0] = 1.0;
      pt.geometric_type[1] = 0.0;
      ret->push_back(pt);
    }
    /// surface downsample
    for (int k = 0; k < surface_results.size(); k++) {
      cvo::CvoPoint pt;
      pt.getVector3fMap() = surface_results[k]->getVector3fMap();
      pt.features[0] = surface_results[k]->intensity;
      if (semantics_vec.size()){
        int index = semantics_vec[surface_pt_to_ind[surface_results[k]]];
        if (index == -1)
          continue;
        std::memset( pt.label_distribution, 0, NUM_CLASSES * sizeof(float));
        pt.label_distribution[index] = 1; 
      }
      pt.geometric_type[0] = 0.0;
      pt.geometric_type[1] = 1.0;
      ret->push_back(pt);
        
    }
    return ret;
    
  }
  
  std::shared_ptr<cvo::CvoPointCloud> downsample_lidar_points(bool is_edge_only,
                                                              pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                              float leaf_size,
                                                              const std::vector<int> & semantics_vec ){




    /*
    // Running edge detection + lego loam point selection
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> selected_edge_inds, selected_loam_inds;
    lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
  
    lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_loam_inds);    
    // *pc_out += *pc_out_edge;
    // *pc_out += *pc_out_surface;
    //
    num_points_ = selected_indexes.size();
    */

    if (is_edge_only) {
      cvo::VoxelMap<pcl::PointXYZI> full_voxel(leaf_size);
      for (int k = 0; k < pc_in->size(); k++) {
        full_voxel.insert_point(&pc_in->points[k]);
      }
      std::vector<pcl::PointXYZI*> downsampled_results = full_voxel.sample_points();
      pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
      for (int k = 0; k < downsampled_results.size(); k++)
        downsampled->push_back(*downsampled_results[k]);
      std::shared_ptr<cvo::CvoPointCloud>  ret(new cvo::CvoPointCloud(downsampled, 5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL));
      return ret;
    } else {
      int expected_points = 10000;
      double intensity_bound = 0.4;
      double depth_bound = 4.0;
      double distance_bound = 40.0;
      int kitti_beam_num = 64;
      cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);
    
      /// edge points
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
      std::vector<int> selected_edge_inds;
      std::unordered_map<pcl::PointXYZI*, int> edge_pt_to_ind, surface_pt_to_ind;
      std::vector <double> output_depth_grad;
      std::vector <double> output_intenstity_grad;
      lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
      std::unordered_set<int> edge_inds;
      for (int l = 0; l < selected_edge_inds.size(); l++){//(auto && j : selected_edge_inds) {
        int j = selected_edge_inds[l];
        edge_inds.insert(j);
        edge_pt_to_ind.insert(std::make_pair(&(pc_out_edge->points[l]), j));
      }

      /// surface points
      std::vector<float> edge_or_surface;
      std::vector<int> selected_loam_inds, loam_semantics;
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);
      lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, selected_loam_inds);
      std::unordered_set<int> loam_inds;
      for (int l = 0; l < selected_loam_inds.size(); l++) { //auto && j : selected_loam_inds) {
        int j = selected_loam_inds[l];
        surface_pt_to_ind.insert(std::make_pair(&(pc_out_loam->points[l]), j));
      }      

      /// declare voxel map
      cvo::VoxelMap<pcl::PointXYZI> edge_voxel(leaf_size / 4); 
      cvo::VoxelMap<pcl::PointXYZI> surface_voxel(leaf_size);

      /// edge and surface downsample
      for (int k = 0; k < pc_out_edge->size(); k++) 
        edge_voxel.insert_point(&pc_out_edge->points[k]);
      std::vector<pcl::PointXYZI*> edge_results = edge_voxel.sample_points();
      for (int k = 0; k < pc_out_loam->size(); k++)  {
        if (edge_or_surface[k] > 0 &&
            edge_inds.find(selected_loam_inds[k]) == edge_inds.end()) {
          surface_voxel.insert_point(&pc_out_loam->points[k]);
        }
      }
      std::vector<pcl::PointXYZI*> surface_results = surface_voxel.sample_points();
      int total_selected_pts_num = edge_results.size() + surface_results.size();    
      std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
      //ret->reserve(total_selected_pts_num, 1, NUM_CLASSES);
      std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
      std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;    

      /// push
      for (int k = 0; k < edge_results.size(); k++) {
        cvo::CvoPoint pt;
        pt.getVector3fMap() = edge_results[k]->getVector3fMap();
        pt.features[0] = edge_results[k]->intensity;
        if (semantics_vec.size()){
          int index = semantics_vec[edge_pt_to_ind[edge_results[k]]];
          if (index == -1)
            continue;
          pt.label_distribution[ index] = 1; 
        }
        pt.geometric_type[0] = 1.0;
        pt.geometric_type[1] = 0.0;
        ret->push_back(pt);
      }
      /// surface downsample
      for (int k = 0; k < surface_results.size(); k++) {
        cvo::CvoPoint pt;
        pt.getVector3fMap() = surface_results[k]->getVector3fMap();
        pt.features[0] = surface_results[k]->intensity;
        if (semantics_vec.size()){
          int index = semantics_vec[surface_pt_to_ind[surface_results[k]]];
          if (index == -1)
            continue;
          pt.label_distribution[index] = 1; 
        }
        pt.geometric_type[0] = 0.0;
        pt.geometric_type[1] = 1.0;
        ret->push_back(pt);
        
      }
      return ret;

    }

  }

  void read_and_downsample_lidar_pc(const std::set<int> & result_selected_frames,
                                    DatasetHandler & dataset,
                                    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & tracking_poses,                                    
                                    int num_merging_sequential_frames,
                                    float edge_voxel_size,
                                    float surface_voxel_size,
                                    int is_edge_only,
                                    int is_semantic,
                                    std::map<int, std::shared_ptr<cvo::CvoPointCloud>> & pcs) {
    std::vector<int> selected_inds;
    for (auto  i : result_selected_frames) selected_inds.push_back(i);

    #pragma omp parallel for
    for (auto it = selected_inds.begin(); it != selected_inds.end(); it++) {
      int i = *it;
      std::cout<<"Processing "<<i<<"\n";
    //for (auto i : result_selected_frames) {
    //for (int i = 0; i<gt_poses.size(); i++) {
      
      pcl::PointCloud<pcl::PointXYZIL>::Ptr pc_edge(new pcl::PointCloud<pcl::PointXYZIL>);
      pcl::PointCloud<pcl::PointXYZIL>::Ptr pc_surface(new pcl::PointCloud<pcl::PointXYZIL>);
      std::vector<int> semantics_local;      
      for (int j = 0; j < 1+num_merging_sequential_frames; j++){

        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZI>);
        std::vector<int> semantics_single;
        int read_result = -1;
        #pragma omp critical                 
        { 
          dataset.set_start_index(i+j);         
          if (is_semantic) {
            read_result = dataset.read_next_lidar(pc_pcl, semantics_single);
          } else {
            read_result = dataset.read_next_lidar(pc_pcl);
          }
        }
        if (read_result == -1) break;
        if (j > 0) {
          Eigen::Matrix4f pose_fi_to_fj = (tracking_poses[i].inverse() * tracking_poses[j+i]).cast<float>();

          for (int k = 0; k < pc_pcl->size(); k++) {
            auto & p = pc_pcl->at(k);
            p.getVector3fMap() = pose_fi_to_fj.block(0,0,3,3) * p.getVector3fMap() + pose_fi_to_fj.block(0,3,3,1);
          }
        }
        std::cout<<"pc before dwonampel size "<<pc_pcl->size()<<"\n";
        select_pc_inds_edge_surface(pc_pcl,
                                    semantics_single,
                                    *pc_edge, *pc_surface);
        std::cout<<"pc after dwonampel size "<<pc_edge->size()<<"\n";        
      }
      //if (i == 0)
      //  pcl::io::savePCDFileASCII(std::to_string(i) + ".pcd", *pc_local);
      //pc_local->write_to_pcd("0.pcd");

      if (pc_edge->size() && pc_surface->size()) {
        std::shared_ptr<cvo::CvoPointCloud> pc = cvo::downsample_edge_surface_with_voxel(*pc_edge,
                                                                                         *pc_surface,
                                                                                         edge_voxel_size,
                                                                                         surface_voxel_size
                                                                                         );

        #pragma omp critical                 
        {
          std::cout<<"new frame "<<i<<" downsampled from  to "<<pc->size()<<"\n";
          pcs.insert(std::make_pair(i, pc));
        }

        /*
        if (i == 0) {
          std::cout<<"is_semantic="<<is_semantic<<"\n";
          if (is_semantic) {
            pcl::PointCloud<pcl::PointXYZIL> pc_local;
            pc_local = *pc_edge + *pc_surface;
            pcl::io::savePCDFileASCII("0_full.pcd", pc_local);
            pc->write_to_label_pcd("0.pcd");
          } else
            pc->write_to_pcd("0.pcd");

        }
        */      
      }
    }
    std::cout<<"Downsample end\n";
    
  }
  


  void read_and_downsample_lidar_pc(const std::set<int> & result_selected_frames,
                                    DatasetHandler & dataset,
 
                                    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & tracking_poses,                                    
                                    int num_merging_sequential_frames,
                                    float voxel_size,
                                    int is_edge_only,
                                    int is_semantic,
                                    std::map<int, std::shared_ptr<cvo::CvoPointCloud>> & pcs) {
    for (auto i : result_selected_frames) {
      //for (int i = 0; i<gt_poses.size(); i++) {
      
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_local(new pcl::PointCloud<pcl::PointXYZI>);
      std::vector<int> semantics_local;      
      for (int j = 0; j < 1+num_merging_sequential_frames; j++){
        dataset.set_start_index(i+j);
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZI>);
        std::vector<int> semantics_single;
        if (is_semantic) {
          if (-1 == dataset.read_next_lidar(pc_pcl, semantics_single))
            break;
        } else {
          if (-1 == dataset.read_next_lidar(pc_pcl)) 
            break;
        }
        if (j > 0) {
          Eigen::Matrix4f pose_fi_to_fj = (tracking_poses[i].inverse() * tracking_poses[j+i]).cast<float>();
          #pragma omp parallel for 
          for (int k = 0; k < pc_pcl->size(); k++) {
            auto & p = pc_pcl->at(k);
            p.getVector3fMap() = pose_fi_to_fj.block(0,0,3,3) * p.getVector3fMap() + pose_fi_to_fj.block(0,3,3,1);
          }
        }
        *pc_local += *pc_pcl;
        semantics_local.insert(semantics_local.end(), semantics_single.begin(), semantics_single.end());
      }
      //if (i == 0)
      //  pcl::io::savePCDFileASCII(std::to_string(i) + ".pcd", *pc_local);
      //pc_local->write_to_pcd("0.pcd");
      float leaf_size = voxel_size;

      if (pc_local->size()) {
        std::shared_ptr<cvo::CvoPointCloud> pc = cvo::downsample_lidar_points(is_edge_only,
                                                                              pc_local,
                                                                              leaf_size,
                                                                              semantics_local);
        std::cout<<"new frame "<<i<<" downsampled from  "<<pc_local->size()<<" to "<<pc->size()<<"\n";
        pcs.insert(std::make_pair(i, pc));
        if (i == 0) {
          std::cout<<"is_semantic="<<is_semantic<<"\n";
          if (is_semantic) {
            cvo::CvoPointCloud pc_full(pc_local, semantics_local, NUM_CLASSES, 5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL);
            pc_full.write_to_label_pcd("0_full.pcd");
            pc->write_to_label_pcd("0.pcd");
          } else
            pc->write_to_pcd("0.pcd");
        }
      }
    }
    
  }
  

}
