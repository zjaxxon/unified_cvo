#include <iostream>
#include <string>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "dataset_handler/PcdHandler.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <map>

using namespace std;
using namespace boost::filesystem;

namespace cvo {
    PcdHandler::PcdHandler(std::string pcd_folder) {
      curr_index = 0;
      folder_name = pcd_folder;

      path pcd_path(pcd_folder.c_str());
      for (auto &p: directory_iterator(pcd_path)) {
        if (is_regular_file(p.path())) {
          string curr_file = p.path().filename().string();
          size_t last_ind = curr_file.find_last_of(".");
          string raw_name = curr_file.substr(0, last_ind);
          names.push_back(raw_name);
        }
      }
      sort(names.begin(), names.end());
      cout << "Source path contains " << names.size() << " pcds\n";
    }

    PcdHandler::~PcdHandler() {}

    int PcdHandler::read_next_pcd(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) {
      if (curr_index >= names.size())
        return -1;

      string pcd_path = folder_name + "/" + names[curr_index] + ".pcd";
      pcl::io::loadPCDFile(pcd_path, *pc);
    }

    void PcdHandler::next_frame_index() {
      curr_index ++;
    }


    void PcdHandler::set_start_index(int start) {
      curr_index = start;
    }

    int PcdHandler::get_current_index() {
      return curr_index;
    }

    int PcdHandler::get_total_number() {
      return names.size();
    }
}