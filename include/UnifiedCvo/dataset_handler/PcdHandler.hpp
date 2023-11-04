#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "DataHandler.hpp"

namespace cvo{

    class PcdHandler : public DatasetHandler {
    public:

        PcdHandler(std::string pcd_path);
        ~PcdHandler();

        int read_next_pcd(pcl::PointCloud<pcl::PointXYZI>::Ptr pc);

        void next_frame_index();
        void set_start_index(int start);
        int get_current_index();
        int get_total_number();
    private:

        int curr_index;
        std::vector<std::string> names;
        std::string folder_name;
    };

}
