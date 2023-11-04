#include "dataset_handler/PcdHandler.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "utils/CvoPointCloud.hpp"

using namespace std;

int main(int argc, char *argv[]) {
  cvo::PcdHandler pcdHandler(argv[1]);
  int total_iters = pcdHandler.get_total_number();
  pcdHandler.set_start_index(0);

  cout << "Total iters: " << total_iters << "\n";
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
  pcdHandler.read_next_pcd(source_pc);
  std::cout<<"read next lidar\n";
  cvo::CvoPointCloud source(source_pc, 5000, 64);
  std::cout<<"Num of source pts is "<<source.num_points()<<"\n";
}