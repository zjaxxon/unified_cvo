cd build
make -j
cd ..
./build/bin/cvo_align_two_nn_feature e2pn_invariant_features/cvo_original_pointcloud_points.pcd e2pn_invariant_features/cvo_transformed_pointcloud_points.pcd e2pn_invariant_features/cvo_original_pointcloud_invariant_feature.bin e2pn_invariant_features/cvo_transformed_pointcloud_invariant_feature.bin  cvo_params/cvo_semantic_params_deep_features.yaml 2.0
