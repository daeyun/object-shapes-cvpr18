add_executable(generate_dataset generate_dataset.cc)
target_link_libraries(generate_dataset mvshape_cpp_core data_io database egl_rendering)

add_executable(generate_metadata generate_metadata.cc)
target_link_libraries(generate_metadata mvshape_cpp_core data_io)

#add_executable(train train.cc)
#target_link_libraries(train mvshape_cpp_core data_io database
#        tf_utils
#        mvshape_helpers)

#add_executable(eval eval.cc)
#target_link_libraries(eval mvshape_cpp_core data_io database
#        tf_utils
#        mvshape_helpers)

add_executable(depth2mesh depth2mesh.cc)
target_link_libraries(depth2mesh mvshape_cpp_core data_io mve)

add_executable(scene scene.cc)
target_link_libraries(scene mvshape_cpp_core data_io mve fssr_utils)

add_executable(mesh_distance meshdist.cc)
target_link_libraries(mesh_distance mvshape_cpp_core data_io mve fssr_utils meshdist)

# TODO
#add_executable(recon recon.cc)
#target_link_libraries(recon mvshape_cpp_core data_io mve fssr_utils meshdist)

add_executable(voxel_iou voxel_iou.cc)
target_link_libraries(voxel_iou mvshape_cpp_core data_io)
