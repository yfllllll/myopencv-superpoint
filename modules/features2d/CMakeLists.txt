set(the_description "2D Features Framework")



ocv_add_dispatched_file(sift SSE4_1 AVX2 AVX512_SKX)

set(debug_modules "")
if(DEBUG_opencv_features2d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(features2d opencv_imgproc ${debug_modules} OPTIONAL opencv_flann WRAP java objc python js)

ocv_install_3rdparty_licenses(mscr "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mscr/chi_table_LICENSE.txt")

#################my add start###############
# 添加ONNX Runtime依赖检查  
# find_package(onnxruntime QUIET)  
# if(onnxruntime_FOUND)  
#     set(HAVE_ONNXRUNTIME TRUE)  
#     list(APPEND opencv_features2d_DEPS onnxruntime)  
# endif()  
  
# 添加源文件  
if(WITH_ONNXRUNTIME AND HAVE_ONNXRUNTIME)
  ocv_target_link_libraries(${the_module} PRIVATE ${ONNXRUNTIME_LIBRARIES})  
  ocv_target_compile_definitions(${the_module} PRIVATE HAVE_ONNXRUNTIME=1)  
  ocv_target_include_directories(${the_module} SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})  
endif()
###################my add end##############################