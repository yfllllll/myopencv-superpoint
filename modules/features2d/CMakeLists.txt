set(the_description "2D Features Framework")

#################my add start###############
# 添加ONNX Runtime依赖检查  
find_package(onnxruntime QUIET)  
if(onnxruntime_FOUND)  
    set(HAVE_ONNXRUNTIME TRUE)  
    list(APPEND opencv_features2d_DEPS onnxruntime)  
endif()  
  
# 添加源文件  
if(HAVE_ONNXRUNTIME)  
    list(APPEND opencv_features2d_SOURCES src/superpoint.cpp) 
    list(APPEND opencv_features2d_DEPS onnxruntime) 
endif()
###################my add end##############################

ocv_add_dispatched_file(sift SSE4_1 AVX2 AVX512_SKX)

set(debug_modules "")
if(DEBUG_opencv_features2d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(features2d opencv_imgproc ${debug_modules} OPTIONAL opencv_flann WRAP java objc python js)

ocv_install_3rdparty_licenses(mscr "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mscr/chi_table_LICENSE.txt")
