set(the_description "Images stitching")

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations -Wshadow -Wstrict-aliasing)
endif()

##########my add start############
# 添加ONNX Runtime依赖  
if(HAVE_ONNXRUNTIME)  
    list(APPEND opencv_stitching_SOURCES src/lightglue.cpp)  
    list(APPEND opencv_stitching_DEPS onnxruntime)    
endif()
##########my add end############
set(STITCHING_CONTRIB_DEPS "opencv_xfeatures2d")
if(BUILD_SHARED_LIBS AND BUILD_opencv_world AND OPENCV_WORLD_EXCLUDE_EXTRA_MODULES)
  set(STITCHING_CONTRIB_DEPS "")
endif()
ocv_define_module(stitching opencv_imgproc opencv_features2d opencv_calib3d opencv_flann
                  OPTIONAL opencv_cudaarithm opencv_cudawarping opencv_cudafeatures2d opencv_cudalegacy opencv_cudaimgproc ${STITCHING_CONTRIB_DEPS}
                  WRAP python)
if(HAVE_ONNXRUNTIME)    
    ocv_target_include_directories(${the_module} SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})    
    ocv_target_link_libraries(${the_module} PRIVATE ${ONNXRUNTIME_LIBRARIES})    
    ocv_target_compile_definitions(${the_module} PRIVATE HAVE_ONNXRUNTIME)    
endif()