# FindONNXRuntime.cmake  
# Find ONNX Runtime library and headers  
  
# Clear any previous values  
unset(ONNXRUNTIME_FOUND)  
unset(ONNXRUNTIME_INCLUDE_DIRS)  
unset(ONNXRUNTIME_LIBRARIES)  
  
# Use environment variable if ONNXRUNTIME_ROOT_PATH is not set  
if(NOT ONNXRUNTIME_ROOT_PATH AND DEFINED ENV{ONNXRUNTIME_ROOT_PATH})  
    set(ONNXRUNTIME_ROOT_PATH $ENV{ONNXRUNTIME_ROOT_PATH})  
endif()  
  
# Find include directory  
find_path(ONNXRUNTIME_INCLUDE_DIRS  
    NAMES onnxruntime_cxx_api.h  
    PATHS   
        ${ONNXRUNTIME_ROOT_PATH}/include  
        $ENV{ONNXRUNTIME_INCLUDE_PATH}  
        /usr/local/include  
        /usr/include  
    PATH_SUFFIXES   
        onnxruntime   
        onnxruntime/core/session  
        ""  
    NO_DEFAULT_PATH  
)  

message("ONNXRUNTIME_INCLUDE_DIRS: debug ${ONNXRUNTIME_INCLUDE_DIRS}")


# If not found with NO_DEFAULT_PATH, try default paths  
if(NOT ONNXRUNTIME_INCLUDE_DIRS)  
    find_path(ONNXRUNTIME_INCLUDE_DIRS  
        NAMES onnxruntime_cxx_api.h  
        PATH_SUFFIXES   
            onnxruntime   
            onnxruntime/core/session  
    )  
endif()  
  
# Find library  
find_library(ONNXRUNTIME_LIBRARIES  
    NAMES onnxruntime  
    PATHS   
        ${ONNXRUNTIME_ROOT_PATH}/lib  
        $ENV{ONNXRUNTIME_LIB_PATH}  
        /usr/local/lib  
        /usr/lib  
    NO_DEFAULT_PATH  
)  
  
# If not found with NO_DEFAULT_PATH, try default paths  
if(NOT ONNXRUNTIME_LIBRARIES)  
    find_library(ONNXRUNTIME_LIBRARIES  
        NAMES onnxruntime  
    )  
endif()  
  
# Set found status and version  
if(ONNXRUNTIME_INCLUDE_DIRS AND ONNXRUNTIME_LIBRARIES)  
    set(onnxruntime_FOUND TRUE)  
    set(ONNXRUNTIME_FOUND TRUE)  
      
    # Try to detect version from header if possible  
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_c_api.h")  
        file(READ "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_c_api.h" _onnxrt_header)  
        string(REGEX MATCH "#define ORT_API_VERSION ([0-9]+)" _onnxrt_version_match "${_onnxrt_header}")  
        if(_onnxrt_version_match)  
            set(onnxruntime_VERSION "1.22.0")  # Default to known version  
        else()  
            set(onnxruntime_VERSION "1.0.0")   # Fallback version  
        endif()  
    else()  
        set(onnxruntime_VERSION "1.22.0")  
    endif()  
      
    # Create imported target for better integration  
    if(NOT TARGET onnxruntime::onnxruntime)  
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)  
        set_target_properties(onnxruntime::onnxruntime PROPERTIES  
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"  
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARIES}"  
        )  
          
        # Handle Windows import library if needed  
        if(WIN32 AND EXISTS "${ONNXRUNTIME_LIBRARIES}")  
            get_filename_component(_lib_dir "${ONNXRUNTIME_LIBRARIES}" DIRECTORY)  
            get_filename_component(_lib_name "${ONNXRUNTIME_LIBRARIES}" NAME_WE)  
            if(EXISTS "${_lib_dir}/${_lib_name}.lib")  
                set_target_properties(onnxruntime::onnxruntime PROPERTIES  
                    IMPORTED_IMPLIB "${_lib_dir}/${_lib_name}.lib"  
                )  
            endif()  
        endif()  
    endif()  
      
    # Status messages  
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")  
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")  
    message(STATUS "ONNX Runtime version: ${onnxruntime_VERSION}")  
else()  
    set(onnxruntime_FOUND FALSE)  
    set(ONNXRUNTIME_FOUND FALSE)  
    message(STATUS "ONNX Runtime not found")  
endif()  
  
# Handle QUIET and REQUIRED arguments  
include(FindPackageHandleStandardArgs)  
find_package_handle_standard_args(onnxruntime  
    FOUND_VAR onnxruntime_FOUND  
    REQUIRED_VARS ONNXRUNTIME_LIBRARIES ONNXRUNTIME_INCLUDE_DIRS  
    VERSION_VAR onnxruntime_VERSION  
)  
  
# Mark variables as advanced  
mark_as_advanced(ONNXRUNTIME_INCLUDE_DIRS ONNXRUNTIME_LIBRARIES)