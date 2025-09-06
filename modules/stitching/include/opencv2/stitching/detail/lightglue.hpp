#ifndef OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP  
#define OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP  
  
#include "opencv2/stitching/detail/matchers.hpp"  
#include "opencv2/stitching.hpp"  
#include "opencv2/core.hpp"  
#include <onnxruntime_cxx_api.h>  // 添加这一行  
#include <iostream>
#include "opencv2/imgproc.hpp"
namespace cv {  
namespace detail {  
  
/** @brief LightGlue feature matcher.  
  
LightGlue is a lightweight feature matcher with high accuracy, based on adaptive pruning  
of keypoint and match candidates. It works particularly well with SuperPoint features.  
  
@note This implementation requires ONNX Runtime and a pre-trained LightGlue model file.  
*/  
class CV_EXPORTS LightGlue : public FeaturesMatcher  
{  
public:  
    /** @brief Constructor  
    @param modelPath Path to the LightGlue ONNX model file  
    @param mode Stitching mode (PANORAMA for perspective, SCANS for affine transformation)  
    @param matchThresh Confidence threshold for match validation (0.0 means no threshold)  
    */  
    CV_WRAP LightGlue(const String& modelPath = String(),   
                      Stitcher::Mode mode = Stitcher::PANORAMA,   
                      float matchThresh = 0.0f);  
  
    /** @brief Creates a LightGlue feature matcher  
    @param modelPath Path to the LightGlue ONNX model file  
    @param mode Stitching mode (PANORAMA for perspective, SCANS for affine transformation)  
    @param matchThresh Confidence threshold for match validation  
    @return Pointer to the created LightGlue instance  
    */  
    CV_WRAP static Ptr<LightGlue> create(const String& modelPath = String(),  
                                         Stitcher::Mode mode = Stitcher::PANORAMA,  
                                         float matchThresh = 0.0f);  
  
    /** @brief Matches features between two images  
    @param features1 Features from the first image  
    @param features2 Features from the second image  
    @param matches_info Output matching information including homography  
    */  
    CV_WRAP void match(const ImageFeatures& features1, const ImageFeatures& features2,  
                       MatchesInfo& matches_info) CV_OVERRIDE;  
  
    /** @brief Functional interface for matching (calls match internally)  
    @param features1 Features from the first image  
    @param features2 Features from the second image  
    @param matches_info Output matching information  
    */  
    CV_WRAP_AS(apply) void operator()(const ImageFeatures& features1, const ImageFeatures& features2,  
        CV_OUT MatchesInfo& matches_info) {  
        match(features1, features2, matches_info);  
    }  
  
    /** @brief Get cached features from all processed images  
    @return Vector of all processed image features  
    */  
    CV_WRAP std::vector<ImageFeatures> features() const { return features_; }  
  
    /** @brief Get cached match information from all processed image pairs  
    @return Vector of all computed pairwise matches  
    */  
    CV_WRAP std::vector<MatchesInfo> matchinfo() const { return pairwise_matches_; }  
  
protected:  
    Stitcher::Mode m_mode;          ///< Transformation mode (affine or perspective)  
    String m_modelPath;             ///< Path to ONNX model file  
    std::vector<ImageFeatures> features_;           ///< Cache of processed image features  
    std::vector<MatchesInfo> pairwise_matches_;     ///< Cache of computed matches  
    float m_matchThresh;            ///< Match confidence threshold  
  
    /** @brief Adds image features to internal cache (prevents duplicates)  
    @param features Image features to add  
    */  
    void AddFeature(const ImageFeatures& features);  
  
    /** @brief Adds match information to internal cache (prevents duplicates)  
    @param matches_info Match information to add  
    */  
    void AddMatcheinfo(const MatchesInfo& matches_info);  
    Ort::SessionOptions createSessionOptions() {  
        Ort::SessionOptions sessionOptions;  
        sessionOptions.SetIntraOpNumThreads(1);  
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  
          
        // 尝试添加CUDA执行提供程序  
        try {  
            OrtCUDAProviderOptions cuda_options{};  
            cuda_options.device_id = 0;  
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);  
            std::cout << "SuperPoint: Using GPU (CUDA) acceleration" << std::endl;  
        } catch (const std::exception& e) {  
            std::cout << "SuperPoint: CUDA not available, falling back to CPU: " << e.what() << std::endl;  
            // CPU是默认的，不需要额外配置  
        }  
          
        return sessionOptions;  
    }  
      
    Ort::MemoryInfo createMemoryInfo() {  
        // 尝试使用GPU内存，失败则回退到CPU  
        try {  
            return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);  
        } catch (const std::exception& e) {  
            return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);  
        }  
    }
};  
  
} // namespace detail  
} // namespace cv  
  
#endif // OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP