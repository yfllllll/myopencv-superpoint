#ifndef OPENCV_FEATURES2D_SUPERPOINT_HPP  
#define OPENCV_FEATURES2D_SUPERPOINT_HPP  
  
#include "opencv2/features2d.hpp"  
#include "opencv2/core.hpp"  
#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif
#include <iostream> 
#include "opencv2/imgproc.hpp"
namespace cv {  
  
/** @brief SuperPoint feature detector and descriptor extractor.  
  
SuperPoint is a self-supervised framework for training interest point detectors and descriptors  
suitable for a large number of multiple-view geometry problems in computer vision.  
  
@note This implementation requires ONNX Runtime and a pre-trained SuperPoint model file.  
*/  
class CV_EXPORTS_W SuperPoint : public Feature2D  
{  
public:  
    /** @brief Constructor  
    @param modelPath Path to the SuperPoint ONNX model file  
    */  
    CV_WRAP SuperPoint(const String& modelPath = String());  
  
    /** @brief Creates a SuperPoint feature detector  
    @param modelPath Path to the SuperPoint ONNX model file  
    @return Pointer to the created SuperPoint instance  
    */  
    CV_WRAP static Ptr<SuperPoint> create(const String& modelPath = String());  
  
    /** @brief Detects keypoints and computes descriptors  
    @param image Input image (preferably grayscale)  
    @param mask Optional mask specifying where to look for keypoints  
    @param keypoints Detected keypoints  
    @param descriptors Computed descriptors (256-dimensional)  
    @param useProvidedKeypoints If true, detection is skipped and only descriptors are computed  
    */  
    CV_WRAP virtual void detectAndCompute(InputArray image, InputArray mask,  
        CV_OUT std::vector<KeyPoint>& keypoints,  
        OutputArray descriptors,  
        bool useProvidedKeypoints = false) CV_OVERRIDE;  
  
    /** @brief Detects keypoints in an image  
    @param image Input image (preferably grayscale)  
    @param keypoints Detected keypoints  
    @param mask Optional mask specifying where to look for keypoints  
    */  
    CV_WRAP virtual void detect(InputArray image,  
        CV_OUT std::vector<KeyPoint>& keypoints,  
        InputArray mask = noArray()) CV_OVERRIDE;  
  
    /** @brief Computes descriptors for given keypoints  
    @param image Input image (preferably grayscale)  
    @param keypoints Input keypoints  
    @param descriptors Computed descriptors (256-dimensional)  
    */  
    CV_WRAP virtual void compute(InputArray image,  
        CV_IN_OUT std::vector<KeyPoint>& keypoints,  
        OutputArray descriptors) CV_OVERRIDE;  
  
protected:  
    String m_modelPath;  
      
    /** @brief Applies image preprocessing for neural network input  
    @param image Input grayscale image  
    @param mean Reference to store computed mean (currently unused)  
    @param std Reference to store computed std (currently unused)  
    @return Preprocessed image data as float vector  
    */  
    std::vector<float> ApplyTransform(const Mat& image, float& mean, float& std);  
#ifdef HAVE_ONNXRUNTIME
    Ort::SessionOptions& createSessionOptions() {
        static Ort::SessionOptions sessionOptions = []() {  
            Ort::SessionOptions options;  
            options.SetIntraOpNumThreads(1);    
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);    
                
            // 尝试添加CUDA执行提供程序    
            try {    
                OrtCUDAProviderOptions cuda_options{};    
                cuda_options.device_id = 0;    
                options.AppendExecutionProvider_CUDA(cuda_options);    
                std::cout << "SuperPoint: Using GPU (CUDA) acceleration" << std::endl;    
            } catch (const std::exception& e) {    
                std::cout << "SuperPoint: CUDA not available, falling back to CPU: " << e.what() << std::endl;    
                // CPU是默认的，不需要额外配置    
            }    
            return options;  
        }();  
        return sessionOptions;    
    }
    Ort::MemoryInfo createMemoryInfo() {  
        // 尝试使用GPU内存，失败则回退到CPU  
            return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);  

    }
    #endif
};  
  
} // namespace cv  
  
#endif // OPENCV_FEATURES2D_SUPERPOINT_HPP