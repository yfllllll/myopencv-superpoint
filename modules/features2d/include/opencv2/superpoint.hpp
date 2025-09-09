#ifndef OPENCV_FEATURES2D_SUPERPOINT_HPP
#define OPENCV_FEATURES2D_SUPERPOINT_HPP

#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <memory>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace cv {

/** @brief SuperPoint feature detector and descriptor extractor.
 *
 * This class implements the SuperPoint algorithm, a self-supervised framework for
 * training interest point detectors and descriptors suitable for a wide range of
 * multiple-view geometry problems in computer vision.
 *
 * @note This implementation requires the ONNX Runtime library and a pre-trained
 * SuperPoint model file in .onnx format.
 */
class CV_EXPORTS_W SuperPoint : public Feature2D
{
public:
    /** @brief Constructs a SuperPoint detector.
     * @param modelPath Path to the ONNX model file.
     */
    CV_WRAP explicit SuperPoint(const String& modelPath = String());

    /** @brief Destructor. */
    virtual ~SuperPoint() = default;

    /** @brief Creates a pointer to a SuperPoint instance.
     * @param modelPath Path to the ONNX model file.
     * @return A Ptr<SuperPoint> object.
     */
    CV_WRAP static Ptr<SuperPoint> create(const String& modelPath = String());

    /** @brief Detects keypoints and computes their descriptors.
     *
     * @param image Input image, preferably grayscale.
     * @param mask Optional mask to specify the region of interest.
     * @param keypoints The detected keypoints.
     * @param descriptors The computed descriptors (256-dimensional).
     * @param useProvidedKeypoints If true, keypoint detection is skipped and descriptors are computed for the provided keypoints.
     */
    CV_WRAP virtual void detectAndCompute(InputArray image, InputArray mask,
        CV_OUT std::vector<KeyPoint>& keypoints,
        OutputArray descriptors,
        bool useProvidedKeypoints = false) CV_OVERRIDE;

    /** @brief Detects keypoints in an image.
     *
     * @param image Input image, preferably grayscale.
     * @param keypoints The detected keypoints.
     * @param mask Optional mask to specify the region of interest.
     */
    CV_WRAP virtual void detect(InputArray image,
        CV_OUT std::vector<KeyPoint>& keypoints,
        InputArray mask = noArray()) CV_OVERRIDE;

    /** @brief Computes descriptors for a set of keypoints.
     *
     * @param image Input image, preferably grayscale.
     * @param keypoints Input keypoints for which descriptors need to be computed.
     * @param descriptors The computed descriptors (256-dimensional).
     */
    CV_WRAP virtual void compute(InputArray image,
        CV_IN_OUT std::vector<KeyPoint>& keypoints,
        OutputArray descriptors) CV_OVERRIDE;

protected:
    /** @brief Initializes the ONNX Runtime session if it hasn't been already.
     *
     * This method is called on-demand before the first inference call.
     */
    void initializeSession() const;

    /** @brief Preprocesses the input image for the neural network.
     *
     * @param image Input grayscale image.
     * @param mean Reference to store the computed mean (currently unused).
     * @param std Reference to store the computed standard deviation (currently unused).
     * @return A vector of floats representing the preprocessed image data.
     */
    std::vector<float> preprocessImage(const Mat& image, float& mean, float& std);

    String modelPath_;

    // ONNX Runtime session management members.
    // 'mutable' allows lazy initialization within const methods.
    mutable std::unique_ptr<Ort::Env> env_;
    mutable std::unique_ptr<Ort::Session> session_;
    mutable std::unique_ptr<Ort::MemoryInfo> memoryInfo_;
    mutable bool initialized_;

#ifdef HAVE_ONNXRUNTIME
private:
    /** @brief Creates and configures ONNX Runtime session options.
     *
     * Sets up threading and attempts to enable CUDA for hardware acceleration.
     * @return Configured Ort::SessionOptions object.
     */
    Ort::SessionOptions createSessionOptions() {
        Ort::SessionOptions sessionOptions;

        // Set thread counts to improve performance through parallelism.
        // Intra-op threads execute a single operator in parallel.
        sessionOptions.SetIntraOpNumThreads(4);
        // Inter-op threads execute different operators in parallel.
        sessionOptions.SetInterOpNumThreads(2);

        // Enable all available graph optimizations.
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Attempt to append the CUDA execution provider for GPU acceleration.
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.gpu_mem_limit = SIZE_MAX; // Use the maximum available GPU memory.
            cuda_options.arena_extend_strategy = 1; // kNextPowerOfTwo
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "SuperPoint: Using GPU (CUDA) for acceleration." << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "SuperPoint: CUDA is not available, falling back to CPU. Reason: " << e.what() << std::endl;
        }

        return sessionOptions;
    }

    /** @brief Creates an ONNX Runtime memory info object.
     *
     * Specifies that tensors should be allocated on the CPU.
     * @return An Ort::MemoryInfo object.
     */
    Ort::MemoryInfo createMemoryInfo() {
        // Fallback is provided for different allocator configurations.
        try {
            return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        }
        catch (const std::exception&) {
            return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        }
    }
#endif // HAVE_ONNXRUNTIME
};

} // namespace cv

#endif // OPENCV_FEATURES2D_SUPERPOINT_HPP