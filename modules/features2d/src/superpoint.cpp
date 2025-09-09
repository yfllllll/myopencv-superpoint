#ifdef HAVE_ONNXRUNTIME
#include "opencv2/superpoint.hpp"
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <locale>
#include <codecvt>
#endif

namespace cv {

SuperPoint::SuperPoint(const String& modelPath)
    : m_modelPath(modelPath), m_initialized(false)
{
}

Ptr<SuperPoint> SuperPoint::create(const String& modelPath)
{
    return makePtr<SuperPoint>(modelPath);
}

std::vector<float> SuperPoint::ApplyTransform(const Mat& image, float& mean, float& std)
{
    Mat floatImage;
    image.convertTo(floatImage, CV_32FC1, 1.0 / 255.0);

    std::vector<float> imgData(image.rows * image.cols);
    if (floatImage.isContinuous()) {
        memcpy(imgData.data(), floatImage.data, image.rows * image.cols * sizeof(float));
    } else {
        for (int h = 0; h < image.rows; ++h) {
            memcpy(imgData.data() + h * image.cols, floatImage.ptr<float>(h), image.cols * sizeof(float));
        }
    }
    return imgData;
}

Ort::SessionOptions SuperPoint::createSessionOptions() const {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4); // Increased for potentially better CPU performance
    options.SetInterOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "SuperPoint: Using GPU (CUDA) acceleration" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "SuperPoint: CUDA not available, falling back to CPU: " << e.what() << std::endl;
    }
    return options;
}

Ort::MemoryInfo SuperPoint::createMemoryInfo() const {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

void SuperPoint::initialize() const {
    if (m_initialized) {
        return;
    }
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
    Ort::SessionOptions sessionOptions = createSessionOptions();

#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::wstring modelPathW = converter.from_bytes(m_modelPath.c_str());
    m_session = std::make_unique<Ort::Session>(*m_env, modelPathW.c_str(), sessionOptions);
#else
    m_session = std::make_unique<Ort::Session>(*m_env, m_modelPath.c_str(), sessionOptions);
#endif
    m_initialized = true;
}

void SuperPoint::detectAndCompute(InputArray image, InputArray mask,
    std::vector<KeyPoint>& keypoints,
    OutputArray descriptors,
    bool useProvidedKeypoints)
{
    initialize();

    Mat img = image.getMat();
    Mat grayImg;
    if (img.channels() == 3)
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
    else
        grayImg = img;

    float mean, std;
    std::vector<float> imgData = ApplyTransform(grayImg, mean, std);

    std::vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

    Ort::MemoryInfo memoryInfo = createMemoryInfo();
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

    const char* input_names[] = { "image" };
    const char* output_names[] = { "keypoints","scores","descriptors" };
    Ort::RunOptions run_options;
    std::vector<Ort::Value> outputs = m_session->Run(run_options, input_names, &inputTensor, 1, output_names, 3);

    std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t* kp = (int64_t*)outputs[0].GetTensorMutableData<void>();
    int keypntcounts = static_cast<int>(kpshape[1]);
    keypoints.resize(keypntcounts);
    for (int i = 0; i < keypntcounts; i++)
    {
        KeyPoint& p = keypoints[i];
        int index = i * 2;
        p.pt.x = static_cast<float>(kp[index]);
        p.pt.y = static_cast<float>(kp[index + 1]);
    }

    std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    float* des = (float*)outputs[2].GetTensorMutableData<void>();

    // Optimized descriptor copy
    Mat desmat(static_cast<int>(desshape[1]), static_cast<int>(desshape[2]), CV_32F, des);
    
    if (!mask.empty()) {
        Mat originalDescriptors;
        desmat.copyTo(originalDescriptors);

        std::vector<int> validIndices;
        Mat maskMat = mask.getMat();
        for (int i = 0; i < keypoints.size(); i++) {
            const KeyPoint& kp_ = keypoints[i];
            if (maskMat.at<uchar>((int)(kp_.pt.y + 0.5f), (int)(kp_.pt.x + 0.5f)) != 0) {
                validIndices.push_back(i);
            }
        }

        KeyPointsFilter::runByPixelsMask(keypoints, maskMat);

        if (descriptors.needed() && !validIndices.empty()) {
            Mat filteredDescriptors(validIndices.size(), originalDescriptors.cols, originalDescriptors.type());
            for (size_t i = 0; i < validIndices.size(); i++) {
                originalDescriptors.row(validIndices[i]).copyTo(filteredDescriptors.row(i));
            }
            filteredDescriptors.copyTo(descriptors);
        } else if (descriptors.needed()) {
            descriptors.clear();
        }
    } else {
        desmat.copyTo(descriptors);
    }
}

void SuperPoint::detect(InputArray image,
    std::vector<KeyPoint>& keypoints,
    InputArray mask)
{
    Mat descriptors; // Dummy descriptor mat
    detectAndCompute(image, mask, keypoints, descriptors, false);
}

void SuperPoint::compute(InputArray image,
    std::vector<KeyPoint>& keypoints,
    OutputArray descriptors)
{
    // This function is inefficient as it re-runs detection. 
    // For optimal performance, use detectAndCompute.
    std::vector<KeyPoint> all_keypoints;
    detectAndCompute(image, noArray(), all_keypoints, descriptors, false);
}

} // namespace cv
#endif // HAVE_ONNXRUNTIME